from datetime import datetime, timezone
import json
import pandas as pd

from strategy.utils_hashing import hash_dataframe
from trading_analysis.db import get_first_candle_from_db, load_ohlcv_from_db, save_model_run
from trading_analysis.feature_selection import select_important_features
from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.risk import calculate_inverse_balance_risk
from trading_analysis.signals import generate_signals_cached
from trading_analysis.backtest import run_backtest
from trading_analysis.charts import plot_backtest_progress

from strategy.objective import estimate_window_size_from_params, optimize_with_validation
from strategy.search_space import search_space
from trading_analysis.utils import prepare_params_for_logging, sanitize_params, split_train_val

def initialize_test(symbol: str, interval: str = "30") -> dict:
    """
    Инициализирует конфигурацию для walk-forward теста.
    Возвращает словарь с параметрами, необходимыми для основного цикла.
    """
    step_candles = int(24 * 60 / int(interval))
    ms_per_candle = int(interval) * 60_000
    last_candle = get_first_candle_from_db(symbol, interval)
    first_ts = int(last_candle.timestamp.timestamp() * 1000)
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "symbol": symbol,
        "interval": interval,
        "window_size": 2000,
        "step_candles": step_candles,
        "ms_per_candle": ms_per_candle,
        "first_ts": first_ts,
        "now_ts": now_ts,
        "total_pnl": 0.0,
        "days_elapsed": 0,
        "trade_log": [],
        "initial_balance": 1000.0,
        "balance": 1000.0,
        "best_params": None,
        "win_streak": 0,
        "risk_history": [],
        "search_space": search_space,
        "open_position": None,
    }

def load_initial_train_data(symbol: str, window_size: int, start_timestamp: int, interval: str):
    """
    Загружает начальные обучающие данные из базы данных по символу.

    :param symbol: Название монеты, например 'BTCUSDT'
    :param window_size: Количество свечей в обучающем окне
    :param start_timestamp: Метка времени начала (в миллисекундах)
    :return: DataFrame с данными
    """
    df = load_ohlcv_from_db(
        symbol=symbol,
        limit=window_size,
        start_timestamp=start_timestamp,
        interval=interval
    )
    return df

def is_end_of_data(test_end_ts: int) -> bool:
    """
    Проверяет, достигли ли конца доступных данных.

    :param test_end_ts: Конец тестового окна в миллисекундах
    :return: True, если больше нет доступных данных
    """
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    return test_end_ts > now_ts

def calculate_test_range(df_train, ms_per_candle, step_candles):
    """
    Вычисляет начало и конец тестового окна на основе df_train.

    :param df_train: DataFrame с историей для обучения
    :param ms_per_candle: длительность одной свечи в миллисекундах
    :param step_candles: количество свечей в тестовом окне
    :return: (test_start_ts, test_end_ts) — временные метки начала и конца теста
    """
    last_ts_train = int(df_train.index[-1].timestamp() * 1000)
    test_start_ts = last_ts_train + ms_per_candle
    test_end_ts = test_start_ts + step_candles * ms_per_candle

    return test_start_ts, test_end_ts

def load_test_window_from_db(symbol: str, interval: str, test_range: tuple) -> pd.DataFrame:
    """
    Загружает тестовое окно свечей из локальной БД.

    :param symbol: тикер, например "PRIMEUSDT"
    :param interval: строка с интервалом, например "30"
    :param test_range: кортеж (start_timestamp_ms, end_timestamp_ms)
    :return: DataFrame со свечами за указанный диапазон
    """
    start_ts, end_ts = test_range
    df = load_ohlcv_from_db(symbol, interval=interval, start_timestamp=start_ts, end_timestamp=end_ts)

    if df.empty:
        print("⚠ Нет данных из БД для указанного тестового диапазона.")
        return None

    return df

def prepare_test_data(df_train: pd.DataFrame, df_test: pd.DataFrame, best_params: dict) -> pd.DataFrame:
    """
    Объединяет тренировочные и тестовые данные, применяет индикаторы и сигналы,
    и возвращает подготовленные данные только для тестового окна.
    """
    df_full = pd.concat([df_train, df_test])
    sanitized_params = sanitize_params(best_params)

    # Генерируем хэши
    df_hash = hash_dataframe(df_full)
    params_serialized = json.dumps(sanitized_params, sort_keys=True)

    # Вызываем кэш-функции
    df_full = calculate_indicators_cached(df_hash, df_full, sanitized_params)
    df_full = generate_signals_cached(df_full, params_serialized)

    # 🔍 Отбор признаков только один раз, если их ещё нет
    if "enabled_long_signals" not in best_params or "enabled_short_signals" not in best_params:
        print("🧠 Отбираем важные признаки с mutual_info_classif...")
        
        # Long
        feature_cols_long = [
            col for col in df_full.columns
            if col.startswith("long_") and "_score" not in col and "_entry" not in col
        ]
        important_feats_long, _ = select_important_features(df_full, feature_cols_long, target_col="long_entry")
        best_params["enabled_long_signals"] = important_feats_long
        print("📌 Важные long признаки:", important_feats_long)

        # Short
        feature_cols_short = [
            col for col in df_full.columns
            if col.startswith("short_") and "_score" not in col and "_entry" not in col
        ]
        important_feats_short, _ = select_important_features(df_full, feature_cols_short, target_col="short_entry")
        best_params["enabled_short_signals"] = important_feats_short
        print("📌 Важные short признаки:", important_feats_short)

    return df_full.iloc[-len(df_test):]

def run_evaluation(df_test_prepared, symbol: str, current_balance: float, risk_pct: float, open_position: dict = None) -> tuple:
    """
    Проводит бэктест на тестовом окне и возвращает результат и обновлённый баланс.

    :param df_test_prepared: DataFrame с тестовыми данными и сигналами
    :param symbol: тикер символа (например, "PRIMEUSDT")
    :param current_balance: текущий баланс на момент входа в период
    :param risk_pct: риск на сделку в долях (например, 0.05 = 5%)
    :return: (результат бэктеста, новый баланс)
    """
    result, final_state = run_backtest(
        df_test_prepared,
        symbol=symbol,
        report=True,
        finalize=False,
        initial_balance=current_balance,
        leverage=1,
        risk_pct=risk_pct,
        open_position=open_position
    )

    if final_state["position_type"]:
        new_open_position = {
            "position_type": final_state["position_type"],
            "entry_price": final_state["entry_price"],
            "position_size": final_state["position"],
            "trades": final_state["trades"]
        }
    else:
        new_open_position = None

    new_balance = current_balance + result["pnl"]
    return result, new_balance, new_open_position

def update_tracking(config: dict, interval, result: dict, df_test, df_test_prepared):
    """
    Обновляет журнал трейдов, сохраняет результаты модели и метрики.

    :param config: конфиг словарь со всем текущим состоянием
    :param result: результат бэктеста
    :param df_test: оригинальный DataFrame теста (без индикаторов)
    :param df_test_prepared: подготовленный DataFrame теста (с индикаторами)
    """
    trade_log = config["trade_log"]
    symbol = config["symbol"]
    risk_pct = config["risk_pct"]
    best_params = config["best_params"]
    balance = config["balance"]
    retrained = result.get("pnl", 0) <= 0
    days_elapsed = config.get("days_elapsed", 0) + 1

    test_date = df_test_prepared.index[0].to_pydatetime()

    trade_log.append({
        "date": test_date.strftime("%Y-%m-%d"),
        "pnl": result["pnl"]
    })
    params_clean = prepare_params_for_logging(config["best_params"], config)
    save_model_run(
        symbol=symbol,
        interval=interval,
        date=test_date,
        params=params_clean,
        loss=-result["winrate"],
        pnl=result["pnl"],
        total_trades=result["total_trades"],
        winrate=result["winrate"],
        risk_pct=risk_pct,
        retrained=retrained
    )

    config["trade_log"] = trade_log
    config["days_elapsed"] = days_elapsed

def update_training_window(df_train, df_test, step_candles):
    """
    Обновляет обучающее окно, гарантируя отсутствие пересечений с df_test.
    """
    print("🧪 Текущий конец df_train:", df_train.index[-1])
    print("🧪 Начало df_test:", df_test.index[0])

    base_cols = ["open", "high", "low", "close", "volume"]
    df_test_clean = df_test[base_cols]

    # Удаляем все строки в df_train, которые перекрываются по времени с df_test
    df_train_filtered = df_train[df_train.index < df_test.index[0]]

    df_train_updated = pd.concat([df_train_filtered, df_test_clean])

    return df_train_updated


def finalize_walkforward(config):
    """
    Завершает walk-forward тест:
    - строит финальный график прогресса;
    - рассчитывает итоговую доходность и APR;
    - выводит статистику по риску.
    """
    trade_log = config["trade_log"]
    initial_balance = config["initial_balance"]
    balance = config["balance"]
    days_elapsed = config["days_elapsed"]
    risk_history = config["risk_history"]
    symbol = config["symbol"]

    if not trade_log:
        print("❗ Нет сделок — нет графика и статистики.")
        return

    plot_backtest_progress(trade_log, title=f"Прогресс стратегии по {symbol}")

    total_pnl = balance - initial_balance
    apr = ((balance / initial_balance) ** (365 / days_elapsed) - 1) * 100 if days_elapsed > 0 else 0

    print(f"\n📊 Итоговая доходность: {total_pnl:.2f} | APR: {apr:.2f}% за {days_elapsed} дней")

    if risk_history:
        print("\n📈 Статистика по риску:")
        print(f"- Средний риск: {sum(risk_history) / len(risk_history):.3f}")
        print(f"- Максимальный риск: {max(risk_history):.3f}")
        print(f"- Минимальный риск: {min(risk_history):.3f}")

def get_next_test_window(df_train, config, symbol, interval):
    test_range = calculate_test_range(df_train, config["ms_per_candle"], config["step_candles"])
    test_end_ts = test_range[1]
    if is_end_of_data(test_end_ts):
        return test_range, None

    df_test = load_test_window_from_db(symbol, interval, test_range)
    if df_test is None or df_test.empty:
        print("⚠ Недостаточно тестовых данных в БД.")
        return test_range, None

    return test_range, df_test

def initialize_best_params(config, df_train_raw, df_val_raw):
    df_full = pd.concat([df_train_raw, df_val_raw])
    df_hash = hash_dataframe(df_full)
    dummy_params = {}

    df_full = calculate_indicators_cached(df_hash, df_full, dummy_params)
    df_full = generate_signals_cached(df_full, json.dumps(dummy_params))

    feature_cols_long = [col for col in df_full.columns if col.startswith("long_") and "_score" not in col and "_entry" not in col]
    feature_cols_short = [col for col in df_full.columns if col.startswith("short_") and "_score" not in col and "_entry" not in col]

    important_feats_long, _ = select_important_features(df_full, feature_cols_long, target_col="long_entry")
    important_feats_short, _ = select_important_features(df_full, feature_cols_short, target_col="short_entry")

    print("\n📌 Отобранные признаки (long):", important_feats_long)
    print("📌 Отобранные признаки (short):", important_feats_short)

    best_params, sharpe_train, sharpe_val = optimize_with_validation(
        df_train_raw, df_val_raw, config["symbol"], search_space,
        initial_params=None,
        enabled_long_signals=important_feats_long,
        enabled_short_signals=important_feats_short
    )

    if not best_params:
        print("⚠ Пропуск дня — стратегия не найдена.")
        return False

    config.update({
        "best_params": best_params,
        "sharpe_train": sharpe_train,
        "sharpe_val": sharpe_val
    })
    return True

def try_prepare_test_data(df_train, df_test, config):
    try:
        return prepare_test_data(df_train, df_test, config["best_params"])
    except Exception as e:
        print(f"⚠ Пропуск дня — ошибка подготовки данных: {e}")
        return None

def update_window_size(config):
    new_window = estimate_window_size_from_params(config["best_params"], verbose=True)

    if config["sharpe_train"] - config["sharpe_val"] > 1.0:
        print("⚠️ Переобучение: window_size не обновляем")
        new_window = config["window_size"]

    max_jump = int(config["window_size"] * 0.3)
    if abs(new_window - config["window_size"]) > max_jump:
        print("⚠️ Резкий скачок window_size — отменён")
        new_window = config["window_size"]

    alpha = 0.3
    config["window_size"] = int((1 - alpha) * config["window_size"] + alpha * new_window)
    print(f"📐 Новый window_size: {config['window_size']}")

def reoptimize_strategy(config, df_train_raw, df_val_raw, symbol):
    print("🔁 Переоптимизация из-за серии убыточных дней.")
    best_params, sharpe_train, sharpe_val = optimize_with_validation(
        df_train_raw, df_val_raw, symbol, search_space,
        initial_params=config["best_params"],
        enabled_long_signals=config["best_params"]["enabled_long_signals"],
        enabled_short_signals=config["best_params"]["enabled_short_signals"]
    )

    if not best_params:
        print("⚠ Переоптимизация не дала результата — best_params сброшены.")
        config["best_params"] = None
    else:
        config.update({
            "best_params": best_params,
            "sharpe_train": sharpe_train,
            "sharpe_val": sharpe_val
        })

def should_trigger_restart(result, config) -> bool:
    if result["final_balance"] < config["initial_balance"] * 0.9:
        return True

    if result["max_loss_streak"] >= 3:
        return True

    if result["total_trades"] == 0:
        config["no_trade_windows"] = config.get("no_trade_windows", 0) + 1
        if config["no_trade_windows"] >= 2:
            return True
    else:
        config["no_trade_windows"] = 0  # сброс, если сделки появились

    if result["winrate"] < 0.4:
        return True

    return False

def walk_forward_test(symbol="PRIMEUSDT", interval="30"):
    config = initialize_test(symbol, interval)
    df_train = load_initial_train_data(
        symbol=symbol,
        window_size=config["window_size"],
        start_timestamp=config["first_ts"],
        interval=interval,
    )

    while True:
        test_range, df_test = get_next_test_window(df_train, config, symbol, interval)
        if df_test is None:
            break

        df_train_raw, df_val_raw = split_train_val(df_train)

        if not config.get("best_params"):
            success = initialize_best_params(config, df_train_raw, df_val_raw)
            if not success:
                df_train = update_training_window(df_train, df_test, config["step_candles"])
                continue

        df_test_prepared = try_prepare_test_data(df_train, df_test, config)
        if df_test_prepared is None:
            df_train = update_training_window(df_train, df_test, config["step_candles"])
            continue

        update_window_size(config)
        config["risk_pct"] = calculate_inverse_balance_risk(config["balance"], config["initial_balance"])

        result, config["balance"], config["open_position"] = run_evaluation(df_test_prepared, symbol, config["balance"], config["risk_pct"], open_position=config.get("open_position"))
        update_tracking(config, interval, result, df_test, df_test_prepared)

        if config["balance"] < 500:
            plot_backtest_progress(config["trade_log"], title="История поражения")
            return

        if should_trigger_restart(result, config):
            print("🔁 Условия перезапуска модели выполнены")
            reoptimize_strategy(config, df_train_raw, df_val_raw, symbol)
            config["no_trade_windows"] = 0  # сброс счётчика

        df_train = update_training_window(df_train, df_test, config["step_candles"])

    finalize_walkforward(config)
