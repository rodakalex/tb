from datetime import datetime, timezone
import json
import pandas as pd

from trading_analysis.db import get_first_candle_from_db, load_ohlcv_from_db, save_model_run
from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.risk import calculate_inverse_balance_risk
from trading_analysis.signals import generate_signals_cached
from trading_analysis.backtest import run_backtest
from trading_analysis.charts import plot_backtest_progress

from strategy.objective import estimate_window_size_from_params, optimize_with_validation
from strategy.search_space import search_space
from trading_analysis.utils import sanitize_params, split_train_val

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

    :param df_train: DataFrame с тренировочными данными
    :param df_test: DataFrame с тестовыми свечами
    :param best_params: параметры, полученные в результате оптимизации
    :return: подготовленный DataFrame для теста
    """
    df_full = pd.concat([df_train, df_test])
    df_full = calculate_indicators_cached(df_full)
    params_serialized = json.dumps(sanitize_params(best_params), sort_keys=True)
    df_full = generate_signals_cached(df_full, params_serialized)

    # Возвращаем только тестовую часть, по длине df_test
    return df_full.iloc[-len(df_test):]

def run_evaluation(df_test_prepared, symbol: str, current_balance: float, risk_pct: float) -> tuple:
    """
    Проводит бэктест на тестовом окне и возвращает результат и обновлённый баланс.

    :param df_test_prepared: DataFrame с тестовыми данными и сигналами
    :param symbol: тикер символа (например, "PRIMEUSDT")
    :param current_balance: текущий баланс на момент входа в период
    :param risk_pct: риск на сделку в долях (например, 0.05 = 5%)
    :return: (результат бэктеста, новый баланс)
    """
    result, _ = run_backtest(
        df_test_prepared,
        symbol=symbol,
        report=True,
        finalize=True,
        initial_balance=current_balance,
        leverage=1,
        risk_pct=risk_pct
    )

    new_balance = current_balance + result["pnl"]
    return result, new_balance

def update_tracking(config: dict, result: dict, df_test, df_test_prepared):
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

    save_model_run(
        symbol=symbol,
        date=test_date,
        params=best_params,
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

def walk_forward_test(symbol="PRIMEUSDT", interval="30"):
    config = initialize_test(symbol, interval)
    df_train = load_initial_train_data(symbol=symbol, window_size=config["window_size"], start_timestamp=config["first_ts"], interval=interval)
    config['bad_days'] = 0

    while True:
        test_range = calculate_test_range(df_train, config["ms_per_candle"], config["step_candles"])
        test_end_ts = test_range[1]

        if is_end_of_data(test_end_ts):
            break

        df_test = load_test_window_from_db(symbol, interval, test_range)
        if df_test is None or df_test.empty:
            print("⚠ Недостаточно тестовых данных в БД.")
            break
        
        df_train_raw, df_val_raw = split_train_val(df_train)
        if config.get("best_params") is None:
            config["best_params"] = optimize_with_validation(
                df_train_raw,
                df_val_raw,
                symbol=symbol,
                search_space=search_space,
                initial_params=config["best_params"]
            )
        best_params = config["best_params"]

        config["window_size"] = estimate_window_size_from_params(config["best_params"])
        print(f"📈 Используемые индикаторы:")
        for k, v in config["best_params"].items():
            if k.startswith("w_") and v > 0:
                print(f"  - {k[2:]}: вес {v}")

        df_test_prepared = prepare_test_data(df_train, df_test, best_params)
        risk_pct = calculate_inverse_balance_risk(
            current_balance=config["balance"],
            initial_balance=config["initial_balance"]
        )
        config["risk_pct"] = risk_pct
        result, config["balance"] = run_evaluation(df_test_prepared, symbol, config["balance"], risk_pct)

        if config["balance"] < 500:
            plot_backtest_progress(config["trade_log"], title="История поражения")
            return

        update_tracking(config, result, df_test, df_test_prepared)

        if result["pnl"] <= 0:
            config["bad_days"] = config.get("bad_days", 0) + 1
        else:
            config["bad_days"] = 0
            config["win_streak"] += 1

        if config["bad_days"] >= 2:
            config["best_params"] = optimize_with_validation(
                df_train_raw,
                df_val_raw,
                symbol=symbol,
                search_space=search_space,
                initial_params=config["best_params"]
            )

        df_train = update_training_window(df_train, df_test, config["step_candles"])

    finalize_walkforward(config)
