# main.py
from trading_analysis.backtest import run_backtest
from trading_analysis.db import init_db, load_ohlcv_from_db
from trading_analysis.bybit_api import periodically_update_history, find_first_kline_timestamp, get_bybit_kline
from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals
from trading_analysis.websocket_kline import listen_kline_async

from optimize import objective_with_df
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
import asyncio
from datetime import datetime, timedelta, timezone

search_space = {
    "w_ema_cross": hp.quniform("w_ema_cross", 0, 3, 1),
    "w_trend": hp.quniform("w_trend", 0, 3, 1),
    "w_rsi": hp.quniform("w_rsi", 0, 3, 1),
    "w_macd": hp.quniform("w_macd", 0, 3, 1),
    "w_stochrsi": hp.quniform("w_stochrsi", 0, 3, 1),
    "w_cci": hp.quniform("w_cci", 0, 3, 1),
    "w_mfi": hp.quniform("w_mfi", 0, 3, 1),
    "w_volume": hp.quniform("w_volume", 0, 3, 1),
    "w_roc": hp.quniform("w_roc", 0, 3, 1),
    "w_volspike": hp.quniform("w_volspike", 0, 3, 1),
    "w_donchian": hp.quniform("w_donchian", 0, 3, 1),
    "w_tema_cross": hp.quniform("w_tema_cross", 0, 3, 1),
    "long_score_threshold": hp.quniform("long_score_threshold", 3, 10, 1),
    "short_score_threshold": hp.quniform("short_score_threshold", 3, 10, 1),
    'tp_pct': hp.uniform('tp_pct', 0.01, 0.50),
    'sl_pct': hp.uniform('sl_pct', 0.005, 0.50),
    "use_dynamic_tp_sl": hp.choice("use_dynamic_tp_sl", [False]),
}

def should_update_data(latest_timestamp_ms: int, interval_minutes: int = 30, tolerance_minutes: int = 1) -> bool:
    last_candle_time = datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=timezone.utc)
    next_expected_time = last_candle_time + timedelta(minutes=interval_minutes)
    now = datetime.now(timezone.utc)
    
    return now >= (next_expected_time - timedelta(minutes=tolerance_minutes))

def strip_indicators(df):
    cols_to_drop = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume", "turnover"]]
    return df.drop(columns=cols_to_drop, errors="ignore")

def objective_with_df(df_outer, symbol):
    def objective(params):
        df = strip_indicators(df_outer.copy())
        df = calculate_indicators(df)
        df = generate_signals(df, params)
        result, _ = run_backtest(df, symbol=symbol, report=False)

        if result["total_trades"] < 10 or result["total_trades"] > 30:
            return {'loss': 1e9, 'status': STATUS_OK}

        return {'loss': -result["winrate"], 'status': STATUS_OK}

    return objective


def walk_forward_test(symbol="PRIMEUSDT", interval="30"):
    window_size = 1000
    step_candles = int(24 * 60 / int(interval))  # 48 свечей = 1 день
    ms_per_candle = int(interval) * 60_000
    total_pnl = 0.0
    days_elapsed = 0

    initial_balance = 1000
    balance = initial_balance
    state = None
    best_params = None
    win_streak = 0
    risk_history = []

    first_ts = find_first_kline_timestamp(symbol, interval)
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    df_train = get_bybit_kline(symbol, interval, limit=window_size, start=first_ts)

    while True:
        last_ts_train = int(df_train.index[-1].timestamp() * 1000)
        test_start = last_ts_train + ms_per_candle
        test_end = test_start + step_candles * ms_per_candle

        if test_end > now_ts:
            print("⏹ Достигнут конец доступных данных.")
            break

        if best_params is None:
            print(f"\n🔍 Первая оптимизация на данных до {df_train.index[-1]}")
            trials = Trials()
            best_params = fmin(
                fn=objective_with_df(df_train, symbol),
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
            )
            if trials.best_trial['result']['loss'] > 1000:
                print("⚠️ Результат слабый — дооптимизируем ещё 200 попыток...")
                best_params = fmin(
                    fn=objective_with_df(df_train, symbol),
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=300,  # ← было 100, теперь 300 (добавляется сверху!)
                    trials=trials,
                )

        df_test = get_bybit_kline(symbol, interval, limit=step_candles, start=test_start, end=test_end)
        if df_test.empty or len(df_test) < step_candles:
            print("⚠ Недостаточно данных для следующего дня.")
            break

        df_full = pd.concat([df_train, df_test])
        df_full = calculate_indicators(df_full)
        df_full = generate_signals(df_full, best_params)
        df_test_prepared = df_full.iloc[-step_candles:]

        # 📌 Расчёт риска на основе win_streak
        risk_pct = min(0.05 + win_streak * 0.015, 0.2)
        risk_history.append(risk_pct)
        print(f"⚖️ Уровень риска: {risk_pct:.3f}")

        result, state = run_backtest(
            df_test_prepared,
            symbol=symbol,
            report=True,
            initial_state=state,
            finalize=False,
            initial_balance=balance,
            leverage=1,
            risk_pct=risk_pct  # ← передаём риск
        )

        total_pnl += result['pnl']
        balance += result['pnl']
        days_elapsed += 1

        print(f"📆 {df_test_prepared.index[0].date()} ▶ PnL: {result['pnl']:.2f} | Сделок: {result['total_trades']} | Общий pnl: {total_pnl:.2f} | Итоговый баланс: {balance:.2f}")

        if result["pnl"] < 0 or result["total_trades"] < 1:
            print("🔁 Результат плохой — переобучение...")
            trials = Trials()
            best_params = fmin(
                fn=objective_with_df(df_train, symbol),
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials
            )
            if trials.best_trial['result']['loss'] > 1000:
                print("⚠️ Результат слабый — дооптимизируем ещё 200 попыток...")
                best_params = fmin(
                    fn=objective_with_df(df_train, symbol),
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=300,  # ← было 100, теперь 300 (добавляется сверху!)
                    trials=trials,
                )
            win_streak = 0
        else:
            win_streak += 1

        base_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df_test_clean = df_test[base_cols]
        df_train = pd.concat([df_train.iloc[step_candles:], df_test_clean])

        apr = ((balance / initial_balance) ** (365 / days_elapsed) - 1) * 100 if days_elapsed > 0 else 0
        print(f"\n📊 Итоговая доходность: {total_pnl:.2f} | APR: {apr:.2f}% за {days_elapsed} дней")

    # Итоговая статистика по риску
    if risk_history:
        avg_risk = sum(risk_history) / len(risk_history)
        print("\n📈 Статистика по риску:")
        print(f"- Средний риск: {avg_risk:.3f}")
        print(f"- Максимальный риск: {max(risk_history):.3f}")
        print(f"- Минимальный риск: {min(risk_history):.3f}")


def backtest_example(symbol):
    start_params = {'long_score_threshold': 5.0,
        'short_score_threshold': 8.0,
        'w_cci': 3.0,
        'w_donchian': 0.0,
        'w_ema_cross': 0.0,
        'w_macd': 0.0,
        'w_mfi': 3.0,
        'w_roc': 0.0,
        'w_rsi': 0.0,
        'w_stochrsi': 3.0,
        'w_tema_cross': 0.0,
        'w_trend': 3.0,
        'w_volspike': 0.0,
        'w_volume': 3.0
    }
    df = load_ohlcv_from_db('PRIMEUSDT')
    if should_update_data():
        periodically_update_history(df)

    df = calculate_indicators(df)
    df = generate_signals(df, params=start_params)
    run_backtest(df, symbol=symbol)

def run():
    init_db()
    symbol = "PRIMEUSDT"
    interval = "30"

    loop = asyncio.get_event_loop()
    loop.create_task(listen_kline_async(symbol, interval))
    loop.create_task(periodically_update_history(symbol, int(interval)))
    loop.run_forever()

if __name__ == "__main__":
    walk_forward_test()
