# optimize.py
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from trading_analysis.db import load_ohlcv_from_db
from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals
from trading_analysis.backtest import run_backtest

import pprint

symbol = "PRIMEUSDT"

def objective(params):
    df = load_ohlcv_from_db(symbol, limit=1000)
    df = calculate_indicators(df)
    df = generate_signals(df, params)

    result = run_backtest(df, symbol=symbol, report=False, )
    score = result["final_balance"]

    return {
        'loss': -score,
        'status': STATUS_OK
    }

def objective_with_df(df_outer, symbol):
    def objective(params):
        df = df_outer.copy()
        df = calculate_indicators(df)
        df = generate_signals(df, params)
        result = run_backtest(df, symbol=symbol, report=False)
        return {
            'loss': -result["final_balance"],
            'status': STATUS_OK
        }
    return objective

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

if __name__ == "__main__":
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials
    )
    print("Лучшие параметры:")
    pprint.pprint(best)
    symbol = "PRIMEUSDT"
    df = load_ohlcv_from_db('PRIMEUSDT')
    df = calculate_indicators(df)
    df = generate_signals(df, params=best)
    run_backtest(df, symbol=symbol)
