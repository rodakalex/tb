from hyperopt import hp

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
    "long_score_threshold": hp.quniform("long_score_threshold", 2, 4, 1),
    "short_score_threshold": hp.quniform("short_score_threshold", 2, 4, 1),
    'tp_pct': hp.uniform('tp_pct', 0.01, 0.50),
    'sl_pct': hp.uniform('sl_pct', 0.005, 0.50),
    "use_dynamic_tp_sl": hp.choice("use_dynamic_tp_sl", [False]),
    "risk_pct": hp.uniform("risk_pct", 0.01, 0.20)
}