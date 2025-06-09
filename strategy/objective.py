from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals
from trading_analysis.backtest import run_backtest
from hyperopt import STATUS_OK

from trading_analysis.utils import strip_indicators

def objective_with_df(df_outer, symbol):
    def objective(params):
        df = strip_indicators(df_outer.copy())
        df = calculate_indicators(df)
        df = generate_signals(df, params)
        result, _ = run_backtest(df, symbol=symbol, report=False)
        loss = -result["winrate"]
        min_trades = 15

        if result["total_trades"] < min_trades:
            loss += (min_trades - result["total_trades"]) * 0.01 

        if result["winrate"] < 0.5:
            loss += (0.5 - result["winrate"]) * 2

        loss -= result["pnl"] * 0.001

        if "max_drawdown" in result:
            loss += result["max_drawdown"] * 0.001

        return {'loss': loss, 'status': STATUS_OK}

    return objective

def estimate_window_size_from_params(best_params: dict) -> int:
    """
    Оценивает необходимый window_size на основе весов и периодов активных индикаторов,
    добавляя надбавку за тяжёлые индикаторы.
    """
    INDICATOR_PERIODS = {
        "rsi": 14,
        "macd": 26,
        "mfi": 14,
        "cci": 20,
        "stochrsi": 14,
        "tema_cross": 21,
        "ema_cross": 200,
        "trend": 50,
        "donchian": 20,
        "roc": 12,
        "volspike": 20,
        "volume": 20,
    }

    weighted_periods = []
    total_weight = 0

    for key, weight in best_params.items():
        if key.startswith("w_"):
            indicator = key[2:]
            if indicator in INDICATOR_PERIODS and weight > 0:
                period = INDICATOR_PERIODS[indicator]
                weighted_periods.append(period * weight)
                total_weight += weight

    if not weighted_periods or total_weight == 0:
        return 500

    avg_weighted_period = sum(weighted_periods) / total_weight

    # Надбавка за тяжёлые индикаторы
    penalty = 0
    if best_params.get("w_ema_cross", 0) >= 2.0:
        penalty += 150
    if best_params.get("w_trend", 0) >= 2.0:
        penalty += 100

    window_size = max(500, int(avg_weighted_period * 8) + penalty)

    print(f"🧠 window_size выбран автоматически: {window_size}")
    return window_size

