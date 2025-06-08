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

        # Основной показатель — winrate (максимизируем)
        loss = -result["winrate"]

        # Штраф за слишком мало сделок
        min_trades = 15
        if result["total_trades"] < min_trades:
            loss += (min_trades - result["total_trades"]) * 0.01  # небольшой штраф за недостающие сделки

        # Штраф за winrate ниже 0.5 (агрессивный)
        if result["winrate"] < 0.5:
            loss += (0.5 - result["winrate"]) * 2

        # Бонус за положительный PnL (0.001 — масштабирующий коэффициент)
        loss -= result["pnl"] * 0.001

        # Необязательный штраф за высокую просадку (если доступно)
        if "max_drawdown" in result:
            loss += result["max_drawdown"] * 0.001  # или другой коэффициент

        return {'loss': loss, 'status': STATUS_OK}

    return objective

