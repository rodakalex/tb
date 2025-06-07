
from hyperopt import STATUS_OK
from main import strip_indicators
from trading_analysis.backtest import run_backtest
from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals


# 🏁 Итоговая суммарная доходность: 183.33
# 📈 APR (годовая доходность): 31.56% за 224 дней
def objective_with_df(df_outer, symbol):
    def objective(params):
        df = strip_indicators(df_outer.copy())
        df = calculate_indicators(df)
        df = generate_signals(df, params)
        result, _ = run_backtest(df, symbol=symbol, report=False)


        avg_trade = result["avg_trade"]
        winrate = result["winrate"]
        score = winrate - 0.1 * (1 / (avg_trade + 1e-6)) - 0.1 * (1 / (result["total_trades"]  + 1e-6))


        return {
            'loss': -score,
            'status': STATUS_OK
        }

    return objective

# 🏁 Итоговая суммарная доходность: 395.07
# 📈 APR (годовая доходность): 72.03% за 224 дней
def objective_with_df1(df_outer, symbol):
    def objective(params):
        df = strip_indicators(df_outer.copy())
        df = calculate_indicators(df)
        df = generate_signals(df, params)
        result, _ = run_backtest(df, symbol=symbol, report=False)

        if result["total_trades"] < 10 or result["total_trades"] > 30:
            return {'loss': 1e9, 'status': STATUS_OK}

        winrate = result["winrate"]
        avg_trade = result["avg_trade"]
        max_loss = abs(result.get("max_loss", 1e-6)) + 1e-6

        score = winrate * avg_trade - 0.5 * abs(avg_trade / max_loss)

        return {'loss': -score, 'status': STATUS_OK}

    return objective