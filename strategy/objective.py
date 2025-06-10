from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals
from trading_analysis.backtest import run_backtest
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials
from trading_analysis.utils import strip_indicators


import json
from hyperopt import fmin, tpe, Trials, STATUS_OK

def optimize_with_validation(df_train, df_val, symbol, search_space, target_loss=7.5, max_rounds=5, initial_params=None):
    """
    Оптимизирует параметры с валидацией. Использует тренировочные данные для переобучения,
    валидационные — для основной оценки. Штрафует за переобучение.
    Если ни одно решение не лучше, чем начальное, возвращает initial_params.
    """
    round_count = 0
    best_loss = float("inf")
    best_params = None
    trials = Trials()
    trial_counter = [0]
    best_local_loss = [float("inf")]
    found_better_than_initial = [False]

    def objective(params):
        trial_counter[0] += 1

        df_train_prep = generate_signals(calculate_indicators(strip_indicators(df_train.copy())), params)
        df_val_prep = generate_signals(calculate_indicators(strip_indicators(df_val.copy())), params)

        train_result, _ = run_backtest(df_train_prep, symbol=symbol, report=False)
        val_result, _ = run_backtest(df_val_prep, symbol=symbol, report=False)

        train_wr = train_result["winrate"]
        val_wr = val_result["winrate"]
        val_trades = val_result["total_trades"]
        val_pnl = val_result["pnl"]
        val_drawdown = val_result.get("max_drawdown", 0)
        val_rr = val_result.get("avg_rr", 1.0)
        val_sharpe = val_result.get("sharpe_ratio", 1.0)

        if val_trades < 10 or val_wr < 0.4:
            return {'loss': 100, 'status': STATUS_OK}

        loss = 0
        loss += (0.6 - val_wr) * 8 if val_wr < 0.6 else 0
        loss += (20 - val_trades) * 0.1 if val_trades < 20 else 0
        loss += abs(val_rr - 1.0) * 2
        loss -= val_pnl * 0.002
        loss += (val_drawdown / val_pnl) * 0.5 if val_pnl > 0 else val_drawdown * 0.001
        loss += max(0, train_wr - val_wr) * 5
        loss += (1 - val_sharpe) * 2 if val_sharpe < 1 else 0

        if loss < best_local_loss[0]:
            best_local_loss[0] = loss
            found_better_than_initial[0] = True
            log_msg = (
                f"\n=== New Best at trial {trial_counter[0]} ===\n"
                f"Loss: {loss:.4f} | Winrate: {val_wr:.2%} | Trades: {val_trades} | PnL: {val_pnl:.2f} | Sharpe: {val_sharpe:.2f}\n"
            )
            print(log_msg)

        return {'loss': loss, 'status': STATUS_OK}

    while best_loss > target_loss and round_count < max_rounds:
        print(f"🔁 Оптимизация раунд {round_count + 1}")
        params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=200 * (round_count + 1),
            trials=trials
        )

        best_trial = trials.best_trial
        best_loss = best_trial['result']['loss']
        best_params = params
        round_count += 1

        print(f"✅ Лучший loss после раунда {round_count}: {best_loss:.4f}")

    if not found_better_than_initial[0] and initial_params is not None:
        print("⚠️ Оптимизация не нашла лучших параметров — возвращаю начальные.")
        return initial_params

    return best_params


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
