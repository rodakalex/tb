from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.signals import generate_signals_cached
from trading_analysis.backtest import run_backtest
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials
from trading_analysis.utils import strip_indicators
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK

def optimize_with_validation(df_train, df_val, symbol, search_space, initial_params=None):
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
    all_wr = []
    all_pnl = []
    all_dd = []
    all_trades = []
    
    no_improve_rounds = 0

    def objective(params):
        nonlocal no_improve_rounds
        trial_counter[0] += 1

        # Подготовка данных
        df_train_prep = generate_signals_cached(calculate_indicators_cached(strip_indicators(df_train.copy())), params)
        df_val_prep = generate_signals_cached(calculate_indicators_cached(strip_indicators(df_val.copy())), params)

        train_result, _ = run_backtest(df_train_prep, symbol=symbol, report=False)
        val_result, _ = run_backtest(df_val_prep, symbol=symbol, report=False)

        # Извлечение метрик
        train_wr = train_result["winrate"]
        val_wr = val_result["winrate"]
        val_trades = val_result["total_trades"]
        val_pnl = val_result["pnl"]
        val_drawdown = val_result.get("max_drawdown", 0)
        val_sharpe = val_result.get("sharpe_ratio", 1.0)
        max_loss_streak = val_result.get("max_loss_streak", 0)
        sharpe_train = train_result.get("sharpe_ratio", 1.0)
        sharpe_gap = max(0, sharpe_train - val_sharpe) * 5
        streak_penalty = max(0, max_loss_streak - 3) * 1.5
        train_pnl = train_result["pnl"]

        # Сохраняем winrate
        all_wr.append(val_wr)
        all_pnl.append(val_pnl)
        all_dd.append(val_drawdown)
        all_trades.append(val_trades)

        # Адаптивная нормализация
        if len(all_wr) >= 20:
            wr_target = np.median(all_wr)
            wr_scale = np.std(all_wr) + 1e-6

            pnl_scale = np.std(all_pnl) + 1e-6
            trades_target = np.median(all_trades)
            trades_scale = np.std(all_trades) + 1e-6
        else:
            wr_target = 0.55
            wr_scale = 0.1
            pnl_scale = 100
            trades_target = 20
            trades_scale = 5

        pnl_gap = max(0, train_pnl - val_pnl) / pnl_scale * 2

        norm_wr = (wr_target - val_wr) / wr_scale
        norm_pnl = -val_pnl / pnl_scale
        norm_dd = val_drawdown / (abs(val_pnl) + 1e-6)
        norm_trades = abs(trades_target - val_trades) / trades_scale

        if val_trades < 10 or val_wr < 0.4:
            return {'loss': 100, 'status': STATUS_OK}

        strategy_score = (
            norm_wr * 5 +
            max(0, norm_trades) * 0.2 +
            norm_pnl +
            max(0, 1 - val_sharpe) * 2
        )

        wr_gap_penalty = max(0, train_wr - val_wr) * 10
        stability_penalty = wr_gap_penalty + norm_dd * 2 + sharpe_gap + pnl_gap

        total_loss = strategy_score + stability_penalty + streak_penalty

        if total_loss < best_local_loss[0]:
            best_local_loss[0] = total_loss
            no_improve_rounds = 0
            found_better_than_initial[0] = True
            print(
                f"\n=== New Best at trial {trial_counter[0]} ===\n"
                f"Loss: {total_loss:.4f} | Winrate: {val_wr:.2%} | Trades: {val_trades} | "
                f"PnL: {val_pnl:.2f} | Sharpe: {val_sharpe:.2f} | "
                f" ⚙️ norm_wr: {norm_wr:.2f}, norm_pnl: {norm_pnl:.2f}, norm_dd: {norm_dd:.2f}, norm_trades: {norm_trades:.2f} | "
                f"🎯 wr_target={wr_target:.3f}, wr_scale={wr_scale:.3f}, pnl_scale={pnl_scale:.2f}, trades_target={trades_target}, trades_scale={trades_scale:.2f}"
            )
        else:
            no_improve_rounds += 1

        return {'loss': total_loss, 'status': STATUS_OK}
    
    patience = 50
    while no_improve_rounds <= patience:
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
