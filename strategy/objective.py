from strategy.utils_hashing import hash_dataframe
from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.signals import generate_signals_cached
from trading_analysis.backtest import run_backtest
from trading_analysis.utils import strip_indicators
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

def prepare_data(df, params):
    df_clean = strip_indicators(df.copy())
    df_hash = hash_dataframe(df_clean)
    df_indicators = calculate_indicators_cached(df_hash, df_clean, params)
    return generate_signals_cached(df_indicators, params)

def optimize_with_validation(df_train, df_val, symbol, search_space, initial_params=None,
                             enabled_long_signals=None, enabled_short_signals=None, verbose=True):
    """
    Оптимизирует параметры с валидацией. Использует тренировочные данные для переобучения,
    валидационные — для основной оценки. Штрафует за переобучение.
    Если ни одно решение не лучше, чем начальное, возвращает initial_params.
    """
    round_count = 0
    best_loss = float("inf")
    best_params = None
    trials = Trials()
    trial_counter = 0
    best_local_loss = float("inf")
    found_better_than_initial = False
    all_wr = []
    all_pnl = []
    all_dd = []
    all_trades = []
    MIN_TRADES = 10
    PATIENCE = 50
    no_improve_rounds = 0
    best_sharpe_train = 1.0
    best_sharpe_val = 1.0

    loss_weights = {
        "norm_wr": 2.0,
        "norm_trades": 2.0,
        "norm_pnl": 1.0,
        "norm_sharpe": 2.0,
        "wr_gap": 10.0,
        "norm_dd": 2.0,
        "pnl_gap": 2.0,
        "streak_penalty": 1.5,
        "sharpe_gap": 5,
    }

    constraints = {
        "min_trades": 10,
        "min_winrate": 0.55,
    }

    def objective(params):
        nonlocal no_improve_rounds, trial_counter, best_local_loss, found_better_than_initial
        trial_counter += 1

        params = params.copy()
        if enabled_long_signals is not None:
            params["enabled_long_signals"] = enabled_long_signals
        if enabled_short_signals is not None:
            params["enabled_short_signals"] = enabled_short_signals

        df_train_prep = prepare_data(df_train, params)
        df_val_prep = prepare_data(df_val, params)

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

        if val_trades < constraints["min_trades"] or val_wr < constraints["min_winrate"]:
            return {"loss": 999, "status": STATUS_OK}

        strategy_score = (
            norm_wr * loss_weights["norm_wr"] +
            max(0, norm_trades) * loss_weights["norm_trades"] +
            norm_pnl * loss_weights["norm_pnl"] +
            max(0, 1 - val_sharpe) * loss_weights["norm_sharpe"]
        )

        wr_gap_penalty = max(0, train_wr - val_wr) * loss_weights["wr_gap"]
        stability_penalty = (
            wr_gap_penalty +
            norm_dd * loss_weights["norm_dd"] +
            sharpe_gap * loss_weights["sharpe_gap"] +
            pnl_gap * loss_weights["pnl_gap"]
        )

        total_loss = (
            strategy_score +
            stability_penalty +
            streak_penalty * loss_weights["streak_penalty"]
        )

        if total_loss < best_local_loss:
            best_local_loss = total_loss
            no_improve_rounds = 0
            found_better_than_initial = True
            best_sharpe_train = sharpe_train
            best_sharpe_val = val_sharpe
            if verbose:
                print(
                    f"\n=== New Best at trial {trial_counter} ===\n"
                    f"Loss: {total_loss:.4f} | Winrate: {val_wr:.2%} | Trades: {val_trades} | "
                    f"PnL: {val_pnl:.2f} | Sharpe: {val_sharpe:.2f} | "
                    f"⚙️ norm_wr: {norm_wr:.2f}, norm_pnl: {norm_pnl:.2f}, norm_dd: {norm_dd:.2f}, norm_trades: {norm_trades:.2f} | "
                    f"🎯 wr_target={wr_target:.3f}, wr_scale={wr_scale:.3f}, pnl_scale={pnl_scale:.2f}, trades_target={trades_target}, trades_scale={trades_scale:.2f}"
                )
        else:
            no_improve_rounds += 1

        return {'loss': total_loss, 'status': STATUS_OK}
    
    if verbose:
        print(f"🔁 Оптимизация раунд {round_count + 1}")
    if initial_params:
        if verbose:
            print("🔍 Оценка initial_params")
        initial_result = objective(initial_params)
        if initial_result["loss"] < best_local_loss:
            best_local_loss = initial_result["loss"]
            best_params = initial_params
            found_better_than_initial = True

    while no_improve_rounds <= PATIENCE and round_count < 1:
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=200 * (round_count + 1),
            trials=trials
        )

        best_trial = trials.best_trial
        trial_loss = best_trial['result']['loss']
        if best_trial['result']['loss'] <= best_local_loss:
            best_local_loss = best_trial['result']['loss']
            flat_vals = {k: v[0] if isinstance(v, list) else v for k, v in best_trial['misc']['vals'].items()}
            best_params = space_eval(search_space, flat_vals)
            best_loss = trial_loss
            found_better_than_initial = True

        round_count += 1
        if verbose:
            print(f"✅ Лучший loss после раунда {round_count}: {best_loss:.4f}")

    if not found_better_than_initial:
        if verbose:
            print("⚠️ Оптимизация не нашла лучших параметров.")
        if initial_params is not None:
            return initial_params, best_sharpe_train, best_sharpe_val
        else:
            return {}, best_sharpe_train, best_sharpe_val

    if best_local_loss == 999:
        return None, None, None
    return best_params, best_sharpe_train, best_sharpe_val

def estimate_window_size_from_params(
    best_params: dict,
    indicator_periods: dict = None,
    heavy_penalties: dict = None,
    period_multiplier: float = 8.0,
    min_window_size: int = 300,
    max_window_size: int = 2000,
    verbose: bool = True
) -> int:
    """
    Оценивает оптимальный window_size на основе активных индикаторов и их весов.
    Учитывает штрафы за тяжёлые индикаторы. Возвращает ограниченный размер окна.
    """
    if not best_params:
        return min_window_size

    indicator_periods = indicator_periods or {
        "rsi": 14, "macd": 26, "mfi": 14, "cci": 20, "stochrsi": 14,
        "tema_cross": 21, "ema_cross": 200, "trend": 50,
        "donchian": 20, "roc": 12, "volspike": 20, "volume": 20
    }

    heavy_penalties = heavy_penalties or {
        "w_ema_cross": (2.0, 150),
        "w_trend": (2.0, 100),
    }

    weighted_sum = 0
    total_weight = 0

    for key, weight in best_params.items():
        if key.startswith("w_") and weight > 0:
            name = key[2:]
            if name in indicator_periods:
                weighted_sum += indicator_periods[name] * weight
                total_weight += weight

    if total_weight == 0:
        return min_window_size

    base = weighted_sum / total_weight
    penalty = sum(p for k, (threshold, p) in heavy_penalties.items()
                  if best_params.get(k, 0) >= threshold)

    window_size = int(base * period_multiplier + penalty)
    window_size = max(min_window_size, min(window_size, max_window_size))

    if verbose:
        print(f"🧠 Auto-selected window_size: {window_size} (base={base:.1f}, penalty={penalty})")

    return window_size
