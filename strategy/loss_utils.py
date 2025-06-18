import numpy as np
from strategy.utils_hashing import hash_dataframe
from trading_analysis.backtest import run_backtest
from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.signals import generate_signals
from trading_analysis.utils import strip_indicators

verbose = False

def prepare_data(df, params):
    df_clean = strip_indicators(df.copy())
    df_hash = hash_dataframe(df_clean)
    df_indicators = calculate_indicators_cached(df_hash, df_clean, params)
    
    return generate_signals(df_indicators, params)

def run_train_val_backtests(df_train, df_val, params, symbol):
    df_train_prep = prepare_data(df_train, params)
    df_val_prep = prepare_data(df_val, params)

    train_result, _ = run_backtest(df_train_prep, symbol=symbol, report=False)
    val_result, _ = run_backtest(df_val_prep, symbol=symbol, report=False)

    return train_result, val_result

def calculate_loss_components(train_result, val_result, history, loss_weights, constraints):
    all_wr, all_pnl, all_dd, all_trades = history

    # Метрики
    train_wr = train_result["winrate"]
    val_wr = val_result["winrate"]
    val_trades = val_result["total_trades"]
    val_pnl = val_result["pnl"]
    val_drawdown = val_result.get("max_drawdown", 0)
    val_sharpe = val_result.get("sharpe_ratio", 1.0)
    max_loss_streak = val_result.get("max_loss_streak", 0)
    sharpe_train = train_result.get("sharpe_ratio", 1.0)
    train_pnl = train_result["pnl"]

    # Запись истории
    all_wr.append(val_wr)
    all_pnl.append(val_pnl)
    all_dd.append(val_drawdown)
    all_trades.append(val_trades)

    if len(all_wr) >= 20:
        wr_target = np.median(all_wr)
        wr_scale = np.std(all_wr) + 1e-6
        pnl_scale = np.std(all_pnl) + 1e-6
        trades_target = np.median(all_trades)
        trades_scale = np.std(all_trades) + 1e-6
    else:
        wr_target, wr_scale = 0.55, 0.1
        pnl_scale = 100
        trades_target, trades_scale = 20, 5

    # Нормализация
    pnl_gap = max(0, train_pnl - val_pnl) / pnl_scale * 2
    norm_wr = (wr_target - val_wr) / wr_scale
    norm_pnl = -val_pnl / pnl_scale
    norm_dd = val_drawdown / (abs(val_pnl) + 1e-6)
    norm_trades = abs(trades_target - val_trades) / trades_scale
    sharpe_gap = max(0, sharpe_train - val_sharpe)
    streak_penalty = max(0, max_loss_streak - 3)

    # # # Early rejection
    # if val_trades < constraints["min_trades"] or val_wr < constraints["min_winrate"]:
    #     return 999, {}

    # Loss components
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

    metrics = {
        "val_wr": val_wr, "val_pnl": val_pnl, "val_trades": val_trades, "val_sharpe": val_sharpe,
        "norm_wr": norm_wr, "norm_pnl": norm_pnl, "norm_dd": norm_dd, "norm_trades": norm_trades,
        "wr_target": wr_target, "wr_scale": wr_scale, "pnl_scale": pnl_scale,
        "trades_target": trades_target, "trades_scale": trades_scale
    }

    return total_loss, metrics

def maybe_update_best(loss, best_loss, metrics, trial_counter):
    improved = loss < best_loss
    if improved and verbose:
        print(
            f"\n=== New Best at trial {trial_counter} ===\n"
            f"Loss: {loss:.4f} | Winrate: {metrics['val_wr']:.2%} | Trades: {metrics['val_trades']} | "
            f"PnL: {metrics['val_pnl']:.2f} | Sharpe: {metrics['val_sharpe']:.2f} | "
            f"⚙️ norm_wr: {metrics['norm_wr']:.2f}, norm_pnl: {metrics['norm_pnl']:.2f}, "
            f"norm_dd: {metrics['norm_dd']:.2f}, norm_trades: {metrics['norm_trades']:.2f} | "
            f"🎯 wr_target={metrics['wr_target']:.3f}, wr_scale={metrics['wr_scale']:.3f}, "
            f"pnl_scale={metrics['pnl_scale']:.2f}, trades_target={metrics['trades_target']}, "
            f"trades_scale={metrics['trades_scale']:.2f}"
        )
    return improved
