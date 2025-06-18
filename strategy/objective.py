# strategy/objective.py
import numpy as np
from strategy.loss_utils import calculate_loss_components, maybe_update_best, run_train_val_backtests
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

verbose = False

class ObjectiveEvaluator:
    def __init__(self, df_train, df_val, symbol, loss_weights, constraints,
                 enabled_long_signals=None, enabled_short_signals=None, verbose=False):
        self.df_train = df_train
        self.df_val = df_val
        self.symbol = symbol
        self.loss_weights = loss_weights
        self.constraints = constraints
        self.enabled_long_signals = enabled_long_signals
        self.enabled_short_signals = enabled_short_signals
        self.trial_counter = 0
        self.best_local_loss = float("inf")
        self.no_improve_rounds = 20
        self.found_better_than_initial = False
        self.all_wr = []
        self.all_pnl = []
        self.all_dd = []
        self.all_trades = []

    def __call__(self, params):
        self.trial_counter += 1
        params = params.copy()
        if self.enabled_long_signals:
            params["enabled_long_signals"] = self.enabled_long_signals
        if self.enabled_short_signals:
            params["enabled_short_signals"] = self.enabled_short_signals

        train_result, val_result = run_train_val_backtests(self.df_train, self.df_val, params, self.symbol)

        loss, metrics = calculate_loss_components(
            train_result, val_result,
            (self.all_wr, self.all_pnl, self.all_dd, self.all_trades),
            self.loss_weights, self.constraints
        )

        if loss == 999:
            return {"loss": 999, "status": STATUS_OK}

        improved = maybe_update_best(loss, self.best_local_loss, metrics, self.trial_counter)
        if improved:
            self.best_local_loss = loss
            self.no_improve_rounds = 1
            self.found_better_than_initial = True
        else:
            self.no_improve_rounds += 1

        return {"loss": loss, "status": STATUS_OK}

def optimize_with_validation(df_train, df_val, symbol, search_space,
                             enabled_long_signals=None, enabled_short_signals=None, verbose=True):
    trials = Trials()
    round_count = 0
    PATIENCE = 50
    max_rounds = 1

    loss_weights = {
        "norm_wr": 2.0, "norm_trades": 2.0, "norm_pnl": 1.0, "norm_sharpe": 2.0,
        "wr_gap": 10.0, "norm_dd": 2.0, "pnl_gap": 2.0, "streak_penalty": 1.5, "sharpe_gap": 5,
    }
    constraints = {
        "min_trades": 10,
        "min_winrate": 0.55,
    }

    evaluator = ObjectiveEvaluator(
        df_train, df_val, symbol,
        loss_weights=loss_weights,
        constraints=constraints,
        enabled_long_signals=enabled_long_signals,
        enabled_short_signals=enabled_short_signals,
        verbose=verbose
    )

    while evaluator.no_improve_rounds <= PATIENCE and round_count < max_rounds:
        if verbose:
            print(f"🔁 Оптимизация раунд {round_count + 1}")

        fmin(
            fn=evaluator,
            space=search_space,
            algo=tpe.suggest,
            max_evals=100 * (round_count + 1),
            trials=trials
        )

        round_count += 1

    best_trial = trials.best_trial
    trial_loss = best_trial["result"]["loss"]

    if not np.isfinite(trial_loss):
        if verbose:
            print("❌ best_trial содержит inf:", trial_loss)
        return None, None, None

    flat_vals = {k: v[0] if isinstance(v, list) else v for k, v in best_trial["misc"]["vals"].items()}
    best_params = space_eval(search_space, flat_vals)

    if trial_loss == 999:
        return None, None, None

    return best_params, 1.0, 1.0  # stub values for sharpe_train, sharpe_val

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
