from strategy.loss_utils import calculate_loss_components, maybe_update_best

default_weights = {
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

default_constraints = {
    "min_trades": 10,
    "min_winrate": 0.55,
}

def dummy(train_wr=0.7, val_wr=0.6, val_trades=20, val_pnl=100, val_dd=5, val_sharpe=1.5,
          streak=2, train_pnl=110, train_sharpe=2.0):
    train = {
        "winrate": train_wr,
        "total_trades": 20,
        "pnl": train_pnl,
        "sharpe_ratio": train_sharpe,
    }
    val = {
        "winrate": val_wr,
        "total_trades": val_trades,
        "pnl": val_pnl,
        "max_drawdown": val_dd,
        "sharpe_ratio": val_sharpe,
        "max_loss_streak": streak,
    }
    return train, val

def test_low_winrate_penalty():
    train_result = {"winrate": 0.8, "pnl": 100, "sharpe_ratio": 1.2}
    val_result = {"winrate": 0.4, "pnl": 50, "total_trades": 15, "sharpe_ratio": 0.9}
    all_metrics = ([], [], [], [])
    loss_weights = {
        "norm_wr": 2.0, "norm_trades": 2.0, "norm_pnl": 1.0,
        "norm_sharpe": 2.0, "wr_gap": 10.0, "norm_dd": 2.0,
        "pnl_gap": 2.0, "streak_penalty": 1.5, "sharpe_gap": 5,
    }
    constraints = {"min_trades": 10, "min_winrate": 0.55}

    loss, _ = calculate_loss_components(train_result, val_result, all_metrics, loss_weights, constraints)
    assert loss == 999

def test_maybe_update_best_prints_on_improvement(capfd):
    metrics = {
        "val_wr": 0.6, "val_trades": 20, "val_pnl": 100, "val_sharpe": 1.2,
        "norm_wr": 0.5, "norm_pnl": -1.0, "norm_dd": 0.2, "norm_trades": 0.5,
        "wr_target": 0.55, "wr_scale": 0.05, "pnl_scale": 50, "trades_target": 20, "trades_scale": 5,
    }

    improved = maybe_update_best(
        loss=0.5,
        best_loss=1.0,
        metrics=metrics,
        trial_counter=5,
        verbose=True
    )

    out, _ = capfd.readouterr()
    assert improved is True
    assert "New Best at trial 5" in out

def test_loss_999_when_trades_too_low():
    train, val = dummy(val_trades=5)
    loss, _ = calculate_loss_components(train, val, ([], [], [], []), default_weights, default_constraints)
    assert loss == 999

def test_loss_999_when_winrate_too_low():
    train, val = dummy(val_wr=0.4)
    loss, _ = calculate_loss_components(train, val, ([], [], [], []), default_weights, default_constraints)
    assert loss == 999

def test_loss_computation_baseline():
    all_wr = [0.6, 0.62]
    all_pnl = [90, 110]
    all_dd = [5, 6]
    all_trades = [20, 22]

    train, val = dummy(val_wr=0.58, val_trades=21, val_pnl=100, val_dd=5, val_sharpe=1.2, train_sharpe=1.5)
    loss, metrics = calculate_loss_components(train, val, (all_wr, all_pnl, all_dd, all_trades), default_weights, default_constraints)

    assert isinstance(loss, float)
    assert loss > 0
    assert "val_wr" in metrics
