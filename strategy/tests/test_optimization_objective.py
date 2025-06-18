import pytest
from unittest.mock import Mock
from strategy.objective import ObjectiveEvaluator


def dummy_result(winrate, trades, pnl=100, sharpe=1.2, dd=10, streak=2):
    return {
        "winrate": winrate,
        "total_trades": trades,
        "pnl": pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "max_loss_streak": streak,
    }, {}

@pytest.fixture
def evaluator_factory():
    def _make(run_fn):
        return ObjectiveEvaluator(
            df_train=None,
            df_val=None,
            symbol="BTCUSDT",
            loss_weights={
                "norm_wr": 2.0,
                "norm_trades": 2.0,
                "norm_pnl": 1.0,
                "norm_sharpe": 2.0,
                "wr_gap": 10.0,
                "norm_dd": 2.0,
                "pnl_gap": 2.0,
                "streak_penalty": 1.5,
                "sharpe_gap": 5,
            },
            constraints={
                "min_trades": 10,
                "min_winrate": 0.55,
            },
            verbose=False
        )
    return _make

def test_maybe_update_best_not_called_on_loss_999(monkeypatch):
    # Arrange
    run_backtests_mock = lambda *args, **kwargs: dummy_result(winrate=0.4, trades=5)
    
    calculate_loss_components_mock = Mock(return_value=(999, {}))
    maybe_update_best_mock = Mock()

    monkeypatch.setattr("strategy.objective.run_train_val_backtests", run_backtests_mock)
    monkeypatch.setattr("strategy.objective.calculate_loss_components", calculate_loss_components_mock)
    monkeypatch.setattr("strategy.objective.maybe_update_best", maybe_update_best_mock)

    evaluator = ObjectiveEvaluator(
        df_train=None, df_val=None, symbol="BTCUSDT",
        loss_weights={"norm_wr": 1},  # not used in this test
        constraints={"min_trades": 10, "min_winrate": 0.55},
        verbose=False
    )

    # Act
    result = evaluator({})

    # Assert
    assert result["loss"] == 999
    maybe_update_best_mock.assert_not_called()
