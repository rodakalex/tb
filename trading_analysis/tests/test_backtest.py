import pytest
import pandas as pd
from trading_analysis.backtest import run_backtest
from unittest.mock import patch

def make_long_test_df():
    return pd.DataFrame([
        {'open': 100, 'high': 102, 'low': 99, 'close': 101, 'long_entry': 1, 'long_exit': 0},
        {'open': 101, 'high': 103, 'low': 100, 'close': 102, 'long_entry': 0, 'long_exit': 1},
        {'open': 102, 'high': 105, 'low': 100, 'close': 104, 'long_exit': 1, 'long_entry': 0}
    ])

def test_run_backtest_basic_long_trade():
    df = make_long_test_df()
    result, state = run_backtest(df, report=False, tp_pct=0.0, sl_pct=0.01, fee=0)

    assert result['total_trades'] == 1

    entries = [t for t in state['trades'] if t[0] == 'OPEN LONG']
    closes = [t for t in state['trades'] if 'CLOSE' in t[0]]

    assert len(entries) == 1
    assert len(closes) == 1
    assert state['balance'] == 1000.0

    for key in ['pnl', 'winrate', 'avg_trade', 'lossrate', 'final_balance']:
        assert key in result

def make_short_test_df():
    return pd.DataFrame([
        {'open': 100, 'high': 101, 'low': 98, 'close': 100, 'long_entry': 0, 'short_entry': 1},
        {'open': 99, 'high': 100, 'low': 95, 'close': 96, 'long_entry': 0, 'short_entry': 0},
        {'open': 96, 'high': 97, 'low': 94, 'close': 95, 'short_exit': 1, 'long_entry': 0, 'short_entry': 0},
    ])

@patch("trading_analysis.backtest._save_and_report")
def test_run_backtest_skips_report(mock_save_and_report):
    df = make_long_test_df()
    result, state = run_backtest(df, report=True)
    mock_save_and_report.assert_called_once()

def test_run_backtest_basic_short_trade():
    df = make_short_test_df()
    result, state = run_backtest(df, report=False)

    assert result['total_trades'] == 1

    entries = [t for t in state['trades'] if t[0] == 'OPEN SHORT']
    closes = [t for t in state['trades'] if 'CLOSE SHORT' in t[0]]

    assert len(entries) == 1
    assert len(closes) == 1
    assert all(t[0] == 'OPEN SHORT' for t in entries)

    assert isinstance(state['balance'], float)
    assert state['balance'] != 1000.0

    for key in ['pnl', 'winrate', 'avg_trade', 'lossrate', 'final_balance']:
        assert key in result

def test_run_backtest_sl_only():
    df = pd.DataFrame([
        {'open': 100, 'high': 101, 'low': 95, 'close': 96, 'long_entry': 1, 'short_entry': 0},
        {'open': 96, 'high': 96.5, 'low': 92, 'close': 93, 'long_entry': 0, 'short_entry': 0},
    ])
    result, state = run_backtest(df, report=False, sl_pct=0.01)
    assert result['total_trades'] == 1
    assert state['balance'] < 1000.0  # SL активировался

def test_run_backtest_with_initial_state():
    df = pd.DataFrame([
        {'open': 104, 'high': 110, 'low': 102, 'close': 109},
    ])
    initial_state = {
        'balance': 1000.0,
        'position': 1,
        'entry_price': 100.0,
        'position_type': 'long',
        'trades': [],
        'leverage': 1.0,
        'fee': 0.00055
    }
    result, state = run_backtest(df, open_position=initial_state, report=False, tp_pct=0.05)
    assert result['total_trades'] == 1
    assert state['balance'] > 1000.0

def test_run_backtest_finalize_false():
    df = pd.DataFrame([
        {'open': 100, 'high': 101.0, 'low': 99, 'close': 100.5, 'long_entry': 1},
        {'open': 100.5, 'high': 101.0, 'low': 99.5, 'close': 100.8},
        {'open': 100.8, 'high': 101.0, 'low': 100, 'close': 100.9},
    ])
    result, state = run_backtest(df, report=False, finalize=False, tp_pct=0.5, sl_pct=0.5)

    assert state['position_type'] == 'long'
    assert any(t[0] == 'OPEN LONG' for t in state['trades'])
    assert not any('CLOSE' in t[0] for t in state['trades'])

def test_run_backtest_tp_only():
    df = pd.DataFrame([
        {'open': 100, 'high': 100, 'low': 99, 'close': 100, 'long_entry': 1},  # вход
        {'open': 100, 'high': 103, 'low': 99, 'close': 101},  # TP сработал
        {'open': 101, 'high': 101, 'low': 100, 'close': 100.5},  # ничего
    ])
    result, state = run_backtest(df, report=False, tp_pct=0.02, sl_pct=0.10, finalize=False)

    # Проверки
    assert result['total_trades'] == 1
    assert state['balance'] > 1000.0
    closes = [t for t in state['trades'] if 'CLOSE' in t[0]]
    assert len(closes) == 1
    assert 'TP' in closes[0][0]

def test_position_carries_over_between_windows():
    df1 = pd.DataFrame({
        "open": [99, 101, 103, 105],
        "close": [100, 102, 104, 106],
        "high": [101, 103, 105, 107],
        "low": [99, 101, 103, 105],
        "long_entry": [0, 1, 0, 0],
        "short_entry": [0, 0, 0, 0]
    })

    result1, state1 = run_backtest(
        df1,
        symbol="TEST",
        initial_balance=1000,
        tp_pct=0.2,
        sl_pct=0.2,
        risk_pct=0.1,
        report=False,
        finalize=False
    )

    # Убедиться, что позиция открыта и НЕ закрыта
    assert state1["position_type"] == "long", "Позиция должна быть открыта"
    closed_trades = [t for t in state1["trades"] if t[0].startswith("CLOSE")]
    assert len(closed_trades) == 0, "Не должно быть закрытых сделок"

    # Можно игнорировать PnL или просто проверить его знак
    assert result1["pnl"] <= 0, "PnL должен быть <= 0 (вход с комиссией)"

    # Окно 2: продолжаем и закрываем по TP
    df2 = pd.DataFrame({
        "open": [106, 108],
        "high": [107, 110],
        "low": [105, 107],
        "close": [107, 109],
        "long_entry": [0, 0],
        "short_entry": [0, 0]
    })

    result2, state2 = run_backtest(
        df2,
        symbol="TEST",
        initial_balance=state1["balance"],
        tp_pct=0.2,
        sl_pct=0.2,
        risk_pct=0.1,
        report=False,
        open_position=state1,
        finalize=True
    )

    assert state2["position_type"] is None, "Позиция должна быть закрыта"
    closed_trades = [t for t in result2["trades"] if t[0].startswith("AUTO SELL")]
    assert len(closed_trades) == 1, "Ожидается один завершённый трейд"
    assert result2["pnl"] > 0, "Должна быть прибыль после TP"
