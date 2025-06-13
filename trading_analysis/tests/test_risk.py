from trading_analysis.risk import calculate_position_size


def test_calculate_position_size_normal():
    size = calculate_position_size(balance=1000, risk_pct=0.01, leverage=10, sl_pct=0.02, price=100)
    expected = (1000 * 0.01 * 10 / 0.02) / 100  # → 5.0
    assert abs(size - expected) < 1e-6

def test_calculate_position_size_invalid_price_or_sl():
    assert calculate_position_size(1000, 0.01, 10, 0.02, 0) == 0
    assert calculate_position_size(1000, 0.01, 10, 0, 100) == 0

def test_calculate_position_size_invalid_risk_or_leverage():
    assert calculate_position_size(1000, 0, 10, 0.02, 100) == 0
    assert calculate_position_size(1000, 0.01, 0, 0.02, 100) == 0

def test_calculate_position_size_type_check():
    assert calculate_position_size("1000", 0.01, 10, 0.02, 100) == 0
    assert calculate_position_size(1000, "0.01", 10, 0.02, 100) == 0

def test_calculate_position_size_high_sl_pct():
    size = calculate_position_size(1000, 0.01, 10, 0.99, 100)
    expected = (1000 * 0.01 * 10 / 0.99) / 100
    assert abs(size - expected) < 1e-6
