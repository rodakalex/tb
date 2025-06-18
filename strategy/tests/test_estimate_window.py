from strategy.objective import estimate_window_size_from_params

def test_window_size_returns_min_if_no_weights():
    result = estimate_window_size_from_params(best_params={}, verbose=False)
    assert result == 300  # default min_window_size

def test_window_size_with_weights():
    params = {
        "w_rsi": 2,  # период 14
        "w_macd": 1,  # период 26
    }
    result = estimate_window_size_from_params(best_params=params, verbose=False)
    # (14*2 + 26*1) / 3 * 8 = 136
    assert 130 <= result <= 150

def test_window_size_with_weights():
    params = {
        "w_rsi": 2,  # период 14
        "w_macd": 1,  # период 26
    }
    result = estimate_window_size_from_params(best_params=params, verbose=False)
    # Расчёт базового значения: (14*2 + 26*1) / 3 * 8 = ~120, но оно ниже min_window_size
    assert result == 300  # потому что min_window_size = 300
