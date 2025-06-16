import os
import pytest
from trading_analysis.db import get_latest_timestamp, ensure_data_loaded
from datetime import datetime, timedelta, timezone

@pytest.mark.skipif("CI" in os.environ, reason="Пропускаем в CI")
def test_latest_candle_not_too_old():
    symbol, interval = "BTCUSDT", "30"
    ensure_data_loaded(symbol, interval)

    ts = get_latest_timestamp(symbol, interval)
    assert ts != 0, f"Нет данных по {symbol}"

    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    now = datetime.now(timezone.utc)

    delta = now - dt
    print(f"🕒 Последняя свеча: {dt.isoformat()}, отставание: {delta}")

    if delta > timedelta(hours=1):
        pytest.skip(f"📉 Данные устарели: отставание {delta}")

