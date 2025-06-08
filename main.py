from trading_analysis.db import check_ohlcv_integrity, fetch_and_save_all_ohlcv, init_db
from strategy.walkforward import walk_forward_test


def run_realtime():
    from trading_analysis.bybit_api import update_hystory
    from trading_analysis.websocket_kline import listen_kline_async
    import asyncio

    symbol = "PRIMEUSDT"
    interval = "30"
    loop = asyncio.get_event_loop()
    loop.create_task(listen_kline_async(symbol, interval))
    loop.create_task(update_hystory(symbol, int(interval)))
    loop.run_forever()


if __name__ == "__main__":
    init_db()
    walk_forward_test(symbol="PRIMEUSDT", interval="30")
