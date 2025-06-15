from trading_analysis.db import fetch_and_save_all_ohlcv, init_db, safe_check_ohlcv_integrity
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
    symbol, interval = 'BTCUSDT', '30'
    # has_data = safe_check_ohlcv_integrity(symbol=symbol, interval=interval)
    # if not has_data:
    #     fetch_and_save_all_ohlcv(symbol=symbol, interval=interval)
    walk_forward_test(symbol=symbol, interval="30")
