import asyncio
import websockets
import json

from trading_analysis.realtime import update_live_kline, run_analysis_for_symbol

WSS_URL = "wss://stream.bybit.com/v5/public/linear"

async def listen_kline_async(symbol: str, interval: str = "30"):
    while True:
        try:
            async with websockets.connect(WSS_URL, ping_interval=1) as ws:
                print(f"[WS] Connected to WebSocket for {symbol}")
                sub_msg = {
                    "op": "subscribe",
                    "args": [f"kline.{interval}.{symbol}"]
                }
                await ws.send(json.dumps(sub_msg))

                async for message in ws:
                    data = json.loads(message)

                    if data.get("type") in ("snapshot", "delta"):
                        for candle in data.get("data", []):
                            ts = int(candle["timestamp"])
                            close = candle["close"]
                            print(f"[WS] {symbol} | close={close} | ts={ts}")

                            if candle.get("confirm"):
                                update_live_kline(symbol, candle)
                                df = run_analysis_for_symbol(symbol, interval=interval, limit=1000)
                                last_row = df.iloc[-1]

                                print(f"[SIGNAL] {symbol} | time={last_row.name} | signal={last_row['signal']}")
        except Exception as e:
            print(f"[WS] Error ({symbol}): {e}, reconnecting in 5s...")
            await asyncio.sleep(5)
