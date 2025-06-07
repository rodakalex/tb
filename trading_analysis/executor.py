from pybit.unified_trading import HTTP

API_KEY = "tC8sAzgnKmAjvMEQcb"
API_SECRET = "aKxUY7bZtlveYu8ezeTPeLgeMJWeTvmpkJCF"

client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)

def place_order(symbol: str, side: str, qty: float, order_type="Market"):
    try:
        response = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,  # "Buy" or "Sell"
            order_type=order_type,
            qty=qty
        )
        print(f"[ORDER] {side} {qty} {symbol} → Order ID: {response['result']['orderId']}")
    except Exception as e:
        print(f"[ORDER ERROR] {e}")
