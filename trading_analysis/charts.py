# trading_analysis/charts.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_volume(df):
    df = df.copy()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    colors = ['green' if close > open_ else 'red' for close, open_ in zip(df['close'], df['open'])]
    plt.figure(figsize=(16, 5))
    plt.bar(df.index, df['volume'], color=colors, width=0.03, alpha=0.5, label="Volume")
    plt.plot(df.index, df["Volume_SMA_20"], label="SMA Volume (20)", color="orange", linewidth=2)
    ymax = df['volume'].quantile(0.995) * 1.1
    plt.ylim(0, ymax)
    plt.title("Volume and Volume SMA")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig("volume.png")

def plot_macd(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["MACD"], label="MACD", color="blue")
    plt.plot(df.index, df["Signal"], label="Signal", color="red")
    histogram = df["Histogram"]
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    plt.bar(df.index, histogram, color=colors, label="Histogram")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    buy_signals = (df["MACD"].shift(1) < df["Signal"].shift(1)) & (df["MACD"] > df["Signal"])
    sell_signals = (df["MACD"].shift(1) > df["Signal"].shift(1)) & (df["MACD"] < df["Signal"])
    plt.plot(df.index[buy_signals], df["MACD"][buy_signals], '^', color='green', label="Buy Signal", markersize=10)
    plt.plot(df.index[sell_signals], df["MACD"][sell_signals], 'v', color='red', label="Sell Signal", markersize=10)
    plt.title("MACD Indicator with Buy/Sell Signals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("macd.png")

def plot_rsi(df):
    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df["RSI"], label="RSI", color="purple")
    plt.axhline(70, color="red", linestyle="--", linewidth=1, label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
    plt.title("Relative Strength Index (RSI)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rsi.png")

def plot_price_with_ema_bb(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["close"], label="Close Price", linewidth=1)
    plt.plot(df.index, df["EMA_9"], label="EMA 9", linestyle="--")
    plt.plot(df.index, df["EMA_21"], label="EMA 21", linestyle="--")
    plt.plot(df.index, df["EMA_100"], label="EMA 100", linestyle="--", color="blue")
    plt.plot(df.index, df["EMA_200"], label="EMA 200", linestyle="--", color="purple")
    plt.fill_between(df.index, df["BBL"], df["BBU"], color='gray', alpha=0.2, label="Bollinger Bands")
    plt.title("Price with EMA and Bollinger Bands")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_ema_bb.png")

def generate_all_charts(df):
    plot_price_with_ema_bb(df)
    plot_rsi(df)
    plot_volume(df)
    plot_macd(df)
