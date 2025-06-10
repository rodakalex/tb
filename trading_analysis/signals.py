# trading_analysis/signals.py
import pandas_ta as ta

_debug = False
_generate_signals_counter = 0

def safe_assign(df, col_name, series, dtype="float64"):
    df[col_name] = series.astype(dtype)

def generate_signals(df):
    df = df.copy()

    # Общие фильтры
    df["volume_above_avg"] = (df["volume"] > df["Volume_SMA_20"]).astype(int)
    df["atr_filter"] = (df["atr"] > df["atr"].rolling(50).mean()).astype(int)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ---- LONG ----
    df["long_ema_cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)
    df["long_trend"] = (df["close"] > df["EMA_200"]).astype(int)
    df["long_rsi"] = ((df["RSI"] > 45) & (df["RSI"] < 75)).astype(int)
    df["long_macd"] = (
        (df["MACD"] > df["Signal"]) &
        (df["MACD"].shift(1) <= df["Signal"].shift(1)) &
        (df["MACD"] > 0)
    ).astype(int)

    df["long_score"] = (
        df["long_ema_cross"] +
        df["long_trend"] +
        df["long_rsi"] +
        df["long_macd"] +
        df["volume_above_avg"]
    )

    # ---- SHORT ----
    df["short_ema_cross"] = (df["EMA_9"] < df["EMA_21"]).astype(int)
    df["short_trend"] = (df["close"] < df["EMA_200"]).astype(int)
    df["short_rsi"] = (df["RSI"] < 50).astype(int)
    df["short_macd"] = (
        (df["MACD"] < df["Signal"]) &
        (df["MACD"].shift(1) >= df["Signal"].shift(1)) &
        (df["MACD"] < 0) &
        (df["Signal"] < 0)
    ).astype(int)
    df["short_bb_rebound"] = (df["close"] > df["BBM"]).astype(int)
    df["short_below_ema9"] = (df["close"] < df["EMA_9"]).astype(int)
    df["ema200_down"] = (df["EMA_200"].diff() < 0).astype(int)
    df["bbp_filter"] = (df["BBP"] > 0.5).astype(int)

    df["short_score"] = (
        df["short_ema_cross"] +
        df["short_trend"] +
        df["short_rsi"] +
        df["short_macd"] +
        df["volume_above_avg"] +
        df["short_bb_rebound"] +
        df["short_below_ema9"]
    )

    # ---- Входы ----
    df["long_entry"] = (df["long_score"] >= 5) & df["atr_filter"]
    # 🎯 Динамический вход в SHORT в падающем тренде
    df["volume_peak"] = (df["volume"] > df["volume"].rolling(10).max().shift(1)).astype(int)
    df["short_entry"] = (
        (df["short_score"] >= 5) &
        df["atr_filter"] &
        (df["ema200_down"] == 1) &
        (df["volume_peak"] == 1)
    )

    return df

def generate_signals(df, params=None):
    global _generate_signals_counter
    _generate_signals_counter += 1
    df = df.copy()
    if params is None:
        params = {}

    def p(key, default=1):
        return params.get(key, default)

    # Основные индикаторы
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volume_above_avg"] = (df["volume"] > df["Volume_SMA_20"]).astype(int)
    df["atr_filter"] = (df["ATR"] > df["ATR"].rolling(50).mean()).astype(int)
    df["ema200_down"] = (df["EMA_200"].diff() < 0).astype(int)
    df["volume_peak"] = (df["volume"] > df["volume"].rolling(10).max().shift(1)).astype(int)
    df["bbp_filter"] = (df["BBP"] > 0.5).astype(int)

    # MACD
    df["long_macd"] = (
        (df["MACD"] > df["Signal"]) & 
        (df["MACD"].shift(1) <= df["Signal"].shift(1)) &
        (df["MACD"] > 0)
    ).astype(int)
    df["short_macd"] = (
        (df["MACD"] < df["Signal"]) &
        (df["MACD"].shift(1) >= df["Signal"].shift(1)) &
        (df["MACD"] < 0) & (df["Signal"] < 0)
    ).astype(int)

    # RSI
    df["long_rsi"] = ((df["RSI"] > 45) & (df["RSI"] < 75)).astype(int)
    df["short_rsi"] = (df["RSI"] < 50).astype(int)

    # EMA
    df["long_ema_cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)
    df["short_ema_cross"] = (df["EMA_9"] < df["EMA_21"]).astype(int)

    # Trend
    df["long_trend"] = (df["close"] > df["EMA_200"]).astype(int)
    df["short_trend"] = (df["close"] < df["EMA_200"]).astype(int)

    # BB
    df["short_bb_rebound"] = (df["close"] > df["BBM"]).astype(int)
    df["short_below_ema9"] = (df["close"] < df["EMA_9"]).astype(int)

    # 🔥 Новые фильтры
    df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    stochrsi = ta.stochrsi(df["close"], length=14)
    df["StochRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
    df["StochRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["ROC"] = ta.roc(df["close"], length=5)
    df["volume_zscore"] = ((df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()).fillna(0)

    # Доп. сигналы
    df["long_stochrsi"] = ((df["StochRSI_K"] < 20) & (df["StochRSI_K"] > df["StochRSI_D"])).astype(int)
    df["short_stochrsi"] = ((df["StochRSI_K"] > 80) & (df["StochRSI_K"] < df["StochRSI_D"])).astype(int)
    df["long_cci"] = (df["CCI"] < -100).astype(int)
    df["short_cci"] = (df["CCI"] > 100).astype(int)
    df["long_mfi"] = (df["MFI"] < 30).astype(int)
    df["short_mfi"] = (df["MFI"] > 70).astype(int)
    df["trend_strength"] = (df["ADX"] > 20).astype(int)
    df["long_roc"] = (df["ROC"] > 1).astype(int)
    df["short_roc"] = (df["ROC"] < -1).astype(int)
    df["volume_spike"] = (df["volume_zscore"] > 2).astype(int)

    # Donchian breakout
    donchian_high = df["high"].rolling(window=20).max()
    donchian_low = df["low"].rolling(window=20).min()
    df["long_donchian"] = (df["close"] > donchian_high.shift(1)).astype(int)
    df["short_donchian"] = (df["close"] < donchian_low.shift(1)).astype(int)

    # TEMA crossover
    df["TEMA_9"] = ta.tema(df["close"], length=9)
    df["TEMA_21"] = ta.tema(df["close"], length=21)
    df["long_tema_cross"] = (df["TEMA_9"] > df["TEMA_21"]).astype(int)
    df["short_tema_cross"] = (df["TEMA_9"] < df["TEMA_21"]).astype(int)

    # Score суммируется динамически (long)
    df["long_score"] = (
        p("w_ema_cross") * df["long_ema_cross"] +
        p("w_trend") * df["long_trend"] +
        p("w_rsi") * df["long_rsi"] +
        p("w_macd") * df["long_macd"] +
        p("w_stochrsi") * df["long_stochrsi"] +
        p("w_cci") * df["long_cci"] +
        p("w_mfi") * df["long_mfi"] +
        p("w_volume") * df["volume_above_avg"] +
        p("w_roc") * df["long_roc"] +
        p("w_volspike") * df["volume_spike"] +
        p("w_donchian") * df["long_donchian"] +
        p("w_tema_cross") * df["long_tema_cross"]
    )

    # Score (short)
    df["short_score"] = (
        p("w_ema_cross") * df["short_ema_cross"] +
        p("w_trend") * df["short_trend"] +
        p("w_rsi") * df["short_rsi"] +
        p("w_macd") * df["short_macd"] +
        p("w_stochrsi") * df["short_stochrsi"] +
        p("w_cci") * df["short_cci"] +
        p("w_mfi") * df["short_mfi"] +
        p("w_bb_rebound") * df["short_bb_rebound"] +
        p("w_below_ema9") * df["short_below_ema9"] +
        p("w_volume") * df["volume_above_avg"] +
        p("w_roc") * df["short_roc"] +
        p("w_volspike") * df["volume_spike"] +
        p("w_donchian") * df["short_donchian"] +
        p("w_tema_cross") * df["short_tema_cross"]
    )

    df["long_entry"] = (df["long_score"] >= p("long_score_threshold", 5)) & (df["atr_filter"] == 1) & (df["trend_strength"] == 1)
    df["short_entry"] = (
        (df["short_score"] >= p("short_score_threshold", 5)) &
        (df["atr_filter"] == 1) &
        (df["ema200_down"] == 1)
    )

    if _generate_signals_counter % 25 == 0 and _debug:
        print("🧪 Генерация сигналов:")
        print(f"→ Вызов #{_generate_signals_counter}")
        print(f"→ Макс long_score: {df['long_score'].max()}")
        print(f"→ Макс short_score: {df['short_score'].max()}")
        print(f"→ long_score_threshold: {p('long_score_threshold')}")
        print(f"→ short_score_threshold: {p('short_score_threshold')}")
        print(f"→ Кол-во long сигналов: {df['long_entry'].sum()}")
        print(f"→ Кол-во short сигналов: {df['short_entry'].sum()}")

    return df
