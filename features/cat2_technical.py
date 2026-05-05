"""Cat 2 — Teknik Göstergeler (RSI, MACD, Stochastic, Bollinger)"""

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period:
        return np.nan
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


def _macd(close: pd.Series):
    if len(close) < 26:
        return np.nan, np.nan, np.nan
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]


def _stochastic(daily: pd.DataFrame, k_period: int = 14):
    if len(daily) < k_period:
        return np.nan, np.nan
    low_min  = daily["Low"].rolling(k_period).min().iloc[-1]
    high_max = daily["High"].rolling(k_period).max().iloc[-1]
    if high_max == low_min:
        return np.nan, np.nan
    k = 100 * (daily["Close"].iloc[-1] - low_min) / (high_max - low_min)
    d = 100 * (daily["Close"] - daily["Low"].rolling(k_period).min()) / \
        (daily["High"].rolling(k_period).max() - daily["Low"].rolling(k_period).min())
    d = d.rolling(3).mean().iloc[-1]
    return k, d


def _bollinger_position(close: pd.Series, period: int = 20):
    if len(close) < period:
        return np.nan
    ma  = close.rolling(period).mean().iloc[-1]
    std = close.rolling(period).std().iloc[-1]
    if std == 0:
        return np.nan
    return (close.iloc[-1] - ma) / (2 * std)   # -1→+1 bandı


def compute_technical(daily: pd.DataFrame) -> dict:
    feat  = {}
    close = daily["Close"]

    if len(close) < 14:
        return feat

    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_21"] = _rsi(close, 21)

    macd_val, macd_sig, macd_hist = _macd(close)
    feat["macd_val"]  = macd_val
    feat["macd_sig"]  = macd_sig
    feat["macd_hist"] = macd_hist
    # Normalize: MACD histogram / fiyat (karşılaştırılabilirlik)
    price = close.iloc[-1]
    feat["macd_hist_norm"] = macd_hist / price if price > 0 else np.nan

    k, d = _stochastic(daily)
    feat["stoch_k"] = k
    feat["stoch_d"] = d

    feat["bb_pos"] = _bollinger_position(close, 20)

    # ATR (Average True Range) normalize
    if len(daily) >= 14:
        high, low, prev_close = daily["High"], daily["Low"], daily["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        feat["atr_norm"] = atr / price if price > 0 else np.nan

    return feat
