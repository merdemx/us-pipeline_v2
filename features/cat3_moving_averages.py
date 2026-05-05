"""Cat 3 — Hareketli Ortalama Oranları & Crossover'lar"""

import numpy as np
import pandas as pd


def compute_moving_averages(daily: pd.DataFrame) -> dict:
    feat  = {}
    close = daily["Close"]
    price = close.iloc[-1]

    if price <= 0:
        return feat

    def sma(n):
        if len(close) < n:
            return np.nan
        return close.rolling(n).mean().iloc[-1]

    def ema(n):
        if len(close) < n:
            return np.nan
        return close.ewm(span=n, adjust=False).mean().iloc[-1]

    sma20  = sma(20);  feat["price_sma20_ratio"]  = price / sma20  - 1 if sma20  else np.nan
    sma50  = sma(50);  feat["price_sma50_ratio"]  = price / sma50  - 1 if sma50  else np.nan
    sma100 = sma(100); feat["price_sma100_ratio"] = price / sma100 - 1 if sma100 else np.nan
    sma200 = sma(200); feat["price_sma200_ratio"] = price / sma200 - 1 if sma200 else np.nan

    ema20  = ema(20);  feat["price_ema20_ratio"]  = price / ema20  - 1 if ema20  else np.nan
    ema50  = ema(50);  feat["price_ema50_ratio"]  = price / ema50  - 1 if ema50  else np.nan
    ema200 = ema(200); feat["price_ema200_ratio"] = price / ema200 - 1 if ema200 else np.nan

    # Crossover durumları (1=golden, 0=death, ara değer=uzaklık oranı)
    if ema50 and ema200:
        feat["ema50_200_ratio"] = ema50 / ema200 - 1
    if sma20 and sma50:
        feat["sma20_50_ratio"]  = sma20 / sma50 - 1
    if sma50 and sma200:
        feat["sma50_200_ratio"] = sma50 / sma200 - 1

    # Fiyatın kaç MA'nın üzerinde olduğu (0-4 arası, trend gücü)
    ma_vals = [sma20, sma50, sma100, sma200]
    above   = sum(1 for m in ma_vals if m and price > m)
    feat["price_above_ma_count"] = above

    return feat
