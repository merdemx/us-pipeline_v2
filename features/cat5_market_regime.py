"""Cat 5 — Piyasa Rejimi (S&P500, VIX)"""

import numpy as np
import pandas as pd


def _adx(daily: pd.DataFrame, period: int = 14) -> float:
    high, low, close = daily["High"], daily["Low"], daily["Close"]
    if len(daily) < period + 1:
        return np.nan
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    dm_plus  = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr     = tr.rolling(period).mean()
    di_plus  = 100 * dm_plus.rolling(period).mean()  / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.rolling(period).mean() / atr.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.rolling(period).mean()
    return adx.iloc[-1]


def compute_market_regime(sp500_daily: pd.DataFrame,
                           vix_daily: pd.DataFrame,
                           month_end: pd.Timestamp) -> dict:
    feat = {}

    # ── S&P500 rejimi ──────────────────────────────────────────────────
    if not sp500_daily.empty:
        sp = sp500_daily["Close"].loc[:month_end]
        if len(sp) >= 50:
            sma50  = sp.rolling(50).mean().iloc[-1]
            sma200 = sp.rolling(200).mean().iloc[-1] if len(sp) >= 200 else np.nan
            price  = sp.iloc[-1]

            feat["sp500_above_sma50"]  = int(price > sma50)
            feat["sp500_above_sma200"] = int(price > sma200) if sma200 else np.nan
            feat["sp500_ret_1m"] = sp.iloc[-1] / sp.iloc[-22] - 1 if len(sp) >= 22 else np.nan
            feat["sp500_ret_3m"] = sp.iloc[-1] / sp.iloc[-63] - 1 if len(sp) >= 63 else np.nan

        sp_df = sp500_daily.loc[:month_end]
        if len(sp_df) >= 28:
            adx = _adx(sp_df)
            feat["sp500_adx"] = adx
            # Rejim: 0=ayı, 1=yatay, 2=boğa
            if adx and not np.isnan(adx):
                above_200 = feat.get("sp500_above_sma200", 0)
                if adx > 25 and above_200:
                    feat["sp500_regime"] = 2
                elif adx > 25 and not above_200:
                    feat["sp500_regime"] = 0
                else:
                    feat["sp500_regime"] = 1

    # ── VIX rejimi ─────────────────────────────────────────────────────
    if not vix_daily.empty:
        vix = vix_daily["Close"].loc[:month_end]
        if len(vix):
            vix_now = vix.iloc[-1]
            feat["vix_level"] = vix_now
            feat["vix_ret_1m"] = vix.iloc[-1] / vix.iloc[-22] - 1 if len(vix) >= 22 else np.nan
            # Rejim: 0=yüksek risk (>25), 1=orta (15-25), 2=düşük (<15)
            if vix_now > 25:
                feat["vix_regime"] = 0
            elif vix_now > 15:
                feat["vix_regime"] = 1
            else:
                feat["vix_regime"] = 2

            # VIX'in kendi ortalamasına oranı (relatif korku seviyesi)
            if len(vix) >= 63:
                feat["vix_vs_avg63"] = vix_now / vix.rolling(63).mean().iloc[-1] - 1

    return feat
