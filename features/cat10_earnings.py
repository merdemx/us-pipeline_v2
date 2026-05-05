"""
Cat 10 — Kazanç Sürprizi & PEAD (Post-Earnings Announcement Drift)

Veri kaynağı: yfinance earnings_dates
Point-in-time: as_of tarihinden önceki açıklamalar kullanılır.
"""

import numpy as np
import pandas as pd


def compute_earnings(earnings_dates: pd.DataFrame,
                      as_of: pd.Timestamp) -> dict:
    """
    earnings_dates: yf.Ticker.earnings_dates ile çekilen DataFrame.
                    Sütunlar: Reported EPS, EPS Estimate, Surprise(%)
    """
    feat = {}

    if earnings_dates is None or earnings_dates.empty:
        return feat

    # Timezone uyumsuzluğunu gider: index'i timezone-naive'e çevir
    idx = earnings_dates.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    as_of_naive = as_of.tz_localize(None) if hasattr(as_of, "tz") and as_of.tz else as_of

    past = earnings_dates.copy()
    past.index = idx
    past = past[past.index < as_of_naive].copy()
    past = past.sort_index(ascending=False)   # en yeniden en eskiye

    if past.empty:
        return feat

    # Son açıklama
    last = past.iloc[0]
    feat["days_since_earnings"] = (as_of - past.index[0]).days

    # Sürpriz yüzdesi
    surp_col = [c for c in past.columns if "Surprise" in c or "surprise" in c]
    if surp_col:
        feat["eps_surprise_last"]  = _safe_float(last[surp_col[0]])
        if len(past) >= 2:
            feat["eps_surprise_prev"]  = _safe_float(past.iloc[1][surp_col[0]])
        if len(past) >= 4:
            # Son 4 çeyreğin ortalama sürprizi (süreklilik)
            feat["eps_surprise_4q_avg"] = float(past.iloc[:4][surp_col[0]].mean())

    # Reported EPS büyümesi (son çeyrek vs yıl önce)
    rep_col = [c for c in past.columns if "Reported" in c or "reported" in c]
    if rep_col and len(past) >= 4:
        eps_q0 = _safe_float(past.iloc[0][rep_col[0]])
        eps_q4 = _safe_float(past.iloc[3][rep_col[0]]) if len(past) > 3 else np.nan
        if eps_q0 and eps_q4 and eps_q4 != 0:
            feat["eps_growth_yoy"] = eps_q0 / abs(eps_q4) - 1

    # PEAD sinyali: son açıklamadan bu yana geçen süre ve sürpriz
    # (kısa vadeli drift için: 0-30 gün içindeyse ve sürpriz pozitifse güçlü sinyal)
    days = feat.get("days_since_earnings", 999)
    surp = feat.get("eps_surprise_last", 0) or 0
    if days <= 30:
        feat["pead_signal"] = np.sign(surp) * max(0, 1 - days / 30)
    else:
        feat["pead_signal"] = 0.0

    return feat


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan
