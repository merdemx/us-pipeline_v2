"""Cat 4 — Hacim & Likidite"""

import numpy as np
import pandas as pd
from features.utils import last_n_bdays


def compute_volume_liquidity(daily: pd.DataFrame) -> dict:
    feat  = {}
    close  = daily["Close"]
    volume = daily["Volume"]

    if len(daily) < 5:
        return feat

    # Ortalama günlük dolar hacmi
    dv = (close * volume).replace(0, np.nan)
    feat["avg_dv_21d"]  = last_n_bdays(dv, 21).mean()
    feat["avg_dv_63d"]  = last_n_bdays(dv, 63).mean() if len(dv) >= 63 else np.nan

    # Relatif hacim: son 21 gün / son 63 gün ortalaması
    vol_21  = last_n_bdays(volume, 21).mean()
    vol_63  = last_n_bdays(volume, 63).mean() if len(volume) >= 63 else np.nan
    feat["rvol_21_63"] = vol_21 / vol_63 if vol_63 else np.nan

    # Hacim trendi: son 21d vs önceki 21d
    if len(volume) >= 42:
        v_recent = volume.iloc[-21:].mean()
        v_prev   = volume.iloc[-42:-21].mean()
        feat["vol_trend"] = v_recent / v_prev - 1 if v_prev > 0 else np.nan

    # Amihud illiquidity (son 21 gün): |ret| / dolar_hacim
    if len(daily) >= 21:
        rets    = last_n_bdays(close.pct_change().abs(), 21)
        dv_21   = last_n_bdays(dv, 21)
        illiq   = (rets / dv_21.replace(0, np.nan)).mean()
        feat["amihud_21d"] = illiq * 1e6   # ölçek

    # Fiyat * hacim son gün (büyüklük sinyali)
    feat["last_dv"] = dv.iloc[-1] if len(dv) else np.nan

    return feat
