"""Cat 8 — Takvim & Mevsimsel Etkiler"""

import numpy as np
import pandas as pd


def compute_calendar(month_end: pd.Timestamp) -> dict:
    feat = {}

    m = month_end.month
    feat["month"]         = m
    feat["quarter"]       = (m - 1) // 3 + 1
    feat["is_q_end"]      = int(m in [3, 6, 9, 12])
    feat["is_january"]    = int(m == 1)    # Ocak anomalisi
    feat["is_december"]   = int(m == 12)   # Tax-loss selling
    feat["is_april"]      = int(m == 4)    # "Sell in May" öncesi
    feat["is_may_oct"]    = int(m in [5, 6, 7, 8, 9, 10])  # zayıf mevsim

    # Sin/cos encoding: döngüsel ay temsili
    feat["month_sin"] = np.sin(2 * np.pi * m / 12)
    feat["month_cos"] = np.cos(2 * np.pi * m / 12)

    return feat
