"""Cat 6 — İstatistiksel Özellikler (Volatilite, Beta, Çarpıklık)"""

import numpy as np
import pandas as pd
from features.utils import last_n_bdays


def compute_statistical(daily: pd.DataFrame,
                         sp500_daily: pd.DataFrame,
                         month_end: pd.Timestamp) -> dict:
    feat  = {}
    close = daily["Close"].loc[:month_end]

    if len(close) < 10:
        return feat

    rets = close.pct_change().dropna()

    # Gerçekleşen volatilite
    def hvol(n):
        r = last_n_bdays(rets, n)
        return r.std() * np.sqrt(252) if len(r) >= n // 2 else np.nan

    feat["hvol_21d"]  = hvol(21)
    feat["hvol_63d"]  = hvol(63)
    feat["hvol_126d"] = hvol(126)

    # Vol rejimi: son 21d / son 63d (volatilite artışı)
    v21 = feat["hvol_21d"]
    v63 = feat["hvol_63d"]
    feat["vol_regime"] = v21 / v63 if (v21 and v63) else np.nan

    # Çarpıklık (21 günlük)
    r21 = last_n_bdays(rets, 21)
    if len(r21) >= 10:
        feat["skew_21d"] = float(r21.skew())

    # Beta (S&P500'e karşı)
    if not sp500_daily.empty:
        sp_close = sp500_daily["Close"].loc[:month_end]
        sp_rets  = sp_close.pct_change().dropna()

        def beta(n):
            stock_r = last_n_bdays(rets, n)
            sp_r    = last_n_bdays(sp_rets, n)
            common  = stock_r.index.intersection(sp_r.index)
            if len(common) < n // 3:
                return np.nan
            cov = stock_r.loc[common].cov(sp_r.loc[common])
            var = sp_r.loc[common].var()
            return cov / var if var > 0 else np.nan

        feat["beta_63d"]  = beta(63)
        feat["beta_252d"] = beta(252)

    # Max drawdown (63 günlük)
    if len(close) >= 63:
        window = last_n_bdays(close, 63)
        roll_max = window.cummax()
        dd = (window - roll_max) / roll_max
        feat["max_dd_63d"] = dd.min()

    return feat
