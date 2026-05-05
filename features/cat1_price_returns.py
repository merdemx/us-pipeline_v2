"""Cat 1 — Fiyat Momentum & Return Features (Aylık)"""

import numpy as np
import pandas as pd
from features.utils import last_n_bdays


def compute_price_returns(daily: pd.DataFrame, month_end: pd.Timestamp) -> dict:
    """
    Aylık snapshot için fiyat bazlı return feature'ları hesaplar.
    daily: tek ticker günlük OHLCV, month_end tarihine kadar filtrelenmiş.
    """
    feat = {}

    close = daily["Close"].sort_index()
    if len(close) < 5:
        return feat

    price_now = close.iloc[-1]

    def ret_n_days(n):
        idx = close.index.searchsorted(month_end - pd.offsets.BDay(n))
        if idx >= len(close):
            return np.nan
        past = close.iloc[idx]
        return price_now / past - 1 if past > 0 else np.nan

    def ret_n_months(n):
        past_date = month_end - pd.DateOffset(months=n)
        idx = close.index.searchsorted(past_date)
        idx = min(idx, len(close) - 1)
        past = close.iloc[idx]
        return price_now / past - 1 if past > 0 else np.nan

    feat["ret_1m"]  = ret_n_months(1)
    feat["ret_2m"]  = ret_n_months(2)
    feat["ret_3m"]  = ret_n_months(3)
    feat["ret_6m"]  = ret_n_months(6)
    feat["ret_9m"]  = ret_n_months(9)
    feat["ret_12m"] = ret_n_months(12)

    # Yıl başından itibaren getiri
    ytd_date = pd.Timestamp(month_end.year, 1, 1)
    idx_ytd  = close.index.searchsorted(ytd_date)
    if idx_ytd < len(close):
        past_ytd = close.iloc[idx_ytd]
        feat["ret_ytd"] = price_now / past_ytd - 1 if past_ytd > 0 else np.nan
    else:
        feat["ret_ytd"] = np.nan

    # 52 haftalık high/low uzaklığı
    window_252 = last_n_bdays(close, 252) if len(close) >= 252 else close
    if len(window_252):
        feat["dist_52w_high"] = price_now / window_252.max() - 1
        feat["dist_52w_low"]  = price_now / window_252.min() - 1

    # Son ay volatilitesi normalize edilmiş return
    last_21 = last_n_bdays(close, 21)
    if len(last_21) >= 5:
        daily_rets = last_21.pct_change().dropna()
        vol = daily_rets.std() * np.sqrt(252)
        feat["ret_1m_vol_adj"] = feat.get("ret_1m", np.nan) / vol if vol > 0 else np.nan

    return feat
