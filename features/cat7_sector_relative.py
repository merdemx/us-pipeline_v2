"""Cat 7 — Sektöre Göre Relatif Özellikler (Sektör ETF bazlı)"""

import numpy as np
import pandas as pd

# GICS sektör → ETF ticker eşlemesi
SECTOR_ETF_MAP = {
    "Technology":              "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Consumer Discretionary":  "XLY",
    "Industrials":             "XLI",
    "Communication Services":  "XLC",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Materials":               "XLB",
    "Basic Materials":         "XLB",
}


def compute_sector_relative(daily: pd.DataFrame,
                              sector_etf_data: dict[str, pd.DataFrame],
                              sector: str,
                              month_end: pd.Timestamp) -> dict:
    feat  = {}
    close = daily["Close"].loc[:month_end]

    if len(close) < 5:
        return feat

    etf_ticker = SECTOR_ETF_MAP.get(sector)
    if not etf_ticker or etf_ticker not in sector_etf_data:
        return feat

    etf_close = sector_etf_data[etf_ticker]["Close"].loc[:month_end]
    if len(etf_close) < 5:
        return feat

    def excess_ret(n_months):
        past = month_end - pd.DateOffset(months=n_months)
        idx_s = close.index.searchsorted(past)
        idx_e = etf_close.index.searchsorted(past)
        if idx_s >= len(close) or idx_e >= len(etf_close):
            return np.nan
        stock_ret = close.iloc[-1] / close.iloc[idx_s] - 1
        etf_ret   = etf_close.iloc[-1] / etf_close.iloc[idx_e] - 1
        return stock_ret - etf_ret

    feat["sector_excess_1m"] = excess_ret(1)
    feat["sector_excess_3m"] = excess_ret(3)
    feat["sector_excess_6m"] = excess_ret(6)

    # ETF'in kendi momentumu (sektör trendi)
    if len(etf_close) >= 22:
        feat["sector_etf_ret_1m"] = etf_close.iloc[-1] / etf_close.iloc[-22] - 1
    if len(etf_close) >= 63:
        feat["sector_etf_ret_3m"] = etf_close.iloc[-1] / etf_close.iloc[-63] - 1

    # Relatif hacim: hisse hacmi / sektör ETF hacmi oranı değil,
    # hissenin kendi hacim trendi sektöre göre normalize (cross-sectional)
    # Bu cross-sectional hesap pipeline.py'de yapılır

    return feat
