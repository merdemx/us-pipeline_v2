"""
Cat 9 — Temel Analiz Özellikleri (Point-in-time)

Veri kaynağı: yfinance quarterly_income_stmt, quarterly_balance_sheet, info
Point-in-time: her snapshot için yalnızca o tarihe kadar açıklanmış
               en son çeyrek raporu kullanılır (look-ahead bias yok).
"""

import numpy as np
import pandas as pd


def _ttm(stmt: pd.DataFrame, row: str, as_of: pd.Timestamp) -> float:
    """Trailing 12 month: as_of tarihinden önce son 4 çeyreğin toplamı."""
    if stmt is None or stmt.empty or row not in stmt.index:
        return np.nan
    cols = [c for c in stmt.columns if pd.Timestamp(c) < as_of]
    if not cols:
        return np.nan
    last4 = stmt.loc[row, cols[:4]]
    return float(last4.sum()) if last4.notna().any() else np.nan


def _latest(stmt: pd.DataFrame, row: str, as_of: pd.Timestamp) -> float:
    """as_of öncesi en son çeyrek değeri."""
    if stmt is None or stmt.empty or row not in stmt.index:
        return np.nan
    cols = [c for c in stmt.columns if pd.Timestamp(c) < as_of]
    if not cols:
        return np.nan
    val = stmt.loc[row, cols[0]]
    return float(val) if pd.notna(val) else np.nan


def compute_fundamental(price: float,
                         shares: float,
                         income_stmt: pd.DataFrame,
                         balance_sheet: pd.DataFrame,
                         info: dict,
                         as_of: pd.Timestamp) -> dict:
    feat = {}

    if price <= 0 or not shares:
        return feat

    mkt_cap = price * shares

    # ── Trailing 12m Gelir Tablosu ─────────────────────────────────────
    revenue_ttm     = _ttm(income_stmt, "Total Revenue",        as_of)
    net_income_ttm  = _ttm(income_stmt, "Net Income",           as_of)
    ebitda_ttm      = _ttm(income_stmt, "EBITDA",               as_of)
    gross_profit_ttm= _ttm(income_stmt, "Gross Profit",         as_of)
    op_income_ttm   = _ttm(income_stmt, "Operating Income",     as_of)

    # EPS (TTM)
    eps_ttm = net_income_ttm / shares if (net_income_ttm and shares) else np.nan

    # ── Değerleme ──────────────────────────────────────────────────────
    feat["pe_ttm"]  = price / eps_ttm       if (eps_ttm and eps_ttm > 0) else np.nan
    feat["ps_ttm"]  = mkt_cap / revenue_ttm if (revenue_ttm and revenue_ttm > 0) else np.nan
    feat["ev_ebitda"] = np.nan  # EV hesabı için net borç gerekli, balance sheet'ten

    # ── Karlılık ───────────────────────────────────────────────────────
    feat["net_margin"]   = net_income_ttm / revenue_ttm  if (net_income_ttm and revenue_ttm) else np.nan
    feat["gross_margin"] = gross_profit_ttm / revenue_ttm if (gross_profit_ttm and revenue_ttm) else np.nan
    feat["op_margin"]    = op_income_ttm / revenue_ttm   if (op_income_ttm and revenue_ttm) else np.nan

    # ── Büyüme ─────────────────────────────────────────────────────────
    # Revenue büyümesi: son çeyrek vs yıl önce aynı çeyrek
    if income_stmt is not None and not income_stmt.empty and "Total Revenue" in income_stmt.index:
        cols = [c for c in income_stmt.columns if pd.Timestamp(c) < as_of]
        if len(cols) >= 5:
            rev_q0 = float(income_stmt.loc["Total Revenue", cols[0]])
            rev_q4 = float(income_stmt.loc["Total Revenue", cols[4]])
            feat["rev_growth_yoy"] = rev_q0 / rev_q4 - 1 if (rev_q4 and rev_q4 != 0) else np.nan

    # ── Bilanço ────────────────────────────────────────────────────────
    book_value   = _latest(balance_sheet, "Stockholders Equity", as_of)
    total_debt   = _latest(balance_sheet, "Total Debt",          as_of)
    total_assets = _latest(balance_sheet, "Total Assets",        as_of)
    cash         = _latest(balance_sheet, "Cash And Cash Equivalents", as_of)

    bvps = book_value / shares if (book_value and shares) else np.nan
    feat["pb"]          = price / bvps         if (bvps and bvps > 0) else np.nan
    feat["debt_equity"] = total_debt / book_value if (total_debt and book_value and book_value > 0) else np.nan
    feat["roe"]         = net_income_ttm / book_value if (net_income_ttm and book_value and book_value > 0) else np.nan
    feat["roa"]         = net_income_ttm / total_assets if (net_income_ttm and total_assets and total_assets > 0) else np.nan

    # EV/EBITDA
    if total_debt and cash and ebitda_ttm and ebitda_ttm > 0:
        ev = mkt_cap + total_debt - cash
        feat["ev_ebitda"] = ev / ebitda_ttm

    # ── Short interest & analist (anlık değer — eğitim için yaklaşım) ──
    feat["short_ratio"]      = info.get("shortRatio",      np.nan)
    feat["short_pct_float"]  = info.get("shortPercentOfFloat", np.nan)
    feat["analyst_rec_mean"] = info.get("recommendationMean",  np.nan)  # 1=güçlü al, 5=sat

    target_price = info.get("targetMeanPrice")
    if target_price and price > 0:
        feat["price_target_upside"] = target_price / price - 1
    else:
        feat["price_target_upside"] = np.nan

    feat["analyst_count"] = info.get("numberOfAnalystOpinions", np.nan)

    return feat
