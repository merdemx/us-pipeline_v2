"""
US Aylık Hisse Tahmin Feature Pipeline — v2
=============================================
Kullanım:
    python pipeline.py --ticker_file us_tickers.xlsx
    python pipeline.py --ticker_file us_tickers.xlsx --start 2018-01-01

Çıktı:
    output/features_monthly_YYYYMMDD.parquet

v2 değişiklikleri:
    - target_pure_momentum_rank kaldırıldı
    - label_alpha_m eklendi   (BIST label_alpha_w aylık karşılığı)
    - label_alpha_reversal_m eklendi (BIST label_reversal_w aylık karşılığı)
"""

import argparse
import logging
import pickle
import random
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from features.cat1_price_returns   import compute_price_returns
from features.cat2_technical       import compute_technical
from features.cat3_moving_averages import compute_moving_averages
from features.cat4_volume_liquidity import compute_volume_liquidity
from features.cat5_market_regime   import compute_market_regime
from features.cat6_statistical     import compute_statistical
from features.cat7_sector_relative import compute_sector_relative, SECTOR_ETF_MAP
from features.cat8_calendar        import compute_calendar
from features.cat9_fundamental     import compute_fundamental
from features.cat10_earnings       import compute_earnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_START  = "2018-01-01"
CACHE_DIR      = Path("cache/raw")
FUND_CACHE_DIR = Path("cache/fundamentals")
OUTPUT_DIR     = Path("output")
BATCH_SIZE     = 50
BATCH_SLEEP    = (8, 15)

MACRO_TICKERS = {
    "sp500":  "^GSPC",
    "vix":    "^VIX",
    "tnx":    "^TNX",
    "dxy":    "DX-Y.NYB",
    "wti":    "CL=F",
    "gold":   "GC=F",
}

SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))


# ── Ticker listesi ────────────────────────────────────────────────────
def load_tickers(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df["ticker"] = df["ticker"].str.strip()
    log.info(f"{len(df)} ticker yüklendi: {xlsx_path}")
    return df


# ── Fiyat verisi indirme (cache + batch) ──────────────────────────────
def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _batch_download(tickers, start, end, max_retries=4) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            return yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False,
                timeout=120, group_by="ticker",
            )
        except Exception as e:
            err = str(e)
            if any(k in err for k in ["429", "Too Many", "RateLimit", "Timeout", "timed out"]):
                wait = (2 ** attempt) * 20 + random.uniform(10, 20)
                log.warning(f"Rate limit — {wait:.0f}s bekleniyor...")
                time.sleep(wait)
            else:
                log.error(f"Kurtarılamaz hata: {err[:100]}")
                return pd.DataFrame()
    return pd.DataFrame()


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()
        df = raw[ticker] if ticker in lvl0 else (
             raw.xs(ticker, axis=1, level=1) if ticker in lvl1 else pd.DataFrame())
    else:
        df = raw.copy()
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c) for c in df.columns]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()
    return df[needed].dropna(subset=["Close"]).sort_index()


def load_price_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result, needs_full, needs_update = {}, [], {}
    end_ts = pd.Timestamp(end)

    for tkr in tickers:
        cp = _cache_path(tkr)
        if cp.exists():
            try:
                df = pd.read_parquet(cp)
                df.index = pd.to_datetime(df.index)
                df.dropna(subset=["Close"], inplace=True)
                result[tkr] = df
                if df.index.max() < end_ts - pd.Timedelta(days=3):
                    needs_update[tkr] = str(df.index.max().date())
            except Exception:
                needs_full.append(tkr)
        else:
            needs_full.append(tkr)

    if needs_update:
        update_list = list(needs_update.keys())
        batch_start = min(needs_update.values())
        log.info(f"Güncelleme: {len(update_list)} ticker")
        for i in range(0, len(update_list), BATCH_SIZE):
            batch = update_list[i:i + BATCH_SIZE]
            raw = _batch_download(batch, batch_start, end)
            if not raw.empty:
                for tkr in batch:
                    fresh = _extract(raw, tkr)
                    if not fresh.empty:
                        combined = pd.concat([result[tkr], fresh]).loc[~pd.Series(
                            pd.concat([result[tkr], fresh]).index).duplicated().values]
                        combined.sort_index(inplace=True)
                        result[tkr] = combined
                        combined.to_parquet(_cache_path(tkr))
            if i + BATCH_SIZE < len(update_list):
                time.sleep(random.uniform(*BATCH_SLEEP))

    if needs_full:
        n = len(needs_full)
        log.info(f"İlk indirme: {n} ticker ({start} → {end})")
        for i in range(0, n, BATCH_SIZE):
            batch = needs_full[i:i + BATCH_SIZE]
            log.info(f"  Batch {i//BATCH_SIZE+1}/{(n-1)//BATCH_SIZE+1}: {len(batch)} ticker")
            raw = _batch_download(batch, start, end)
            if not raw.empty:
                for tkr in batch:
                    df = _extract(raw, tkr)
                    if not df.empty:
                        result[tkr] = df
                        df.to_parquet(_cache_path(tkr))
            if i + BATCH_SIZE < n:
                time.sleep(random.uniform(*BATCH_SLEEP))

    return result


# ── Makro & Sektör ETF verisi ─────────────────────────────────────────
def load_macro(start: str, end: str) -> dict[str, pd.DataFrame]:
    all_tickers = list(MACRO_TICKERS.values()) + SECTOR_ETFS
    log.info(f"Makro & sektör ETF verisi çekiliyor ({len(all_tickers)} ticker)...")

    for attempt in range(3):
        raw = _batch_download(all_tickers, start, end)
        missing_critical = [tkr for tkr in MACRO_TICKERS.values() if _extract(raw, tkr).empty]
        if not missing_critical:
            break
        log.warning(f"  Makro retry {attempt+1}/3 — eksik: {missing_critical}")
        import time; time.sleep(5)

    result = {}
    for name, tkr in MACRO_TICKERS.items():
        df = _extract(raw, tkr)
        if df.empty:
            log.warning(f"  {name} ({tkr}) indirilemedi — boş DataFrame kullanılacak")
        result[name] = df
    sector_etf_data = {}
    for etf in SECTOR_ETFS:
        df = _extract(raw, etf)
        if not df.empty:
            sector_etf_data[etf] = df
    return result, sector_etf_data


# ── Fundamental veri (cache) ──────────────────────────────────────────
def load_fundamental_cache(ticker: str) -> dict:
    FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = FUND_CACHE_DIR / f"{ticker.replace('/', '_')}.pkl"
    if cp.exists():
        try:
            with open(cp, "rb") as f:
                cached = pickle.load(f)
            if (datetime.now() - cached["fetched_at"]).days < 7:
                return cached
        except Exception:
            pass
    return {}


def save_fundamental_cache(ticker: str, data: dict):
    FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["fetched_at"] = datetime.now()
    cp = FUND_CACHE_DIR / f"{ticker.replace('/', '_')}.pkl"
    with open(cp, "wb") as f:
        pickle.dump(data, f)


def fetch_fundamental_data(tickers: list[str]) -> dict[str, dict]:
    log.info(f"Fundamental veri çekiliyor ({len(tickers)} ticker)...")
    result = {}
    for i, tkr in enumerate(tickers):
        if i % 100 == 0:
            log.info(f"  Fundamental: {i}/{len(tickers)}")
        cached = load_fundamental_cache(tkr)
        if cached:
            result[tkr] = cached
            continue
        try:
            t = yf.Ticker(tkr)
            data = {
                "info":            t.info or {},
                "income_stmt":     t.quarterly_income_stmt,
                "balance_sheet":   t.quarterly_balance_sheet,
                "earnings_dates":  t.earnings_dates,
                "shares":          (t.info or {}).get("sharesOutstanding"),
            }
            result[tkr] = data
            save_fundamental_cache(tkr, data)
        except Exception as e:
            log.warning(f"  Fundamental hata ({tkr}): {e}")
            result[tkr] = {}
        time.sleep(0.2)
    return result


# ── Ay sonu tarihleri ──────────────────────────────────────────────────
def get_month_ends(start: str, end: str, price_data: dict) -> list[pd.Timestamp]:
    all_dates = set()
    for df in price_data.values():
        if not df.empty:
            monthly = df.resample("BME").last()
            all_dates.update(monthly.index.tolist())
    all_dates = sorted(d for d in all_dates
                       if pd.Timestamp(start) <= d <= pd.Timestamp(end))
    return all_dates


# ── Tek ticker × tek ay özellik hesabı ────────────────────────────────
def compute_features_one(ticker: str,
                          daily: pd.DataFrame,
                          month_end: pd.Timestamp,
                          macro_data: dict,
                          sector_etf_data: dict,
                          fund_data: dict,
                          sector: str) -> dict:
    d = daily.loc[:month_end]
    if len(d) < 20:
        return {}

    feat = {"ticker": ticker, "month_end": month_end, "sector": sector}

    feat.update(compute_price_returns(d, month_end))
    feat.update(compute_technical(d))
    feat.update(compute_moving_averages(d))
    feat.update(compute_volume_liquidity(d))
    feat.update(compute_market_regime(
        macro_data.get("sp500", pd.DataFrame()),
        macro_data.get("vix",   pd.DataFrame()),
        month_end,
    ))
    feat.update(compute_statistical(d, macro_data.get("sp500", pd.DataFrame()), month_end))
    feat.update(compute_sector_relative(d, sector_etf_data, sector, month_end))
    feat.update(compute_calendar(month_end))

    for name in ["tnx", "dxy", "wti", "gold"]:
        df_m = macro_data.get(name, pd.DataFrame())
        if not df_m.empty:
            close_m = df_m["Close"].loc[:month_end]
            if len(close_m):
                feat[f"macro_{name}"] = close_m.iloc[-1]
                if len(close_m) >= 22:
                    feat[f"macro_{name}_ret1m"] = close_m.iloc[-1] / close_m.iloc[-22] - 1

    fdata = fund_data.get(ticker, {})
    if fdata:
        price  = d["Close"].iloc[-1]
        shares = fdata.get("shares")
        feat.update(compute_fundamental(
            price=price,
            shares=shares,
            income_stmt=fdata.get("income_stmt"),
            balance_sheet=fdata.get("balance_sheet"),
            info=fdata.get("info", {}),
            as_of=month_end,
        ))
        feat.update(compute_earnings(fdata.get("earnings_dates"), month_end))

    return feat


# ── Target hesabı ─────────────────────────────────────────────────────
RANK_TOP_PCT     = 0.80   # üst %20
REVERSAL_MOM_PCT = 0.35   # alt %35 momentum → reversal adayı


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    LABEL_RANK_TOP20  : cross-sectional üst %20 getiri (binary)
    LABEL_REVERSAL    : üst %20 AND alt %35 momentum (binary)
    ret_1m_rank       : hot-stock tespiti için 1 aylık getiri sırası
    ret_2m_rank       : hot-stock tespiti için 2 aylık getiri sırası
    """
    df = df.copy()
    df = df.sort_values(["ticker", "month_end"])

    df["next_month_ret"] = df.groupby("ticker")["month_close"].pct_change().shift(-1)

    index_ret = (
        df.groupby("month_end")["next_month_ret"]
        .mean()
        .rename("index_next_ret")
    )
    df = df.join(index_ret, on="month_end")

    df["TARGET_REL"] = df["next_month_ret"] - df["index_next_ret"]
    lo = df["TARGET_REL"].quantile(0.005)
    hi = df["TARGET_REL"].quantile(0.995)
    df["TARGET_REL"] = df["TARGET_REL"].clip(lo, hi)

    rel_rank = df.groupby("month_end")["TARGET_REL"].rank(pct=True)
    df["LABEL_RANK_TOP20"] = (rel_rank >= RANK_TOP_PCT).astype(int)

    if "ret_3m" in df.columns:
        mom_rank = df.groupby("month_end")["ret_3m"].rank(pct=True)
        df["LABEL_REVERSAL"] = (
            (rel_rank >= RANK_TOP_PCT) & (mom_rank < REVERSAL_MOM_PCT)
        ).astype(int)
    else:
        df["LABEL_REVERSAL"] = np.nan

    df["ret_1m_rank"] = df.groupby("month_end")["ret_1m"].rank(pct=True)
    if "ret_2m" in df.columns:
        df["ret_2m_rank"] = df.groupby("month_end")["ret_2m"].rank(pct=True)

    df.drop(columns=["index_next_ret"], inplace=True)
    return df


# ── Cross-sectional normalize ─────────────────────────────────────────
def add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ret_1m", "ret_3m", "ret_6m", "avg_dv_21d"]:
        if col in df.columns:
            df[f"{col}_sector_rank"] = df.groupby(
                ["month_end", "sector"])[col].rank(pct=True)
    return df


# ── Winsorize ─────────────────────────────────────────────────────────
_SKIP_COLS = frozenset([
    "ticker", "month_end", "sector", "month_close",
    "next_month_ret", "TARGET_REL",
    "LABEL_RANK_TOP20", "LABEL_REVERSAL",
    "ret_1m_rank", "ret_2m_rank",
])


def fit_winsorizer(df: pd.DataFrame, low=0.01, high=0.99) -> dict:
    bounds = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in _SKIP_COLS:
            continue
        lo = df[col].quantile(low)
        hi = df[col].quantile(high)
        if lo < hi:
            bounds[col] = (lo, hi)
    return bounds


def apply_winsorizer(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).clip(lower=lo, upper=hi)
    return df


# ── Ana akış ──────────────────────────────────────────────────────────
def main(tickers_df: pd.DataFrame, start: str = DEFAULT_START,
         end: str = date.today().isoformat()):

    tickers = tickers_df["ticker"].tolist()
    sector_map = dict(zip(tickers_df["ticker"], tickers_df["sector"].fillna("Unknown")))

    log.info("=" * 60)
    log.info("ADIM 1/5 — Fiyat verisi")
    price_data = load_price_data(tickers, start, end)
    log.info(f"{len(price_data)} ticker'da veri var.")

    log.info("=" * 60)
    log.info("ADIM 2/5 — Makro & sektör ETF")
    macro_data, sector_etf_data = load_macro(start, end)

    log.info("=" * 60)
    log.info("ADIM 3/5 — Fundamental veri")
    fund_data = fetch_fundamental_data(tickers)

    log.info("=" * 60)
    log.info("ADIM 4/5 — Feature hesabı")
    month_ends = get_month_ends(start, end, price_data)
    log.info(f"{len(month_ends)} ay sonu tarihi")

    records = []
    for m_idx, month_end in enumerate(month_ends):
        if m_idx % 12 == 0:
            log.info(f"  {month_end.strftime('%Y-%m')} ({m_idx+1}/{len(month_ends)})")
        for tkr in tickers:
            daily = price_data.get(tkr, pd.DataFrame())
            if daily.empty:
                continue
            feat = compute_features_one(
                tkr, daily, month_end,
                macro_data, sector_etf_data, fund_data,
                sector_map.get(tkr, "Unknown"),
            )
            if feat:
                feat["month_close"] = daily["Close"].loc[:month_end].iloc[-1]
                records.append(feat)

    if not records:
        raise RuntimeError("Feature hesabı başarısız — hiç kayıt üretilemedi.")

    df = pd.DataFrame(records)
    log.info(f"Ham veri: {len(df):,} satır × {len(df.columns)} sütun")

    log.info("=" * 60)
    log.info("ADIM 5/5 — Target & cross-sectional")
    df = add_cross_sectional(df)
    df = build_targets(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"features_monthly_{date.today().strftime('%Y%m%d')}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Kaydedildi → {out_path}  ({len(df):,} satır)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker_file", default="us_tickers.xlsx")
    parser.add_argument("--start",       default=DEFAULT_START)
    parser.add_argument("--end",         default=date.today().isoformat())
    args = parser.parse_args()

    tickers_df = load_tickers(args.ticker_file)
    main(tickers_df, args.start, args.end)
