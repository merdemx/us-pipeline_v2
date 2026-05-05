"""
US Hisse Evreni Oluşturucu
==========================
Kullanım:
    python fetch_universe.py
    python fetch_universe.py --min-volume 10  # min $10M/gün

Akış:
    1. Nasdaq 100  → Wikipedia
    2. Russell 1000 → iShares IWB ETF holdings CSV
    3. Birleştir, tekrarları sil
    4. yfinance'ten 60 günlük ort. günlük dolar hacmi çek
    5. Hacim filtresi uygula (varsayılan: $5M/gün)
    6. Sektör bilgisi ekle (yfinance)
    7. us_tickers.xlsx kaydet

Çıktı sütunları:
    ticker, company_name, sector, industry, avg_daily_volume_usd, source
"""

import argparse
import time
import warnings
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
IWB_HOLDINGS_URL   = (
    "https://www.ishares.com/us/products/239707/"
    "ISHARES-RUSSELL-1000-ETF/1467271812596.ajax"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

MIN_VOLUME_DEFAULT = 5_000_000   # $5M/gün
VOLUME_LOOKBACK    = 60          # gün


# ── 1. Nasdaq 100 ─────────────────────────────────────────────────────
def fetch_nasdaq100() -> pd.DataFrame:
    print("[1/5] Nasdaq 100 çekiliyor (Wikipedia)...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(NASDAQ100_WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
    df = tables[0]
    df = df.rename(columns={"Ticker": "ticker", "Company": "company_name"})
    df["ticker"] = df["ticker"].str.strip()
    df["source"] = "nasdaq100"
    print(f"      {len(df)} hisse bulundu.")
    return df[["ticker", "company_name", "source"]]


# ── 2. Russell 1000 ───────────────────────────────────────────────────
def fetch_russell1000() -> pd.DataFrame:
    print("[2/5] Russell 1000 çekiliyor (iShares IWB)...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(IWB_HOLDINGS_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    # Asıl veri başlığını bul (Ticker ile başlayan satır)
    data_start = next(
        (i for i, l in enumerate(lines)
         if l.startswith("Ticker") or l.startswith('"Ticker"')),
        None,
    )
    if data_start is None:
        raise ValueError("iShares CSV formatı değişmiş, Ticker sütunu bulunamadı.")

    csv_text = "\n".join(lines[data_start:])
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={"Ticker": "ticker", "Name": "company_name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()

    # Geçerli ticker: 1-5 karakter, sadece harf/rakam/- içerir
    valid = df["ticker"].str.match(r"^[A-Z]{1,5}(-[A-Z])?$", na=False)
    df = df[valid].copy()

    df["source"] = "russell1000"
    print(f"      {len(df)} hisse bulundu.")
    return df[["ticker", "company_name", "source"]]


# ── 3. Birleştir ──────────────────────────────────────────────────────
def merge_universes(ndx: pd.DataFrame, russ: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Evrenler birleştiriliyor...")
    combined = pd.concat([ndx, russ], ignore_index=True)

    # Kaynak etiketini birleştir (ikisinde de olanlar)
    source_map = (
        combined.groupby("ticker")["source"]
        .apply(lambda s: "+".join(sorted(set(s))))
        .reset_index()
    )
    combined = (
        combined.drop_duplicates(subset="ticker", keep="first")
        .merge(source_map, on="ticker", suffixes=("_old", ""))
        .drop(columns=["source_old"], errors="ignore")
    )
    print(f"      Toplam {len(combined)} benzersiz hisse.")
    return combined


# ── 4. Hacim filtresi ─────────────────────────────────────────────────
def filter_by_volume(df: pd.DataFrame, min_vol: float) -> pd.DataFrame:
    print(f"[4/5] Hacim filtresi uygulanıyor (min ${min_vol/1e6:.0f}M/gün, son {VOLUME_LOOKBACK} gün)...")
    tickers = df["ticker"].tolist()

    # Batch'ler halinde indir (rate limit önlemi)
    batch_size = 100
    vol_records = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"      Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch)} ticker)...")
        try:
            raw = yf.download(
                batch,
                period=f"{VOLUME_LOOKBACK}d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
            for tkr in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close  = raw[tkr]["Close"].dropna()
                        volume = raw[tkr]["Volume"].dropna()
                    else:
                        close  = raw["Close"].dropna()
                        volume = raw["Volume"].dropna()
                    avg_dv = (close * volume).mean()
                    vol_records[tkr] = avg_dv
                except Exception:
                    vol_records[tkr] = 0
        except Exception as e:
            print(f"      Batch hatası: {e}")
            for tkr in batch:
                vol_records[tkr] = 0

        if i + batch_size < len(tickers):
            time.sleep(3)

    df["avg_daily_volume_usd"] = df["ticker"].map(vol_records).fillna(0)
    before = len(df)
    df = df[df["avg_daily_volume_usd"] >= min_vol].copy()
    print(f"      {before} → {len(df)} hisse kaldı (filtre: ${min_vol/1e6:.0f}M/gün).")
    return df


# ── 5. Sektör bilgisi ─────────────────────────────────────────────────
def _get_sector_info(tkr: str) -> tuple[str, str]:
    """Tek ticker için sektör/sektör bilgisini çeker, birden fazla anahtar dener."""
    sector_keys   = ["sector", "sectorDisp", "sectorKey"]
    industry_keys = ["industry", "industryDisp", "industryKey"]
    for attempt in range(3):
        try:
            info = yf.Ticker(tkr).info
            if not info or len(info) < 5:   # boş veya yarım yanıt
                raise ValueError("boş info")
            sector = next((info[k] for k in sector_keys if info.get(k)), "Unknown")
            industry = next((info[k] for k in industry_keys if info.get(k)), "Unknown")
            return sector, industry
        except Exception:
            time.sleep(1 * (attempt + 1))
    return "Unknown", "Unknown"


def enrich_sector(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Sektör bilgisi çekiliyor (yfinance)...")
    sectors, industries = {}, {}
    tickers = df["ticker"].tolist()

    for i, tkr in enumerate(tickers):
        if i % 50 == 0:
            print(f"      {i}/{len(tickers)}...")
        sec, ind = _get_sector_info(tkr)
        sectors[tkr]    = sec
        industries[tkr] = ind
        time.sleep(0.15)

    df["sector"]   = df["ticker"].map(sectors)
    df["industry"] = df["ticker"].map(industries)

    filled = (df["sector"] != "Unknown").sum()
    print(f"      Sektör dolu: {filled}/{len(df)} hisse.")
    return df


# ── Ana akış ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME_DEFAULT / 1e6,
                        help="Minimum ort. günlük dolar hacmi (milyon $, varsayılan: 5)")
    parser.add_argument("--skip-sector", action="store_true",
                        help="Sektör bilgisi çekme (hızlı test için)")
    args = parser.parse_args()
    min_vol = args.min_volume * 1_000_000

    print("\n=== US Hisse Evreni Oluşturucu ===\n")

    ndx  = fetch_nasdaq100()
    russ = fetch_russell1000()
    df   = merge_universes(ndx, russ)
    df   = filter_by_volume(df, min_vol)

    if not args.skip_sector:
        df = enrich_sector(df)
    else:
        df["sector"]   = "Unknown"
        df["industry"] = "Unknown"

    df = df.sort_values("avg_daily_volume_usd", ascending=False).reset_index(drop=True)

    out_path = "us_tickers.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nKaydedildi → {out_path}")
    print(f"Toplam {len(df)} hisse | Sektörler: {df['sector'].nunique()}")
    print(f"\nİlk 10:")
    print(df[["ticker", "company_name", "sector", "avg_daily_volume_usd"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
