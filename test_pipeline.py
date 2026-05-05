"""
Hızlı pipeline smoke testi — 10 ticker, 1 yıl veri.
Kullanım: python test_pipeline.py
"""

import pandas as pd
from pipeline import main, load_tickers

# 10 farklı sektörden temsili ticker
TEST_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "AMZN", "TSLA", "PG", "NEE", "CAT"]

tickers_df = pd.DataFrame({
    "ticker":   TEST_TICKERS,
    "company_name": TEST_TICKERS,
    "sector":   ["Technology", "Technology", "Financials", "Health Care",
                 "Energy", "Consumer Discretionary", "Consumer Discretionary",
                 "Consumer Staples", "Utilities", "Industrials"],
    "industry": [""] * 10,
    "avg_daily_volume_usd": [0] * 10,
    "source": ["test"] * 10,
})

print(f"Test tickers: {TEST_TICKERS}")
print(f"Tarih aralığı: 2023-01-01 → bugün\n")

df = main(tickers_df, start="2023-01-01")

print(f"\n{'='*60}")
print(f"SONUÇ")
print(f"{'='*60}")
print(f"Satır sayısı  : {len(df):,}")
print(f"Sütun sayısı  : {len(df.columns)}")
print(f"Ay sayısı     : {df['month_end'].nunique()}")
print(f"Ticker sayısı : {df['ticker'].nunique()}")
print(f"\nTarget dolu oranı:")
for col in ["target_rank", "target_reversal_rank"]:
    if col in df.columns:
        pct = df[col].notna().mean() * 100
        print(f"  {col}: %{pct:.1f}")
print(f"\nİlk 5 feature sütunu:")
feat_cols = [c for c in df.columns if c not in
             ["ticker","month_end","sector","month_close",
              "target_rank","target_reversal_rank",
              "target_ret_1m","target_excess_ret","target_adj_excess"]]
print(f"  {feat_cols[:10]}")
print(f"\nÖrnek satır (son ay, AAPL):")
sample = df[(df["ticker"]=="AAPL")].tail(1)
if len(sample):
    print(sample[["ticker","month_end","ret_1m","ret_3m","rsi_14","pe_ttm","eps_surprise_last","target_rank"]].to_string(index=False))
