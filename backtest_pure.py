"""
US Aylık Model Backtest — M2 + M3
====================================
OOT dönemi boyunca her ay top-N tahmini gerçek getirilerle simüle eder.
M2 (reversal) + M3 (cezasız momentum), her biri %50 ağırlık.

Kullanım:
    python backtest_pure.py
    python backtest_pure.py --top 5 --start 2024-01-01 --capital 100

Çıktı:
    output/backtest_pure_YYYYMMDD.xlsx
    - Portfoy    : aylık portföy değeri ve S&P500 karşılaştırması
    - Pozisyonlar: her ay hangi hisseler, gerçek getirileri
    - Ozet       : toplam getiri, Sharpe, max drawdown, alpha
"""

import argparse
import pickle
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import apply_winsorizer

SAVE_DIR        = Path("model/saved")
OUTPUT_DIR      = Path("output")
W_REVERSAL      = 0.5
W_PURE_MOMENTUM = 0.5


# ── Model yükleme ────────────────────────────────────────────────────
def load_art(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Tahmin ───────────────────────────────────────────────────────────
def predict(art: dict, df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["sector_enc"] = art["label_enc"].transform(
        df["sector"].fillna("Unknown").astype(str)
    )
    df = apply_winsorizer(df, art["bounds"])
    missing = [c for c in art["feat_cols"] if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[art["feat_cols"]].values
    w = art["weights"]
    preds = (w[0] * art["xgb_model"].predict(X) +
             w[1] * art["lgb_model"].predict(X) +
             w[2] * art["cat_model"].predict(X))
    return pd.Series(preds, index=df.index)


# ── Backtest ─────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame,
                 art2: dict,
                 art3: dict | None,
                 top_n: int,
                 start_date: str,
                 initial_capital: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Her ay için:
      1. O aydaki feature'larla tahmin üret (M2 reversal, M3 cezasız)
      2. Top-N hisseyi seç
      3. Bir sonraki ay gerçek getirisini uygula
      4. Portföy değerini güncelle
    """
    months = sorted(df["month_end"].unique())
    months = [m for m in months if m >= pd.Timestamp(start_date)]

    portfolio_rows = []
    position_rows  = []
    capital        = initial_capital

    w2 = W_REVERSAL      / (W_REVERSAL + (W_PURE_MOMENTUM if art3 else 0))
    w3 = W_PURE_MOMENTUM / (W_REVERSAL + (W_PURE_MOMENTUM if art3 else 0)) if art3 else 0

    for i, month in enumerate(months[:-1]):   # son ay için gerçek getiri yok
        month_df = df[df["month_end"] == month].copy()
        if len(month_df) < top_n:
            continue

        p2_rank = predict(art2, month_df).rank(pct=True)
        p3_rank = predict(art3, month_df).rank(pct=True) if art3 else None

        combined = w2 * p2_rank
        if p3_rank is not None:
            combined = combined + w3 * p3_rank

        month_df["combined_score"] = combined.values
        month_df["pred_rank"]      = combined.rank(pct=True).values

        top         = month_df.nlargest(top_n, "combined_score")
        actual_rets = top["target_ret_1m"].fillna(0).values
        port_ret    = actual_rets.mean()
        new_capital = capital * (1 + port_ret)

        next_month = months[i + 1]
        portfolio_rows.append({
            "month":         next_month,
            "port_value":    new_capital,
            "port_ret":      port_ret,
            "capital_start": capital,
        })

        for _, row in top.iterrows():
            position_rows.append({
                "month":      month,
                "ticker":     row["ticker"],
                "sector":     row.get("sector", ""),
                "pred_rank":  row["pred_rank"],
                "actual_ret": row.get("target_ret_1m", np.nan),
                "m2_rank":    p2_rank.loc[row.name] if row.name in p2_rank.index else np.nan,
                "m3_rank":    p3_rank.loc[row.name] if (p3_rank is not None and row.name in p3_rank.index) else np.nan,
            })

        capital = new_capital

    port_df = pd.DataFrame(portfolio_rows)
    pos_df  = pd.DataFrame(position_rows)
    return port_df, pos_df


# ── S&P500 getirisi ───────────────────────────────────────────────────
def fetch_sp500_monthly(start: str, end: str) -> pd.Series:
    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    monthly = raw["Close"].resample("BME").last()
    return monthly.pct_change().dropna()


# ── Metrik hesabı ─────────────────────────────────────────────────────
def compute_metrics(port_df: pd.DataFrame,
                    sp500_rets: pd.Series,
                    initial_capital: float) -> dict:
    if port_df.empty:
        return {}

    rets = port_df.set_index("month")["port_ret"]

    sp_aligned = sp500_rets.reindex(rets.index).fillna(0)
    sp_value   = initial_capital * (1 + sp_aligned).cumprod()
    port_df    = port_df.copy()
    port_df["sp500_value"] = sp_value.values
    port_df["sp500_ret"]   = sp_aligned.values

    n_months     = len(rets)
    total_ret    = port_df["port_value"].iloc[-1] / initial_capital - 1
    sp_total_ret = port_df["sp500_value"].iloc[-1] / initial_capital - 1
    ann_ret      = (1 + total_ret) ** (12 / n_months) - 1
    sp_ann_ret   = (1 + sp_total_ret) ** (12 / n_months) - 1

    excess_rets  = rets.values - sp_aligned.values
    sharpe       = (rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else np.nan
    info_ratio   = (excess_rets.mean() / excess_rets.std() * np.sqrt(12)) if excess_rets.std() > 0 else np.nan

    roll_max = port_df["port_value"].cummax()
    drawdown = (port_df["port_value"] - roll_max) / roll_max
    max_dd   = drawdown.min()

    win_rate = (rets > 0).mean()

    metrics = {
        "Başlangıç Tarihi":     port_df["month"].iloc[0].strftime("%Y-%m"),
        "Bitiş Tarihi":         port_df["month"].iloc[-1].strftime("%Y-%m"),
        "Ay Sayısı":            n_months,
        "─── OOT Portföy ───":  "",
        "Başlangıç Sermaye":    f"${initial_capital:.2f}",
        "Bitiş Sermaye (Port)": f"${port_df['port_value'].iloc[-1]:.2f}",
        "Bitiş Sermaye (SP500)":f"${port_df['sp500_value'].iloc[-1]:.2f}",
        "Toplam Getiri (Port)": f"%{total_ret*100:.1f}",
        "Toplam Getiri (SP500)":f"%{sp_total_ret*100:.1f}",
        "Yıllık Getiri (Port)": f"%{ann_ret*100:.1f}",
        "Yıllık Getiri (SP500)":f"%{sp_ann_ret*100:.1f}",
        "Alpha (yıllık)":       f"%{(ann_ret - sp_ann_ret)*100:.1f}",
        "Sharpe Ratio":         f"{sharpe:.2f}",
        "Information Ratio":    f"{info_ratio:.2f}",
        "Max Drawdown":         f"%{max_dd*100:.1f}",
        "Kazanan Ay Oranı":     f"%{win_rate*100:.1f}",
    }
    return metrics, port_df


def add_ic_metrics(metrics: dict, art2: dict, art3: dict | None) -> dict:
    def _ic(art, label):
        val = art.get("val_ics", {})
        oot = art.get("oot_ics", {})
        w   = art.get("weights", [1/3]*3)
        return {
            f"─── {label} Eğitim IC ───": "",
            f"{label} Val IC — XGBoost":  f"{val.get('xgb', 'N/A'):.4f}" if isinstance(val.get('xgb'), float) else "N/A",
            f"{label} Val IC — LightGBM": f"{val.get('lgb', 'N/A'):.4f}" if isinstance(val.get('lgb'), float) else "N/A",
            f"{label} Val IC — CatBoost": f"{val.get('cat', 'N/A'):.4f}" if isinstance(val.get('cat'), float) else "N/A",
            f"{label} OOT IC — XGBoost":  f"{oot.get('xgb', 'N/A'):.4f}" if isinstance(oot.get('xgb'), float) else "N/A",
            f"{label} OOT IC — LightGBM": f"{oot.get('lgb', 'N/A'):.4f}" if isinstance(oot.get('lgb'), float) else "N/A",
            f"{label} OOT IC — CatBoost": f"{oot.get('cat', 'N/A'):.4f}" if isinstance(oot.get('cat'), float) else "N/A",
            f"{label} OOT IC — Ensemble": f"{oot.get('ensemble', 'N/A'):.4f}" if isinstance(oot.get('ensemble'), float) else "N/A",
            f"{label} Ağırlık XGB/LGB/CAT": f"{w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f}",
        }
    metrics.update(_ic(art2, "Reversal"))
    if art3:
        metrics.update(_ic(art3, "PureMomentum"))
    return metrics


# ── Excel çıktısı ────────────────────────────────────────────────────
def _autofit(ws):
    for col in ws.columns:
        w = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max(w + 2, 12)


def save_excel(port_df: pd.DataFrame,
               pos_df: pd.DataFrame,
               metrics: dict,
               top_n: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = pd.Timestamp.now().strftime("%Y%m%d")
    out = OUTPUT_DIR / f"backtest_pure_{ts}.xlsx"

    # Portföy tablosu
    pf = port_df.copy()
    pf["month"]     = pf["month"].dt.strftime("%Y-%m")
    pf["port_ret"]  = (pf["port_ret"]  * 100).round(2)
    pf["sp500_ret"] = (pf["sp500_ret"] * 100).round(2)
    pf["port_value"]  = pf["port_value"].round(2)
    pf["sp500_value"] = pf["sp500_value"].round(2)
    pf = pf.rename(columns={
        "month": "Ay", "port_value": "Portföy ($)",
        "sp500_value": "S&P500 ($)",
        "port_ret": "Portföy Getiri (%)",
        "sp500_ret": "S&P500 Getiri (%)",
        "capital_start": "Dönem Başı ($)",
    })

    # Pozisyonlar
    ps = pos_df.copy()
    ps["month"]      = ps["month"].dt.strftime("%Y-%m")
    ps["actual_ret"] = (ps["actual_ret"] * 100).round(2)
    ps["pred_rank"]  = ps["pred_rank"].round(4)
    ps = ps.rename(columns={
        "month": "Ay", "ticker": "Hisse", "sector": "Sektör",
        "pred_rank": "Tahmin Rank", "actual_ret": "Gerçek Getiri (%)",
        "m2_rank": "M2 Rank (Reversal)", "m3_rank": "M3 Rank (Cezasız)",
    })

    # Özet
    oz = pd.DataFrame([{"Metrik": k, "Değer": v} for k, v in metrics.items()])

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pf.to_excel(writer, sheet_name="Portfoy",      index=False)
        ps.to_excel(writer, sheet_name="Pozisyonlar",  index=False)
        oz.to_excel(writer, sheet_name="Ozet",         index=False)
        for name in ["Portfoy", "Pozisyonlar", "Ozet"]:
            _autofit(writer.sheets[name])

    print(f"Backtest Excel → {out.name}")
    return out


# ── Ana akış ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",     type=int,   default=5,           help="Top-N hisse")
    parser.add_argument("--start",   default="2024-01-01",            help="OOT başlangıcı")
    parser.add_argument("--capital", type=float, default=100.0,       help="Başlangıç sermayesi ($)")
    parser.add_argument("--model2",  default="model/saved/ensemble_artifacts_reversal.pkl")
    parser.add_argument("--model3",  default="model/saved/ensemble_artifacts_pure_momentum.pkl")
    args = parser.parse_args()

    print("\n=== US Aylık Model Backtest (M2 + M3) ===\n")

    # Parquet yükle
    files = sorted(OUTPUT_DIR.glob("features_monthly*.parquet"))
    if not files:
        raise FileNotFoundError("Parquet bulunamadı. Önce pipeline çalıştırın.")
    df = pd.read_parquet(files[-1])
    df["month_end"] = pd.to_datetime(df["month_end"])
    print(f"Parquet yüklendi: {files[-1].name}  ({len(df):,} satır)")

    art2 = load_art(Path(args.model2))
    art3 = load_art(Path(args.model3))
    if art2 is None:
        raise FileNotFoundError("Model 2 (Reversal) bulunamadı. Önce train_reversal.py çalıştırın.")
    print(f"Model 2 (Reversal): {'OK' if art2 else 'YOK'}  |  "
          f"Model 3 (Cezasız Mom): {'OK' if art3 else 'YOK'}")

    # Backtest
    port_df, pos_df = run_backtest(df, art2, art3, args.top, args.start, args.capital)
    if port_df.empty:
        print("Yeterli OOT verisi yok.")
        return

    # S&P500
    sp_rets = fetch_sp500_monthly(args.start, date.today().isoformat())

    metrics, port_df = compute_metrics(port_df, sp_rets, args.capital)
    metrics = add_ic_metrics(metrics, art2, art3)

    # Konsol özeti
    print(f"\n{'='*45}")
    for k, v in metrics.items():
        if v:
            print(f"  {k:<32} {v}")
    print(f"{'='*45}")

    save_excel(port_df, pos_df, metrics, args.top)


if __name__ == "__main__":
    main()
