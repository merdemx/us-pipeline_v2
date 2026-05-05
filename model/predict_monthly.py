"""
US Aylık Tahmin v2 — BIST-uyumlu (Momentum + Reversal, ABCDEF stratejileri)
==============================================================================
Kullanım:
    python model/predict_monthly.py
    python model/predict_monthly.py --skip-pipeline
    python model/predict_monthly.py --skip-pipeline --oot-start 2025-01-01

Çıktı:
    output/tahmin_YYYYMMDD_HHMM.xlsx

    Tahmin sayfaları:
        Rehber            — açıklama kılavuzu
        Momentum_Top5     — A/B/C/D/E/F kombinasyonları yan yana
        Birlesik_Top5     — A/B/C/D/E/F kombinasyonları yan yana
        Birlesik          — A_1m_Cap1 birleşik (retrain.py uyumlu, ticker sütunlu)
        Reversal_Top5     — reversal top-5
        Karma_Top5        — 4 Momentum (A_1m_Cap1) + 1 Reversal
        Tum_Hisseler      — tam skorlama + tüm seçim bayrakları

    Backtest sayfaları (OOT $100 → n ay portföy simülasyonu):
        Backtest          — 14 strateji sermaye evrimi + S&P500 + Gold benchmark
        Backtest_Detay    — seçilen hisseler (strateji / ticker / getiri)
        Backtest_Skorlar  — her OOT ayındaki tüm hisse skorları

Hot-cap stratejileri (A–D):
    Momentum sırasında üst %5'te yer alan hisseler "hot" sayılır.
    Cap = 1 veya 2: top-5'te bu kadar hot hisseye izin verilir;
    fazlası en düşük skorlu hot'tan başlayarak soğuk (cold) hisse ile değiştirilir.

Dışlama stratejileri:
    E_NoDualHot: hem 1m hem 2m hot olan hisseler tamamen dışlanır (AND)
    F_NoAnyHot : 1m VEYA 2m hot olan hisseler tamamen dışlanır (OR)
"""

import sys
import argparse
import pickle
import subprocess
import warnings
from pathlib import Path
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import apply_winsorizer

# ── Sabitler ─────────────────────────────────────────────────────────
HOT_MOM_THRESHOLD = 0.95
W_MOMENTUM        = 0.8
W_REVERSAL        = 0.2
INIT_CASH_BT      = 100.0

CONFIGS = {
    "A_1m_Cap1": {"lookback": "ret_1m_rank", "cap": 1},
    "B_1m_Cap2": {"lookback": "ret_1m_rank", "cap": 2},
    "C_2m_Cap1": {"lookback": "ret_2m_rank", "cap": 1},
    "D_2m_Cap2": {"lookback": "ret_2m_rank", "cap": 2},
}

SAVE_DIR    = Path(__file__).parent / "saved"
OUTPUT_DIR  = Path(__file__).parent.parent / "output"
PIPELINE    = Path(__file__).parent.parent / "pipeline.py"
TICKER_FILE = Path(__file__).parent.parent / "us_tickers.xlsx"

REVERSAL_ARTIFACTS = SAVE_DIR / "ensemble_artifacts_reversal.pkl"

EXTRA_COLS = [
    "sector", "sp500_regime", "vix_regime", "vix_level",
    "beta_63d", "hvol_63d", "analyst_rec_mean", "price_target_upside",
    "eps_surprise_last", "pead_signal", "short_ratio",
]


# ── Pipeline & yükleme ────────────────────────────────────────────────
def run_pipeline():
    start = (date.today() - relativedelta(years=3)).isoformat()
    cmd   = [sys.executable, str(PIPELINE), "--ticker_file", str(TICKER_FILE), "--start", start]
    print(f"\n[1/4] Pipeline çalıştırılıyor... (start={start})")
    subprocess.run(cmd, check=True)
    print("[1/4] Pipeline tamamlandı.\n")


def load_artifacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_latest_parquet() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("features_monthly*.parquet"))
    if not files:
        raise FileNotFoundError("Parquet bulunamadı. Önce pipeline çalıştırın.")
    df = pd.read_parquet(files[-1])
    df["month_end"] = pd.to_datetime(df["month_end"])
    print(f"[3/4] Parquet yüklendi: {files[-1].name}  ({len(df):,} satır)")
    return df


# ── Rank hesabı (eski parquet uyumu) ──────────────────────────────────
def ensure_ret_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """ret_1m_rank / ret_2m_rank parquet'te yoksa hesapla (eski pipeline uyumu)."""
    df = df.copy()
    for col in ["ret_1m", "ret_2m"]:
        rank_col = col + "_rank"
        if col in df.columns and rank_col not in df.columns:
            df[rank_col] = df.groupby("month_end")[col].rank(pct=True)
    return df


# ── Tahmin (classifier) ───────────────────────────────────────────────
def predict_proba(art: dict, df: pd.DataFrame) -> pd.Series:
    le  = art["label_enc"]
    df  = df.copy()
    df["sector_enc"] = le.transform(df["sector"].fillna("Unknown").astype(str))
    df  = apply_winsorizer(df, art["bounds"])
    for c in art["feat_cols"]:
        if c not in df.columns:
            df[c] = 0.0
    X = df[art["feat_cols"]].values
    w = art["weights"]

    def _p(m):
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X)
            cls   = list(m.classes_) if hasattr(m, "classes_") else [0, 1]
            return proba[:, cls.index(1)] if 1 in cls else np.zeros(len(X))
        return m.predict(X)

    preds = w[0]*_p(art["xgb_model"]) + w[1]*_p(art["lgb_model"]) + w[2]*_p(art["cat_model"])
    return pd.Series(preds, index=df.index)


# ── Hot-cap ve dışlama ────────────────────────────────────────────────
def apply_hot_cap(md: pd.DataFrame, score_col: str, n: int = 5,
                  lookback_col: str = "ret_1m_rank", cap: int = 2) -> pd.DataFrame:
    """Top-n seçiminde hot hisselerden max cap adet alınır; fazlası cold ile değiştirilir."""
    top_n   = md.nlargest(n, score_col)
    is_hot  = (top_n[lookback_col] > HOT_MOM_THRESHOLD
               if lookback_col in top_n.columns
               else pd.Series(False, index=top_n.index))
    hot_cnt = int(is_hot.sum())
    if hot_cnt <= cap:
        return top_n
    excess     = hot_cnt - cap
    excess_hot = top_n[is_hot].nsmallest(excess, score_col)
    remaining  = top_n[~top_n.index.isin(excess_hot.index)]
    not_in_top = ~md.index.isin(top_n.index)
    is_cold    = (~(md[lookback_col] > HOT_MOM_THRESHOLD)
                  if lookback_col in md.columns
                  else pd.Series(True, index=md.index))
    fill = md[not_in_top & is_cold].nlargest(excess, score_col)
    return pd.concat([remaining, fill]).sort_values(score_col, ascending=False)


def apply_dual_hot_excl(md: pd.DataFrame, score_col: str, n: int = 5) -> pd.DataFrame:
    """Hem 1m hem 2m hot olan hisseleri dışlar (AND). Yeterli aday yoksa filtre kalkar."""
    hot_1m = (md["ret_1m_rank"] > HOT_MOM_THRESHOLD
              if "ret_1m_rank" in md.columns else pd.Series(False, index=md.index))
    hot_2m = (md["ret_2m_rank"] > HOT_MOM_THRESHOLD
              if "ret_2m_rank" in md.columns else pd.Series(False, index=md.index))
    cands  = md[~(hot_1m & hot_2m)]
    return (cands if len(cands) >= n else md).nlargest(n, score_col)


def apply_any_hot_excl(md: pd.DataFrame, score_col: str, n: int = 5) -> pd.DataFrame:
    """1m VEYA 2m hot olan hisseleri dışlar (OR). Yeterli aday yoksa filtre kalkar."""
    hot_1m = (md["ret_1m_rank"] > HOT_MOM_THRESHOLD
              if "ret_1m_rank" in md.columns else pd.Series(False, index=md.index))
    hot_2m = (md["ret_2m_rank"] > HOT_MOM_THRESHOLD
              if "ret_2m_rank" in md.columns else pd.Series(False, index=md.index))
    cands  = md[~(hot_1m | hot_2m)]
    return (cands if len(cands) >= n else md).nlargest(n, score_col)


# ── Tahmin üretimi (güncel ay) ────────────────────────────────────────
def build_prediction(df: pd.DataFrame, art_mom: dict, art_rev: dict | None) -> dict:
    latest = df["month_end"].max()
    md     = df[df["month_end"] == latest].copy()
    print(f"[4/4] Tahmin üretiliyor — ay: {latest.strftime('%Y-%m')}  ({len(md)} hisse)")

    md["prob_mom"] = predict_proba(art_mom, md).values
    if art_rev is not None:
        md["prob_rev"] = predict_proba(art_rev, md).values
        md["prob_bir"] = W_MOMENTUM * md["prob_mom"] + W_REVERSAL * md["prob_rev"]
    else:
        md["prob_rev"] = np.nan
        md["prob_bir"] = md["prob_mom"]

    # A/B/C/D hot-cap stratejileri
    mom_tops: dict = {}
    bir_tops: dict = {}
    for cfg_name, cfg in CONFIGS.items():
        lb, cap = cfg["lookback"], cfg["cap"]
        mom_tops[cfg_name] = apply_hot_cap(md, "prob_mom", lookback_col=lb, cap=cap)
        bir_tops[cfg_name] = apply_hot_cap(md, "prob_bir", lookback_col=lb, cap=cap)

    # E/F dışlama stratejileri
    mom_tops["E_NoDualHot"] = apply_dual_hot_excl(md, "prob_mom")
    bir_tops["E_NoDualHot"] = apply_dual_hot_excl(md, "prob_bir")
    mom_tops["F_NoAnyHot"]  = apply_any_hot_excl(md, "prob_mom")
    bir_tops["F_NoAnyHot"]  = apply_any_hot_excl(md, "prob_bir")

    reversal_top5 = (
        md[["ticker", "prob_rev"] + [c for c in EXTRA_COLS if c in md.columns]]
        .nlargest(5, "prob_rev").reset_index(drop=True)
    ) if art_rev is not None else pd.DataFrame()

    # Karma: 4 Mom (A_1m_Cap1) + 1 Reversal
    top4_mom_k = apply_hot_cap(md, "prob_mom", n=4, lookback_col="ret_1m_rank", cap=1)
    excl_k     = set(top4_mom_k["ticker"])
    cand_k     = md[~md["ticker"].isin(excl_k)]
    rev_col    = "prob_rev" if art_rev is not None else "prob_mom"
    top1_rev_k = (cand_k if not cand_k.empty else md).nlargest(1, rev_col)
    karma_df = pd.concat([
        top4_mom_k[["ticker", "prob_mom"]].rename(columns={"prob_mom": "skor"}).assign(kaynak="Momentum"),
        top1_rev_k[["ticker", rev_col]].rename(columns={rev_col: "skor"}).assign(
            kaynak="Reversal" if art_rev is not None else "Momentum"),
    ]).reset_index(drop=True)

    # Tam skorlama
    scoring = md[["ticker"]].copy()
    scoring["Momentum_Skor"] = md["prob_mom"].round(4)
    scoring["Momentum_Sira"] = md["prob_mom"].rank(ascending=False, method="min").astype(int)
    if art_rev is not None:
        scoring["Reversal_Skor"] = md["prob_rev"].round(4)
        scoring["Reversal_Sira"] = md["prob_rev"].rank(ascending=False, method="min").astype(int)
        scoring["Birlesik_Skor"] = md["prob_bir"].round(4)
        scoring["Birlesik_Sira"] = md["prob_bir"].rank(ascending=False, method="min").astype(int)
    for col in ["ret_1m_rank", "ret_2m_rank"]:
        if col in md.columns:
            scoring[col] = md[col].round(3)
    if "ret_1m_rank" in md.columns:
        scoring["Hot_1m"] = (md["ret_1m_rank"] > HOT_MOM_THRESHOLD).map({True: "✓", False: ""})
    if "ret_2m_rank" in md.columns:
        scoring["Hot_2m"] = (md["ret_2m_rank"] > HOT_MOM_THRESHOLD).map({True: "✓", False: ""})
    for col in EXTRA_COLS:
        if col in md.columns:
            scoring[col] = md[col].values
    for cfg_name in list(CONFIGS.keys()) + ["E_NoDualHot", "F_NoAnyHot"]:
        scoring[f"Sec_Mom_{cfg_name}"] = md["ticker"].isin(
            set(mom_tops[cfg_name]["ticker"])).map({True: "✓", False: ""})
        scoring[f"Sec_Bir_{cfg_name}"] = md["ticker"].isin(
            set(bir_tops[cfg_name]["ticker"])).map({True: "✓", False: ""})
    scoring["Sec_Karma"] = md["ticker"].isin(set(karma_df["ticker"])).map({True: "✓", False: ""})

    sort_col = "Birlesik_Sira" if "Birlesik_Sira" in scoring.columns else "Momentum_Sira"
    scoring  = scoring.sort_values(sort_col).reset_index(drop=True)

    bir_main = bir_tops["A_1m_Cap1"].copy()
    extra_in = [c for c in EXTRA_COLS if c in bir_main.columns]
    bir_main = bir_main[["ticker", "prob_bir"] + extra_in].reset_index(drop=True)
    bir_main.insert(0, "sira", range(1, len(bir_main) + 1))

    return {
        "date":     latest,
        "mom_tops": mom_tops,
        "bir_tops": bir_tops,
        "reversal": reversal_top5,
        "karma":    karma_df,
        "scoring":  scoring,
        "bir_main": bir_main,
        "md":       md,
    }


# ── Benchmark verisi ─────────────────────────────────────────────────
def _fetch_benchmark(ticker: str, month_ends: list) -> dict:
    months = sorted(pd.Timestamp(m) for m in month_ends)
    if len(months) < 2:
        return {}
    start_s = (months[0]  - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end_s   = (months[-1] + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(ticker, start=start_s, end=end_s,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return {}
        prices = raw["Close"].squeeze()
        prices.index = pd.to_datetime(prices.index)
        ret_map = {}
        for i, m in enumerate(months[:-1]):
            nxt = months[i + 1]
            before_m = prices.index[prices.index <= m]
            before_n = prices.index[prices.index <= nxt]
            if len(before_m) == 0 or len(before_n) == 0:
                continue
            p0, p1 = float(prices.loc[before_m[-1]]), float(prices.loc[before_n[-1]])
            ret_map[m] = (p1 - p0) / p0
        return ret_map
    except Exception as e:
        print(f"  Benchmark {ticker} alınamadı: {e}")
        return {}


# ── Backtest ──────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, art_mom: dict, art_rev: dict | None,
                 oot_start: str = "2024-01-01") -> tuple:
    oot    = df[df["month_end"] >= pd.Timestamp(oot_start)].copy()
    months = sorted(oot["month_end"].unique())
    if len(months) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    strategy_keys = (
        [f"Mom_{k}" for k in CONFIGS] +
        [f"Bir_{k}" for k in CONFIGS] +
        ["Mom_E_NoDualHot", "Bir_E_NoDualHot",
         "Mom_F_NoAnyHot",  "Bir_F_NoAnyHot",
         "Reversal", "Karma_4M1R"]
    )
    capital = {k: INIT_CASH_BT for k in strategy_keys}

    print(f"\n[Backtest] {oot_start} → {months[-1].strftime('%Y-%m')} ({len(months)} ay)")
    print("[Backtest] Benchmark verileri çekiliyor (S&P500, Gold)...")
    sp_map   = _fetch_benchmark("^GSPC", months)
    gold_map = _fetch_benchmark("GC=F",  months)
    sp_port   = INIT_CASH_BT
    gold_port = INIT_CASH_BT

    history_rows = []
    detail_rows  = []
    scores_parts = []

    for month in months:
        md       = oot[oot["month_end"] == month].copy()
        md_valid = md.dropna(subset=["next_month_ret"]) if "next_month_ret" in md.columns else md
        is_last  = (month == months[-1])
        has_rets = len(md_valid) >= 5

        if not has_rets:
            if is_last and len(md) >= 5:
                md_valid = md.copy()
            else:
                continue

        try:
            md_valid["prob_mom"] = predict_proba(art_mom, md_valid).values
        except Exception as e:
            print(f"  Momentum hata ({month.strftime('%Y-%m')}): {e}")
            continue

        if art_rev is not None:
            try:
                md_valid["prob_rev"] = predict_proba(art_rev, md_valid).values
                md_valid["prob_bir"] = W_MOMENTUM * md_valid["prob_mom"] + W_REVERSAL * md_valid["prob_rev"]
            except Exception:
                md_valid["prob_rev"] = np.nan
                md_valid["prob_bir"] = md_valid["prob_mom"]
        else:
            md_valid["prob_rev"] = np.nan
            md_valid["prob_bir"] = md_valid["prob_mom"]

        sp_ret    = sp_map.get(month, np.nan)
        gold_ret  = gold_map.get(month, np.nan)
        sp_port   = sp_port   * (1 + sp_ret)   if pd.notna(sp_ret)   else sp_port
        gold_port = gold_port * (1 + gold_ret)  if pd.notna(gold_ret) else gold_port

        label   = month.strftime("%Y-%m") + ("*" if not has_rets else "")
        h_row   = {"Tarih": label}
        strat_picks: dict = {}

        # A/B/C/D hot-cap
        for cfg_name, cfg in CONFIGS.items():
            lb, cap = cfg["lookback"], cfg["cap"]
            for prefix, scol in [("Mom", "prob_mom"), ("Bir", "prob_bir")]:
                skey  = f"{prefix}_{cfg_name}"
                picks = apply_hot_cap(md_valid, scol, lookback_col=lb, cap=cap)
                strat_picks[skey] = picks
                if has_rets and "next_month_ret" in picks.columns:
                    capital[skey] *= (1 + float(picks["next_month_ret"].fillna(0).mean()))
                h_row[skey] = round(capital[skey], 4)

        # Karma: 4 Mom + 1 Rev
        rev_col    = "prob_rev" if art_rev is not None else "prob_mom"
        top4_k     = apply_hot_cap(md_valid, "prob_mom", n=4, lookback_col="ret_1m_rank", cap=1)
        excl_k     = set(top4_k["ticker"])
        cand_k     = md_valid[~md_valid["ticker"].isin(excl_k)]
        top1_k     = (cand_k if not cand_k.empty else md_valid).nlargest(1, rev_col)
        karma_picks = pd.concat([top4_k, top1_k])
        strat_picks["Karma_4M1R"] = karma_picks
        if has_rets and "next_month_ret" in karma_picks.columns:
            capital["Karma_4M1R"] *= (1 + float(karma_picks["next_month_ret"].fillna(0).mean()))
        h_row["Karma_4M1R"] = round(capital["Karma_4M1R"], 4)

        # E: dual-hot dışlama
        for prefix, scol in [("Mom", "prob_mom"), ("Bir", "prob_bir")]:
            skey  = f"{prefix}_E_NoDualHot"
            picks = apply_dual_hot_excl(md_valid, scol)
            strat_picks[skey] = picks
            if has_rets and "next_month_ret" in picks.columns:
                capital[skey] *= (1 + float(picks["next_month_ret"].fillna(0).mean()))
            h_row[skey] = round(capital[skey], 4)

        # F: any-hot dışlama
        for prefix, scol in [("Mom", "prob_mom"), ("Bir", "prob_bir")]:
            skey  = f"{prefix}_F_NoAnyHot"
            picks = apply_any_hot_excl(md_valid, scol)
            strat_picks[skey] = picks
            if has_rets and "next_month_ret" in picks.columns:
                capital[skey] *= (1 + float(picks["next_month_ret"].fillna(0).mean()))
            h_row[skey] = round(capital[skey], 4)

        # Reversal
        if art_rev is not None:
            rev_picks = md_valid.nlargest(5, "prob_rev")
        else:
            rev_picks = md_valid.nlargest(5, "prob_mom")
        strat_picks["Reversal"] = rev_picks
        if has_rets and "next_month_ret" in rev_picks.columns:
            capital["Reversal"] *= (1 + float(rev_picks["next_month_ret"].fillna(0).mean()))
        h_row["Reversal"] = round(capital["Reversal"], 4)

        h_row["SP500"] = round(sp_port, 4)
        h_row["Gold"]  = round(gold_port, 4)
        history_rows.append(h_row)

        per_stock = INIT_CASH_BT / 5
        for skey, picks in strat_picks.items():
            for _, r in picks.iterrows():
                ret_v = r.get("next_month_ret", np.nan) if has_rets else np.nan
                detail_rows.append({
                    "Tarih":    label,
                    "Strateji": skey,
                    "Ticker":   r["ticker"],
                    "Sektor":   r.get("sector", ""),
                    "Prob_Mom": round(float(r["prob_mom"]), 4) if pd.notna(r.get("prob_mom")) else "",
                    "Prob_Rev": round(float(r.get("prob_rev", np.nan)), 4) if pd.notna(r.get("prob_rev")) else "",
                    "Prob_Bir": round(float(r.get("prob_bir", np.nan)), 4) if pd.notna(r.get("prob_bir")) else "",
                    "Hot_1m":   "✓" if pd.notna(r.get("ret_1m_rank")) and r["ret_1m_rank"] > HOT_MOM_THRESHOLD else "",
                    "Hot_2m":   "✓" if pd.notna(r.get("ret_2m_rank")) and r["ret_2m_rank"] > HOT_MOM_THRESHOLD else "",
                    "Getiri_%": round(float(ret_v) * 100, 2) if pd.notna(ret_v) else "",
                    "Giris_$":  round(per_stock, 4),
                    "Cikis_$":  round(per_stock * (1 + float(ret_v)), 4) if pd.notna(ret_v) and has_rets else "",
                })

        sc_df = md_valid[["ticker"]].copy()
        sc_df["Tarih"] = month.strftime("%Y-%m")
        for col, lbl in [("prob_mom", "Momentum"), ("prob_rev", "Reversal"), ("prob_bir", "Birlesik")]:
            if col in md_valid.columns and md_valid[col].notna().any():
                sc_df[f"{lbl}_Skor"] = md_valid[col].round(4)
                sc_df[f"{lbl}_Sira"] = md_valid[col].rank(ascending=False, method="min").astype(int)
        for col in ["ret_1m_rank", "ret_2m_rank"]:
            if col in md_valid.columns:
                sc_df[col] = md_valid[col].round(3)
        if "ret_1m_rank" in md_valid.columns:
            sc_df["Hot_1m"] = (md_valid["ret_1m_rank"] > HOT_MOM_THRESHOLD).map({True: "✓", False: ""})
        if "ret_2m_rank" in md_valid.columns:
            sc_df["Hot_2m"] = (md_valid["ret_2m_rank"] > HOT_MOM_THRESHOLD).map({True: "✓", False: ""})
        if "next_month_ret" in md_valid.columns:
            sc_df["Gerceklesen_%"] = (md_valid["next_month_ret"] * 100).round(2)
        for skey, picks in strat_picks.items():
            sc_df[f"Sec_{skey}"] = md_valid["ticker"].isin(
                set(picks["ticker"])).map({True: "✓", False: ""})
        sort_col = "Birlesik_Sira" if "Birlesik_Sira" in sc_df.columns else "Momentum_Sira"
        scores_parts.append(sc_df.sort_values(sort_col).reset_index(drop=True))

    if not history_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    history_df = pd.DataFrame(history_rows)
    ordered    = (["Tarih"] + [s for s in strategy_keys if s in history_df.columns]
                  + [b for b in ["SP500", "Gold"] if b in history_df.columns])
    history_df = history_df[ordered]

    n = len(history_df)
    if n > 0:
        print(f"\n[Backtest] {n} ay simüle edildi  |  başlangıç: ${INIT_CASH_BT:.0f}")
        print(f"  {'Strateji':<20} {'Son Değer':>12} {'Getiri':>10}")
        print(f"  {'-'*46}")
        for s in strategy_keys:
            if s in history_df.columns:
                fv = history_df[s].iloc[-1]
                print(f"  {s:<20} ${fv:>11.2f} {(fv/INIT_CASH_BT-1)*100:>+9.1f}%")
        for bname in ["SP500", "Gold"]:
            if bname in history_df.columns:
                fv = history_df[bname].iloc[-1]
                print(f"  {bname:<20} ${fv:>11.2f} {(fv/INIT_CASH_BT-1)*100:>+9.1f}%")

    detail_df = pd.DataFrame(detail_rows)
    scores_df = pd.concat(scores_parts, ignore_index=True) if scores_parts else pd.DataFrame()
    return history_df, detail_df, scores_df


# ── Excel yardımcıları ────────────────────────────────────────────────
def _comparison_df(tops: dict, score_col: str) -> pd.DataFrame:
    frames = []
    for cfg_name, picks in tops.items():
        if picks.empty:
            continue
        p = picks[["ticker", score_col]].copy().reset_index(drop=True)
        p.index = range(1, len(p) + 1)
        p[score_col] = p[score_col].round(4)
        frames.append(p.rename(columns={"ticker": cfg_name, score_col: f"{cfg_name}_skor"}))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1)
    result.index.name = "Sira"
    return result.reset_index()


def _write_sheet(writer, df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    for col in ws.columns:
        w = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max(w + 2, 12)


def _build_rehber(art_mom: dict, preds: dict, art_rev: dict | None) -> pd.DataFrame:
    hot_pct = round((1 - HOT_MOM_THRESHOLD) * 100)
    rows = [
        ("GENEL BİLGİ", ""),
        ("Pipeline", "Nasdaq + Russell ~1000 hisse, aylık güncelleme"),
        ("Modeller", "XGBoost + LightGBM + CatBoost ensemble (classifier)"),
        ("Momentum (M1)", "LABEL_RANK_TOP20 — cross-sectional üst %20 fazla getiri (binary)"),
        ("Reversal (M2)", "LABEL_REVERSAL — üst %20 fazla getiri VE alt %35 momentum (binary)"),
        ("Birleşik", f"M1 × {W_MOMENTUM:.0%} + M2 × {W_REVERSAL:.0%} ağırlıklı olasılık"),
        ("", ""),
        ("SEÇİM STRATEJİLERİ", ""),
        ("A_1m_Cap1", f"1m momentum sırası; hot (üst %{hot_pct}) hisseden portföye max 1 adet."),
        ("B_1m_Cap2", f"1m momentum sırası; hot hisseden portföye max 2 adet."),
        ("C_2m_Cap1", f"2m momentum sırası; hot hisseden portföye max 1 adet."),
        ("D_2m_Cap2", f"2m momentum sırası; hot hisseden portföye max 2 adet."),
        ("E_NoDualHot", f"Hem 1m hem 2m hot olan hisseler tamamen dışlanır (AND)."),
        ("F_NoAnyHot",  f"1m VEYA 2m hot olan hisseler tamamen dışlanır (OR)."),
        ("Karma_Top5",  "4 Momentum (A_1m_Cap1 bazlı, n=4) + 1 Reversal."),
        ("", ""),
        ("BACKTEST", ""),
        ("Backtest", "OOT dönemi 14 strateji, $100 başlangıç, aylık eşit ağırlık. İşlem maliyeti yok."),
        ("Benchmarks", "S&P500 (^GSPC) ve Gold (GC=F)"),
        ("", ""),
        ("BU AY ÖZETİ", ""),
        ("Tahmin ayı",    preds["date"].strftime("%Y-%m")),
        ("Hisse sayısı",  str(len(preds["scoring"]))),
        ("Birleşik Top-5 (A_1m_Cap1)", ", ".join(preds["bir_main"]["ticker"].head(5).tolist())),
        ("Karma Top-5",  ", ".join(preds["karma"]["ticker"].tolist())),
    ]
    for label, key in [("Momentum Holdout F-beta", "ho_ic"), ("Momentum Top-1 Prec", "top1_precision")]:
        val = art_mom.get(key)
        if val is not None:
            rows.append((label, f"{val:.4f}"))
    oot = (art_mom.get("oot_ics") or {}).get("ensemble")
    if oot is not None:
        rows.append(("Momentum OOT F-beta", f"{oot:.4f}"))
    rows += [
        ("", ""),
        ("UYARILAR", ""),
        ("", "Yatırım tavsiyesi değildir. Geçmiş performans gelecek getirileri garanti etmez."),
    ]
    return pd.DataFrame(rows, columns=["Konu / Strateji", "Açıklama"])


# ── Konsol çıktısı ────────────────────────────────────────────────────
def print_top(label: str, df: pd.DataFrame, score_col: str, n: int = 10):
    print(f"\n{'='*65}\n  {label}\n{'='*65}")
    print(f"  {'#':>3}  {'Ticker':<8}  {'Skor':>6}  {'Sektör':<30}")
    print(f"  {'-'*55}")
    for i, (_, row) in enumerate(df.head(n).iterrows(), 1):
        sector = str(row.get("sector", ""))[:28]
        print(f"  {i:>3}  {row['ticker']:<8}  {row.get(score_col, 0):>6.4f}  {sector:<30}")
    print(f"{'='*65}")


# ── Ana akış ─────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--oot-start", dest="oot_start", default="2024-01-01")
    args = parser.parse_args()

    print("\n=== US Aylık Tahmin v2 ===")

    if not args.skip_pipeline:
        run_pipeline()
    else:
        print("[1/4] Pipeline atlandı.\n")

    df = load_latest_parquet()
    df = ensure_ret_ranks(df)

    art_mom = load_artifacts(args.artifacts or SAVE_DIR / "ensemble_artifacts.pkl")
    if art_mom is None:
        raise FileNotFoundError("ensemble_artifacts.pkl bulunamadı — önce train.py çalıştırın.")
    art_rev = load_artifacts(REVERSAL_ARTIFACTS)

    loaded = ["Momentum"]
    if art_rev:
        loaded.append("Reversal")
    print(f"[2/4] Modeller: {', '.join(loaded)}")

    preds = build_prediction(df, art_mom, art_rev)

    print_top(f"Momentum A_1m_Cap1 ({preds['date'].strftime('%Y-%m')})",
              preds["mom_tops"]["A_1m_Cap1"], "prob_mom")
    if art_rev:
        print_top(f"Birleşik A_1m_Cap1 ({preds['date'].strftime('%Y-%m')})",
                  preds["bir_tops"]["A_1m_Cap1"], "prob_bir")
        if not preds["reversal"].empty:
            print_top(f"Reversal ({preds['date'].strftime('%Y-%m')})", preds["reversal"], "prob_rev")
    print(f"\nKarma Top-5: {preds['karma']['ticker'].tolist()}")

    bt_history, bt_detail, bt_scores = run_backtest(
        df, art_mom, art_rev, args.oot_start
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = OUTPUT_DIR / f"tahmin_{ts}.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        _write_sheet(writer, _build_rehber(art_mom, preds, art_rev), "Rehber")

        mom_cmp = _comparison_df(preds["mom_tops"], "prob_mom")
        _write_sheet(writer, mom_cmp, "Momentum_Top5")

        bir_cmp = _comparison_df(preds["bir_tops"], "prob_bir")
        _write_sheet(writer, bir_cmp, "Birlesik_Top5")

        _write_sheet(writer, preds["bir_main"], "Birlesik")

        if art_rev and not preds["reversal"].empty:
            _write_sheet(writer, preds["reversal"], "Reversal_Top5")

        _write_sheet(writer, preds["karma"], "Karma_Top5")

        _write_sheet(writer, preds["scoring"], "Tum_Hisseler")

        if not bt_history.empty:
            _write_sheet(writer, bt_history, "Backtest")
        if not bt_detail.empty:
            _write_sheet(writer, bt_detail,  "Backtest_Detay")
        if not bt_scores.empty:
            _write_sheet(writer, bt_scores,  "Backtest_Skorlar")

    print(f"\nExcel kaydedildi → {out.name}")


if __name__ == "__main__":
    main()
