"""
US Aylık Otomatik Retraining — v2
===================================
Kullanım:
    python retrain.py              # tam HPO (75 trial)
    python retrain.py --fast       # warm-start HPO (25 trial, önceki best_params başlangıç)
    python retrain.py --force      # çok kriterli karşılaştırmayı atla, her zaman güncelle
    python retrain.py --dry-run    # adımları logla, hiçbir şeyi kaydetme

Akış:
    1. pipeline.py           → veri güncelle
    2. train.py              → Momentum modeli (LABEL_RANK_TOP20, classifier)
    3. train.py              → Reversal modeli (LABEL_REVERSAL, classifier)
    4. Versiyon kaydet       → model/saved/versions/YYYYMMDD
    5. Çok kriterli karşılaştır → kabul/red (2/3 kural), gerekirse rollback
    6. predict_monthly.py    → güncel tahmin

Model Registry:
    model/saved/versions/
        ensemble_artifacts_YYYYMMDD.pkl
        ensemble_artifacts_reversal_YYYYMMDD.pkl
    Son N_KEEP=4 versiyon tutulur (≈1 yıl).

Kabul kriterleri (2/3 geçmeli):
    1. Holdout F-beta degradasyonu ≤ IC_DEGRADATION_THRESHOLD (0.02)
    2. Holdout top-1 precision ≥ MIN_TOP1_PRECISION (0.35)
    3. Canlı precision / holdout top-1 precision ≥ LIVE_PRECISION_RATIO (0.65)
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from glob import glob as _glob
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "model"
SAVED_DIR    = MODEL_DIR / "saved"
VERSIONS_DIR = SAVED_DIR / "versions"
OUTPUT_DIR   = BASE_DIR / "output"
HISTORY_FILE = SAVED_DIR / "retrain_history.json"
TICKER_FILE  = BASE_DIR / "us_tickers.xlsx"

OOT_WINDOW_MONTHS        = 3
VAL_WINDOW_YEARS         = 1
BUFFER_MONTHS            = 1
IC_DEGRADATION_THRESHOLD = 0.02
MIN_TOP1_PRECISION       = 0.35
LIVE_PRECISION_RATIO     = 0.65
LIVE_PRECISION_MONTHS    = 6
N_TRIALS_FAST            = 25
N_KEEP                   = 4

ARTIFACT_NAMES = [
    "ensemble_artifacts.pkl",
    "ensemble_artifacts_reversal.pkl",
]


# ── Tarih hesabı ──────────────────────────────────────────────────────

def compute_date_splits() -> dict:
    today     = date.today()
    oot_start = today - relativedelta(months=OOT_WINDOW_MONTHS)
    val_start = oot_start - relativedelta(years=VAL_WINDOW_YEARS)
    val_end   = oot_start - relativedelta(days=1)
    train_end = val_start - relativedelta(months=BUFFER_MONTHS)
    splits = dict(
        train_end  = train_end.strftime("%Y-%m-%d"),
        val_start  = val_start.strftime("%Y-%m-%d"),
        val_end    = val_end.strftime("%Y-%m-%d"),
        oot_start  = oot_start.strftime("%Y-%m-%d"),
    )
    log.info(f"Split → TRAIN_END:{splits['train_end']}  "
             f"VAL:{splits['val_start']}→{splits['val_end']}  OOT:{splits['oot_start']}")
    return splits


# ── Modül yükleme ─────────────────────────────────────────────────────

def _load_mod(path: Path, splits: dict):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TRAIN_END  = splits["train_end"]
    mod.VAL_START  = splits["val_start"]
    mod.VAL_END    = splits["val_end"]
    mod.OOT_START  = splits["oot_start"]
    return mod


# ── Eğitim adımları ──────────────────────────────────────────────────

def step_pipeline(dry_run=False):
    log.info("ADIM 1/6 — Veri güncelleme (pipeline.py)")
    if dry_run:
        return
    cmd = [sys.executable, str(BASE_DIR / "pipeline.py"), "--ticker_file", str(TICKER_FILE)]
    subprocess.run(cmd, check=True)


def step_train(splits: dict, dry_run=False, fast=False, warm_start: dict | None = None) -> dict:
    log.info(f"ADIM 2/6 — Momentum modeli (LABEL_RANK_TOP20, classifier) {'[FAST]' if fast else ''}")
    if dry_run:
        return {}

    if fast:
        os.environ["N_TRIALS"] = str(N_TRIALS_FAST)
    if warm_start and warm_start.get("momentum"):
        os.environ["WARM_START_PARAMS"] = json.dumps(warm_start["momentum"])
        log.info("  Warm-start: momentum best_params yüklendi")

    try:
        mod = _load_mod(MODEL_DIR / "train.py", splits)
        mod.main()
    finally:
        os.environ.pop("N_TRIALS", None)
        os.environ.pop("WARM_START_PARAMS", None)

    with open(SAVED_DIR / "ensemble_artifacts.pkl", "rb") as f:
        return pickle.load(f)


def step_train_reversal(splits: dict, dry_run=False, fast=False, warm_start: dict | None = None):
    log.info(f"ADIM 3/6 — Reversal modeli (LABEL_REVERSAL, classifier) {'[FAST]' if fast else ''}")
    if dry_run:
        return

    if fast:
        os.environ["N_TRIALS"] = str(N_TRIALS_FAST)
    if warm_start and warm_start.get("reversal"):
        os.environ["WARM_START_PARAMS"] = json.dumps(warm_start["reversal"])

    try:
        mod = _load_mod(MODEL_DIR / "train.py", splits)
        mod.main(
            target="LABEL_REVERSAL",
            save_name="ensemble_artifacts_reversal.pkl",
            classification=True,
        )
    finally:
        os.environ.pop("N_TRIALS", None)
        os.environ.pop("WARM_START_PARAMS", None)


def step_predict(dry_run=False, oot_start: str = ""):
    log.info("ADIM 6/6 — Aylık tahmin")
    if dry_run:
        return
    cmd = [sys.executable, str(MODEL_DIR / "predict_monthly.py"), "--skip-pipeline"]
    if oot_start:
        cmd += ["--oot-start", oot_start]
    subprocess.run(cmd, check=True)


# ── Model Registry ───────────────────────────────────────────────────

def _save_versioned(tag: str, dry_run=False):
    log.info(f"ADIM 4/6 — Versiyon kaydediliyor: {tag}")
    if dry_run:
        return
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        src = SAVED_DIR / name
        if src.exists():
            stem = name.replace(".pkl", "")
            dst  = VERSIONS_DIR / f"{stem}_{tag}.pkl"
            shutil.copy(src, dst)
            log.info(f"  → {dst.name}")
    _cleanup_versions()


def _cleanup_versions():
    for name in ARTIFACT_NAMES:
        stem  = name.replace(".pkl", "")
        files = sorted(VERSIONS_DIR.glob(f"{stem}_????????.pkl"))
        for old in files[:-N_KEEP]:
            old.unlink()
            log.info(f"  Eski versiyon silindi: {old.name}")


def _rollback_to_last_accepted(hist: list) -> bool:
    for entry in reversed(hist):
        if not entry.get("accepted"):
            continue
        tag = entry.get("version_tag", "")
        if not tag:
            continue
        success = True
        for name in ARTIFACT_NAMES:
            stem = name.replace(".pkl", "")
            src  = VERSIONS_DIR / f"{stem}_{tag}.pkl"
            if src.exists():
                shutil.copy(src, SAVED_DIR / name)
                log.info(f"  Rollback: {src.name} → {name}")
            else:
                log.warning(f"  Rollback dosyası bulunamadı: {src.name}")
                success = False
        if success:
            log.info(f"  Rollback tamamlandı: {tag} versiyonuna döndü.")
        return success
    log.warning("  Rollback için geçerli versiyon bulunamadı.")
    return False


# ── Warm-start yardımcısı ─────────────────────────────────────────────

def _load_warm_start_params() -> dict:
    ws   = {}
    hist = _load_history()
    for entry in reversed(hist):
        if entry.get("accepted") and entry.get("best_params"):
            ws["momentum"] = entry["best_params"]
            log.info("  Warm-start momentum params yüklendi: " + entry.get("version_tag", ""))
            break

    rev_path = SAVED_DIR / "ensemble_artifacts_reversal.pkl"
    if rev_path.exists():
        try:
            with open(rev_path, "rb") as f:
                rev_art = pickle.load(f)
            if rev_art.get("best_params"):
                ws["reversal"] = rev_art["best_params"]
                log.info("  Warm-start reversal params yüklendi.")
        except Exception:
            pass

    return ws


# ── Canlı performans ─────────────────────────────────────────────────

def compute_live_precision(n_months: int = LIVE_PRECISION_MONTHS) -> float | None:
    import pandas as pd

    parquet_files = sorted(_glob(str(OUTPUT_DIR / "features_monthly*.parquet")))
    if not parquet_files:
        return None

    df = pd.read_parquet(parquet_files[-1])
    df["month_end"] = pd.to_datetime(df["month_end"])

    if "LABEL_RANK_TOP20" not in df.columns or "ticker" not in df.columns:
        return None

    pred_files = sorted(_glob(str(OUTPUT_DIR / "tahmin_[0-9]*.xlsx")))[-n_months:]
    if not pred_files:
        return None

    month_scores = []
    for pf in pred_files:
        stem = Path(pf).stem
        try:
            yyyymm = stem.split("_")[1][:6]
            year   = int(yyyymm[:4])
            month  = int(yyyymm[4:6])
        except Exception:
            continue

        f_year  = year if month > 1 else year - 1
        f_month = month - 1 if month > 1 else 12

        month_data = df[
            (df["month_end"].dt.year  == f_year) &
            (df["month_end"].dt.month == f_month) &
            df["LABEL_RANK_TOP20"].notna()
        ].copy()
        if len(month_data) < 20:
            continue

        try:
            dp      = pd.read_excel(pf, sheet_name="Birlesik")
            tickers = dp["ticker"].dropna().head(5).tolist()
        except Exception:
            continue
        if not tickers:
            continue

        matched   = month_data[month_data["ticker"].isin(tickers)]
        n_picked  = len(set(tickers) & set(month_data["ticker"].tolist()))
        if n_picked == 0:
            continue

        precision = matched["LABEL_RANK_TOP20"].sum() / n_picked
        month_scores.append(float(precision))

    if not month_scores:
        return None

    live_prec = float(sum(month_scores) / len(month_scores))
    log.info(f"  Canlı precision ({len(month_scores)} ay): {live_prec:.4f}")
    return live_prec


# ── History & Karşılaştırma ──────────────────────────────────────────

def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_history(art: dict, accepted: bool, reason: str,
                  version_tag: str = "", live_precision: float | None = None):
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    hist = _load_history()

    fi_raw   = art.get("feature_importance") or {}
    fi_top10 = [k for k, _ in sorted(fi_raw.items(), key=lambda x: x[1], reverse=True)[:10]]

    entry = {
        "retrained_at":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version_tag":              version_tag,
        "accepted":                 accepted,
        "reason":                   reason,
        "holdout_ic":               art.get("ho_ic"),
        "top1_precision":           art.get("top1_precision"),
        "live_precision":           live_precision,
        "oot_ensemble_ic":          (art.get("oot_ics") or {}).get("ensemble"),
        "cv_ics":                   art.get("cv_ics"),
        "oot_ics":                  art.get("oot_ics"),
        "best_params":              art.get("best_params"),
        "feature_importance_top10": fi_top10,
    }
    hist.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    log.info(f"Geçmiş güncellendi ({len(hist)} kayıt)")


def _check_feature_drift(new_art: dict, hist: list) -> float | None:
    new_fi = dict(new_art.get("feature_importance") or {})
    if not new_fi:
        return None
    new_top10 = set(sorted(new_fi, key=new_fi.get, reverse=True)[:10])

    for entry in reversed(hist):
        if not entry.get("accepted"):
            continue
        old_top10 = set(entry.get("feature_importance_top10") or [])
        if not old_top10:
            continue
        jaccard = len(new_top10 & old_top10) / len(new_top10 | old_top10)
        log.info(f"  Feature drift → Jaccard top-10: {jaccard:.2f}")
        if jaccard < 0.60:
            log.warning(f"  UYARI: Feature önemi önemli ölçüde değişti (Jaccard={jaccard:.2f} < 0.60)")
        return jaccard
    return None


def step_compare(new_art: dict, version_tag: str = "",
                 dry_run=False, force=False,
                 live_precision: float | None = None) -> bool:
    log.info("ADIM 5/6 — Çok kriterli model karşılaştırması (2/3 kural)")

    if dry_run or force:
        reason = "dry-run" if dry_run else "force flag"
        if not dry_run:
            _save_history(new_art, accepted=True, reason=reason,
                          version_tag=version_tag, live_precision=live_precision)
        return True

    hist = _load_history()
    if not hist:
        _save_history(new_art, accepted=True, reason="ilk eğitim",
                      version_tag=version_tag, live_precision=live_precision)
        return True

    _check_feature_drift(new_art, hist)

    prev     = next((e for e in reversed(hist) if e.get("accepted")), None)
    old_ic   = prev.get("holdout_ic") if prev else None
    new_ic   = new_art.get("ho_ic")
    new_top1 = new_art.get("top1_precision")

    if old_ic is not None and new_ic is not None:
        crit1 = (old_ic - new_ic) <= IC_DEGRADATION_THRESHOLD
        log.info(f"  Kriter 1 — Holdout IC: eski={old_ic:.4f} yeni={new_ic:.4f} "
                 f"fark={old_ic - new_ic:+.4f} → {'✓' if crit1 else '✗'}")
    else:
        crit1 = True
        log.info("  Kriter 1 — Holdout IC: karşılaştırılamadı → ✓")

    if new_top1 is not None:
        crit2 = new_top1 >= MIN_TOP1_PRECISION
        log.info(f"  Kriter 2 — Top-1 precision: {new_top1:.4f} (eşik {MIN_TOP1_PRECISION}) → {'✓' if crit2 else '✗'}")
    else:
        crit2 = True
        log.info("  Kriter 2 — Top-1 precision: veri yok → ✓")

    holdout_top1 = prev.get("top1_precision") if prev else None
    if live_precision is not None and holdout_top1 and holdout_top1 > 0:
        ratio = live_precision / holdout_top1
        crit3 = ratio >= LIVE_PRECISION_RATIO
        log.info(f"  Kriter 3 — Canlı/holdout oranı: {ratio:.2f} (eşik {LIVE_PRECISION_RATIO}) → {'✓' if crit3 else '✗'}")
    else:
        crit3 = True
        log.info("  Kriter 3 — Canlı precision: yeterli veri yok → ✓")

    passed = sum([crit1, crit2, crit3])
    log.info(f"  Sonuç: {passed}/3 kriter geçildi")

    if passed >= 2:
        reason = f"Kabul ({passed}/3): IC={'✓' if crit1 else '✗'} Top1={'✓' if crit2 else '✗'} Canlı={'✓' if crit3 else '✗'}"
        _save_history(new_art, accepted=True, reason=reason,
                      version_tag=version_tag, live_precision=live_precision)
        return True
    else:
        rolled_back = _rollback_to_last_accepted(hist)
        rb_msg = "rollback yapıldı" if rolled_back else "rollback başarısız"
        reason = (f"Reddedildi ({passed}/3): IC={'✓' if crit1 else '✗'} "
                  f"Top1={'✓' if crit2 else '✗'} Canlı={'✓' if crit3 else '✗'} — {rb_msg}")
        log.warning(f"  Yeni model reddedildi. {reason}")
        _save_history(new_art, accepted=False, reason=reason,
                      version_tag=version_tag, live_precision=live_precision)
        return False


# ── Ana akış ─────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--fast",    action="store_true", help=f"Warm-start HPO ({N_TRIALS_FAST} trial)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║       US Aylık Pipeline v2 — Çeyreklik Retrain       ║")
    if args.fast:
        log.info("║       MOD: Fast (warm-start HPO)                     ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    splits      = compute_date_splits()
    version_tag = date.today().strftime("%Y%m%d")

    warm_start = _load_warm_start_params() if args.fast else {}

    live_precision = None
    if not args.dry_run:
        log.info("Canlı precision hesaplanıyor...")
        live_precision = compute_live_precision()

    step_pipeline(dry_run=args.dry_run)
    new_art = step_train(splits, dry_run=args.dry_run,
                         fast=args.fast, warm_start=warm_start)
    step_train_reversal(splits, dry_run=args.dry_run,
                        fast=args.fast, warm_start=warm_start)
    _save_versioned(version_tag, dry_run=args.dry_run)
    accepted = step_compare(new_art, version_tag=version_tag,
                            dry_run=args.dry_run, force=args.force,
                            live_precision=live_precision)
    step_predict(dry_run=args.dry_run, oot_start=splits.get("train_end", ""))

    log.info("=" * 55)
    log.info(f"Retraining tamamlandı — versiyon: {version_tag} — "
             f"{'KABUL' if accepted else 'REDDEDİLDİ'}")


if __name__ == "__main__":
    main()
