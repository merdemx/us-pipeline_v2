"""
US Aylık Ensemble Model Eğitimi — v2
======================================
Modeller  : XGBoost + LightGBM + CatBoost
HPO       : Optuna, MedianPruner, per-fold skor maximize
Hedef     : target_rank / target_reversal_rank  → Regressor, metrik = IC
            label_alpha_m / label_alpha_reversal_m → Classifier, metrik = top_k_fbeta
Split     : Walk-forward CV (fold_size ≥36 ay), holdout %20
Ensemble  : Holdout üzerinde grid-search ağırlık optimizasyonu
             Regresyon → holdout IC | Sınıflandırma → holdout fbeta(k=5, β=1.5)
"""

import json as _json
import os
import sys
import pickle
import warnings
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import fit_winsorizer, apply_winsorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_END        = "2022-12-31"
VAL_START        = "2023-01-01"
VAL_END          = "2023-12-31"
OOT_START        = "2024-01-01"
N_TRIALS         = int(os.environ.get("N_TRIALS", 75))
N_CV_FOLDS       = 3
HOLDOUT_FRAC     = 0.20
MIN_FOLD_MONTHS  = 36
MAX_TRAIN_MONTHS = int(os.environ.get("MAX_TRAIN_MONTHS", 120))
SEED             = 42
SAVE_DIR         = Path(__file__).parent / "saved"
OUTPUT_DIR       = Path(__file__).parent.parent / "output"

_ws_env = os.environ.get("WARM_START_PARAMS")
WARM_START_PARAMS: dict = _json.loads(_ws_env) if _ws_env else {}

_SKIP = {
    "ticker", "month_end", "sector", "month_close",
    "next_month_ret", "TARGET_REL",
    "LABEL_RANK_TOP20", "LABEL_REVERSAL",
}
TARGET    = "LABEL_RANK_TOP20"
SAVE_NAME = "ensemble_artifacts.pkl"


# ── Yardımcı ──────────────────────────────────────────────────────────

def load_latest_parquet() -> pd.DataFrame:
    files = sorted(OUTPUT_DIR.glob("features_monthly*.parquet"))
    if not files:
        raise FileNotFoundError(f"Parquet bulunamadı: {OUTPUT_DIR}")
    log.info(f"Yükleniyor: {files[-1].name}")
    df = pd.read_parquet(files[-1])
    df["month_end"] = pd.to_datetime(df["month_end"])
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _SKIP
        and df[c].dtype != object
        and df[c].notna().mean() > 0.3
    ]


def compute_ic(y_true, y_pred: np.ndarray, dates) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ds = np.asarray(dates)
    ics = []
    for d in np.unique(ds):
        mask = ds == d
        yt_d, yp_d = yt[mask], yp[mask]
        ok = ~np.isnan(yt_d)
        if ok.sum() < 5:
            continue
        ic, _ = spearmanr(yt_d[ok], yp_d[ok])
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


# ── Classification metrikleri ─────────────────────────────────────────

def top_k_precision(y_true, probs, groups, k=5):
    y, p, g = np.array(y_true), np.array(probs), np.array(groups)
    scores = []
    for grp in np.unique(g):
        m = g == grp
        if m.sum() < k:
            continue
        idx = np.argsort(p[m])[-k:]
        scores.append(y[m][idx].sum() / k)
    return float(np.mean(scores)) if scores else 0.0


def top_k_recall(y_true, probs, groups, k=5):
    y, p, g = np.array(y_true), np.array(probs), np.array(groups)
    scores = []
    for grp in np.unique(g):
        m = g == grp
        if m.sum() < k or y[m].sum() == 0:
            continue
        idx = np.argsort(p[m])[-k:]
        scores.append(y[m][idx].sum() / y[m].sum())
    return float(np.mean(scores)) if scores else 0.0


def top_k_fbeta(y_true, probs, groups, k=5, beta=1.5):
    pr = top_k_precision(y_true, probs, groups, k)
    rc = top_k_recall(y_true, probs, groups, k)
    denom = beta**2 * pr + rc
    if denom < 1e-9:
        return 0.0
    return float((1 + beta**2) * pr * rc / denom)


def get_prob(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    return model.predict(X)


# ── Ensemble ağırlık optimizasyonu ───────────────────────────────────

def optimize_weights_ic(models, X_ho, y_ho, dates_ho):
    step = 0.1
    grid = np.round(np.arange(0, 1 + step, step), 10)
    best_ic, best_w = -99.0, [1/3, 1/3, 1/3]
    preds = [m.predict(X_ho) for m in models]
    for w0 in grid:
        for w1 in grid:
            w2 = round(1.0 - w0 - w1, 10)
            if w2 < -1e-9:
                continue
            w2 = max(w2, 0.0)
            w_arr = np.array([w0, w1, w2])
            w_arr /= w_arr.sum()
            combined = sum(wi * p for wi, p in zip(w_arr, preds))
            ic = compute_ic(y_ho, combined, dates_ho)
            if ic > best_ic:
                best_ic = ic
                best_w = [w0, w1, w2]
    t = sum(best_w)
    return [w / t for w in best_w], best_ic


def optimize_weights_fbeta(models, X_ho, y_ho, dates_ho, k=5, beta=1.5):
    y, d = np.array(y_ho), np.array(dates_ho)
    step = 0.1
    grid = np.round(np.arange(0, 1 + step, step), 10)
    best_score, best_w = -1.0, [1/3, 1/3, 1/3]
    for w0 in grid:
        for w1 in grid:
            w2 = round(1.0 - w0 - w1, 10)
            if w2 < -1e-9:
                continue
            w2 = max(w2, 0.0)
            probs = [get_prob(m, X_ho) for m in models]
            w_arr = np.array([w0, w1, w2])
            w_arr /= w_arr.sum()
            combined = sum(wi * p for wi, p in zip(w_arr, probs))
            s = top_k_fbeta(y, combined, d, k=k, beta=beta)
            if s > best_score:
                best_score = s
                best_w = [w0, w1, w2]
    t = sum(best_w)
    return [w / t for w in best_w], best_score


# ── Walk-forward splitter ─────────────────────────────────────────────

def _make_wf_folds(dates_series: pd.Series):
    unique_dates = sorted(dates_series.unique())
    n_total      = len(unique_dates)
    n_holdout    = max(1, int(n_total * HOLDOUT_FRAC))
    cv_dates     = unique_dates[:n_total - n_holdout]
    ho_dates     = set(unique_dates[n_total - n_holdout:])
    n_cv         = len(cv_dates)

    eff_folds = N_CV_FOLDS
    fold_size = max(1, n_cv // (eff_folds + 1))
    while fold_size < MIN_FOLD_MONTHS and eff_folds > 2:
        eff_folds -= 1
        fold_size = max(1, n_cv // (eff_folds + 1))

    folds = []
    for i in range(eff_folds):
        tr_end = (i + 1) * fold_size
        va_end = min(tr_end + fold_size, n_cv)
        if tr_end >= va_end:
            break
        folds.append((cv_dates[:tr_end], cv_dates[tr_end:va_end]))

    log.info(f"  WF-CV: {len(folds)} fold × ~{fold_size} ay | holdout={len(ho_dates)} ay | N_TRIALS={N_TRIALS}")
    return folds, ho_dates


# ── HPO objective fonksiyonları ───────────────────────────────────────

def score_fold(model, X_va, y_va, dates_va, classification: bool) -> float:
    if classification:
        probs = get_prob(model, X_va)
        return top_k_fbeta(y_va, probs, dates_va)
    else:
        return compute_ic(y_va, model.predict(X_va), dates_va)


def _xgb_objective(trial, X, y, dates, folds, classification: bool):
    n_est = trial.suggest_int("n_estimators", 200, 800)
    p = dict(
        n_estimators     = n_est,
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    if classification:
        p["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.5, 20.0, log=True)
        Cls = xgb.XGBClassifier
    else:
        Cls = xgb.XGBRegressor

    dates_s = pd.Series(dates)
    fold_scores = []
    for fi, (tr_d, va_d) in enumerate(folds):
        tr_m = dates_s.isin(tr_d)
        va_m = dates_s.isin(va_d)
        m = Cls(**p)
        m.fit(X[tr_m], y[tr_m])
        fold_scores.append(score_fold(m, X[va_m], y[va_m], dates[va_m], classification))
        trial.report(float(np.mean(fold_scores)), step=fi)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(fold_scores))


def _lgb_objective(trial, X, y, dates, folds, classification: bool):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 800),
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    if classification:
        cw = trial.suggest_float("cw_pos", 0.5, 20.0, log=True)
        p["class_weight"] = {0: 1.0, 1: cw}
        Cls = lgb.LGBMClassifier
    else:
        Cls = lgb.LGBMRegressor

    dates_s = pd.Series(dates)
    fold_scores = []
    for fi, (tr_d, va_d) in enumerate(folds):
        tr_m = dates_s.isin(tr_d)
        va_m = dates_s.isin(va_d)
        m = Cls(**p)
        m.fit(X[tr_m], y[tr_m])
        fold_scores.append(score_fold(m, X[va_m], y[va_m], dates[va_m], classification))
        trial.report(float(np.mean(fold_scores)), step=fi)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(fold_scores))


def _cat_objective(trial, X, y, dates, folds, classification: bool):
    p = dict(
        iterations          = trial.suggest_int("iterations", 200, 800),
        depth               = trial.suggest_int("depth", 3, 8),
        learning_rate       = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True),
        subsample           = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bylevel   = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        random_seed=SEED, verbose=False,
    )
    if classification:
        cw = trial.suggest_float("cw_pos", 0.5, 20.0, log=True)
        p["class_weights"] = {0: 1.0, 1: cw}
        Cls = CatBoostClassifier
    else:
        Cls = CatBoostRegressor

    dates_s = pd.Series(dates)
    fold_scores = []
    for fi, (tr_d, va_d) in enumerate(folds):
        tr_m = dates_s.isin(tr_d)
        va_m = dates_s.isin(va_d)
        m = Cls(**p)
        m.fit(X[tr_m], y[tr_m], verbose=False)
        fold_scores.append(score_fold(m, X[va_m], y[va_m], dates[va_m], classification))
        trial.report(float(np.mean(fold_scores)), step=fi)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(fold_scores))


def run_hpo(name: str, objective_fn, n_trials: int = N_TRIALS):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
    )
    study.optimize(objective_fn, n_trials=n_trials)
    log.info(f"  {name:10s} | Best CV skor: {study.best_value:.4f}")
    return study.best_params, study.best_value


# ── Ana eğitim ────────────────────────────────────────────────────────

def main(target: str = None, save_name: str = None, classification: bool = True):
    sys.stdout.reconfigure(encoding="utf-8")
    _target    = target    or TARGET
    _save_name = save_name or SAVE_NAME
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    mode_str = "Sınıflandırma" if classification else "Regresyon"
    log.info(f"Target: {_target} | Mod: {mode_str} | Artifact: {_save_name} | N_TRIALS={N_TRIALS}")

    df = load_latest_parquet()
    log.info(f"Toplam: {len(df):,} satır | {len(df.columns)} sütun")

    train_df = df[df["month_end"] <= TRAIN_END].copy().sort_values("month_end").reset_index(drop=True)

    if MAX_TRAIN_MONTHS > 0:
        cutoff = pd.to_datetime(TRAIN_END) - pd.DateOffset(months=MAX_TRAIN_MONTHS)
        before = len(train_df)
        train_df = train_df[train_df["month_end"] > cutoff].copy().reset_index(drop=True)
        log.info(f"Kayan pencere: {before:,} → {len(train_df):,} satır (son {MAX_TRAIN_MONTHS} ay)")

    oot_df = df[df["month_end"] >= OOT_START].copy()
    log.info(f"Train: {len(train_df):,} | OOT: {len(oot_df):,}")

    le = LabelEncoder()
    le.fit(df["sector"].fillna("Unknown").astype(str))
    train_df["sector_enc"] = le.transform(train_df["sector"].fillna("Unknown").astype(str))
    oot_df = oot_df.copy()
    oot_df["sector_enc"] = le.transform(oot_df["sector"].fillna("Unknown").astype(str))

    feat_cols = get_feature_cols(df) + ["sector_enc"]
    feat_cols = [c for c in feat_cols if c in train_df.columns]
    log.info(f"Feature sayısı: {len(feat_cols)}")

    # Winsorizer: CV kısmına fit (holdout sızıntısı önlenir)
    folds_tmp, ho_dates_tmp = _make_wf_folds(train_df["month_end"])
    cv_mask_tmp = ~train_df["month_end"].isin(ho_dates_tmp)
    bounds = fit_winsorizer(train_df[cv_mask_tmp])
    train_w = apply_winsorizer(train_df, bounds)
    oot_w   = apply_winsorizer(oot_df,   bounds)

    X_train_all = train_w[feat_cols].values
    y_train_all = train_df[_target].values
    d_train_all = train_df["month_end"].values

    ok_mask = ~np.isnan(y_train_all)
    X_ok    = X_train_all[ok_mask]
    y_ok    = y_train_all[ok_mask]
    d_ok    = d_train_all[ok_mask]

    folds, ho_dates = _make_wf_folds(pd.Series(d_ok))
    ho_mask  = pd.Series(d_ok).isin(ho_dates).values
    cv_mask  = ~ho_mask

    X_cv, y_cv, d_cv = X_ok[cv_mask], y_ok[cv_mask], d_ok[cv_mask]
    X_ho, y_ho, d_ho = X_ok[ho_mask], y_ok[ho_mask], d_ok[ho_mask]
    log.info(f"CV: {len(y_cv):,} | Holdout: {len(y_ho):,}")

    _ws = WARM_START_PARAMS

    # ── XGBoost HPO ──
    log.info(f"XGBoost HPO ({N_TRIALS} trial, {len(folds)}-fold WF-CV)...")
    xgb_obj = lambda t: _xgb_objective(t, X_cv, y_cv, d_cv, folds, classification)
    s_xgb = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if _ws.get("xgb"):
        try:
            s_xgb.enqueue_trial(_ws["xgb"])
        except Exception:
            pass
    s_xgb.optimize(xgb_obj, n_trials=N_TRIALS)
    xgb_params, xgb_cv_score = s_xgb.best_params, s_xgb.best_value

    # ── LightGBM HPO ──
    log.info(f"LightGBM HPO ({N_TRIALS} trial, {len(folds)}-fold WF-CV)...")
    lgb_obj = lambda t: _lgb_objective(t, X_cv, y_cv, d_cv, folds, classification)
    s_lgb = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if _ws.get("lgb"):
        try:
            s_lgb.enqueue_trial(_ws["lgb"])
        except Exception:
            pass
    s_lgb.optimize(lgb_obj, n_trials=N_TRIALS)
    lgb_params, lgb_cv_score = s_lgb.best_params, s_lgb.best_value

    # ── CatBoost HPO ──
    log.info(f"CatBoost HPO ({N_TRIALS} trial, {len(folds)}-fold WF-CV)...")
    cat_obj = lambda t: _cat_objective(t, X_cv, y_cv, d_cv, folds, classification)
    s_cat = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if _ws.get("cat"):
        try:
            s_cat.enqueue_trial(_ws["cat"])
        except Exception:
            pass
    s_cat.optimize(cat_obj, n_trials=N_TRIALS)
    cat_params, cat_cv_score = s_cat.best_params, s_cat.best_value

    # ── Final modeller: tüm CV verisi üzerinde ──
    log.info("Final modeller CV verisi üzerinde eğitiliyor...")
    if classification:
        xgb_final = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items()},
                                       random_state=SEED, n_jobs=-1, tree_method="hist")
        _lgb_p  = {k: v for k, v in lgb_params.items() if k != "cw_pos"}
        lgb_final = lgb.LGBMClassifier(**_lgb_p,
                                        class_weight={0: 1.0, 1: lgb_params.get("cw_pos", 1.0)},
                                        random_state=SEED, n_jobs=-1, verbosity=-1)
        _cat_p  = {k: v for k, v in cat_params.items() if k != "cw_pos"}
        cat_final = CatBoostClassifier(**_cat_p,
                                        class_weights={0: 1.0, 1: cat_params.get("cw_pos", 1.0)},
                                        random_seed=SEED, verbose=False)
    else:
        xgb_final = xgb.XGBRegressor(**xgb_params, random_state=SEED, n_jobs=-1, tree_method="hist")
        lgb_final = lgb.LGBMRegressor(**lgb_params, random_state=SEED, n_jobs=-1, verbosity=-1)
        cat_final = CatBoostRegressor(**cat_params, random_seed=SEED, verbose=False)

    xgb_final.fit(X_cv, y_cv)
    lgb_final.fit(X_cv, y_cv)
    cat_final.fit(X_cv, y_cv, verbose=False)

    # ── Ensemble ağırlık optimizasyonu (holdout grid-search) ──
    log.info("Ensemble ağırlıkları holdout üzerinde optimize ediliyor...")
    models = [xgb_final, lgb_final, cat_final]
    if classification:
        weights, holdout_score = optimize_weights_fbeta(models, X_ho, y_ho, d_ho)
        metric_name = "Holdout F-beta"
    else:
        weights, holdout_score = optimize_weights_ic(models, X_ho, y_ho, d_ho)
        metric_name = "Holdout IC"

    log.info(f"  {metric_name}: {holdout_score:.4f} | "
             f"W: XGB={weights[0]:.2f} LGB={weights[1]:.2f} CAT={weights[2]:.2f}")

    # ── OOT değerlendirme ──
    X_oot  = oot_w[feat_cols].values
    y_oot  = oot_df[_target].values
    d_oot  = oot_df["month_end"].values
    ok_oot = ~np.isnan(y_oot)

    if ok_oot.sum() > 0:
        if classification:
            p_xgb = get_prob(xgb_final, X_oot[ok_oot])
            p_lgb = get_prob(lgb_final, X_oot[ok_oot])
            p_cat = get_prob(cat_final, X_oot[ok_oot])
        else:
            p_xgb = xgb_final.predict(X_oot[ok_oot])
            p_lgb = lgb_final.predict(X_oot[ok_oot])
            p_cat = cat_final.predict(X_oot[ok_oot])

        p_ens = weights[0]*p_xgb + weights[1]*p_lgb + weights[2]*p_cat

        if classification:
            oot_score_xgb = top_k_fbeta(y_oot[ok_oot], p_xgb, d_oot[ok_oot])
            oot_score_lgb = top_k_fbeta(y_oot[ok_oot], p_lgb, d_oot[ok_oot])
            oot_score_cat = top_k_fbeta(y_oot[ok_oot], p_cat, d_oot[ok_oot])
            oot_score_ens = top_k_fbeta(y_oot[ok_oot], p_ens, d_oot[ok_oot])
            ens_oot_ic    = compute_ic(y_oot[ok_oot], p_ens, d_oot[ok_oot])
            score_label   = "OOT F-beta"
        else:
            oot_score_xgb = compute_ic(y_oot[ok_oot], p_xgb, d_oot[ok_oot])
            oot_score_lgb = compute_ic(y_oot[ok_oot], p_lgb, d_oot[ok_oot])
            oot_score_cat = compute_ic(y_oot[ok_oot], p_cat, d_oot[ok_oot])
            oot_score_ens = compute_ic(y_oot[ok_oot], p_ens, d_oot[ok_oot])
            ens_oot_ic    = oot_score_ens
            score_label   = "OOT IC"
    else:
        oot_score_xgb = oot_score_lgb = oot_score_cat = oot_score_ens = ens_oot_ic = 0.0
        score_label = "OOT IC"

    # top-1 precision: OOT'ta en yüksek skorlu tahmin gerçekte label=1 mi?
    top1_pr = 0.0
    if ok_oot.sum() > 0:
        hits = []
        for dt in np.unique(d_oot[ok_oot]):
            m = d_oot[ok_oot] == dt
            if m.sum() < 1:
                continue
            hits.append(int(y_oot[ok_oot][m][np.argmax(p_ens[m])] > 0.5))
        top1_pr = float(np.mean(hits)) if hits else 0.0

    # ── Feature importance ──
    fi = pd.Series(xgb_final.feature_importances_, index=feat_cols)
    fi_norm  = fi / fi.sum() if fi.sum() > 0 else fi
    fi_top20 = fi_norm.sort_values(ascending=False).head(20).to_dict()

    fi_path = SAVE_DIR / f"feature_importance_{date.today().strftime('%Y%m%d')}.json"
    with open(fi_path, "w", encoding="utf-8") as f:
        _json.dump(fi_top20, f, indent=2, ensure_ascii=False)

    # ── Artifacts ──
    artifacts = {
        "xgb_model":       xgb_final,
        "lgb_model":       lgb_final,
        "cat_model":       cat_final,
        "weights":         np.array(weights),
        "bounds":          bounds,
        "label_enc":       le,
        "feat_cols":       feat_cols,
        "classification":  classification,
        "ho_ic":           holdout_score if not classification else ens_oot_ic,
        "top1_precision":  top1_pr,
        "best_params":     {"xgb": dict(xgb_params), "lgb": dict(lgb_params), "cat": dict(cat_params)},
        "feature_importance": fi_top20,
        "cv_ics":  {"xgb": xgb_cv_score, "lgb": lgb_cv_score, "cat": cat_cv_score},
        "val_ics": {"xgb": xgb_cv_score, "lgb": lgb_cv_score, "cat": cat_cv_score},
        "oot_ics": {
            "xgb":      oot_score_xgb,
            "lgb":      oot_score_lgb,
            "cat":      oot_score_cat,
            "ensemble": ens_oot_ic,
        },
        "holdout_score": {metric_name: holdout_score},
        "xgb_params":  xgb_params,
        "lgb_params":  lgb_params,
        "cat_params":  cat_params,
        "train_end":   TRAIN_END,
    }
    save_path = SAVE_DIR / _save_name
    with open(save_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n{'='*58}\n  MODEL SONUÇLARI — {_target}  ({mode_str})\n{'='*58}")
    print(f"  {'Model':<12} {'CV Skor':>10} {score_label:>12}")
    print(f"  {'-'*36}")
    print(f"  {'XGBoost':<12} {xgb_cv_score:>10.4f} {oot_score_xgb:>12.4f}")
    print(f"  {'LightGBM':<12} {lgb_cv_score:>10.4f} {oot_score_lgb:>12.4f}")
    print(f"  {'CatBoost':<12} {cat_cv_score:>10.4f} {oot_score_cat:>12.4f}")
    print(f"  {'-'*36}")
    print(f"  {'Ensemble':<12} {'':>10} {oot_score_ens:>12.4f}")
    print(f"  {metric_name}: {holdout_score:.4f}")
    print(f"{'='*58}")
    print(f"  Artifacts: {save_path}")


if __name__ == "__main__":
    main()
