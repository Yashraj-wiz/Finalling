#!/usr/bin/env python3
"""
scoring/analyse.py — Stage 7: C-PROFILE regression, C-FAIR, C-INJECT analysis.

Reads scored CSVs from results/ and descriptors/battery.parquet.
Produces:
  results/profile_regression.csv   (Table 2 in paper)
  results/fairness_summary.csv
  results/injection_summary.csv

Usage:
  python scoring/analyse.py --smoke-test
  python scoring/analyse.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import ROOT, get_logger

log = get_logger("analyse")
RESULTS = ROOT / "results"


def _load(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        log.warning(f"[load] {name} not found — skipping.")
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_battery() -> pd.DataFrame:
    p = ROOT / "descriptors" / "battery.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


# ── C-PROFILE: mixed-effects regression ──────────────────────────────────────
def c_profile(smoke: bool) -> None:
    """
    Δ ~ SNR + speech_likeness + linguistic_content + mod_2to8Hz +
        spectral_overlap + stationarity + onset_density + (1|item) + (1|model) + (1|background)

    Uses statsmodels OLS as a practical approximation; random effects via
    dummy-encoding item/model/background then dropping them from output.
    """
    log.info("[C-PROFILE] Running descriptor regression...")
    asr = _load("e1_asr.csv")
    battery = _load_battery()

    if asr.empty or battery.empty:
        log.warning("[C-PROFILE] Missing data. Skipping.")
        return

    df = asr.merge(battery, on="background_id", how="left")
    df = df[df["condition"] == "noisy"].dropna(subset=["dwer"])
    if smoke:
        df = df.head(50)

    DESCRIPTORS = [
        "snr_db", "speech_likeness", "linguistic_content", "mod_2to8Hz",
        "spectral_overlap", "stationarity", "onset_density",
    ]
    avail = [c for c in DESCRIPTORS if c in df.columns]
    if len(avail) < 2:
        log.warning("[C-PROFILE] Too few descriptor columns available.")
        return

    # Standardise predictors
    df_model = df[avail + ["dwer", "model", "speech_id", "background_id"]].dropna()
    for col in avail:
        mu, sd = df_model[col].mean(), df_model[col].std() + 1e-9
        df_model[col] = (df_model[col] - mu) / sd

    # Group dummies as random-effect proxies (mean-center per group)
    for grp in ["model", "speech_id", "background_id"]:
        if grp in df_model.columns:
            group_means = df_model.groupby(grp)["dwer"].transform("mean")
            df_model["_re_" + grp] = group_means

    import statsmodels.api as sm
    X_cols = avail + [c for c in df_model.columns if c.startswith("_re_")]
    X = sm.add_constant(df_model[X_cols].astype(float))
    y = df_model["dwer"].astype(float)
    res = sm.OLS(y, X).fit()

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = float("nan")
        vif_data.append({"predictor": col, "vif": round(v, 2)})

    # Build output table
    records = []
    for col in avail:
        if col not in res.params.index:
            continue
        records.append({
            "descriptor": col,
            "coeff_std":  round(float(res.params[col]), 4),
            "pvalue":     round(float(res.pvalues[col]), 4),
            "ci_lo":      round(float(res.conf_int().loc[col, 0]), 4),
            "ci_hi":      round(float(res.conf_int().loc[col, 1]), 4),
            "vif":        next((d["vif"] for d in vif_data if d["predictor"] == col), float("nan")),
            "task": "asr",
        })

    pd.DataFrame(records).to_csv(RESULTS / "profile_regression.csv", index=False)
    log.info(f"[C-PROFILE] Saved profile_regression.csv. R²={res.rsquared:.3f}")

    # KWS version (if available)
    kws = _load("e1_kws.csv")
    if not kws.empty:
        # Use miss rate as the outcome (higher miss = worse)
        kws_merged = kws.merge(battery, on="background_id", how="left")
        kws_merged = kws_merged[kws_merged["condition"] == "noisy"]
        if smoke:
            kws_merged = kws_merged.head(50)
        avail_kws = [c for c in DESCRIPTORS if c in kws_merged.columns]
        if avail_kws and "miss" in kws_merged.columns:
            for col in avail_kws:
                mu, sd = kws_merged[col].mean(), kws_merged[col].std() + 1e-9
                kws_merged[col] = (kws_merged[col] - mu) / sd
            X2 = sm.add_constant(kws_merged[avail_kws].dropna().astype(float))
            y2 = kws_merged.loc[X2.index, "miss"].astype(float)
            res2 = sm.OLS(y2, X2).fit()
            records2 = []
            for col in avail_kws:
                if col not in res2.params.index:
                    continue
                records2.append({
                    "descriptor": col,
                    "coeff_std": round(float(res2.params[col]), 4),
                    "pvalue":    round(float(res2.pvalues[col]), 4),
                    "ci_lo":     round(float(res2.conf_int().loc[col, 0]), 4),
                    "ci_hi":     round(float(res2.conf_int().loc[col, 1]), 4),
                    "vif":       float("nan"),
                    "task": "kws",
                })
            existing = pd.read_csv(RESULTS / "profile_regression.csv")
            combined = pd.concat([existing, pd.DataFrame(records2)])
            combined.to_csv(RESULTS / "profile_regression.csv", index=False)
            log.info(f"[C-PROFILE] KWS added. R²={res2.rsquared:.3f}")


# ── C-FAIR: disparity analysis ────────────────────────────────────────────────
def c_fair(smoke: bool) -> None:
    log.info("[C-FAIR] Fairness analysis...")
    e3 = _load("e3_fairness.csv")
    if e3.empty:
        log.warning("[C-FAIR] e3_fairness.csv not found.")
        return

    records = []
    for sg_type in e3["subgroup_type"].unique():
        sub = e3[e3["subgroup_type"] == sg_type].dropna(subset=["mean_dwer"])
        gap = float(sub["mean_dwer"].max() - sub["mean_dwer"].min())
        mean_d = float(sub["mean_dwer"].mean())
        dri = gap / (mean_d + 1e-9)
        records.append({
            "subgroup_type": sg_type,
            "robustness_gap": round(gap, 4),
            "mean_dwer": round(mean_d, 4),
            "dri": round(dri, 4),
            "n_groups": len(sub),
        })

    pd.DataFrame(records).to_csv(RESULTS / "fairness_summary.csv", index=False)
    log.info("[C-FAIR] → fairness_summary.csv")

    # Paired permutation test for disparity claim
    e1 = _load("e1_asr.csv")
    if not e1.empty and "accent" in e1.columns and "dwer" in e1.columns:
        _permutation_test_disparity(e1, smoke)


def _permutation_test_disparity(df: pd.DataFrame, smoke: bool, n_perm: int = 1000) -> None:
    """Paired permutation test: is ΔWER gap across accents significant?"""
    df = df[df["condition"] == "noisy"].dropna(subset=["dwer", "accent"])
    if smoke:
        n_perm = 100
    accents = df["accent"].unique()
    if len(accents) < 2:
        return

    group_means = df.groupby("accent")["dwer"].mean()
    observed_gap = float(group_means.max() - group_means.min())

    rng = np.random.default_rng(42)
    null_gaps = []
    labels = df["accent"].values.copy()
    for _ in range(n_perm):
        rng.shuffle(labels)
        perm_means = pd.Series(df["dwer"].values).groupby(labels).mean()
        null_gaps.append(perm_means.max() - perm_means.min())

    p = float(np.mean(np.array(null_gaps) >= observed_gap))
    log.info(f"[C-FAIR] Permutation test: observed gap={observed_gap:.4f}, p={p:.4f}")

    perm_out = RESULTS / "fairness_permutation.json"
    perm_out.write_text(
        __import__("json").dumps({
            "observed_gap": round(observed_gap, 4),
            "p_value": round(p, 4),
            "n_permutations": n_perm,
            "significant": bool(p < 0.05),
        }, indent=2)
    )


# ── C-INJECT: injection analysis ─────────────────────────────────────────────
def c_inject(smoke: bool) -> None:
    log.info("[C-INJECT] Injection analysis...")
    e2_asr = _load("e2_asr.csv")
    e2_tir = _load("e2_tir.csv")

    records = []
    if not e2_asr.empty:
        for bg_id, grp in e2_asr.groupby("background_id"):
            records.append({
                "background_id": bg_id,
                "mean_semantic_gap": round(float(grp["semantic_gap"].mean()), 4)
                    if "semantic_gap" in grp else float("nan"),
                "mean_bir":         round(float(grp["bir"].mean()), 4)
                    if "bir" in grp else float("nan"),
                "n": len(grp),
            })

    if not e2_tir.empty:
        mean_tir = float(e2_tir["tir"].mean()) if "tir" in e2_tir else float("nan")
        log.info(f"[C-INJECT] Mean TIR = {mean_tir:.4f}")

    pd.DataFrame(records).to_csv(RESULTS / "injection_summary.csv", index=False)
    log.info("[C-INJECT] → injection_summary.csv")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7 — Analysis")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Use a small subset for quick sanity check.")
    args = ap.parse_args()

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    RESULTS.mkdir(parents=True, exist_ok=True)
    c_profile(smoke=args.smoke_test)
    c_fair(smoke=args.smoke_test)
    c_inject(smoke=args.smoke_test)
    log.info("=== Stage 7 complete. ===")


if __name__ == "__main__":
    main()
