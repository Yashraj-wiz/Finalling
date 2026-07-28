#!/usr/bin/env python3
"""
scoring/figures.py — Stage 8: produce all paper figures and appendix plots.

Main paper figures (Figs 1–3):
  Fig 1: ΔWER vs speech_likeness and vs mod_2to8Hz (partial-dependence, 2 panels)
  Fig 2: Semantic gap
  Fig 3: Disparate robustness: ΔWER by accent/gender (grouped bars)

Appendix figures (A1–A5):
  A1: SNR dose-response per background cluster
  A2: Error/injection taxonomy (sub/del/ins/injection stacked bars)
  A3: Steerability RER per model
  A4: Battery descriptor-space map (PCA scatter)
  A5: Architecture view: Δ vs encoder-coupling tier

All figures saved to results/figures/

Usage:
  python scoring/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import ROOT, get_logger

log = get_logger("figures")
RESULTS  = ROOT / "results"
FIG_DIR  = RESULTS / "figures"

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _load(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_battery() -> pd.DataFrame:
    p = ROOT / "descriptors" / "battery.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "category" in df.columns:
        pass
    return df


# ── Fig 1: Partial-dependence (descriptor law) ────────────────────────────────
def fig1_descriptor_law() -> None:
    asr = _load("e1_asr.csv")
    battery = _load_battery()
    if asr.empty or battery.empty:
        log.warning("[Fig1] Missing data.")
        return

    df = asr[asr["condition"] == "noisy"].merge(battery, on="background_id", how="left")
    df = df.dropna(subset=["dwer"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), tight_layout=True)

    for ax, desc, xlabel in zip(
        axes,
        ["speech_likeness", "mod_2to8Hz"],
        ["Speech-likeness (P(speech))", "Modulation energy 2–8 Hz"],
    ):
        if desc not in df.columns:
            ax.text(0.5, 0.5, f"{desc}\nnot available", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        xs = df[desc].dropna()
        ys = df.loc[xs.index, "dwer"]
        # Partial-dependence proxy: bin and plot mean ± SE
        bins = np.percentile(xs, np.linspace(0, 100, 11))
        bin_idx = np.digitize(xs, bins)
        means, ses, centers = [], [], []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            v = ys[mask]
            means.append(v.mean())
            ses.append(v.sem())
            centers.append((bins[b-1] + bins[b]) / 2)
        ax.errorbar(centers, means, yerr=ses, fmt="o-", color=PALETTE[0], capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ΔWER")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_title(f"Partial dependence: ΔWER vs {desc}")

    fig.suptitle("Fig 1 — Descriptor Law: which background properties drive ASR failure?",
                 fontsize=11)
    out = FIG_DIR / "fig1_descriptor_law.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[Fig1] → {out}")


# ── Fig 2: Semantic gap ───────────────────────────────────────────────────────
def fig2_semantic_gap() -> None:
    e2 = _load("e2_asr.csv")
    if e2.empty:
        log.warning("[Fig2] e2_asr.csv not found.")
        return

    grouped = e2.groupby("background_id")["dwer_real"].mean().reset_index()
    grouped = grouped.dropna(subset=["dwer_real"])

    x = np.arange(len(grouped))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    ax.bar(x, grouped["dwer_real"], w, label="Real", color=PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["background_id"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ΔWER @ 0 dB")
    ax.set_title("Fig 2 — Real: Semantic Bias (E2 / H2–H3)")
    ax.legend()
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    out = FIG_DIR / "fig2_semantic_gap.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[Fig2] → {out}")


# ── Fig 3: Disparate robustness ───────────────────────────────────────────────
def fig3_disparate_robustness() -> None:
    e3 = _load("e3_fairness.csv")
    asr = _load("e1_asr.csv")
    if e3.empty:
        log.warning("[Fig3] e3_fairness.csv not found.")
        return

    # Accent subgroup
    sub_accent = e3[e3["subgroup_type"] == "accent"].copy()
    if sub_accent.empty:
        log.warning("[Fig3] No accent subgroup data.")
        return

    x = np.arange(len(sub_accent))
    w = 0.28
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

    ax.bar(x - w, sub_accent["mean_dwer"], w, label="Overall ΔWER", color=PALETTE[0])
    if "dwer_speech_like" in sub_accent.columns:
        ax.bar(x,     sub_accent["dwer_speech_like"].fillna(0), w,
               label="Speech-like bg", color=PALETTE[2])
    if "dwer_non_speech" in sub_accent.columns:
        ax.bar(x + w, sub_accent["dwer_non_speech"].fillna(0), w,
               label="Non-speech bg", color=PALETTE[3])

    # Clean gap (baseline WER difference across accents — overlaid as scatter)
    if not asr.empty and "accent" in asr.columns and "wer" in asr.columns:
        clean_wer = (asr[asr["condition"] == "clean"]
                     .groupby("accent")["wer"].mean().reset_index()
                     .rename(columns={"accent": "subgroup_value", "wer": "clean_wer"}))
        merged = sub_accent.merge(clean_wer, on="subgroup_value", how="left")
        if "clean_wer" in merged.columns:
            ax.scatter(x, merged["clean_wer"].fillna(0), marker="D", color="black",
                       zorder=5, label="Clean WER (baseline)", s=40)

    ax.set_xticks(x)
    ax.set_xticklabels(sub_accent["subgroup_value"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ΔWER")
    ax.set_title("Fig 3 — Disparate Robustness: ΔWER by Accent (E3 / H4)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    out = FIG_DIR / "fig3_disparate_robustness.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[Fig3] → {out}")


# ── Fig A1: SNR dose-response ─────────────────────────────────────────────────
def figa1_snr_dose_response() -> None:
    asr = _load("e1_asr.csv")
    battery = _load_battery()
    if asr.empty:
        return
    df = asr[asr["condition"] == "noisy"]
    if not battery.empty:
        df = df.merge(battery[["bg_id", "category"]], left_on="background_id",
                      right_on="bg_id", how="left")
        cat_col = "category"
    else:
        df["category"] = "unknown"
        cat_col = "category"

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    for i, (cat, grp) in enumerate(df.groupby(cat_col)):
        by_snr = grp.groupby("snr_db")["dwer"].mean()
        ax.plot(by_snr.index, by_snr.values, "o-", color=PALETTE[i % len(PALETTE)], label=cat)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean ΔWER")
    ax.set_title("Fig A1 — SNR Dose–Response per Background Cluster")
    ax.legend(fontsize=8)
    out = FIG_DIR / "figa1_snr_dose_response.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[FigA1] → {out}")


# ── Fig A2: Error/injection taxonomy ─────────────────────────────────────────
def figa2_error_taxonomy() -> None:
    asr = _load("e1_asr.csv")
    if asr.empty or "substitutions" not in asr.columns:
        log.warning("[FigA2] Error taxonomy columns not found.")
        return

    models = asr["model"].unique()
    x = np.arange(len(models))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    for i, col in enumerate(["substitutions", "deletions", "insertions"]):
        means = [asr[asr["model"] == m][col].mean() for m in models]
        ax.bar(x + (i - 1) * w, means, w, label=col.capitalize(), color=PALETTE[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean count per utterance")
    ax.set_title("Fig A2 — Error Taxonomy (sub / del / ins) per Model")
    ax.legend()
    out = FIG_DIR / "figa2_error_taxonomy.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[FigA2] → {out}")


# ── Fig A3: Steerability RER ─────────────────────────────────────────────────
def figa3_steerability() -> None:
    e4 = _load("e4_steerability.csv")
    if e4.empty:
        log.warning("[FigA3] e4_steerability.csv not found.")
        return

    models = e4["model"].unique()
    rers = []
    for m in models:
        base  = e4[(e4["model"] == m) & (e4["condition_label"] == "base")]["dwer"].mean()
        steer = e4[(e4["model"] == m) & (e4["condition_label"] == "steer")]["dwer"].mean()
        rers.append(steer / (base + 1e-9))

    fig, ax = plt.subplots(figsize=(6, 3.5), tight_layout=True)
    ax.bar(models, rers, color=PALETTE[:len(models)])
    ax.axhline(1.0, color="grey", ls="--", lw=0.8, label="No effect of instruction")
    ax.set_ylabel("RER (effect with / without instruction)")
    ax.set_title("Fig A3 — Steerability: Residual-Effect Ratio per Model")
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    out = FIG_DIR / "figa3_steerability.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[FigA3] → {out}")


# ── Fig A4: Battery descriptor-space map ─────────────────────────────────────
def figa4_battery_map() -> None:
    battery = _load_battery()
    if battery.empty:
        return

    desc_cols = ["speech_likeness", "linguistic_content", "mod_2to8Hz",
                 "spectral_overlap", "stationarity", "onset_density"]
    avail = [c for c in desc_cols if c in battery.columns]
    if len(avail) < 2:
        log.warning("[FigA4] Too few descriptor columns for PCA.")
        return

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = battery[avail].fillna(0).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    cats = battery.get("category", pd.Series(["unknown"] * len(battery)))
    cat_list = list(cats.unique())
    for i, cat in enumerate(cat_list):
        mask = cats == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cat,
                   color=PALETTE[i % len(PALETTE)], s=80, alpha=0.85)
    for j, bg_id in enumerate(battery.get("bg_id", pd.RangeIndex(len(battery)))):
        ax.annotate(str(bg_id), (coords[j, 0], coords[j, 1]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Fig A4 — Battery Map: Backgrounds in Descriptor Space")
    ax.legend(fontsize=8)
    out = FIG_DIR / "figa4_battery_map.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[FigA4] → {out}")


# ── Fig A5: Architecture view ─────────────────────────────────────────────────
COUPLING_TIER = {
    "qwen25_omni_3b":  "end-to-end",
    "qwen2_audio_7b":  "end-to-end",
    "phi4_multimodal": "LoRA adapter",
    "gemma3n_e4b":     "USM + LLM",
}

def figa5_architecture_view() -> None:
    asr = _load("e1_asr.csv")
    if asr.empty:
        return

    models = [m for m in asr["model"].unique() if m in COUPLING_TIER]
    tiers  = [COUPLING_TIER[m] for m in models]
    means  = [asr[(asr["model"] == m) & (asr["condition"] == "noisy")]["dwer"].mean()
              for m in models]

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    colors = [PALETTE[["end-to-end", "LoRA adapter", "USM + LLM"].index(t) % len(PALETTE)]
              for t in tiers]
    ax.bar(models, means, color=colors)
    ax.set_ylabel("Mean ΔWER (all noisy conditions)")
    ax.set_title("Fig A5 — Architecture View: ΔWER vs Coupling Tier")
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    # Add tier legend
    handles = [mpatches.Patch(color=PALETTE[i], label=t)
               for i, t in enumerate(["end-to-end", "LoRA adapter", "USM + LLM"])]
    ax.legend(handles=handles, fontsize=8)
    out = FIG_DIR / "figa5_architecture_view.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    log.info(f"[FigA5] → {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Try importing sklearn for A4; it's optional
    try:
        import sklearn  # noqa: F401
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False
        log.warning("[figures] scikit-learn not installed — Fig A4 will be skipped.")

    fig1_descriptor_law()
    fig2_semantic_gap()
    fig3_disparate_robustness()
    figa1_snr_dose_response()
    figa2_error_taxonomy()
    figa3_steerability()
    if _has_sklearn:
        figa4_battery_map()
    figa5_architecture_view()
    log.info(f"=== Stage 8 complete. Figures in {FIG_DIR} ===")


if __name__ == "__main__":
    main()
