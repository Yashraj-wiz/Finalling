#!/usr/bin/env python3
"""
scripts/generate_plots_new.py

Generates all experimental results plots into results/plots_new/
without modifying the existing results/plots/ folder.
Uses data from results/ (CSVs and JSONL files).

All deltas are strictly calculated as:
  Delta = Noisy - Clean
  - dwer = wer_noisy - wer_clean (positive indicates error increase)
  - dacc = acc_noisy - acc_clean (negative indicates accuracy drop)
  - df1  = f1_noisy - f1_clean   (negative indicates F1 score drop)
"""

import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OLD_PLOTS_DIR = RESULTS_DIR / "plots"
NEW_PLOTS_DIR = RESULTS_DIR / "plots_new"
NEW_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── AACL / EMNLP Publication Style ──────────────────────────────────────────
def set_paper_style():
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'legend.title_fontsize': 14,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'figure.dpi': 300
    })
    sns.set_style("whitegrid")
    sns.set_palette("muted")

PALETTE_CAT = {'speech_like': '#d62728', 'non_speech': '#1f77b4'}
MODELS = ['gemma3n_e4b', 'phi4_multimodal', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b']

BG_CATEGORY_MAP = {
    "esc_rain": "non_speech",
    "esc_seawave": "non_speech",
    "esc_engine": "non_speech",
    "esc_vacuum": "non_speech",
    "esc_footsteps": "non_speech",
    "esc_fire": "non_speech",
    "esc_helicopter": "non_speech",
    "esc_clock": "non_speech",
    "esc_keyboard": "non_speech",
    "esc_dog": "non_speech",
    "esc_rooster": "non_speech",
    "esc_wind": "non_speech",
    "snsd_airconditioner": "non_speech",
    "snsd_office": "non_speech",
    "snsd_cafeteria": "speech_like",
    "snsd_restaurant": "speech_like",
    "snsd_square": "speech_like",
    "snsd_airport": "speech_like",
    "noisex_babble": "speech_like",
    "musan_music": "speech_like",
    "musan_hubbub": "speech_like",
}


def load_asr_data():
    csv_p = OLD_PLOTS_DIR / "asr_results.csv"
    if csv_p.exists():
        df = pd.read_csv(csv_p)
    else:
        rows = []
        for m in MODELS:
            jf = RESULTS_DIR / m / "asr.jsonl"
            if jf.exists():
                for line in open(jf, encoding="utf-8"):
                    if line.strip():
                        try:
                            d = json.loads(line)
                            d["model"] = m
                            rows.append(d)
                        except:
                            pass
        df = pd.DataFrame(rows)
    
    if "dwer" not in df.columns or df["dwer"].isnull().all():
        clean = df[df["condition"] == "clean"][["model", "speech_id", "wer"]].rename(columns={"wer": "wer_clean"})
        df = df.merge(clean, on=["model", "speech_id"], how="left")
        df["dwer"] = df["wer"] - df["wer_clean"]

    if "category" not in df.columns or df["category"].isnull().any():
        df["category"] = df["bg_id"].map(BG_CATEGORY_MAP)
    return df


def load_kws_data():
    csv_p = OLD_PLOTS_DIR / "kws_results.csv"
    if csv_p.exists():
        df = pd.read_csv(csv_p)
    else:
        rows = []
        for m in MODELS:
            jf = RESULTS_DIR / m / "kws.jsonl"
            if jf.exists():
                for line in open(jf, encoding="utf-8"):
                    if line.strip():
                        try:
                            d = json.loads(line)
                            d["model"] = m
                            rows.append(d)
                        except:
                            pass
        df = pd.DataFrame(rows)
        
    if "acc_clean" not in df.columns:
        clean = df[df["condition"] == "clean"][["model", "speech_id", "correct"]].rename(columns={"correct": "acc_clean"})
        df = df.merge(clean, on=["model", "speech_id"], how="left")

    # Delta = Noisy - Clean
    if "acc_clean" in df.columns and "correct" in df.columns:
        df["dacc"] = df["correct"] - df["acc_clean"]

    if "category" not in df.columns or df["category"].isnull().any():
        df["category"] = df["bg_id"].map(BG_CATEGORY_MAP)
    return df


def load_sqa_data():
    csv_p = RESULTS_DIR / "e1_sqa.csv"
    if csv_p.exists():
        df = pd.read_csv(csv_p)
    else:
        rows = []
        for m in MODELS:
            jf = RESULTS_DIR / m / "sqa.jsonl"
            if jf.exists():
                for line in open(jf, encoding="utf-8"):
                    if line.strip():
                        try:
                            d = json.loads(line)
                            d["model"] = m
                            rows.append(d)
                        except:
                            pass
        df = pd.DataFrame(rows)
    
    if "bg_id" not in df.columns and "background_id" in df.columns:
        df["bg_id"] = df["background_id"]
        
    df["category"] = df["bg_id"].map(BG_CATEGORY_MAP)
    
    if "f1_clean" not in df.columns:
        clean = df[df["condition"] == "clean"][["model", "speech_id", "f1"]].rename(columns={"f1": "f1_clean"})
        df = df.merge(clean, on=["model", "speech_id"], how="left")
    
    # Delta = Noisy - Clean
    if "f1" in df.columns and "f1_clean" in df.columns:
        df["df1"] = df["f1"] - df["f1_clean"]
        
    return df


def compute_dri(df, group_col, category_col):
    results = []
    for (model, cat), group in df.groupby(["model", category_col]):
        means = group.groupby(group_col)["dwer"].mean().dropna()
        if len(means) >= 2:
            gap = float(means.max() - means.min())
            mean_val = float(means.mean())
            dri = gap / (mean_val + 1e-9)
            results.append({"model": model, "category": cat, "dri": dri})
    return pd.DataFrame(results)


def plot_bar_dwer_vs_accents(asr_df):
    noisy = asr_df[(asr_df["condition"] == "noisy") & asr_df["accent"].notnull()]
    if noisy.empty: return
    plt.figure(figsize=(14, 8))
    sns.barplot(data=noisy, x="accent", y="dwer", hue="model", errorbar=None)
    plt.title("ASR: Mean ΔWER across Accents (Noisy - Clean)", pad=20)
    plt.ylabel("ΔWER (Noisy - Clean)")
    plt.xlabel("Accent")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_dwer_vs_accents.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_dwer_vs_gender(asr_df):
    noisy = asr_df[(asr_df["condition"] == "noisy") & asr_df["gender"].notnull()]
    if noisy.empty: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=noisy, x="gender", y="dwer", hue="model", errorbar=None)
    plt.title("ASR: Mean ΔWER across Genders (Noisy - Clean)", pad=20)
    plt.ylabel("ΔWER (Noisy - Clean)")
    plt.xlabel("Gender")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_dwer_vs_gender.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_line_wer_vs_snr(asr_df):
    if "snr_db" not in asr_df.columns: return
    valid = asr_df[asr_df["snr_db"].notnull()]
    if valid.empty: return
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=valid, x="snr_db", y="wer", hue="model", marker="o", linewidth=2.5, markersize=10)
    plt.title("ASR: WER across Signal-to-Noise Ratios (SNR)", pad=20)
    plt.ylabel("WER")
    plt.xlabel("SNR (dB)")
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "line_wer_vs_snr.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_dwer_speech_vs_noise(asr_df):
    noisy = asr_df[asr_df["condition"] == "noisy"]
    comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
    if comp.empty: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=comp, x="model", y="dwer", hue="category", errorbar=None, palette=PALETTE_CAT)
    plt.title("ASR: Mean ΔWER by Noise Type (Noisy - Clean)", pad=20)
    plt.ylabel("ΔWER (Noisy - Clean)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_dwer_speech_vs_noise.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_kws_acc_speech_vs_noise(kws_df):
    noisy = kws_df[kws_df["condition"] == "noisy"]
    comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
    if comp.empty: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=comp, x="model", y="dacc", hue="category", errorbar=None, palette=PALETTE_CAT)
    plt.title("KWS: Mean ΔAccuracy by Noise Type (Noisy - Clean)", pad=20)
    plt.ylabel("ΔAccuracy (Noisy - Clean)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_kws_acc_speech_vs_noise.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_kws_acc_vs_accents(kws_df):
    noisy = kws_df[(kws_df["condition"] == "noisy") & kws_df["accent"].notnull()]
    if noisy.empty: return
    plt.figure(figsize=(14, 8))
    sns.barplot(data=noisy, x="accent", y="dacc", hue="model", errorbar=None)
    plt.title("KWS: Mean ΔAccuracy across Accents (Noisy - Clean)", pad=20)
    plt.ylabel("ΔAccuracy (Noisy - Clean)")
    plt.xlabel("Accent")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_kws_acc_vs_accents.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_kws_acc_vs_gender(kws_df):
    noisy = kws_df[(kws_df["condition"] == "noisy") & kws_df["gender"].notnull()]
    if noisy.empty: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=noisy, x="gender", y="dacc", hue="model", errorbar=None)
    plt.title("KWS: Mean ΔAccuracy across Genders (Noisy - Clean)", pad=20)
    plt.ylabel("ΔAccuracy (Noisy - Clean)")
    plt.xlabel("Gender")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_kws_acc_vs_gender.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_line_dri_speech_vs_noise(asr_df):
    noisy = asr_df[asr_df["condition"] == "noisy"]
    comp = noisy[noisy["category"].isin(["speech_like", "non_speech"]) & noisy["accent"].notnull()]
    if comp.empty: return
    dri_df = compute_dri(comp, group_col="accent", category_col="category")
    if dri_df.empty: return
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=dri_df, x="model", y="dri", hue="category", marker="s", linewidth=3, markersize=12, palette=PALETTE_CAT)
    plt.title("ASR: Degradation Robustness Index (DRI) by Noise Type", pad=20)
    plt.ylabel("DRI")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "line_dri_speech_vs_noise.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_sqa_f1_clean_vs_noisy(sqa_df):
    if sqa_df.empty or "f1" not in sqa_df.columns: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=sqa_df, x="model", y="f1", hue="condition", errorbar=None)
    plt.title("SQA: Mean F1 Score (Clean vs Noisy Baseline)", pad=20)
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_sqa_f1_clean_vs_noisy.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_line_sqa_f1_vs_snr(sqa_df):
    if sqa_df.empty or "snr_db" not in sqa_df.columns: return
    noisy = sqa_df[sqa_df["snr_db"].notnull() & (sqa_df["condition"] == "noisy")]
    if noisy.empty: return
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=noisy, x="snr_db", y="f1", hue="model", marker="o", linewidth=2.5, markersize=10)
    plt.title("SQA: F1 Score across Signal-to-Noise Ratios (SNR)", pad=20)
    plt.ylabel("F1 Score")
    plt.xlabel("SNR (dB)")
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "line_sqa_f1_vs_snr.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_sqa_f1_speech_vs_noise(sqa_df):
    if sqa_df.empty or "df1" not in sqa_df.columns: return
    noisy = sqa_df[sqa_df["condition"] == "noisy"]
    comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
    if comp.empty: return
    plt.figure(figsize=(12, 8))
    sns.barplot(data=comp, x="model", y="df1", hue="category", errorbar=None, palette=PALETTE_CAT)
    plt.title("SQA: Mean ΔF1 by Noise Type (Noisy - Clean)", pad=20)
    plt.ylabel("ΔF1 (Noisy - Clean)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_sqa_f1_speech_vs_noise.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_bar_silero_vad_speech_vs_nonspeech(asr_df):
    if "speech_likeness" not in asr_df.columns or "bg_id" not in asr_df.columns: return
    df_bg = asr_df.groupby("bg_id").agg({
        "speech_likeness": "mean",
        "category": "first"
    }).reset_index().dropna()
    
    if df_bg.empty: return
    df_bg = df_bg.sort_values("speech_likeness", ascending=False)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_bg, x="bg_id", y="speech_likeness", hue="category", palette=PALETTE_CAT, dodge=False)
    plt.title("Silero VAD Speech Likeness Score by Background Sound Class", pad=20)
    plt.ylabel("Speech Likeness (Silero VAD Voiced Ratio)")
    plt.xlabel("Background Class")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    out = NEW_PLOTS_DIR / "bar_silero_vad_speech_vs_nonspeech.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Generated {out.name}")


def plot_holistic_and_task_degradations(asr_df, kws_df, sqa_df):
    # Prepare ASR (dwer = noisy - clean; higher positive dwer = worse performance)
    df_asr = asr_df[asr_df['snr_db'] == 0].copy()
    if df_asr.empty: return
    if 'bg_id' not in df_asr.columns and 'background_id' in df_asr.columns:
        df_asr['bg_id'] = df_asr['background_id']
    df_asr['bg_class'] = df_asr['bg_id']
    df_asr['degradation'] = df_asr['dwer']
    df_asr['task'] = 'ASR'
    df_asr['dataset'] = 'ASR_Dataset'

    # Prepare KWS (dacc = noisy - clean; -dacc = clean - noisy, higher positive = worse performance)
    df_kws = kws_df[(kws_df['condition'] == 'noisy') & (kws_df['snr_db'] == 0)].copy()
    if 'bg_id' not in df_kws.columns and 'background_id' in df_kws.columns:
        df_kws['bg_id'] = df_kws['background_id']
    df_kws['bg_class'] = df_kws['bg_id']
    df_kws['degradation'] = -df_kws['dacc'] if 'dacc' in df_kws.columns else 0
    df_kws['task'] = 'KWS'
    df_kws['dataset'] = 'KWS_Dataset'

    # Prepare SQA (df1 = noisy - clean; -df1 = clean - noisy, higher positive = worse performance)
    df_sqa = sqa_df[(sqa_df['condition'] == 'noisy') & (sqa_df['snr_db'] == 0)].copy() if not sqa_df.empty else pd.DataFrame()
    if not df_sqa.empty:
        if 'bg_id' not in df_sqa.columns and 'background_id' in df_sqa.columns:
            df_sqa['bg_id'] = df_sqa['background_id']
        df_sqa['bg_class'] = df_sqa['bg_id']
        df_sqa['degradation'] = -df_sqa['df1'] if 'df1' in df_sqa.columns else 0
        df_sqa['task'] = 'SQA'
        df_sqa['dataset'] = 'SQA_Dataset'

    cols_to_keep = ['task', 'model', 'dataset', 'gender', 'accent', 'bg_class', 'degradation']

    def extract_cols(d):
        if d.empty: return pd.DataFrame(columns=cols_to_keep)
        for col in ['gender', 'accent']:
            if col not in d.columns: d[col] = 'unknown'
            else: d[col] = d[col].fillna('unknown')
        return d[cols_to_keep].dropna(subset=['bg_class', 'degradation'])

    df_a = extract_cols(df_asr)
    df_k = extract_cols(df_kws)
    df_s = extract_cols(df_sqa)

    combined = pd.concat([df_a, df_k, df_s], ignore_index=True)
    if combined.empty: return

    buckets = ['task', 'model', 'dataset', 'gender', 'accent']
    combined['z_score'] = combined.groupby(buckets)['degradation'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    def draw_bg_plot(df_sub, title, fname):
        scores = df_sub.groupby('bg_class')['z_score'].mean().sort_values(ascending=False).reset_index()
        scores['category'] = scores['bg_class'].map(BG_CATEGORY_MAP)
        plt.figure(figsize=(16, 8))
        sns.barplot(data=scores, x='bg_class', y='z_score', hue='category', dodge=False, palette=PALETTE_CAT)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(title, pad=15)
        plt.ylabel("Average Z-Score (Higher = More Destructive)")
        plt.xlabel("Background Class")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = NEW_PLOTS_DIR / fname
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Generated {out.name}")

    draw_bg_plot(
        combined,
        "Holistic Background Degradation Score (Averaged across ASR, KWS, SQA at 0 dB)\nNormalized across Task, Model, Dataset, Gender, Accent",
        "e1_holistic_bg_degradation.png"
    )

    draw_bg_plot(
        combined[combined['task'] == 'ASR'],
        "ASR Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
        "e1_asr_bg_degradation.png"
    )

    draw_bg_plot(
        combined[combined['task'] == 'KWS'],
        "KWS Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
        "e1_kws_bg_degradation.png"
    )

    if not df_s.empty:
        draw_bg_plot(
            combined[combined['task'] == 'SQA'],
            "SQA Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
            "e1_sqa_bg_degradation.png"
        )


def main():
    set_paper_style()
    print("Loading datasets from results...")
    asr_df = load_asr_data()
    kws_df = load_kws_data()
    sqa_df = load_sqa_data()

    print(f"ASR records: {len(asr_df)}, KWS records: {len(kws_df)}, SQA records: {len(sqa_df)}")

    # Save CSV files with delta = noisy - clean in plots_new
    asr_df.to_csv(NEW_PLOTS_DIR / "asr_results.csv", index=False)
    kws_df.to_csv(NEW_PLOTS_DIR / "kws_results.csv", index=False)

    print("\nGenerating all plots in results/plots_new/...")
    plot_bar_dwer_vs_accents(asr_df)
    plot_bar_dwer_vs_gender(asr_df)
    plot_line_wer_vs_snr(asr_df)
    plot_bar_dwer_speech_vs_noise(asr_df)
    plot_bar_kws_acc_speech_vs_noise(kws_df)
    plot_bar_kws_acc_vs_accents(kws_df)
    plot_bar_kws_acc_vs_gender(kws_df)
    plot_line_dri_speech_vs_noise(asr_df)
    plot_bar_sqa_f1_clean_vs_noisy(sqa_df)
    plot_line_sqa_f1_vs_snr(sqa_df)
    plot_bar_sqa_f1_speech_vs_noise(sqa_df)
    plot_bar_silero_vad_speech_vs_nonspeech(asr_df)
    plot_holistic_and_task_degradations(asr_df, kws_df, sqa_df)

    print(f"\nSuccessfully generated all plots in: {NEW_PLOTS_DIR}")

if __name__ == '__main__':
    main()
