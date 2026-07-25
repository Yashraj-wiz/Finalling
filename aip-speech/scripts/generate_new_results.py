import sys
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jiwer
import re
import numpy as np

# AACL Paper Styling
def set_paper_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'figure.titlesize': 20,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
        'font.family': 'sans-serif'
    })
    sns.set_style("whitegrid")
    sns.set_palette("muted")

ROOT = Path(__file__).resolve().parent.parent
INFER_ASR_DIR = ROOT / "inference_1"
INFER_KWS_DIR = ROOT / "inference_sync_all_1784759190"
OUT_DIR = ROOT / "result_new_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATTERY = ROOT / "descriptors" / "battery.parquet"

def clean_hyp(hyp):
    if not isinstance(hyp, str): return ""
    return re.sub(r'<\|.*?\|>', '', hyp).strip()

def normalize_text(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def calc_wer(ref, hyp):
    try:
        return float(jiwer.wer(normalize_text(ref), normalize_text(hyp)))
    except:
        return float("nan")

def extract_asr():
    rows = []
    if not INFER_ASR_DIR.exists(): return pd.DataFrame()
    for model_dir in INFER_ASR_DIR.iterdir():
        if not model_dir.is_dir(): continue
        asr_f = model_dir / "asr.jsonl"
        if not asr_f.exists(): continue
        model = model_dir.name
        
        with open(asr_f, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip(): continue
                try: data = json.loads(l)
                except: continue
                
                raw = clean_hyp(data.get("raw", ""))
                ref = data.get("transcript", "")
                
                rows.append({
                    "model": model,
                    "id": data.get("id"),
                    "speech_id": data.get("speech_id"),
                    "bg_id": data.get("background_id"),
                    "condition": data.get("condition"),
                    "snr_db": data.get("snr_db"),
                    "accent": data.get("accent"),
                    "gender": data.get("gender"),
                    "wer": calc_wer(ref, raw)
                })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        clean = df[df["condition"] == "clean"][["model", "speech_id", "wer"]].rename(columns={"wer": "wer_clean"})
        df = df.merge(clean, on=["model", "speech_id"], how="left")
        df["dwer"] = df["wer"] - df["wer_clean"]
    return df

def extract_kws():
    rows = []
    if not INFER_KWS_DIR.exists(): return pd.DataFrame()
    for model_dir in INFER_KWS_DIR.iterdir():
        if not model_dir.is_dir(): continue
        model = model_dir.name
        kws_f = model_dir / "kws.jsonl"
        if not kws_f.exists(): continue
        
        with open(kws_f, "r", encoding="utf-8") as f:
            for l in f:
                if not l.strip(): continue
                try: data = json.loads(l)
                except: continue
                
                raw = clean_hyp(data.get("raw", ""))
                ref = str(data.get("keyword", "")).lower()
                hyp = str(raw).lower()
                correct = 1 if ref in hyp else 0
                
                rows.append({
                    "model": model,
                    "id": data.get("id"),
                    "speech_id": data.get("speech_id"),
                    "bg_id": data.get("background_id"),
                    "condition": data.get("condition"),
                    "snr_db": data.get("snr_db"),
                    "accent": data.get("accent"),
                    "gender": data.get("gender"),
                    "correct": correct
                })
                
    df = pd.DataFrame(rows)
    if not df.empty:
        clean = df[df["condition"] == "clean"][["model", "speech_id", "correct"]].rename(columns={"correct": "acc_clean"})
        df = df.merge(clean, on=["model", "speech_id"], how="left")
        # dacc = acc_clean - correct (Accuracy drop relative to clean baseline)
        df["dacc"] = df["acc_clean"] - df["correct"]
    return df

def compute_dri(df, group_col, category_col):
    results = []
    for (model, cat), group in df.groupby(["model", category_col]):
        means = group.groupby(group_col)["dwer"].mean().dropna()
        if len(means) >= 2:
            gap = float(means.max() - means.min())
            mean_val = float(means.mean())
            dri = gap / (mean_val + 1e-9)
            results.append({
                "model": model,
                "category": cat,
                "dri": dri
            })
    return pd.DataFrame(results)

def main():
    set_paper_style()
    print("Loading data...")
    asr_df = extract_asr()
    kws_df = extract_kws()
    
    try:
        bat = pd.read_parquet(BATTERY)
        bat = bat.rename(columns={"bg_id": "bg_id"}) if "bg_id" in bat.columns else bat.rename(columns={"id": "bg_id"})
    except:
        bat = pd.DataFrame(columns=["bg_id", "category"])
        
    if not asr_df.empty:
        asr_df = asr_df.merge(bat, on="bg_id", how="left")
        asr_df.to_csv(OUT_DIR / "asr_results.csv", index=False)
        
    if not kws_df.empty:
        kws_df = kws_df.merge(bat, on="bg_id", how="left")
        kws_df.to_csv(OUT_DIR / "kws_results.csv", index=False)

    # 1. ASR Bar DWER vs Accents
    if not asr_df.empty and "accent" in asr_df.columns:
        noisy = asr_df[asr_df["condition"] == "noisy"]
        if not noisy.empty:
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=noisy, x="accent", y="dwer", hue="model", errorbar=None)
            plt.title("ASR: Mean ΔWER across Accents", pad=20)
            plt.ylabel("ΔWER")
            plt.xlabel("Accent")
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_dwer_vs_accents.png", dpi=300)
            plt.close()

    # 2. ASR Bar DWER vs Gender
    if not asr_df.empty and "gender" in asr_df.columns:
        noisy = asr_df[asr_df["condition"] == "noisy"]
        if not noisy.empty:
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=noisy, x="gender", y="dwer", hue="model", errorbar=None)
            plt.title("ASR: Mean ΔWER across Genders", pad=20)
            plt.ylabel("ΔWER")
            plt.xlabel("Gender")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_dwer_vs_gender.png", dpi=300)
            plt.close()

    # 3. ASR Line for WER vs SNR
    if not asr_df.empty and "snr_db" in asr_df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=asr_df[asr_df["snr_db"].notnull()], x="snr_db", y="wer", hue="model", marker="o", linewidth=2.5, markersize=10)
        plt.title("ASR: WER across Signal-to-Noise Ratios (SNR)", pad=20)
        plt.ylabel("WER")
        plt.xlabel("SNR (dB)")
        plt.gca().invert_xaxis()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "line_wer_vs_snr.png", dpi=300)
        plt.close()

    # 4. ASR Bar Speech vs Noise avg DWER
    if not asr_df.empty and "category" in asr_df.columns:
        noisy = asr_df[asr_df["condition"] == "noisy"]
        comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
        if not comp.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=comp, x="model", y="dwer", hue="category", errorbar=None)
            plt.title("ASR: Mean ΔWER by Noise Type", pad=20)
            plt.ylabel("ΔWER")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_dwer_speech_vs_noise.png", dpi=300)
            plt.close()

    # 5. KWS Bar Speech vs Non-Speech ΔAcc (Relative to Clean)
    if not kws_df.empty and "category" in kws_df.columns:
        noisy_kws = kws_df[kws_df["condition"] == "noisy"]
        comp = noisy_kws[noisy_kws["category"].isin(["speech_like", "non_speech"])]
        if not comp.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=comp, x="model", y="dacc", hue="category", errorbar=None)
            plt.title("KWS: Mean ΔAccuracy (Clean - Noisy) by Noise Type", pad=20)
            plt.ylabel("ΔAccuracy (Accuracy Drop)")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_kws_acc_speech_vs_noise.png", dpi=300)
            plt.close()

    # 6. KWS Bar ΔAcc vs Accents
    if not kws_df.empty and "accent" in kws_df.columns:
        noisy_kws = kws_df[kws_df["condition"] == "noisy"]
        if not noisy_kws.empty and noisy_kws["accent"].notnull().any():
            plt.figure(figsize=(14, 8))
            sns.barplot(data=noisy_kws, x="accent", y="dacc", hue="model", errorbar=None)
            plt.title("KWS: Mean ΔAccuracy across Accents", pad=20)
            plt.ylabel("ΔAccuracy (Accuracy Drop)")
            plt.xlabel("Accent")
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_kws_acc_vs_accents.png", dpi=300)
            plt.close()

    # 7. KWS Bar ΔAcc vs Gender
    if not kws_df.empty and "gender" in kws_df.columns:
        noisy_kws = kws_df[kws_df["condition"] == "noisy"]
        if not noisy_kws.empty and noisy_kws["gender"].notnull().any():
            plt.figure(figsize=(12, 8))
            sns.barplot(data=noisy_kws, x="gender", y="dacc", hue="model", errorbar=None)
            plt.title("KWS: Mean ΔAccuracy across Genders", pad=20)
            plt.ylabel("ΔAccuracy (Accuracy Drop)")
            plt.xlabel("Gender")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_kws_acc_vs_gender.png", dpi=300)
            plt.close()

    # 8. DRI vs Speech/Non-speech Line Plot
    if not asr_df.empty and "category" in asr_df.columns and "accent" in asr_df.columns:
        noisy = asr_df[asr_df["condition"] == "noisy"]
        comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
        
        dri_df = compute_dri(comp, group_col="accent", category_col="category")
        if not dri_df.empty:
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=dri_df, x="model", y="dri", hue="category", marker="s", linewidth=3, markersize=12)
            plt.title("ASR: Degradation Robustness Index (DRI) by Noise Type", pad=20)
            plt.ylabel("DRI")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "line_dri_speech_vs_noise.png", dpi=300)
            plt.close()
            print("Generated DRI plot.")

    print(f"All requested plots saved to {OUT_DIR.name}/")

if __name__ == '__main__':
    main()
