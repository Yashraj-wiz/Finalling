import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
OUT_DIR = ROOT / "result_new_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATTERY = ROOT / "descriptors" / "battery.parquet"
SQA_CSV = ROOT / "results" / "e1_sqa.csv"

def main():
    set_paper_style()
    if not SQA_CSV.exists():
        print("SQA CSV not found.")
        return
        
    df = pd.read_csv(SQA_CSV)
    try:
        bat = pd.read_parquet(BATTERY)
        bat = bat.rename(columns={"bg_id": "bg_id"}) if "bg_id" in bat.columns else bat.rename(columns={"id": "bg_id"})
        df = df.merge(bat, on="bg_id", how="left")
    except:
        pass

    # Merge clean baseline F1 to calculate Delta F1 = F1_noisy - F1_clean
    clean = df[df["condition"] == "clean"][["model", "speech_id", "f1"]].rename(columns={"f1": "f1_clean"})
    df = df.merge(clean, on=["model", "speech_id"], how="left")
    df["df1"] = df["f1"] - df["f1_clean"]

    # 1. Bar SQA Clean vs Noisy F1
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x="model", y="f1", hue="condition", errorbar=None)
    plt.title("SQA: Mean F1 Score (Clean vs Noisy Baseline)", pad=20)
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_sqa_f1_clean_vs_noisy.png", dpi=300)
    plt.close()

    # 2. Line SQA F1 vs SNR
    if "snr_db" in df.columns:
        noisy = df[df["snr_db"].notnull() & (df["condition"] == "noisy")]
        if not noisy.empty:
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=noisy, x="snr_db", y="f1", hue="model", marker="o", linewidth=2.5, markersize=10)
            plt.title("SQA: F1 Score across Signal-to-Noise Ratios (SNR)", pad=20)
            plt.ylabel("F1 Score")
            plt.xlabel("SNR (dB)")
            plt.gca().invert_xaxis()
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "line_sqa_f1_vs_snr.png", dpi=300)
            plt.close()

    # 3. Bar SQA ΔF1 by Noise Category (Speech-like vs Non-speech)
    if "category" in df.columns:
        noisy = df[df["condition"] == "noisy"]
        comp = noisy[noisy["category"].isin(["speech_like", "non_speech"])]
        if not comp.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=comp, x="model", y="df1", hue="category", errorbar=None)
            plt.title("SQA: Mean ΔF1 (F1 Drop relative to Clean) by Noise Type", pad=20)
            plt.ylabel("ΔF1 (F1 Score Drop)")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Noise Category", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bar_sqa_f1_speech_vs_noise.png", dpi=300)
            plt.close()

    print("SQA plots generated successfully in result_new_1/")

if __name__ == '__main__':
    main()
