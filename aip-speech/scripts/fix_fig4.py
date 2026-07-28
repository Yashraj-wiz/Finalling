#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PLOTS_NEW_DIR = ROOT / "results" / "plots_new"
OUT_DIR = ROOT / "results_new"

def set_paper_style():
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.titlesize': 22,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'figure.dpi': 300
    })
    sns.set_style("whitegrid")

def main():
    set_paper_style()
    asr_path = PLOTS_NEW_DIR / "asr_results.csv"
    asr_df = pd.read_csv(asr_path)
    df = asr_df[(asr_df["condition"] == "noisy") & asr_df["accent"].notnull()].copy()
    
    accent_clean_map = {
        'american_english': 'US English (anon)',
        'United States English': 'US English',
        'India and South Asia (India, Pakistan, Sri Lanka)': 'India & S. Asia',
        'England English': 'England English',
        'Canadian English': 'Canadian English',
        'West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)': 'West Indies & Bermuda',
        'Scottish English': 'Scottish English',
        'Irish English': 'Irish English',
        'Australian English': 'Australian English',
        'Southern African (South Africa, Zimbabwe, Namibia)': 'Southern African',
        'Hong Kong English': 'Hong Kong English',
        'New Zealand English': 'New Zealand English'
    }
    df['accent_clean'] = df['accent'].map(accent_clean_map).fillna(df['accent'])
    group_df = df.groupby(['model', 'accent_clean'])['dwer'].mean().reset_index()
    
    plt.figure(figsize=(14, 7.5))
    ax = sns.lineplot(
        data=group_df,
        x='accent_clean',
        y='dwer',
        hue='model',
        style='model',
        markers=True,
        dashes=False,
        linewidth=3.0,
        markersize=11
    )
    plt.title("ASR: Mean Performance Degradation (ΔWER) across Accent Subgroups", pad=35, fontsize=19, fontweight='bold')
    plt.ylabel("Mean ΔWER (Noisy - Clean)", fontsize=17, labelpad=12)
    plt.xlabel("Accent Subgroup", fontsize=17, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=16)
    plt.xticks(rotation=40, ha='right')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=14, frameon=True, framealpha=0.95)
    plt.tight_layout()
    out = OUT_DIR / "fig4_accent_degradation_lines.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Successfully generated updated {out.name}")

if __name__ == '__main__':
    main()
