#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PLOTS_NEW_DIR = ROOT / "results" / "plots_new"
OUT_DIR = ROOT / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['gemma3n_e4b', 'phi4_multimodal', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b']

MODEL_NAMES_MAP = {
    'gemma3n_e4b': 'Gemma 3n E4B',
    'phi4_multimodal': 'Phi-4 Multimodal',
    'qwen25_omni_3b': 'Qwen2.5-Omni 3B',
    'qwen25_omni_7b': 'Qwen2.5-Omni 7B',
    'qwen2_audio_7b': 'Qwen2-Audio 7B'
}

def set_paper_style():
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 15,
        'legend.title_fontsize': 16,
        'figure.titlesize': 22,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'figure.dpi': 300
    })
    sns.set_style("whitegrid")

def generate_snr_plot():
    set_paper_style()
    asr_path = PLOTS_NEW_DIR / "asr_results.csv"
    kws_path = PLOTS_NEW_DIR / "kws_results.csv"
    
    if not asr_path.exists() or not kws_path.exists():
        print("Missing ASR or KWS results CSV files!")
        return

    asr_df = pd.read_csv(asr_path)
    kws_df = pd.read_csv(kws_path)
    
    snr_order = [99, 10, 5, 0]
    snr_labels = ['Clean', '10 dB', '5 dB', '0 dB']
    
    asr_grp = asr_df.groupby(['model', 'snr_db'])['wer'].mean().reset_index()
    kws_grp = kws_df.groupby(['model', 'snr_db'])['correct'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    
    markers = {
        'gemma3n_e4b': 'o',
        'phi4_multimodal': 's',
        'qwen25_omni_3b': '^',
        'qwen25_omni_7b': 'D',
        'qwen2_audio_7b': 'v'
    }
    
    # 1. Left Subplot: ASR (WER vs SNR)
    ax1 = axes[0]
    for model in MODELS:
        m_df = asr_grp[asr_grp['model'] == model].copy()
        if m_df.empty: continue
        m_df['snr_pos'] = m_df['snr_db'].map(lambda x: snr_order.index(x) if x in snr_order else -1)
        m_df = m_df.sort_values('snr_pos')
        
        display_name = MODEL_NAMES_MAP.get(model, model)
        ax1.plot(
            range(len(snr_labels)),
            m_df['wer'],
            marker=markers.get(model, 'o'),
            linewidth=3.0,
            markersize=10,
            label=display_name
        )
        
    ax1.set_xticks(range(len(snr_labels)))
    ax1.set_xticklabels(snr_labels, fontsize=16)
    ax1.set_title("(a) ASR Performance across SNRs (WER ↓)", pad=15, fontsize=18, fontweight='bold')
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=17, labelpad=10)
    ax1.set_ylabel("Word Error Rate (WER)", fontsize=17, labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.legend(title="Model", loc='upper left', fontsize=14, title_fontsize=15, frameon=True, framealpha=0.95)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Right Subplot: KWS (Accuracy vs SNR)
    ax2 = axes[1]
    for model in MODELS:
        if model == 'qwen2_audio_7b':
            continue  # Exclude Qwen2-Audio 7B from KWS plot
        m_df = kws_grp[kws_grp['model'] == model].copy()
        if m_df.empty: continue
        m_df['snr_pos'] = m_df['snr_db'].map(lambda x: snr_order.index(x) if x in snr_order else -1)
        m_df = m_df.sort_values('snr_pos')
        
        display_name = MODEL_NAMES_MAP.get(model, model)
        ax2.plot(
            range(len(snr_labels)),
            m_df['correct'],
            marker=markers.get(model, 'o'),
            linewidth=3.0,
            markersize=10,
            label=display_name
        )
        
    ax2.set_xticks(range(len(snr_labels)))
    ax2.set_xticklabels(snr_labels, fontsize=16)
    ax2.set_title("(b) KWS Performance across SNRs (Accuracy ↑)", pad=15, fontsize=18, fontweight='bold')
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=17, labelpad=10)
    ax2.set_ylabel("Accuracy", fontsize=17, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(title="Model", loc='lower left', fontsize=14, title_fontsize=15, frameon=True, framealpha=0.95)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "fig6_snr_performance_side_by_side.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Successfully generated side-by-side SNR plot: {out.name}")

if __name__ == '__main__':
    generate_snr_plot()
