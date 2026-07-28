#!/usr/bin/env python3
"""
scripts/generate_results_new_2.py

Generates figures and tables for results_new/:
  - Computes a broad pool of acoustic descriptors for every background recording.
  - Computes Spearman correlations with task degradation (averaged over 10, 5, 0 dB SNR).
  - Selects the top 3 descriptors driven by data.
  - Figure 1: 1x3 Grid of Regression lines (ASR, KWS, SQA using RAW degradation values).
  - Figure 2: Spearman correlation heatmap (Descriptors x Tasks).
  - Figure 3: Per-background degradation ranking horizontal bar plot.
  - Saves full descriptor table & full correlation table to CSV for Appendix.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG_DIR = ROOT / "data" / "bg"
PLOTS_NEW_DIR = ROOT / "results" / "plots_new"

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

TASK_COLORS = {
    'ASR': '#4C72B0',  # Blue
    'KWS': '#DD8452',  # Orange
    'SQA': '#55A868'   # Green
}

def set_paper_style():
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 17,
        'figure.titlesize': 22,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'figure.dpi': 300
    })
    sns.set_style("whitegrid")


def extract_acoustic_descriptors():
    print("Extracting acoustic descriptor pool from background audio files...")
    wav_files = sorted(BG_DIR.glob("*.wav"))
    records = []
    
    for f in wav_files:
        y, sr = sf.read(f)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=16000)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=16000)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=16000)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=16000)
        flux = float(np.mean(onset_env))
        
        contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=16000)))
        rms_var = float(np.var(librosa.feature.rms(y=y)[0]))
        
        records.append({
            'bg_id': f.stem,
            'Zero Crossing Rate (ZCR)': round(zcr, 4),
            'Spectral Centroid (Hz)': round(centroid, 2),
            'Spectral Rolloff (Hz)': round(rolloff, 2),
            'Spectral Bandwidth (Hz)': round(bandwidth, 2),
            'Spectral Flux': round(flux, 4),
            'Spectral Flatness': round(flatness, 4),
            'Spectral Contrast (dB)': round(contrast, 4),
            'RMS Energy Variance': round(rms_var, 6)
        })
        
    df_desc = pd.DataFrame(records).set_index('bg_id')
    
    # Load Silero VAD voiced fraction
    vad_csv = ROOT / "descriptors" / "vad_bg_labels.csv"
    if vad_csv.exists():
        vad_df = pd.read_csv(vad_csv).set_index('bg_id')
        df_desc['Silero VAD Speech Fraction'] = vad_df['silero_speech_frac']
        
    return df_desc


def load_task_degradations():
    print("Loading evaluation metrics averaged over 10, 5, 0 dB SNR...")
    asr = pd.read_csv(PLOTS_NEW_DIR / "asr_results.csv")
    kws = pd.read_csv(PLOTS_NEW_DIR / "kws_results.csv")
    sqa = pd.read_csv(PLOTS_NEW_DIR / "sqa_results.csv")

    # Filter noisy conditions across 10, 5, 0 dB
    asr_noisy = asr[asr['condition'] == 'noisy']
    kws_noisy = kws[kws['condition'] == 'noisy']
    sqa_noisy = sqa[sqa['condition'] == 'noisy']

    # Mean raw degradation per background:
    # ASR: dwer = noisy - clean (higher = worse)
    asr_deg = asr_noisy.groupby('bg_id')['dwer'].mean()

    # KWS: dacc = noisy - clean -> degradation = -dacc = clean - noisy (higher = worse)
    kws_deg = -kws_noisy.groupby('bg_id')['dacc'].mean()

    # SQA: df1 = noisy - clean -> degradation = -df1 = clean - noisy (higher = worse)
    sqa_deg = -sqa_noisy.groupby('bg_id')['df1'].mean()

    deg_df = pd.DataFrame({'ASR': asr_deg, 'KWS': kws_deg, 'SQA': sqa_deg})
    return deg_df


def compute_correlations(desc_df, deg_df):
    merged = desc_df.join(deg_df).dropna()
    corrs = []
    for col in desc_df.columns:
        r_asr, p_asr = stats.spearmanr(merged[col], merged['ASR'])
        r_kws, p_kws = stats.spearmanr(merged[col], merged['KWS'])
        r_sqa, p_sqa = stats.spearmanr(merged[col], merged['SQA'])
        mean_r = np.mean([r_asr, r_kws, r_sqa])
        mean_abs_r = np.mean([abs(r_asr), abs(r_kws), abs(r_sqa)])
        corrs.append({
            'Descriptor': col,
            'ASR (r_s)': round(r_asr, 3), 'ASR (p)': round(p_asr, 3),
            'KWS (r_s)': round(r_kws, 3), 'KWS (p)': round(p_kws, 3),
            'SQA (r_s)': round(r_sqa, 3), 'SQA (p)': round(p_sqa, 3),
            'Mean r_s': round(mean_r, 3),
            'Mean |r_s|': round(mean_abs_r, 3)
        })
    corr_df = pd.DataFrame(corrs).sort_values('Mean |r_s|', ascending=False)
    return merged, corr_df


def generate_figure1_scatter(merged, top_descriptors):
    print("Generating Figure 1: 1x3 Scatter + Regression plots for top 3 descriptors...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    for idx, desc in enumerate(top_descriptors):
        ax = axes[idx]
        for task in ['ASR', 'KWS']:
            sns.regplot(
                data=merged,
                x=desc,
                y=task,
                label=task,
                color=TASK_COLORS[task],
                ax=ax,
                scatter_kws={'s': 65, 'alpha': 0.85},
                line_kws={'linewidth': 3.0}
            )
        ax.set_title(f"({chr(97+idx)}) {desc} vs Degradation", pad=15, fontsize=18, fontweight='bold')
        ax.set_xlabel(desc, fontsize=17, labelpad=10)
        ax.set_ylabel("Raw Degradation Value", fontsize=17, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(title="Task", loc='upper left', fontsize=15, title_fontsize=16, frameon=True, framealpha=0.95)
        
    plt.tight_layout()
    out = OUT_DIR / "fig1_descriptor_regressions.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure2_heatmap(corr_df):
    print("Generating Figure 2: Spearman correlation heatmap across descriptors and tasks...")
    heatmap_df = corr_df.set_index('Descriptor')[['ASR (r_s)', 'KWS (r_s)']].copy()
    heatmap_df.columns = ['ASR', 'KWS']
    
    plt.figure(figsize=(10, 8.5))
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 15},
        cmap="vlag",
        center=0,
        cbar_kws={'label': 'Spearman Correlation'},
        linewidths=1.2,
        linecolor='white',
        square=True
    )
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Spearman Correlation', fontsize=17, labelpad=12)

    # Bold the best (highest absolute correlation) descriptor in each column
    max_row_per_col = heatmap_df.abs().idxmax()
    for r_idx, row_name in enumerate(heatmap_df.index):
        for c_idx, col_name in enumerate(heatmap_df.columns):
            if row_name == max_row_per_col[col_name]:
                text_obj = ax.texts[r_idx * len(heatmap_df.columns) + c_idx]
                text_obj.set_weight('bold')
                text_obj.set_fontsize(17)
                
    plt.title("Spearman Correlation between Acoustic Descriptors\nand Task Degradations", pad=20, fontsize=19, fontweight='bold')
    plt.ylabel("Acoustic Descriptor", labelpad=12, fontsize=17, fontweight='bold')
    plt.xlabel("Evaluation Task", labelpad=12, fontsize=17, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16, labelrotation=0)
    ax.tick_params(axis='y', labelsize=15, labelrotation=0)
    plt.tight_layout()
    out = OUT_DIR / "fig2_spearman_correlation_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure2_heatmap_horizontal(corr_df):
    print("Generating Figure 2 Horizontal: Spearman correlation heatmap...")
    heatmap_df = corr_df.set_index('Descriptor')[['ASR (r_s)', 'KWS (r_s)']].copy()
    heatmap_df.columns = ['ASR', 'KWS']
    heatmap_df = heatmap_df.T
    
    plt.figure(figsize=(15, 5))
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 15},
        cmap="vlag",
        center=0,
        cbar_kws={'label': 'Spearman Correlation'},
        linewidths=1.2,
        linecolor='white',
        square=True
    )
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Spearman Correlation', fontsize=17, labelpad=12)

    max_col_per_row = heatmap_df.abs().idxmax(axis=1)
    for r_idx, row_name in enumerate(heatmap_df.index):
        for c_idx, col_name in enumerate(heatmap_df.columns):
            if col_name == max_col_per_row[row_name]:
                text_obj = ax.texts[r_idx * len(heatmap_df.columns) + c_idx]
                text_obj.set_weight('bold')
                text_obj.set_fontsize(17)
                
    plt.title("Spearman Correlation between Acoustic Descriptors and Task Degradations", pad=20, fontsize=19, fontweight='bold')
    plt.ylabel("Evaluation Task", labelpad=12, fontsize=17, fontweight='bold')
    plt.xlabel("Acoustic Descriptor", labelpad=12, fontsize=17, fontweight='bold')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=17, labelrotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out = OUT_DIR / "fig2_spearman_correlation_heatmap_horizontal.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure3_ranking(deg_df):
    print("Generating Figure 3: Per-background mean degradation ranking horizontal bar plot...")
    deg = deg_df[['ASR', 'KWS']].copy()
    deg['Mean_Degradation'] = deg.mean(axis=1)
    sorted_deg = deg.sort_values('Mean_Degradation', ascending=True).reset_index()
    
    plt.figure(figsize=(11, 9))
    ax = sns.barplot(
        data=sorted_deg,
        y='bg_id',
        x='Mean_Degradation',
        color='#4C72B0'
    )
    plt.title("Per-Background Mean Degradation Ranking (Averaged across ASR and KWS)", pad=18, fontsize=19, fontweight='bold')
    plt.xlabel("Mean Raw Degradation (Higher = More Harmful)", fontsize=17, labelpad=12)
    plt.ylabel("Background Sound Class", fontsize=17, labelpad=12)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=16)
    plt.tight_layout()
    out = OUT_DIR / "fig3_per_background_degradation_ranking.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_accent_degradation_plot():
    print("Generating Figure 4: Demographic robustness across accents line plot...")
    asr_path = PLOTS_NEW_DIR / "asr_results.csv"
    if not asr_path.exists():
        print(f"ASR results not found at {asr_path}, skipping Fig 4.")
        return
        
    asr_df = pd.read_csv(asr_path)
    df = asr_df[(asr_df["condition"] == "noisy") & asr_df["accent"].notnull()].copy()
    if df.empty:
        print("No noisy accent data found in ASR results, skipping Fig 4.")
        return
    
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
    
    fig, ax = plt.subplots(figsize=(14, 7.5))
    
    sns.lineplot(
        data=group_df,
        x='accent_clean',
        y='dwer',
        hue='model',
        style='model',
        markers=True,
        dashes=False,
        linewidth=3.0,
        markersize=11,
        ax=ax
    )
    
    fig.suptitle("ASR: Mean Performance Degradation (ΔWER) across Accent Subgroups", fontsize=19, fontweight='bold', y=0.97)
    plt.ylabel("Mean ΔWER (Noisy - Clean)", fontsize=17, labelpad=12)
    plt.xlabel("Accent Subgroup", fontsize=17, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=16)
    plt.xticks(rotation=40, ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=14, frameon=True, framealpha=0.95)
    plt.subplots_adjust(top=0.82, bottom=0.22, left=0.08, right=0.96)
    out = OUT_DIR / "fig4_accent_degradation_lines.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_prompt_rer_plot():
    print("Generating Figure 5: Prompt steering Residual Effect Ratio (RER) bar plot...")
    import jiwer
    import string
    import json
    import re
    
    models = ['gemma3n_e4b', 'phi4_multimodal', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b']
    
    def clean_txt(s):
        s = str(s).lower()
        s = re.sub(rf'[{re.escape(string.punctuation)}]', ' ', s)
        return ' '.join(s.split())

    def wer(r, h):
        try:
            return jiwer.wer(clean_txt(r), clean_txt(h))
        except:
            return 1.0

    KWS_TARGETS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    
    records = []
    
    for m in models:
        # 1. ASR RER
        try:
            with open(ROOT / "results" / m / "asr.jsonl", encoding='utf-8') as f:
                base_data = [json.loads(line) for line in f]
            with open(ROOT / "results" / m / "asr_steer_p5.jsonl", encoding='utf-8') as f:
                steer_data = [json.loads(line) for line in f]
            
            df_base = pd.DataFrame(base_data)
            df_steer = pd.DataFrame(steer_data)
            
            df_base['wer'] = df_base.apply(lambda r: wer(r['transcript'], r['raw']), axis=1)
            df_steer['wer'] = df_steer.apply(lambda r: wer(r['transcript'], r['raw']), axis=1)
            
            clean_base = df_base[df_base['condition'] == 'clean'].set_index('speech_id')['wer']
            noisy_base = df_base[df_base['condition'] == 'noisy']
            dwer_base = noisy_base['wer'] - noisy_base['speech_id'].map(clean_base)
            
            clean_steer = df_steer[df_steer['condition'] == 'clean'].set_index('speech_id')['wer']
            noisy_steer = df_steer[df_steer['condition'] == 'noisy']
            dwer_steer = noisy_steer['wer'] - noisy_steer['speech_id'].map(clean_steer)
            
            rer_asr = dwer_steer.mean() / (dwer_base.mean() + 1e-9)
            records.append({'Model': m, 'Task': 'ASR', 'RER': rer_asr})
        except Exception as e:
            print(f"Error computing ASR RER for {m}: {e}")
            
        # 2. KWS RER
        try:
            with open(ROOT / "results" / m / "kws.jsonl", encoding='utf-8') as f:
                base_data = [json.loads(line) for line in f]
            with open(ROOT / "results" / m / "kws_steer_p5.jsonl", encoding='utf-8') as f:
                steer_data = [json.loads(line) for line in f]
                
            df_base = pd.DataFrame(base_data)
            df_steer = pd.DataFrame(steer_data)
            
            def is_correct(row):
                ref_kw = str(row['keyword']).lower().strip()
                hyp_raw = str(row['raw']).lower().strip()
                is_target = bool(row['is_target'])
                hyp_kw = hyp_raw.split()[0] if hyp_raw else 'silence'
                hyp_kw = hyp_kw.strip(string.punctuation)
                if is_target:
                    return int(hyp_kw == ref_kw)
                else:
                    return int(hyp_kw not in KWS_TARGETS)
                    
            df_base['correct'] = df_base.apply(is_correct, axis=1)
            df_steer['correct'] = df_steer.apply(is_correct, axis=1)
            
            clean_acc_base = df_base[df_base['condition'] == 'clean']['correct'].mean()
            noisy_acc_base = df_base[df_base['condition'] == 'noisy']['correct'].mean()
            dacc_base = clean_acc_base - noisy_acc_base
            
            clean_acc_steer = df_steer[df_steer['condition'] == 'clean']['correct'].mean()
            noisy_acc_steer = df_steer[df_steer['condition'] == 'noisy']['correct'].mean()
            dacc_steer = clean_acc_steer - noisy_acc_steer
            
            rer_kws = dacc_steer / (dacc_base + 1e-9)
            if abs(dacc_base) < 0.01 and abs(dacc_steer) < 0.01:
                rer_kws = 1.0
            records.append({'Model': m, 'Task': 'KWS', 'RER': rer_kws})
        except Exception as e:
            print(f"Error computing KWS RER for {m}: {e}")
            
    df_rer = pd.DataFrame(records)
    if df_rer.empty:
        return
        
    plt.figure(figsize=(11, 6.5))
    ax = sns.barplot(
        data=df_rer,
        x='Model',
        y='RER',
        hue='Task',
        palette=TASK_COLORS
    )
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.8, label='Baseline (RER = 1.0)')
    plt.title("Prompt Effectiveness: Residual Effect Ratio (RER) by Model", pad=18, fontsize=19, fontweight='bold')
    plt.ylabel("Residual Effect Ratio (RER)", fontsize=17, labelpad=12)
    plt.xlabel("Model", fontsize=17, labelpad=12)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.legend(loc='upper right', fontsize=15, title_fontsize=16, frameon=True, framealpha=0.95)
    
    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=14, fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / "fig5_prompt_rer.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


MODELS = ['gemma3n_e4b', 'phi4_multimodal', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b']

MODEL_NAMES_MAP = {
    'gemma3n_e4b': 'Gemma 3n E4B',
    'phi4_multimodal': 'Phi-4 Multimodal',
    'qwen25_omni_3b': 'Qwen2.5-Omni 3B',
    'qwen25_omni_7b': 'Qwen2.5-Omni 7B',
    'qwen2_audio_7b': 'Qwen2-Audio 7B'
}


def generate_snr_performance_side_by_side_plot():
    print("Generating Figure: SNR Effect on Performance for ASR and KWS (Side-by-Side)...")
    asr_path = PLOTS_NEW_DIR / "asr_results.csv"
    kws_path = PLOTS_NEW_DIR / "kws_results.csv"
    
    if not asr_path.exists() or not kws_path.exists():
        print("ASR or KWS results file missing for SNR side-by-side plot.")
        return

    asr_df = pd.read_csv(asr_path)
    kws_df = pd.read_csv(kws_path)
    
    snr_order = [99, 10, 5, 0]
    snr_labels = ['Clean', '10 dB', '5 dB', '0 dB']
    
    asr_grp = asr_df.groupby(['model', 'snr_db'])['wer'].mean().reset_index()
    
    if 'correct' in kws_df.columns:
        kws_grp = kws_df.groupby(['model', 'snr_db'])['correct'].mean().reset_index()
    else:
        kws_grp = pd.DataFrame()

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
    if not kws_grp.empty:
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
    print(f"Saved {out.name}")


def main():
    set_paper_style()
    
    # 1. Extract acoustic descriptor pool
    desc_df = extract_acoustic_descriptors()
    desc_csv = OUT_DIR / "appendix_acoustic_descriptors_table.csv"
    desc_df.to_csv(desc_csv)
    print(f"Saved complete descriptor table: {desc_csv}")
    
    # 2. Load task degradations
    deg_df = load_task_degradations()
    
    # 3. Compute correlations
    merged, corr_df = compute_correlations(desc_df, deg_df)
    corr_csv = OUT_DIR / "appendix_correlation_matrix_table.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved complete correlation table: {corr_csv}")
    
    # Select top 3 descriptors by mean |r_s|
    top_3 = corr_df.head(3)['Descriptor'].tolist()
    print(f"\nTop 3 data-driven descriptors selected: {top_3}")
    
    # 4. Generate Figures
    generate_figure1_scatter(merged, top_3)
    generate_figure2_heatmap(corr_df)
    generate_figure2_heatmap_horizontal(corr_df)
    generate_figure3_ranking(deg_df)
    generate_accent_degradation_plot()
    generate_prompt_rer_plot()
    generate_snr_performance_side_by_side_plot()
    
    print(f"\nAll figures and tables successfully generated in: {OUT_DIR}")

if __name__ == '__main__':
    main()
