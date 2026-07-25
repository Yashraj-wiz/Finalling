import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. Load Background Map
battery = pd.read_parquet(ROOT / "descriptors" / "battery.parquet")
bg_map = dict(zip(battery['bg_id'], battery['category']))

# Prefer result_new_1 if available
RESULT_NEW_1 = ROOT / "result_new_1"
RESULTS_OLD = ROOT / "results"

asr_path = RESULT_NEW_1 / "asr_results.csv" if (RESULT_NEW_1 / "asr_results.csv").exists() else RESULTS_OLD / "e1_asr.csv"
kws_path = RESULT_NEW_1 / "kws_results.csv" if (RESULT_NEW_1 / "kws_results.csv").exists() else RESULTS_OLD / "e1_kws.csv"
sqa_path = RESULTS_OLD / "e1_sqa.csv"

print(f"Reading ASR from: {asr_path}")
print(f"Reading KWS from: {kws_path}")
print(f"Reading SQA from: {sqa_path}")

# 2. Process ASR
df_asr = pd.read_csv(asr_path)
if 'background_id' in df_asr.columns and 'bg_id' not in df_asr.columns:
    df_asr['bg_id'] = df_asr['background_id']
df_asr = df_asr[df_asr['snr_db'] == 0].copy()
df_asr['bg_class'] = df_asr['bg_id']
df_asr['degradation'] = df_asr['dwer'] # Positive = worse
df_asr['task'] = 'ASR'
df_asr['dataset'] = 'ASR_Dataset'

# 3. Process KWS
df_kws = pd.read_csv(kws_path)
if 'background_id' in df_kws.columns and 'bg_id' not in df_kws.columns:
    df_kws['bg_id'] = df_kws['background_id']
df_kws_0db = df_kws[(df_kws['condition'] == 'noisy') & (df_kws['snr_db'] == 0)].copy()

if 'dacc' in df_kws_0db.columns:
    # dacc is (noisy_acc - clean_acc), so degradation = -dacc (Positive = worse performance)
    df_kws_0db['degradation'] = -df_kws_0db['dacc']
else:
    clean_kws = df_kws[df_kws['condition'] == 'clean'].groupby(['model', 'speech_id'])['correct'].mean()
    df_kws_0db['clean_correct'] = df_kws_0db.set_index(['model', 'speech_id']).index.map(clean_kws)
    df_kws_0db['degradation'] = df_kws_0db['clean_correct'] - df_kws_0db['correct']

df_kws_0db['bg_class'] = df_kws_0db['bg_id']
df_kws_0db['task'] = 'KWS'
df_kws_0db['dataset'] = 'KWS_Dataset'

# 4. Process SQA
df_sqa = pd.read_csv(sqa_path)
if 'background_id' in df_sqa.columns and 'bg_id' not in df_sqa.columns:
    df_sqa['bg_id'] = df_sqa['background_id']
df_sqa_0db = df_sqa[(df_sqa['condition'] == 'noisy') & (df_sqa['snr_db'] == 0)].copy()

if 'df1' in df_sqa_0db.columns:
    df_sqa_0db['degradation'] = -df_sqa_0db['df1']
else:
    clean_sqa = df_sqa[df_sqa['condition'] == 'clean'].groupby(['model', 'speech_id'])['f1'].mean()
    df_sqa_0db['clean_f1'] = df_sqa_0db.set_index(['model', 'speech_id']).index.map(clean_sqa)
    df_sqa_0db['degradation'] = df_sqa_0db['clean_f1'] - df_sqa_0db['f1']

df_sqa_0db['bg_class'] = df_sqa_0db['bg_id']
df_sqa_0db['task'] = 'SQA'
df_sqa_0db['dataset'] = 'SQA_Dataset'

cols_to_keep = ['task', 'model', 'dataset', 'gender', 'accent', 'bg_class', 'degradation']

def extract_cols(df):
    for col in ['gender', 'accent']:
        if col not in df.columns:
            df[col] = 'unknown'
        else:
            df[col] = df[col].fillna('unknown')
    return df[cols_to_keep].dropna(subset=['bg_class', 'degradation'])

print("Combining data...")
df_asr_prep = extract_cols(df_asr)
df_kws_prep = extract_cols(df_kws_0db)
df_sqa_prep = extract_cols(df_sqa_0db)

combined = pd.concat([df_asr_prep, df_kws_prep, df_sqa_prep], ignore_index=True)

buckets = ['task', 'model', 'dataset', 'gender', 'accent']

print("Calculating Z-scores...")
combined['z_score'] = combined.groupby(buckets)['degradation'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

def generate_and_save_plot(df_data, title, filename):
    scores = df_data.groupby('bg_class')['z_score'].mean().sort_values(ascending=False).reset_index()
    scores['category'] = scores['bg_class'].map(bg_map)

    plt.figure(figsize=(16, 8))
    palette = {'speech_like': '#d62728', 'non_speech': '#1f77b4'}
    sns.barplot(data=scores, x='bg_class', y='z_score', hue='category', dodge=False, palette=palette)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(title, pad=15)
    plt.ylabel("Average Z-Score (Higher = More Destructive)")
    plt.xlabel("Background Class")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_paths = [
        ROOT / "result_new_1" / filename,
        ROOT / "results_new_1" / filename,
        ROOT / "results_new_1" / "plots" / filename,
        ROOT / "results" / "plots" / filename
    ]

    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot: {out_path}")

    plt.close()

# 1. Combined Holistic Plot
print("Plotting Combined Holistic Score...")
generate_and_save_plot(
    combined, 
    "Holistic Background Degradation Score (Averaged across ASR, KWS, SQA at 0 dB)\nNormalized across Task, Model, Dataset, Gender, Accent",
    "e1_holistic_bg_degradation.png"
)

# 2. ASR Plot
print("Plotting ASR Background Degradation...")
generate_and_save_plot(
    combined[combined['task'] == 'ASR'],
    "ASR Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
    "e1_asr_bg_degradation.png"
)

# 3. KWS Plot
print("Plotting KWS Background Degradation...")
generate_and_save_plot(
    combined[combined['task'] == 'KWS'],
    "KWS Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
    "e1_kws_bg_degradation.png"
)

# 4. SQA Plot
print("Plotting SQA Background Degradation...")
generate_and_save_plot(
    combined[combined['task'] == 'SQA'],
    "SQA Background Degradation Score (at 0 dB SNR)\nNormalized across Model, Gender, Accent",
    "e1_sqa_bg_degradation.png"
)

print("All plots regenerated and saved successfully!")
