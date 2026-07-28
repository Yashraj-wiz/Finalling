#!/usr/bin/env python3
import json
import re
import string
from pathlib import Path
import pandas as pd
import jiwer

ROOT = Path(__file__).resolve().parent.parent
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

res = []
for m in models:
    # ASR RER
    rer_asr = 1.0
    try:
        p_base = ROOT / "results" / m / "asr.jsonl"
        p_steer = ROOT / "results" / m / "asr_steer_p5.jsonl"
        if p_base.exists() and p_steer.exists():
            with open(p_base, encoding='utf-8') as f:
                base_data = [json.loads(line) for idx, line in enumerate(f) if idx < 500]
            with open(p_steer, encoding='utf-8') as f:
                steer_data = [json.loads(line) for idx, line in enumerate(f) if idx < 500]
            
            df_base = pd.DataFrame(base_data)
            df_steer = pd.DataFrame(steer_data)
            
            df_base['wer'] = df_base.apply(lambda r: wer(r['transcript'], r['raw']), axis=1)
            df_steer['wer'] = df_steer.apply(lambda r: wer(r['transcript'], r['raw']), axis=1)
            
            clean_b = df_base[df_base['condition'] == 'clean'].set_index('speech_id')['wer']
            noisy_b = df_base[df_base['condition'] == 'noisy']
            dwer_b = noisy_b['wer'] - noisy_b['speech_id'].map(clean_b)
            
            clean_s = df_steer[df_steer['condition'] == 'clean'].set_index('speech_id')['wer']
            noisy_s = df_steer[df_steer['condition'] == 'noisy']
            dwer_s = noisy_s['wer'] - noisy_s['speech_id'].map(clean_s)
            
            if dwer_b.mean() != 0:
                rer_asr = dwer_s.mean() / dwer_b.mean()
    except Exception as e:
        print(f"ASR error {m}: {e}")
        
    # KWS RER
    rer_kws = 1.0
    try:
        p_base = ROOT / "results" / m / "kws.jsonl"
        p_steer = ROOT / "results" / m / "kws_steer_p5.jsonl"
        if p_base.exists() and p_steer.exists():
            with open(p_base, encoding='utf-8') as f:
                base_data = [json.loads(line) for idx, line in enumerate(f) if idx < 500]
            with open(p_steer, encoding='utf-8') as f:
                steer_data = [json.loads(line) for idx, line in enumerate(f) if idx < 500]
            
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
            
            clean_acc_b = df_base[df_base['condition'] == 'clean']['correct'].mean()
            noisy_acc_b = df_base[df_base['condition'] == 'noisy']['correct'].mean()
            dacc_b = clean_acc_b - noisy_acc_b
            
            clean_acc_s = df_steer[df_steer['condition'] == 'clean']['correct'].mean()
            noisy_acc_s = df_steer[df_steer['condition'] == 'noisy']['correct'].mean()
            dacc_s = clean_acc_s - noisy_acc_s
            
            if abs(dacc_b) > 0.001:
                rer_kws = dacc_s / dacc_b
    except Exception as e:
        print(f"KWS error {m}: {e}")
        
    res.append({'Model': m, 'ASR_RER': round(rer_asr, 3), 'KWS_RER': round(rer_kws, 3)})

df_out = pd.DataFrame(res)
print(df_out.to_string(index=False))
