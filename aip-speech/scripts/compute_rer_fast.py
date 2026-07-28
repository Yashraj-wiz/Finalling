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

res = {}
for m in models:
    res[m] = {}
    # ASR
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
        
        res[m]['ASR'] = dwer_steer.mean() / (dwer_base.mean() + 1e-9)
    except Exception as e:
        res[m]['ASR'] = None

    # KWS
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
        res[m]['KWS'] = rer_kws
    except Exception as e:
        res[m]['KWS'] = None

df_res = pd.DataFrame(res).T
print(df_res)
