#!/usr/bin/env python3
"""
06_score_metrics.py — Stage 6: compute all metrics from inference JSONL outputs.

Experiments scored:
  E1  WER/CER + sub/del/ins + ΔWER (ASR);  Accuracy/FAR/Miss (KWS)
  E2  ΔWER on speech-like noise;   TIR on injection probes (KWS)
  E3  Per-subgroup ΔWER, Robustness Gap, DRI;  SAA controlled cut
  E4  Residual-Effect Ratio (RER), compliance rate

Outputs → results/*.csv  +  results/summary_table.csv

Usage:
  python scripts/06_score_metrics.py --smoke-test   # quick validation
  python scripts/06_score_metrics.py                # full scoring
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, get_logger, jsonl_read

log = get_logger("06_score_metrics")

RESULTS   = ROOT / "results"
INFER     = ROOT / "inference_1"
BATTERY_F = ROOT / "descriptors" / "battery.parquet"
SMOKE_N   = 10

KWS_TARGETS = ["yes", "no", "up", "down", "left", "right",
               "on", "off", "stop", "go"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _all_models() -> list[str]:
    if not INFER.exists():
        return []
    return sorted(d.name for d in INFER.iterdir() if d.is_dir())


import re

def _clean_hyp(hyp: str) -> str:
    hyp = str(hyp).strip()
    if hyp == "__ERROR__":
        return hyp
    # Strip common Qwen2-Audio preambles
    hyp = re.sub(r"^(The transcription of the speech is|The speech transcribed from the audio is|The keyword is|The word is)[^:]*:\s*['\"]?", "", hyp, flags=re.IGNORECASE)
    # Also remove trailing quotes if we removed a leading quote
    if hyp.endswith("'") or hyp.endswith('"'):
        hyp = hyp[:-1]
    return hyp.strip()

def _load_inference(model_id: str, task: str) -> pd.DataFrame:
    path = INFER / model_id / f"{task}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = jsonl_read(path)
    df = pd.DataFrame(rows)
    if "raw" in df.columns:
        df["raw"] = df["raw"].apply(_clean_hyp)
    return df


def _load_battery() -> pd.DataFrame:
    if BATTERY_F.exists():
        df = pd.read_parquet(BATTERY_F)
        return df
    return pd.DataFrame()


def _normalize_asr_text(s: str) -> str:
    import re, string
    s = str(s).lower()
    # Replace punctuation with spaces
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    # Normalize whitespaces
    s = " ".join(s.split())
    return s


def _wer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        ref_norm = _normalize_asr_text(ref)
        hyp_norm = _normalize_asr_text(hyp)
        return float(jiwer.wer(ref_norm, hyp_norm))
    except Exception:
        return float("nan")


def _cer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        ref_norm = _normalize_asr_text(ref)
        hyp_norm = _normalize_asr_text(hyp)
        return float(jiwer.cer(ref_norm, hyp_norm))
    except Exception:
        return float("nan")


def _error_counts(ref: str, hyp: str) -> dict:
    """Return substitution, deletion, insertion counts."""
    import jiwer
    try:
        ref_norm = _normalize_asr_text(ref)
        hyp_norm = _normalize_asr_text(hyp)
        out = jiwer.process_words(ref_norm, hyp_norm)
        return {"substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions)}
    except Exception:
        return {"substitutions": 0, "deletions": 0, "insertions": 0}


def _save(df: pd.DataFrame, name: str) -> Path:
    out = RESULTS / name
    if df.empty:
        if out.exists():
            try:
                out.unlink()
            except Exception:
                pass
        return out
    df.to_csv(out, index=False)
    log.info(f"  → {out}  ({len(df)} rows)")
    return out


# ── E1-ASR: WER, CER, ΔWER ───────────────────────────────────────────────────
def score_e1_asr(smoke: bool) -> pd.DataFrame:
    log.info("[E1-ASR] Scoring WER / CER / ΔWER ...")
    records = []
    for mid in _all_models():
        df = _load_inference(mid, "asr")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N)
        for _, row in df.iterrows():
            ref = str(row.get("transcript", ""))
            hyp = str(row.get("raw", ""))
            if hyp == "__ERROR__":
                continue
            wer = _wer(ref, hyp)
            cer = _cer(ref, hyp)
            ec  = _error_counts(ref, hyp)
            records.append({
                "model": mid,
                "speech_id":     row.get("speech_id", ""),
                "background_id": row.get("background_id", ""),
                "snr_db":        row.get("snr_db", ""),
                "condition":     row.get("condition", ""),
                "accent":        row.get("accent", ""),
                "gender":        row.get("gender", ""),
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                **ec,
            })

    out = pd.DataFrame(records)
    if not out.empty:
        # Paired ΔWER = wer(noisy) − wer(clean) per (model, speech_id)
        clean = (out[out["condition"] == "clean"]
                 [["model", "speech_id", "wer"]]
                 .rename(columns={"wer": "wer_clean"}))
        out = out.merge(clean, on=["model", "speech_id"], how="left")
        out["dwer"] = out["wer"] - out["wer_clean"]

    _save(out, "e1_asr.csv")
    return out


# ── E1-KWS: Accuracy, FAR, Miss ──────────────────────────────────────────────
def score_e1_kws(smoke: bool) -> pd.DataFrame:
    log.info("[E1-KWS] Scoring Accuracy / FAR / Miss ...")
    records = []
    for mid in _all_models():
        df = _load_inference(mid, "kws")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N)
        for _, row in df.iterrows():
            ref_kw    = str(row.get("keyword", "")).lower().strip()
            hyp_raw   = str(row.get("raw", "")).lower().strip()
            is_target = bool(row.get("is_target", False))
            if hyp_raw == "__error__":
                continue
            hyp_kw   = hyp_raw.split()[0] if hyp_raw else "silence"
            import string
            hyp_kw   = hyp_kw.strip(string.punctuation)
            correct  = (hyp_kw == ref_kw) if is_target else (hyp_kw not in KWS_TARGETS)
            false_alarm = (not is_target) and (hyp_kw in KWS_TARGETS)
            records.append({
                "model": mid,
                "speech_id":     row.get("speech_id", ""),
                "background_id": row.get("background_id", ""),
                "snr_db":        row.get("snr_db", ""),
                "condition":     row.get("condition", ""),
                "keyword":       ref_kw,
                "is_target":     is_target,
                "predicted":     hyp_kw,
                "correct":       int(correct),
                "false_alarm":   int(false_alarm),
                "miss":          int(is_target and not correct),
            })

    out = pd.DataFrame(records)
    _save(out, "e1_kws.csv")
    return out


def _normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re, string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))

def _f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if not common:
        return 0.0
    # count how many times common tokens appear in prediction (to handle duplicates if needed)
    num_same = sum(1 for token in prediction_tokens if token in common)
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)

def _hallucination_rate(prediction: str, passage: str) -> float:
    """Percentage of predicted tokens that do not appear in the source passage."""
    pred_tokens = set(_normalize_answer(prediction).split())
    passage_tokens = set(_normalize_answer(passage).split())
    if not pred_tokens:
        return 0.0
    hallucinated = pred_tokens - passage_tokens
    return len(hallucinated) / len(pred_tokens)

# ── E1-SQA: EM, F1, Hallucination ───────────────────────────────────────────
def score_e1_sqa(smoke: bool) -> pd.DataFrame:
    log.info("[E1-SQA] Scoring EM / F1 / Hallucination ...")
    records = []
    for mid in _all_models():
        df = _load_inference(mid, "sqa")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N)
        for _, row in df.iterrows():
            ref_ans   = str(row.get("answer", ""))
            passage   = str(row.get("passage_text", ""))
            hyp_raw   = str(row.get("raw", ""))
            if hyp_raw == "__ERROR__":
                continue
                
            em = _exact_match_score(hyp_raw, ref_ans)
            f1 = _f1_score(hyp_raw, ref_ans)
            hal = _hallucination_rate(hyp_raw, passage)
            
            records.append({
                "model": mid,
                "speech_id":     row.get("speech_id", ""),
                "background_id": row.get("background_id", ""),
                "snr_db":        row.get("snr_db", ""),
                "condition":     row.get("condition", ""),
                "em": round(em, 4),
                "f1": round(f1, 4),
                "hallucination": round(hal, 4),
            })
            
    out = pd.DataFrame(records)
    if not out.empty:
        _save(out, "e1_sqa.csv")
    return out


def _score_e1_profile(smoke: bool) -> None:
    log.info("[E1-Profile] Scoring descriptor regressions ...")
    asr_f = RESULTS / "e1_asr.csv"
    if not asr_f.exists():
        return
    try:
        df_asr = pd.read_csv(asr_f)
    except Exception as e:
        log.warning(f"[E1-Profile] Could not read {asr_f}: {e}. Skipping profiling.")
        return
    bat = _load_battery()
    if bat.empty:
        return
    merged = df_asr.merge(bat, left_on="background_id", right_on="bg_id", how="inner")
    if not merged.empty:
        _save(merged, "e1_profile.csv")
        
        # Calculate speech_like vs non-speech dwer comparison
        if "category" in merged.columns:
            comp = merged[merged["category"].isin(["speech_like", "non_speech"])]
            if not comp.empty:
                summary = comp.groupby(["model", "category"])["dwer"].mean().reset_index()
                _save(summary, "e1_speech_vs_non_speech.csv")
                log.info("[E1-Profile] Speech-like vs Non-speech ΔWER comparison:")
                for _, row in summary.iterrows():
                    log.info(f"  {row['model']} / {row['category']}: Mean ΔWER = {row['dwer']:.4f}")


# ── E2-ASR ───────────────────────────────────────────────────────────────────

def score_e2(smoke: bool) -> None:
    log.info("[E2] Scoring real + TIR ...")
    bat = _load_battery()
    speech_like = (bat[bat.get("category", pd.Series()) == "speech_like"]["bg_id"].tolist()
                   if "category" in bat.columns else [])

    records = []
    for mid in _all_models():
        df = _load_inference(mid, "asr")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N * 4)

        clean = df[df["condition"] == "clean"]

        for bg_id in speech_like:
            real = df[(df["background_id"] == bg_id) & (df["snr_db"] == 0)]

            for _, rr in real.iterrows():
                ref = str(rr.get("transcript", ""))
                hyp_r = str(rr.get("raw", ""))
                if hyp_r == "__ERROR__":
                    continue
                cm = clean[clean["speech_id"] == rr["speech_id"]]
                hyp_c = str(cm.iloc[0]["raw"]) if len(cm) > 0 else ""

                wer_r = _wer(ref, hyp_r)
                wer_c = _wer(ref, hyp_c) if hyp_c else float("nan")
                dwer_r = wer_r - wer_c

                records.append({
                    "model": mid, "speech_id": rr["speech_id"],
                    "background_id": bg_id,
                    "dwer_real": round(dwer_r, 4),
                })

    _save(pd.DataFrame(records), "e2_semantic.csv")

    # TIR
    _score_tir(smoke)


def _score_tir(smoke: bool) -> None:
    """TIR = FAR on injection probes vs generic noise."""
    pf = ROOT / "prereg" / "injection_probe_ids.json"
    if not pf.exists():
        log.warning("[TIR] No injection probe IDs. Skipping.")
        return
    probe_ids = set(json.loads(pf.read_text()))

    records = []
    for mid in _all_models():
        df = _load_inference(mid, "kws")
        if df.empty or "speech_id" not in df.columns:
            continue
        probes = df[df["speech_id"].isin(probe_ids)]
        if smoke:
            probes = probes.head(SMOKE_N)
        for _, row in probes.iterrows():
            raw = str(row.get("raw", "")).lower().strip()
            predicted = raw.split()[0] if raw else ""
            import string
            predicted = predicted.strip(string.punctuation)
            records.append({
                "model": mid,
                "probe_id":      row.get("speech_id", ""),
                "background_id": row.get("background_id", ""),
                "condition":     row.get("condition", ""),
                "predicted":     predicted,
                "tir":           int(predicted in KWS_TARGETS),
            })

    _save(pd.DataFrame(records), "e2_tir.csv")


# ── E3: Disparate Robustness ─────────────────────────────────────────────────
def score_e3(smoke: bool) -> None:
    log.info("[E3] Disparate-robustness scoring ...")
    asr_path = RESULTS / "e1_asr.csv"
    if not asr_path.exists():
        log.warning("[E3] Run E1-ASR first.")
        return
    asr = pd.read_csv(asr_path)

    # Bring in source from itembank
    asr_items = {r["id"]: r.get("source", "") for r in jsonl_read(ROOT / "itembanks" / "asr.jsonl")}
    asr["source"] = asr["speech_id"].map(asr_items)
    
    # We no longer filter only for Common Voice, so LibriSpeech is included in gender/accent analysis!
    # asr = asr[asr["source"] == "common_voice_17"]

    bat = _load_battery()
    sl_ids = (bat[bat["category"] == "speech_like"]["bg_id"].tolist()
              if "category" in bat.columns else [])
    ns_ids = (bat[bat["category"] == "non_speech"]["bg_id"].tolist()
              if "category" in bat.columns else [])

    records = []
    for sg in ["accent", "gender"]:
        if sg not in asr.columns:
            continue
        for model_id in asr["model"].unique():
            mdf = asr[asr["model"] == model_id]
            for gval, gdf in mdf.groupby(sg):
                noisy = gdf[gdf["condition"] == "noisy"]
                mean_dw = float(noisy["dwer"].mean()) if "dwer" in noisy.columns else float("nan")
                dw_sl = float(noisy[noisy["background_id"].isin(sl_ids)]["dwer"].mean()) \
                    if sl_ids and "dwer" in noisy.columns else float("nan")
                dw_ns = float(noisy[noisy["background_id"].isin(ns_ids)]["dwer"].mean()) \
                    if ns_ids and "dwer" in noisy.columns else float("nan")
                records.append({
                    "model": model_id,
                    "subgroup_type":  sg,
                    "subgroup_value": gval,
                    "mean_dwer":         round(mean_dw, 4),
                    "dwer_speech_like":  round(dw_sl, 4),
                    "dwer_non_speech":   round(dw_ns, 4),
                    "n_items":           len(noisy),
                })

    df_e3 = pd.DataFrame(records)
    # Robustness Gap + DRI per (model, subgroup_type)
    if not df_e3.empty:
        gaps = []
        for (mid, sg), sub in df_e3.groupby(["model", "subgroup_type"]):
            vals = sub["mean_dwer"].dropna()
            if len(vals) >= 2:
                gap = float(vals.max() - vals.min())
                dri = gap / (float(vals.mean()) + 1e-9)
                gaps.append({"model": mid, "subgroup_type": sg,
                             "robustness_gap": round(gap, 4),
                             "dri": round(dri, 4)})
                log.info(f"  {mid} / {sg}: Gap={gap:.4f}, DRI={dri:.4f}")
        if gaps:
            _save(pd.DataFrame(gaps), "e3_gaps.csv")

    _save(df_e3, "e3_fairness.csv")

    # SAA controlled cut
    _score_e3_saa(smoke)


def _score_e3_saa(smoke: bool) -> None:
    """Content-matched per-accent ΔWER on Speech Accent Archive."""
    saa_items = {r["id"]: r for r in jsonl_read(ROOT / "itembanks" / "saa.jsonl")}
    if not saa_items:
        log.info("[E3-SAA] No SAA bank. Skipping.")
        return

    records = []
    for mid in _all_models():
        df = _load_inference(mid, "saa")
        if df.empty or "speech_id" not in df.columns:
            continue
        sdf = df[df["speech_id"].isin(saa_items.keys())]
        if smoke:
            sdf = sdf.head(SMOKE_N)
        for _, row in sdf.iterrows():
            sp = saa_items.get(row["speech_id"], {})
            ref = sp.get("transcript", "")
            hyp = str(row.get("raw", ""))
            if hyp == "__ERROR__":
                continue
            records.append({
                "model": mid,
                "speech_id":     row["speech_id"],
                "accent":        sp.get("accent", ""),
                "gender":        sp.get("gender", ""),
                "background_id": row.get("background_id", ""),
                "snr_db":        row.get("snr_db", ""),
                "condition":     row.get("condition", ""),
                "wer":           round(_wer(ref, hyp), 4),
            })

    df_saa = pd.DataFrame(records)
    if not df_saa.empty:
        # Calculate paired clean WER and ΔWER (dwer)
        clean = (df_saa[df_saa["condition"] == "clean"]
                 [["model", "speech_id", "wer"]]
                 .rename(columns={"wer": "wer_clean"}))
        df_saa = df_saa.merge(clean, on=["model", "speech_id"], how="left")
        df_saa["dwer"] = df_saa["wer"] - df_saa["wer_clean"]
        _save(df_saa, "e3_saa.csv")

        # Save accent-wise aggregate stats (fairness)
        bat = _load_battery()
        sl_ids = (bat[bat["category"] == "speech_like"]["bg_id"].tolist()
                  if "category" in bat.columns else [])
        ns_ids = (bat[bat["category"] == "non_speech"]["bg_id"].tolist()
                  if "category" in bat.columns else [])

        saa_records = []
        for model_id in df_saa["model"].unique():
            mdf = df_saa[df_saa["model"] == model_id]
            for acc, acc_df in mdf.groupby("accent"):
                noisy = acc_df[acc_df["condition"] == "noisy"]
                mean_dw = float(noisy["dwer"].mean()) if "dwer" in noisy.columns else float("nan")
                dw_sl = float(noisy[noisy["background_id"].isin(sl_ids)]["dwer"].mean()) \
                    if sl_ids and "dwer" in noisy.columns else float("nan")
                dw_ns = float(noisy[noisy["background_id"].isin(ns_ids)]["dwer"].mean()) \
                    if ns_ids and "dwer" in noisy.columns else float("nan")
                saa_records.append({
                    "model": model_id,
                    "accent": acc,
                    "mean_dwer": round(mean_dw, 4),
                    "dwer_speech_like": round(dw_sl, 4),
                    "dwer_non_speech": round(dw_ns, 4),
                    "n_items": len(noisy),
                })
        _save(pd.DataFrame(saa_records), "e3_saa_fairness.csv")
    else:
        _save(df_saa, "e3_saa.csv")


# ── E4: Steerability (RER) ───────────────────────────────────────────────────
def score_e4(smoke: bool) -> None:
    log.info("[E4] Steerability / RER scoring ...")
    records = []
    steer_tasks = ["asr_steer", "asr_steer_p1", "asr_steer_p2", "asr_steer_p3", "asr_steer_p4", "asr_steer_p5"]
    for mid in _all_models():
        df_base = _load_inference(mid, "asr")
        if df_base.empty:
            continue
        
        available_steers = []
        for st in steer_tasks:
            df_st = _load_inference(mid, st)
            if not df_st.empty:
                available_steers.append((df_st, st.replace("asr_", "")))
                
        if not available_steers:
            continue
            
        all_evals = [(df_base, "base")] + available_steers
        for df, label in all_evals:
            if "condition" not in df.columns:
                continue
            df_copy = df.copy()
            if smoke:
                df_copy = df_copy.head(SMOKE_N)
            noisy = df_copy[df_copy["condition"] == "noisy"]
            clean = (df_copy[df_copy["condition"] == "clean"]
                     [["speech_id", "raw"]].rename(columns={"raw": "raw_clean"}))
            merged = noisy.merge(clean, on="speech_id", how="left")
            for _, row in merged.iterrows():
                ref = str(row.get("transcript", ""))
                hyp_n = str(row.get("raw", ""))
                hyp_c = str(row.get("raw_clean", ""))
                if hyp_n == "__ERROR__":
                    continue
                wn = _wer(ref, hyp_n)
                wc = _wer(ref, hyp_c) if hyp_c else float("nan")
                records.append({
                    "model": mid,
                    "speech_id":     row.get("speech_id", ""),
                    "background_id": row.get("background_id", ""),
                    "snr_db":        row.get("snr_db", ""),
                    "prompt":        label,
                    "dwer":          round(wn - wc, 4),
                })

    df_e4 = pd.DataFrame(records)
    if not df_e4.empty:
        for mid in df_e4["model"].unique():
            sub = df_e4[df_e4["model"] == mid]
            base_eff = sub[sub["prompt"] == "base"]["dwer"].mean()
            for label in sub["prompt"].unique():
                if label == "base":
                    continue
                steer_eff = sub[sub["prompt"] == label]["dwer"].mean()
                rer = steer_eff / (base_eff + 1e-9)
                compl = float((sub[sub["prompt"] == label]["dwer"] < base_eff).mean())
                log.info(f"  {mid} ({label}): RER={rer:.4f}, compliance={compl:.2%}")

    _save(df_e4, "e4_steer.csv")


# ── E4-KWS: Steerability for keyword spotting ─────────────────────────────────
def score_e4_kws(smoke: bool) -> None:
    log.info("[E4-KWS] Steerability / KWS prompt engineering scoring ...")
    records = []
    steer_tasks = ["kws_steer", "kws_steer_p1", "kws_steer_p2", "kws_steer_p3", "kws_steer_p4", "kws_steer_p5"]
    for mid in _all_models():
        df_base = _load_inference(mid, "kws")
        if df_base.empty:
            continue

        available_steers = []
        for st in steer_tasks:
            df_st = _load_inference(mid, st)
            if not df_st.empty:
                available_steers.append((df_st, st.replace("kws_", "")))

        if not available_steers:
            continue

        all_evals = [(df_base, "base")] + available_steers
        for df, label in all_evals:
            df_copy = df.copy()
            if smoke:
                df_copy = df_copy.head(SMOKE_N)
            for _, row in df_copy.iterrows():
                ref_kw   = str(row.get("keyword", "")).lower().strip()
                hyp_raw  = str(row.get("raw", "")).lower().strip()
                is_target = bool(row.get("is_target", False))
                if hyp_raw == "__error__":
                    continue
                hyp_kw   = hyp_raw.split()[0] if hyp_raw else "silence"
                import string
                hyp_kw   = hyp_kw.strip(string.punctuation)
                correct      = (hyp_kw == ref_kw) if is_target else (hyp_kw not in KWS_TARGETS)
                false_alarm  = (not is_target) and (hyp_kw in KWS_TARGETS)
                records.append({
                    "model": mid,
                    "speech_id":     row.get("speech_id", ""),
                    "background_id": row.get("background_id", ""),
                    "snr_db":        row.get("snr_db", ""),
                    "condition":     row.get("condition", ""),
                    "prompt":        label,
                    "is_target":     is_target,
                    "correct":       int(correct),
                    "false_alarm":   int(false_alarm),
                })

    df_e4k = pd.DataFrame(records)
    if not df_e4k.empty:
        base_acc = df_e4k[df_e4k["prompt"] == "base"]["correct"].mean()
        log.info(f"[E4-KWS] Base KWS accuracy: {base_acc:.4f}")
        for lbl in df_e4k["prompt"].unique():
            if lbl == "base":
                continue
            acc = df_e4k[df_e4k["prompt"] == lbl]["correct"].mean()
            improvement = acc - base_acc
            log.info(f"  KWS ({lbl}): Acc={acc:.4f}, Δacc={improvement:+.4f}")

    _save(df_e4k, "e4_kws_steer.csv")


# ── Plots (Visualizations) ───────────────────────────────────────────────────
def plot_all_results(smoke: bool) -> None:
    log.info("[Plots] Generating result visualizations ...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        log.warning("[Plots] matplotlib or seaborn not installed.")
        return

    sns.set_theme(style="whitegrid")
    PLOT_DIR = RESULTS / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    bat = _load_battery()
    
    # ── E1 General Robustness (Box, Violin, Line Graphs) ──
    asr_f = RESULTS / "e1_asr.csv"
    if asr_f.exists() and asr_f.stat().st_size > 10:
        df_asr = pd.read_csv(asr_f)
        noisy = df_asr[df_asr["condition"] == "noisy"]
        if "dwer" in df_asr.columns and not noisy.empty:
            # Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=noisy, x="model", y="dwer")
            plt.title("E1 ASR: ΔWER Box Plot across Models")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e1_asr_dwer_box.png")
            plt.close()
            
            # Violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=noisy, x="model", y="dwer", inner="quartile")
            plt.title("E1 ASR: ΔWER Violin Plot across Models")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e1_asr_dwer_violin.png")
            plt.close()

            # Point plot (Line graph) over SNR
            if "snr_db" in df_asr.columns:
                plt.figure(figsize=(10, 6))
                sns.pointplot(data=noisy, x="snr_db", y="wer", hue="model", markers="o")
                plt.title("E1 ASR: Mean WER by SNR (Line Graph)")
                plt.tight_layout()
                plt.savefig(PLOT_DIR / "e1_asr_wer_by_snr_line.png")
                plt.close()

    # ── Fig 1: E1 Profile Law ──
    prof_f = RESULTS / "e1_profile.csv"
    if prof_f.exists() and prof_f.stat().st_size > 10:
        df_prof = pd.read_csv(prof_f)
        noisy = df_prof[df_prof["condition"] == "noisy"]
        if "speech_likeness" in noisy.columns:
            # Regression plot
            g = sns.lmplot(data=noisy, x="speech_likeness", y="dwer", hue="model", scatter_kws={'alpha':0.3})
            g.fig.suptitle("Fig 1a: E1 Profile Law (ΔWER vs Speech-Likeness) - Reg")
            g.fig.tight_layout()
            g.savefig(PLOT_DIR / "fig1a_profile_speech_likeness.png")
            plt.close(g.fig)
            
            # Line plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=noisy, x="speech_likeness", y="dwer", hue="model", marker="o")
            plt.title("Fig 1a: E1 Profile Law (ΔWER vs Speech-Likeness) - Line")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "fig1a_profile_speech_likeness_line.png")
            plt.close()
            
        if "mod_2to8Hz" in noisy.columns:
            # Regression plot
            g = sns.lmplot(data=noisy, x="mod_2to8Hz", y="dwer", hue="model", scatter_kws={'alpha':0.3})
            g.fig.suptitle("Fig 1b: E1 Profile Law (ΔWER vs Syllabic Modulation 2-8Hz) - Reg")
            g.fig.tight_layout()
            g.savefig(PLOT_DIR / "fig1b_profile_modulation.png")
            plt.close(g.fig)
            
            # Line plot
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=noisy, x="mod_2to8Hz", y="dwer", hue="model", marker="o")
            plt.title("Fig 1b: E1 Profile Law (ΔWER vs Syllabic Modulation 2-8Hz) - Line")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "fig1b_profile_modulation_line.png")
            plt.close()

        # Speech-like vs Non-speech category comparison plots (ASR, KWS, SQA)
        comp_asr = pd.DataFrame()
        comp_kws = pd.DataFrame()
        comp_sqa = pd.DataFrame()

        if "category" in noisy.columns:
            comp_asr = noisy[noisy["category"].isin(["speech_like", "non_speech"])].copy()

        # Load and process KWS
        kws_f = RESULTS / "e1_kws.csv"
        if kws_f.exists() and kws_f.stat().st_size > 10:
            try:
                df_kws = pd.read_csv(kws_f)
                if not df_kws.empty and "condition" in df_kws.columns:
                    clean_kws = df_kws[df_kws["condition"] == "clean"][["model", "speech_id", "correct"]].rename(columns={"correct": "correct_clean"})
                    df_kws = df_kws.merge(clean_kws, on=["model", "speech_id"], how="left")
                    df_kws["dacc"] = df_kws["correct"] - df_kws["correct_clean"]
                    # Merge with battery
                    df_kws = df_kws.merge(bat, left_on="background_id", right_on="bg_id", how="inner")
                    if "category" in df_kws.columns:
                        comp_kws = df_kws[(df_kws["condition"] == "noisy") & (df_kws["category"].isin(["speech_like", "non_speech"]))].copy()
            except Exception as e:
                log.warning(f"[Plots] Could not process KWS category comparison: {e}")

        # Load and process SQA
        sqa_f = RESULTS / "e1_sqa.csv"
        if sqa_f.exists() and sqa_f.stat().st_size > 10:
            try:
                df_sqa = pd.read_csv(sqa_f)
                if not df_sqa.empty and "condition" in df_sqa.columns:
                    clean_sqa = df_sqa[df_sqa["condition"] == "clean"][["model", "speech_id", "f1"]].rename(columns={"f1": "f1_clean"})
                    df_sqa = df_sqa.merge(clean_sqa, on=["model", "speech_id"], how="left")
                    df_sqa["df1"] = df_sqa["f1"] - df_sqa["f1_clean"]
                    # Merge with battery
                    df_sqa = df_sqa.merge(bat, left_on="background_id", right_on="bg_id", how="inner")
                    if "category" in df_sqa.columns:
                        comp_sqa = df_sqa[(df_sqa["condition"] == "noisy") & (df_sqa["category"].isin(["speech_like", "non_speech"]))].copy()
            except Exception as e:
                log.warning(f"[Plots] Could not process SQA category comparison: {e}")

        # Generate the multi-column Bar plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        has_bar_data = False
        
        # ASR Subplot
        if not comp_asr.empty:
            sns.barplot(data=comp_asr, x="model", y="dwer", hue="category", ax=axes[0])
            axes[0].set_title("ASR Task: Mean ΔWER\n(Lower/Near 0 is Better)")
            axes[0].set_ylabel("ΔWER (noisy - clean)")
            axes[0].tick_params(axis='x', rotation=45)
            has_bar_data = True
        else:
            axes[0].text(0.5, 0.5, "No ASR Data", ha="center", va="center")
            
        # KWS Subplot
        if not comp_kws.empty:
            sns.barplot(data=comp_kws, x="model", y="dacc", hue="category", ax=axes[1])
            axes[1].set_title("KWS Task: Mean ΔAccuracy\n(Higher/Near 0 is Better)")
            axes[1].set_ylabel("ΔAccuracy (noisy - clean)")
            axes[1].tick_params(axis='x', rotation=45)
            has_bar_data = True
        else:
            axes[1].text(0.5, 0.5, "No KWS Data", ha="center", va="center")

        # SQA Subplot
        if not comp_sqa.empty:
            sns.barplot(data=comp_sqa, x="model", y="df1", hue="category", ax=axes[2])
            axes[2].set_title("SQA Task: Mean ΔF1 Score\n(Higher/Near 0 is Better)")
            axes[2].set_ylabel("ΔF1 (noisy - clean)")
            axes[2].tick_params(axis='x', rotation=45)
            has_bar_data = True
        else:
            axes[2].text(0.5, 0.5, "No SQA Data", ha="center", va="center")

        if has_bar_data:
            plt.suptitle("E1 Robustness: Speech-like vs Non-speech Noise across ASR, KWS, and SQA", fontsize=14, y=0.98)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e1_speech_vs_non_speech_bar.png")
            plt.close()

        # Generate the multi-column Box plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        has_box_data = False
        
        # ASR Subplot
        if not comp_asr.empty:
            sns.boxplot(data=comp_asr, x="model", y="dwer", hue="category", ax=axes[0])
            axes[0].set_title("ASR Task: ΔWER Distribution")
            axes[0].set_ylabel("ΔWER (noisy - clean)")
            axes[0].tick_params(axis='x', rotation=45)
            has_box_data = True
        else:
            axes[0].text(0.5, 0.5, "No ASR Data", ha="center", va="center")
            
        # KWS Subplot
        if not comp_kws.empty:
            sns.boxplot(data=comp_kws, x="model", y="dacc", hue="category", ax=axes[1])
            axes[1].set_title("KWS Task: ΔAccuracy Distribution")
            axes[1].set_ylabel("ΔAccuracy (noisy - clean)")
            axes[1].tick_params(axis='x', rotation=45)
            has_box_data = True
        else:
            axes[1].text(0.5, 0.5, "No KWS Data", ha="center", va="center")

        # SQA Subplot
        if not comp_sqa.empty:
            sns.boxplot(data=comp_sqa, x="model", y="df1", hue="category", ax=axes[2])
            axes[2].set_title("SQA Task: ΔF1 Distribution")
            axes[2].set_ylabel("ΔF1 (noisy - clean)")
            axes[2].tick_params(axis='x', rotation=45)
            has_box_data = True
        else:
            axes[2].text(0.5, 0.5, "No SQA Data", ha="center", va="center")

        if has_box_data:
            plt.suptitle("E1 Robustness Distribution: Speech-like vs Non-speech Noise across ASR, KWS, and SQA", fontsize=14, y=0.98)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e1_speech_vs_non_speech_box.png")
            plt.close()

    # ── Task Generalisation (ASR vs KWS) ──
    kws_f = RESULTS / "e1_kws.csv"
    if asr_f.exists() and kws_f.exists() and asr_f.stat().st_size > 10 and kws_f.stat().st_size > 10:
        df_asr = pd.read_csv(asr_f)
        df_kws = pd.read_csv(kws_f)
        if "dwer" in df_asr.columns and "correct" in df_kws.columns:
            df_asr_grp = df_asr[df_asr["condition"] == "noisy"].groupby(["model", "snr_db"])["dwer"].mean().reset_index()
            df_kws_grp = df_kws[df_kws["condition"] == "noisy"].groupby(["model", "snr_db"])["correct"].mean().reset_index()
            gen = pd.merge(df_asr_grp, df_kws_grp, on=["model", "snr_db"], suffixes=("_asr", "_kws"))
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=gen, x="dwer", y="correct", hue="model", style="snr_db", s=100)
            plt.title("Task Generalisation: ASR ΔWER vs KWS Accuracy")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e2_task_generalisation_scatter.png")
            plt.close()

    # ── Fig 2: E2 Semantic Bias & Injection ──
    e2_f = RESULTS / "e2_semantic.csv"
    if e2_f.exists() and e2_f.stat().st_size > 10:
        df_e2 = pd.read_csv(e2_f)
        if "dwer_real" in df_e2.columns:
            # Bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_e2, x="model", y="dwer_real")
            plt.title("Fig 2a: E2 ASR ΔWER under Speech-like Noise at 0 dB")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "fig2a_dwer_real_bar.png")
            plt.close()

    tir_f = RESULTS / "e2_tir.csv"
    if tir_f.exists() and tir_f.stat().st_size > 10:
        df_tir = pd.read_csv(tir_f)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_tir, x="model", y="tir")
        plt.title("Fig 2c: E2 Trigger Injection Rate (TIR)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "fig2c_tir.png")
        plt.close()

    # ── Fig 3: E3 Disparate Robustness (Ecological & SAA) ──
    e3_f = RESULTS / "e3_gaps.csv"
    if e3_f.exists() and e3_f.stat().st_size > 10:
        df_e3 = pd.read_csv(e3_f)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_e3, x="model", y="robustness_gap", hue="subgroup_type")
        plt.title("Fig 3a: E3 Disparate Robustness Gap (Ecological)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "fig3a_robustness_gap.png")
        plt.close()
        
    e3_f_fair = RESULTS / "e3_fairness.csv"
    if e3_f_fair.exists() and e3_f_fair.stat().st_size > 10:
        df_fair = pd.read_csv(e3_f_fair)
        if "dwer_speech_like" in df_fair.columns and "dwer_non_speech" in df_fair.columns:
            # Melt for speech_like vs non-speech comparison
            df_fair_melt = df_fair.melt(id_vars=["model", "subgroup_type", "subgroup_value"], value_vars=["dwer_speech_like", "dwer_non_speech"], var_name="BgType", value_name="ΔWER")
            
            # 1. Accent-only plots
            df_acc = df_fair_melt[df_fair_melt["subgroup_type"] == "accent"]
            if not df_acc.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_acc, x="subgroup_value", y="ΔWER", hue="BgType")
                plt.title("Fig 3b: E3 Ecological Accent ΔWER (Speech-like vs Non-speech) - Bar")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(PLOT_DIR / "fig3b_ecological_accent_bar.png")
                plt.close()
                
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=df_acc, x="subgroup_value", y="ΔWER", hue="BgType")
                plt.title("Fig 3b: E3 Ecological Accent ΔWER - Box")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(PLOT_DIR / "fig3b_ecological_accent_box.png")
                plt.close()

            # 2. Gender-only plots
            df_gen = df_fair_melt[df_fair_melt["subgroup_type"] == "gender"]
            if not df_gen.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df_gen, x="subgroup_value", y="ΔWER", hue="BgType")
                plt.title("Fig 3c: E3 Ecological Gender ΔWER (Speech-like vs Non-speech) - Bar")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(PLOT_DIR / "fig3c_ecological_gender_bar.png")
                plt.close()

    # 3. SAA Content-Controlled Accent plots
    saa_fair_f = RESULTS / "e3_saa_fairness.csv"
    if saa_fair_f.exists() and saa_fair_f.stat().st_size > 10:
        df_saa_fair = pd.read_csv(saa_fair_f)
        if "dwer_speech_like" in df_saa_fair.columns and "dwer_non_speech" in df_saa_fair.columns:
            df_saa_melt = df_saa_fair.melt(id_vars=["model", "accent"], value_vars=["dwer_speech_like", "dwer_non_speech"], var_name="BgType", value_name="ΔWER")
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_saa_melt, x="accent", y="ΔWER", hue="BgType")
            plt.title("Fig 3d: E3 SAA Content-Controlled Accent ΔWER (Speech-like vs Non-speech) - Bar")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "fig3d_saa_accent_bar.png")
            plt.close()
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_saa_melt, x="accent", y="ΔWER", hue="BgType")
            plt.title("Fig 3d: E3 SAA Content-Controlled Accent ΔWER - Box")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "fig3d_saa_accent_box.png")
            plt.close()

    # ── E4 ASR Steerability ──
    e4_f = RESULTS / "e4_steer.csv"
    if e4_f.exists() and e4_f.stat().st_size > 10:
        df_e4 = pd.read_csv(e4_f)

        # Bar plot: mean ΔWER per prompt per model
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_e4, x="model", y="dwer", hue="prompt")
        plt.title("E4 ASR: Instruction Steerability (ΔWER by Prompt) - Bar")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "e4_asr_steerability_bar.png")
        plt.close()

        # Line plot: mean ΔWER per prompt per model
        plt.figure(figsize=(12, 6))
        sns.pointplot(data=df_e4, x="model", y="dwer", hue="prompt", markers="o", linestyles="-")
        plt.title("E4 ASR: Instruction Steerability (ΔWER by Prompt) - Line")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "e4_asr_steerability_line.png")
        plt.close()

        # Box plot: ΔWER distribution per prompt
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_e4, x="prompt", y="dwer", hue="model")
        plt.title("E4 ASR: ΔWER Distribution per Steering Prompt - Box")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "e4_asr_steerability_box.png")
        plt.close()

    # ── E4 KWS Steerability ──
    e4_kws_f = RESULTS / "e4_kws_steer.csv"
    if e4_kws_f.exists() and e4_kws_f.stat().st_size > 10:
        df_e4k = pd.read_csv(e4_kws_f)

        # 1. Bar: mean accuracy per prompt per model
        if "correct" in df_e4k.columns:
            acc_grp = df_e4k.groupby(["model", "prompt"])["correct"].mean().reset_index()
            acc_grp.rename(columns={"correct": "accuracy"}, inplace=True)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=acc_grp, x="model", y="accuracy", hue="prompt")
            plt.title("E4 KWS: Keyword Accuracy by Steering Prompt - Bar")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e4_kws_steerability_acc_bar.png")
            plt.close()

            # 2. Line: accuracy per prompt per model
            plt.figure(figsize=(12, 6))
            sns.pointplot(data=acc_grp, x="model", y="accuracy", hue="prompt", markers="o", linestyles="-")
            plt.title("E4 KWS: Keyword Accuracy by Steering Prompt - Line")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e4_kws_steerability_acc_line.png")
            plt.close()

        # 3. Bar: false alarm rate per prompt per model
        if "false_alarm" in df_e4k.columns:
            far_grp = df_e4k.groupby(["model", "prompt"])["false_alarm"].mean().reset_index()
            far_grp.rename(columns={"false_alarm": "FAR"}, inplace=True)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=far_grp, x="model", y="FAR", hue="prompt")
            plt.title("E4 KWS: False Alarm Rate by Steering Prompt - Bar")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e4_kws_steerability_far_bar.png")
            plt.close()

        # 4. Box: per-utterance accuracy distribution per prompt
        if "correct" in df_e4k.columns:
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=df_e4k, x="prompt", y="correct", hue="model")
            plt.title("E4 KWS: Per-Utterance Correct Rate Distribution per Prompt - Box")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e4_kws_steerability_box.png")
            plt.close()

        # 5. Side-by-side comparison: noisy-only accuracy per prompt
        if "condition" in df_e4k.columns and "correct" in df_e4k.columns:
            noisy_k = df_e4k[df_e4k["condition"] == "noisy"]
            plt.figure(figsize=(12, 6))
            sns.barplot(data=noisy_k, x="prompt", y="correct", hue="model")
            plt.title("E4 KWS: Noisy Accuracy per Prompt (all models) - Bar")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "e4_kws_noisy_acc_by_prompt_bar.png")
            plt.close()

    log.info(f"  → Plots saved to {PLOT_DIR}")


# ── Summary Table (proposal Table 1) ─────────────────────────────────────────
def build_summary(smoke: bool) -> None:
    log.info("[Summary] Building headline table ...")
    rows = []
    asr_f = RESULTS / "e1_asr.csv"
    kws_f = RESULTS / "e1_kws.csv"

    for mid in _all_models():
        entry = {"model": mid}

        # ASR headline
        if asr_f.exists():
            asr = pd.read_csv(asr_f)
            asr = asr[asr["model"] == mid]
            clean = asr[asr["condition"] == "clean"]
            noisy = asr[asr["condition"] == "noisy"]
            entry["asr_wer_clean"] = round(float(clean["wer"].mean()), 4) if not clean.empty else None
            entry["asr_wer_noisy_mean"] = round(float(noisy["wer"].mean()), 4) if not noisy.empty else None
            if "dwer" in noisy.columns and not noisy.empty:
                entry["asr_dwer_mean"] = round(float(noisy["dwer"].mean()), 4)
                entry["asr_dwer_worst_bg"] = round(float(
                    noisy.groupby("background_id")["dwer"].mean().max()), 4)

        # KWS headline
        if kws_f.exists():
            kws = pd.read_csv(kws_f)
            kws = kws[kws["model"] == mid]
            clean_k = kws[kws["condition"] == "clean"]
            noisy_k = kws[kws["condition"] == "noisy"]
            entry["kws_acc_clean"] = round(float(clean_k["correct"].mean()), 4) if not clean_k.empty else None
            entry["kws_acc_noisy"] = round(float(noisy_k["correct"].mean()), 4) if not noisy_k.empty else None
            entry["kws_far"] = round(float(kws["false_alarm"].mean()), 4) if not kws.empty else None

        rows.append(entry)

    _save(pd.DataFrame(rows), "summary_table.csv")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 — Score all metrics")
    ap.add_argument("--smoke-test", action="store_true",
                    help=f"Score only {SMOKE_N} rows per scorer.")
    args = ap.parse_args()

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    RESULTS.mkdir(parents=True, exist_ok=True)

    score_e1_asr(smoke=args.smoke_test)
    score_e1_kws(smoke=args.smoke_test)
    score_e1_sqa(smoke=args.smoke_test)
    _score_e1_profile(smoke=args.smoke_test)
    score_e2(smoke=args.smoke_test)
    score_e3(smoke=args.smoke_test)
    score_e4(smoke=args.smoke_test)
    score_e4_kws(smoke=args.smoke_test)
    build_summary(smoke=args.smoke_test)
    plot_all_results(smoke=args.smoke_test)

    log.info("=== Stage 6 complete. Results in results/ ===")


if __name__ == "__main__":
    main()
