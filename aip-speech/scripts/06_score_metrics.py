#!/usr/bin/env python3
"""
06_score_metrics.py — Stage 6: compute all metrics from inference JSONL outputs.

Experiments scored:
  E1  WER/CER + sub/del/ins + ΔWER (ASR);  Accuracy/FAR/Miss (KWS)
  E2  ΔWER real-vs-scrambled, BIR (ASR);   TIR on injection probes (KWS)
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
INFER     = ROOT / "inference"
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
        return pd.read_parquet(BATTERY_F)
    return pd.DataFrame()


def _wer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        return float(jiwer.wer(str(ref).strip(), str(hyp).strip()))
    except Exception:
        return float("nan")


def _cer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        return float(jiwer.cer(str(ref).strip(), str(hyp).strip()))
    except Exception:
        return float("nan")


def _error_counts(ref: str, hyp: str) -> dict:
    """Return substitution, deletion, insertion counts."""
    import jiwer
    try:
        out = jiwer.process_words(str(ref).strip(), str(hyp).strip())
        return {"substitutions": int(out.substitutions),
                "deletions": int(out.deletions),
                "insertions": int(out.insertions)}
    except Exception:
        return {"substitutions": 0, "deletions": 0, "insertions": 0}


def _save(df: pd.DataFrame, name: str) -> Path:
    out = RESULTS / name
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


# ── E2-ASR: real vs scrambled, BIR ───────────────────────────────────────────
def _compute_bir(hyp: str, ref: str, bg_id: str) -> float:
    """Background-Injection Rate: fraction of inserted tokens matching bg vocab."""
    import jiwer
    try:
        out = jiwer.process_words(str(ref).strip(), str(hyp).strip())
        # Extract inserted words from alignment
        inserted = []
        for align in out.alignments:
            for chunk in align:
                if chunk.type == "insert":
                    hyp_tokens = str(hyp).strip().split()
                    for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        if i < len(hyp_tokens):
                            inserted.append(hyp_tokens[i].lower())
    except Exception:
        inserted = []
    if not inserted:
        return 0.0
    bg_vocab = set(bg_id.lower().replace("_", " ").split())
    hits = sum(1 for w in inserted if w in bg_vocab)
    return hits / len(inserted)


def score_e2(smoke: bool) -> None:
    log.info("[E2] Scoring real-vs-scrambled + BIR + TIR ...")
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
            scram = df[(df["background_id"] == f"{bg_id}_scrambled") & (df["snr_db"] == 0)]

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

                sm = scram[scram["speech_id"] == rr["speech_id"]]
                if len(sm) > 0:
                    hyp_s = str(sm.iloc[0]["raw"])
                    dwer_s = _wer(ref, hyp_s) - wer_c
                    semantic_gap = dwer_r - dwer_s
                else:
                    dwer_s = semantic_gap = float("nan")

                bir = _compute_bir(hyp_r, ref, bg_id)

                records.append({
                    "model": mid, "speech_id": rr["speech_id"],
                    "background_id": bg_id,
                    "dwer_real": round(dwer_r, 4),
                    "dwer_scrambled": round(dwer_s, 4) if not np.isnan(dwer_s) else None,
                    "semantic_gap": round(semantic_gap, 4) if not np.isnan(semantic_gap) else None,
                    "bir": round(bir, 4),
                })

    _save(pd.DataFrame(records), "e2_semantic.csv")

    # TIR
    _score_tir(smoke)


def _score_tir(smoke: bool) -> None:
    """TIR = FAR on injection probes vs scrambled vs generic noise."""
    pf = ROOT / "prereg" / "injection_probe_ids.json"
    if not pf.exists():
        log.warning("[TIR] No injection probe IDs. Skipping.")
        return
    probe_ids = set(json.loads(pf.read_text()))

    records = []
    for mid in _all_models():
        df = _load_inference(mid, "kws")
        if df.empty:
            continue
        probes = df[df["speech_id"].isin(probe_ids)]
        if smoke:
            probes = probes.head(SMOKE_N)
        for _, row in probes.iterrows():
            raw = str(row.get("raw", "")).lower().strip()
            predicted = raw.split()[0] if raw else ""
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

    bat = _load_battery()
    sl_ids = (bat[bat["category"] == "speech_like"]["bg_id"].tolist()
              if "category" in bat.columns else [])
    st_ids = (bat[bat["category"] == "stationary"]["bg_id"].tolist()
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
                dw_st = float(noisy[noisy["background_id"].isin(st_ids)]["dwer"].mean()) \
                    if st_ids and "dwer" in noisy.columns else float("nan")
                records.append({
                    "model": model_id,
                    "subgroup_type":  sg,
                    "subgroup_value": gval,
                    "mean_dwer":         round(mean_dw, 4),
                    "dwer_speech_like":  round(dw_sl, 4),
                    "dwer_stationary":   round(dw_st, 4),
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
        df = _load_inference(mid, "asr")
        if df.empty:
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

    _save(pd.DataFrame(records), "e3_saa.csv")


# ── E4: Steerability (RER) ───────────────────────────────────────────────────
def score_e4(smoke: bool) -> None:
    log.info("[E4] Steerability / RER scoring ...")
    records = []
    for mid in _all_models():
        df_base  = _load_inference(mid, "asr")
        df_steer = _load_inference(mid, "asr_steer")
        if df_base.empty or df_steer.empty:
            continue
        if smoke:
            df_base  = df_base.head(SMOKE_N)
            df_steer = df_steer.head(SMOKE_N)

        for df, label in [(df_base, "base"), (df_steer, "steer")]:
            noisy = df[df["condition"] == "noisy"]
            clean = (df[df["condition"] == "clean"]
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
            base_eff  = sub[sub["prompt"] == "base"]["dwer"].mean()
            steer_eff = sub[sub["prompt"] == "steer"]["dwer"].mean()
            rer = steer_eff / (base_eff + 1e-9)
            compl = float((sub[sub["prompt"] == "steer"]["dwer"] < base_eff).mean())
            log.info(f"  {mid}: RER={rer:.4f}, compliance={compl:.2%}")

    _save(df_e4, "e4_steer.csv")


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

        # E2 semantic gap
        e2_f = RESULTS / "e2_semantic.csv"
        if e2_f.exists() and e2_f.stat().st_size > 10:
            try:
                e2 = pd.read_csv(e2_f)
                e2m = e2[e2["model"] == mid]
                if not e2m.empty and "semantic_gap" in e2m.columns:
                    entry["semantic_gap_mean"] = round(float(e2m["semantic_gap"].mean()), 4)
            except Exception:
                pass

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
    score_e2(smoke=args.smoke_test)
    score_e3(smoke=args.smoke_test)
    score_e4(smoke=args.smoke_test)
    build_summary(smoke=args.smoke_test)

    log.info("=== Stage 6 complete. Results in results/ ===")


if __name__ == "__main__":
    main()
