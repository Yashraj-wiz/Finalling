#!/usr/bin/env python3
"""
scoring/score_all.py — Stage 6: compute all metrics from inference JSONL outputs.

Runs scorers for every experiment:
  E1: WER/CER (ASR), Accuracy/FAR/Miss (KWS)
  E2: TIR
  E3: per-subgroup ΔWER, Robustness Gap, DRI
  E4: RER (if steer outputs exist)

All results → results/ as CSVs + updates results.md.

Usage:
  python scoring/score_all.py --smoke-test
  python scoring/score_all.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import ROOT, get_logger, jsonl_read, csv_append

log = get_logger("score_all")

RESULTS  = ROOT / "results"
INFER    = ROOT / "inference"
MANIFESTS = ROOT / "manifests"
BATTERY_F = ROOT / "descriptors" / "battery.parquet"

SMOKE_N = 10  # rows per check in smoke mode


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_battery_df() -> pd.DataFrame:
    if BATTERY_F.exists():
        return pd.read_parquet(BATTERY_F)
    return pd.DataFrame()


def _load_inference(model_id: str, task: str) -> pd.DataFrame:
    path = INFER / model_id / f"{task}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = jsonl_read(path)
    return pd.DataFrame(rows)


def _all_models() -> list[str]:
    if not INFER.exists():
        return []
    return [d.name for d in INFER.iterdir() if d.is_dir()]


def _wer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        return float(jiwer.wer(str(ref), str(hyp)))
    except Exception:
        return float("nan")


def _cer(ref: str, hyp: str) -> float:
    import jiwer
    try:
        return float(jiwer.cer(str(ref), str(hyp)))
    except Exception:
        return float("nan")


def _error_counts(ref: str, hyp: str) -> dict:
    """Return substitutions, deletions, insertions counts."""
    import jiwer
    try:
        out = jiwer.process_words(str(ref), str(hyp))
        return {
            "substitutions": int(out.substitutions),
            "deletions":     int(out.deletions),
            "insertions":    int(out.insertions),
        }
    except Exception:
        return {"substitutions": 0, "deletions": 0, "insertions": 0}


# ── E1 ASR scoring ─────────────────────────────────────────────────────────────
def score_e1_asr(smoke: bool) -> pd.DataFrame:
    log.info("[E1-ASR] Scoring...")
    records = []
    for model_id in _all_models():
        df = _load_inference(model_id, "asr")
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
                "model":        model_id,
                "speech_id":    row.get("speech_id", ""),
                "background_id":row.get("background_id", ""),
                "snr_db":       row.get("snr_db", ""),
                "condition":    row.get("condition", ""),
                "accent":       row.get("accent", ""),
                "gender":       row.get("gender", ""),
                "wer":          round(wer, 4),
                "cer":          round(cer, 4),
                **{k: v for k, v in ec.items()},
            })

    df_all = pd.DataFrame(records)
    if not df_all.empty:
        # ΔWER = wer(noisy) - wer(clean), paired per (model, speech_id)
        clean = df_all[df_all["condition"] == "clean"][["model", "speech_id", "wer"]].rename(
            columns={"wer": "wer_clean"})
        df_all = df_all.merge(clean, on=["model", "speech_id"], how="left")
        df_all["dwer"] = df_all["wer"] - df_all["wer_clean"]

    out = RESULTS / "e1_asr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, index=False)
    log.info(f"[E1-ASR] {len(df_all)} rows → {out}")
    return df_all


# ── E1 KWS scoring ─────────────────────────────────────────────────────────────
def score_e1_kws(smoke: bool) -> pd.DataFrame:
    log.info("[E1-KWS] Scoring...")
    records = []
    for model_id in _all_models():
        df = _load_inference(model_id, "kws")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N)
        for _, row in df.iterrows():
            ref_kw   = str(row.get("keyword", "")).lower().strip()
            hyp_raw  = str(row.get("raw", "")).lower().strip()
            is_target = bool(row.get("is_target", False))
            hyp_kw   = hyp_raw.split()[0] if hyp_raw else "silence"
            correct  = (hyp_kw == ref_kw) if is_target else (hyp_kw != ref_kw)
            # For FAR: non-target clip predicted as a target keyword
            false_alarm = (not is_target) and (hyp_kw in [kw.lower() for kw in _kws_targets()])
            records.append({
                "model":         model_id,
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

    df_all = pd.DataFrame(records)
    out = RESULTS / "e1_kws.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out, index=False)
    log.info(f"[E1-KWS] {len(df_all)} rows → {out}")
    return df_all


def _kws_targets() -> list[str]:
    return ["yes","no","up","down","left","right","on","off","stop","go"]


# ── E2 scoring: TIR ───────────────────────────────────────────────────────────
def score_e2(smoke: bool) -> None:
    log.info("[E2] Scoring real + injection rates...")
    battery_df = _load_battery_df()
    speech_like_ids = battery_df[battery_df["category"] == "speech_like"]["bg_id"].tolist() \
        if not battery_df.empty else []

    records_asr = []
    for model_id in _all_models():
        df = _load_inference(model_id, "asr")
        if df.empty:
            continue
        if smoke:
            df = df.head(SMOKE_N * 2)

        for bg_id in speech_like_ids:
            real_rows      = df[(df["background_id"] == bg_id) & (df["snr_db"] == 0)]
            clean_rows     = df[df["condition"] == "clean"]

            for _, r_row in real_rows.iterrows():
                ref = str(r_row.get("transcript", ""))
                hyp_real = str(r_row.get("raw", ""))
                clean_match = clean_rows[clean_rows["speech_id"] == r_row["speech_id"]]
                hyp_clean = str(clean_match.iloc[0]["raw"]) if len(clean_match) > 0 else ""

                wer_real  = _wer(ref, hyp_real)
                wer_clean = _wer(ref, hyp_clean) if hyp_clean else float("nan")
                dwer_real = wer_real - wer_clean

                records_asr.append({
                    "model": model_id,
                    "speech_id": r_row["speech_id"],
                    "background_id": bg_id,
                    "dwer_real": round(dwer_real, 4),
                })

    out = RESULTS / "e2_asr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records_asr).to_csv(out, index=False)
    log.info(f"[E2-ASR] → {out}")

    # TIR: FAR on injection probes vs generic noise
    _score_tir(smoke)



def _score_tir(smoke: bool) -> None:
    """TIR = FAR specifically on injection probes vs generic noise."""
    probe_ids_file = ROOT / "prereg" / "injection_probe_ids.json"
    if not probe_ids_file.exists():
        log.warning("[TIR] No injection probe IDs found. Skipping TIR.")
        return
    probe_ids = set(json.loads(probe_ids_file.read_text()))

    records = []
    for model_id in _all_models():
        df = _load_inference(model_id, "kws")
        if df.empty:
            continue
        probe_rows = df[df["speech_id"].isin(probe_ids)]
        if smoke:
            probe_rows = probe_rows.head(SMOKE_N)
        for _, row in probe_rows.iterrows():
            predicted = str(row.get("raw", "")).lower().strip().split()[0] if row.get("raw") else ""
            tir = int(predicted in _kws_targets())
            records.append({
                "model": model_id,
                "probe_id": row.get("speech_id", ""),
                "background_id": row.get("background_id", ""),
                "condition": row.get("condition", ""),
                "predicted": predicted,
                "tir": tir,
            })

    out = RESULTS / "e2_tir.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    log.info(f"[E2-TIR] → {out}")


# ── E3 scoring: disparate robustness ─────────────────────────────────────────
def score_e3(smoke: bool) -> None:
    log.info("[E3] Disparate robustness scoring...")
    asr_df = pd.read_csv(RESULTS / "e1_asr.csv") if (RESULTS / "e1_asr.csv").exists() else pd.DataFrame()
    if asr_df.empty:
        log.warning("[E3] E1 ASR results not found. Run E1 first.")
        return

    battery_df = _load_battery_df()
    speech_like_bgs = battery_df[battery_df["category"] == "speech_like"]["bg_id"].tolist() \
        if not battery_df.empty else []
    non_speech_bgs  = battery_df[battery_df["category"] == "non_speech"]["bg_id"].tolist() \
        if not battery_df.empty else []

    # Per-subgroup ΔWER
    subgroups = ["accent", "gender"]
    records = []
    for sg in subgroups:
        if sg not in asr_df.columns:
            continue
        for group_val, group_df in asr_df.groupby(sg):
            noisy = group_df[group_df["condition"] == "noisy"]
            mean_dwer = float(noisy["dwer"].mean()) if "dwer" in noisy.columns else float("nan")
            # Speech-like vs non-speech breakdown
            dwer_sl = float(noisy[noisy["background_id"].isin(speech_like_bgs)]["dwer"].mean()) \
                if speech_like_bgs else float("nan")
            dwer_ns = float(noisy[noisy["background_id"].isin(non_speech_bgs)]["dwer"].mean()) \
                if non_speech_bgs else float("nan")
            records.append({
                "subgroup_type":  sg,
                "subgroup_value": group_val,
                "mean_dwer":      round(mean_dwer, 4),
                "dwer_speech_like":  round(dwer_sl, 4),
                "dwer_non_speech":   round(dwer_ns, 4),
                "n_items":        len(noisy),
            })

    df_e3 = pd.DataFrame(records)
    if not df_e3.empty:
        # Robustness Gap and DRI per subgroup type
        for sg in df_e3["subgroup_type"].unique():
            sub = df_e3[df_e3["subgroup_type"] == sg]["mean_dwer"].dropna()
            if len(sub) >= 2:
                gap = float(sub.max() - sub.min())
                dri = gap / (float(sub.mean()) + 1e-9)
                log.info(f"[E3] {sg}: Robustness Gap={gap:.4f}, DRI={dri:.4f}")

    out = RESULTS / "e3_fairness.csv"
    df_e3.to_csv(out, index=False)
    log.info(f"[E3] → {out}")

    # SAA controlled (if saa inference exists)
    _score_e3_saa(smoke)


def _score_e3_saa(smoke: bool) -> None:
    """Content-matched per-accent ΔWER on Speech Accent Archive."""
    saa_items = {r["id"]: r for r in jsonl_read(ROOT / "itembanks" / "saa.jsonl")}
    if not saa_items:
        log.info("[E3-SAA] No SAA bank. Skipping.")
        return

    records = []
    for model_id in _all_models():
        df = _load_inference(model_id, "asr")
        if df.empty:
            continue
        saa_df = df[df["speech_id"].isin(saa_items.keys())]
        if smoke:
            saa_df = saa_df.head(SMOKE_N)
        for _, row in saa_df.iterrows():
            sp_info = saa_items.get(row["speech_id"], {})
            ref = sp_info.get("transcript", "")
            hyp = str(row.get("raw", ""))
            wer = _wer(ref, hyp)
            records.append({
                "model": model_id,
                "speech_id": row["speech_id"],
                "accent": sp_info.get("accent", ""),
                "gender": sp_info.get("gender", ""),
                "background_id": row.get("background_id", ""),
                "snr_db": row.get("snr_db", ""),
                "condition": row.get("condition", ""),
                "wer": round(wer, 4),
            })

    out = RESULTS / "e3_saa.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    log.info(f"[E3-SAA] → {out}")


# ── E4 scoring: steerability ──────────────────────────────────────────────────
def score_e4(smoke: bool) -> None:
    """RER = effect_with_instruction / effect_without."""
    log.info("[E4] Steerability scoring...")
    records = []
    for model_id in _all_models():
        df_base  = _load_inference(model_id, "asr")
        df_steer = _load_inference(model_id, "asr_steer")
        if df_base.empty or df_steer.empty:
            continue
        if smoke:
            df_base  = df_base.head(SMOKE_N)
            df_steer = df_steer.head(SMOKE_N)

        # Compute effect for each condition
        for df, label in [(df_base, "base"), (df_steer, "steer")]:
            noisy = df[df["condition"] == "noisy"]
            clean = df[df["condition"] == "clean"][["speech_id", "raw"]].rename(
                columns={"raw": "raw_clean"})
            merged = noisy.merge(clean, on="speech_id", how="left")
            for _, row in merged.iterrows():
                ref = str(row.get("transcript", ""))
                wer_noisy = _wer(ref, str(row.get("raw", "")))
                wer_clean = _wer(ref, str(row.get("raw_clean", "")))
                records.append({
                    "model": model_id,
                    "speech_id": row.get("speech_id", ""),
                    "background_id": row.get("background_id", ""),
                    "snr_db": row.get("snr_db", ""),
                    "condition_label": label,
                    "dwer": round(wer_noisy - wer_clean, 4),
                })

    df_e4 = pd.DataFrame(records)
    if not df_e4.empty:
        base_effect  = df_e4[df_e4["condition_label"] == "base"]["dwer"].mean()
        steer_effect = df_e4[df_e4["condition_label"] == "steer"]["dwer"].mean()
        rer = steer_effect / (base_effect + 1e-9)
        log.info(f"[E4] Overall RER = {rer:.4f}")

    out = RESULTS / "e4_steerability.csv"
    df_e4.to_csv(out, index=False)
    log.info(f"[E4] → {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 — Score all experiments")
    ap.add_argument("--smoke-test", action="store_true",
                    help=f"Score only {SMOKE_N} rows per check.")
    args = ap.parse_args()

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    RESULTS.mkdir(parents=True, exist_ok=True)

    score_e1_asr(smoke=args.smoke_test)
    score_e1_kws(smoke=args.smoke_test)
    score_e2(smoke=args.smoke_test)
    score_e3(smoke=args.smoke_test)
    score_e4(smoke=args.smoke_test)

    log.info("=== Stage 6 complete. Results in results/ ===")


if __name__ == "__main__":
    main()
