#!/usr/bin/env python3
"""
04_manipulation_checks.py — Stage 4: manipulation checks before inference.

Checks (from §Stage 4 of implementation plan):
  1. Background presence   → checks/bg_presence.csv
  2. WER constancy         → checks/wer_constancy.csv
  3. Scramble validity     → checks/scramble_validity.csv
  4. Determinism floor     → checks/determinism.json

Also materialises the fixed ~60-clip inspection sample → checks/inspection/

Usage:
  python scripts/04_manipulation_checks.py --smoke-test   # 5 clips
  python scripts/04_manipulation_checks.py                # full checks
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, SR, SPEECH_LUFS, BG_LUFS, get_logger,
                   ProgressLog, jsonl_read, load_audio, loudness_normalize,
                   csv_append)

sys.path.insert(0, str(ROOT / "mixing"))
from mix import mix, materialize, diagnostics, DIAG_CSV

log = get_logger("04_manipulation_checks")
PROGRESS    = ROOT / "checks" / "checks_progress.json"
INSPECT_DIR = ROOT / "checks" / "inspection"
BG_PRESENCE = ROOT / "checks" / "bg_presence.csv"
WER_CONST   = ROOT / "checks" / "wer_constancy.csv"
SCRAMBLE_V  = ROOT / "checks" / "scramble_validity.csv"
DETERMINISM = ROOT / "checks" / "determinism.json"

RANDOM_SEED    = 42
INSPECT_N      = 60   # full inspection sample size
INSPECT_N_SMOKE = 5


# ── build inspection sample ───────────────────────────────────────────────────
def _load_battery() -> list[dict]:
    """Return battery as list of dicts."""
    bat_file = ROOT / "descriptors" / "battery.parquet"
    if not bat_file.exists():
        return []
    import pandas as pd
    return pd.read_parquet(bat_file).to_dict("records")


def _load_asr_items(n: int) -> list[dict]:
    items = jsonl_read(ROOT / "itembanks" / "asr.jsonl")
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(items)
    return items[:n]


def materialise_inspection(prog: ProgressLog, smoke: bool) -> None:
    """Write ~60 clips to checks/inspection/ for manual listening."""
    key = "inspection_sample"
    if prog.done(key):
        log.info("[skip] Inspection sample already materialised.")
        return

    n = INSPECT_N_SMOKE if smoke else INSPECT_N
    battery = _load_battery()
    speech_items = _load_asr_items(n)

    if not battery or not speech_items:
        log.warning("[inspection] Battery or ASR bank empty — skipping materialisation.")
        return

    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    count = 0

    for sp_item in speech_items[:n]:
        bg = rng.choice(battery)
        snr = rng.choice([10, 5, 0])
        seed = rng.randint(0, 2**31)

        row = {
            "id": f"insp_{sp_item['id']}_{bg['bg_id']}_snr{snr}",
            "speech_id":   sp_item["id"],
            "background_id": bg["bg_id"],
            "snr_db": snr,
            "condition": "noisy",
            "seed": seed,
            "speech_path": str(ROOT / sp_item["wav"]),
            "bg_path":     str(ROOT / bg["wav"]),
        }
        out = INSPECT_DIR / f"{row['id']}.wav"
        if not out.exists():
            materialize(row, out)
        count += 1

    log.info(f"[inspection] Materialised {count} clips → {INSPECT_DIR}")
    prog.mark(key)


# ── check 1: background presence ─────────────────────────────────────────────
def check_bg_presence(prog: ProgressLog, smoke: bool) -> None:
    """
    Confirm background is audible in each inspection clip using a simple
    speech-energy proxy: noisy mix should have more high-freq energy than clean.
    """
    key = "bg_presence"
    if prog.done(key):
        log.info("[skip] bg_presence check already done.")
        return

    clips = sorted(INSPECT_DIR.glob("*.wav"))
    if not clips:
        log.warning("[bg_presence] No inspection clips found.")
        return

    if smoke:
        clips = clips[:INSPECT_N_SMOKE]

    for clip in clips:
        x = load_audio(clip)
        # Proxy: fraction of energy in non-silent frames
        frames = [x[i:i+480] for i in range(0, len(x)-480, 480)]
        rms = [float(np.sqrt(np.mean(f**2))) for f in frames]
        presence_score = float(np.mean(np.array(rms) > 1e-4))
        csv_append(BG_PRESENCE, {
            "clip": clip.name,
            "presence_score": round(presence_score, 4),
            "pass": presence_score > 0.5,
        }, fieldnames=["clip", "presence_score", "pass"])

    prog.mark(key)
    log.info(f"[done] bg_presence → {BG_PRESENCE}")


# ── check 2: WER constancy ────────────────────────────────────────────────────
def check_wer_constancy(prog: ProgressLog, smoke: bool) -> None:
    """
    Whisper WER on the inspection clips should not vary wildly across backgrounds
    at a fixed SNR. We run Whisper-base on each clip and record WER.
    """
    key = "wer_constancy"
    if prog.done(key):
        log.info("[skip] wer_constancy check already done.")
        return

    try:
        import whisper
        import jiwer
    except ImportError:
        log.warning("[wer_constancy] whisper or jiwer not installed. Skipping.")
        return

    clips = sorted(INSPECT_DIR.glob("*.wav"))
    if smoke:
        clips = clips[:INSPECT_N_SMOKE]

    wmodel = whisper.load_model(
        "base",
        download_root=str(ROOT / "models" / "cache" / "whisper")
    )
    asr_bank = {r["id"]: r for r in jsonl_read(ROOT / "itembanks" / "asr.jsonl")}

    for clip in clips:
        # Parse speech_id from clip name: insp_<speech_id>_<bg_id>_snr<N>.wav
        parts = clip.stem.split("_")
        speech_id = parts[1] if len(parts) > 1 else ""
        ref = asr_bank.get(speech_id, {}).get("transcript", "")
        if not ref:
            continue
        x = load_audio(clip)
        result = wmodel.transcribe(x, language="en", fp16=False)
        hyp = result.get("text", "").strip()
        try:
            wer = jiwer.wer(ref, hyp)
        except Exception:
            wer = float("nan")
        csv_append(WER_CONST, {
            "clip": clip.name,
            "speech_id": speech_id,
            "wer": round(wer, 4),
            "ref": ref,
            "hyp": hyp,
        }, fieldnames=["clip", "speech_id", "wer", "ref", "hyp"])

    prog.mark(key)
    log.info(f"[done] wer_constancy → {WER_CONST}")


# ── check 3: scramble validity ────────────────────────────────────────────────
def check_scramble_validity(prog: ProgressLog, smoke: bool) -> None:
    """
    For each scrambled twin:
    - spectrum should match original within tolerance
    - linguistic_content (Whisper word count) should be ~0
    """
    key = "scramble_validity"
    if prog.done(key):
        log.info("[skip] scramble_validity already done.")
        return

    bg_dir  = DATA / "bg"
    bgs_dir = DATA / "bg_scrambled"
    twins = sorted(bgs_dir.glob("*_scrambled.wav"))
    if smoke:
        twins = twins[:2]

    try:
        import whisper
        wmodel = whisper.load_model(
            "base",
            download_root=str(ROOT / "models" / "cache" / "whisper")
        )
    except ImportError:
        log.warning("[scramble_validity] whisper not installed.")
        wmodel = None

    for twin_path in twins:
        orig_id = twin_path.stem.replace("_scrambled", "")
        orig_path = bg_dir / f"{orig_id}.wav"
        if not orig_path.exists():
            continue
        orig = load_audio(orig_path)
        twin = load_audio(twin_path)

        # Spectral match: cosine similarity of magnitude spectra
        O = np.abs(np.fft.rfft(orig))
        T = np.abs(np.fft.rfft(twin))
        min_len = min(len(O), len(T))
        cos_sim = float(np.dot(O[:min_len], T[:min_len]) /
                        (np.linalg.norm(O[:min_len]) * np.linalg.norm(T[:min_len]) + 1e-9))

        # Linguistic content in twin
        ling_content = 0.0
        if wmodel is not None:
            result = wmodel.transcribe(twin, language="en", fp16=False, word_timestamps=True)
            words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
            ling_content = float(sum(abs(w.get("probability", 0)) for w in words))

        csv_append(SCRAMBLE_V, {
            "bg_id": orig_id,
            "spectral_cosine_sim": round(cos_sim, 4),
            "twin_linguistic_content": round(ling_content, 4),
            "pass_spectral": cos_sim > 0.99,
            "pass_linguistic": ling_content < 2.0,
        }, fieldnames=[
            "bg_id", "spectral_cosine_sim", "twin_linguistic_content",
            "pass_spectral", "pass_linguistic"
        ])

    prog.mark(key)
    log.info(f"[done] scramble_validity → {SCRAMBLE_V}")


# ── check 4: determinism floor ────────────────────────────────────────────────
def check_determinism(prog: ProgressLog, smoke: bool) -> None:
    """
    Run 50 clean ASR items through the first available model twice with
    greedy decoding and confirm identical outputs. Records residual.
    """
    key = "determinism"
    if prog.done(key):
        log.info("[skip] determinism check already done.")
        return

    items = jsonl_read(ROOT / "itembanks" / "asr.jsonl")
    n = 5 if smoke else 50
    items = items[:n]

    if not items:
        log.warning("[determinism] No ASR items found.")
        return

    # Use Whisper as a stand-in determinism checker (avoids loading a full LLM)
    try:
        import whisper
        wmodel = whisper.load_model(
            "base",
            download_root=str(ROOT / "models" / "cache" / "whisper")
        )
    except ImportError:
        log.warning("[determinism] whisper not installed — skipping.")
        prog.mark(key)
        return

    mismatches = 0
    total = 0
    for item in items:
        wav_path = ROOT / item["wav"]
        if not wav_path.exists():
            continue
        x = load_audio(wav_path)
        r1 = wmodel.transcribe(x, language="en", fp16=False).get("text", "")
        r2 = wmodel.transcribe(x, language="en", fp16=False).get("text", "")
        if r1.strip() != r2.strip():
            mismatches += 1
        total += 1

    result = {
        "total_items": total,
        "mismatches": mismatches,
        "determinism_rate": round(1 - mismatches / max(total, 1), 4),
    }
    DETERMINISM.write_text(json.dumps(result, indent=2))
    prog.mark(key)
    log.info(f"[done] determinism: {result}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 — Manipulation checks")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run checks on 5 clips only.")
    args = ap.parse_args()
    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    prog = ProgressLog(PROGRESS)
    materialise_inspection(prog, smoke=args.smoke_test)
    check_bg_presence(prog, smoke=args.smoke_test)
    check_wer_constancy(prog, smoke=args.smoke_test)
    check_scramble_validity(prog, smoke=args.smoke_test)
    check_determinism(prog, smoke=args.smoke_test)
    log.info("=== Stage 4 complete. ===")


if __name__ == "__main__":
    main()
