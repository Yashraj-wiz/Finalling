#!/usr/bin/env python3
"""
02_build_banks.py — Stage 2: build foreground item banks.

Reads downloaded corpora and produces:
  itembanks/asr.jsonl  : {id, wav, transcript, source, accent, gender, age}
  itembanks/kws.jsonl  : {id, wav, keyword, is_target, probe_id?}
  itembanks/saa.jsonl  : {id, wav, transcript, accent, gender}

All audio paths are RELATIVE to ROOT so the project is portable.
Common Voice is STREAMED (never written to disk).

Usage:
  python scripts/02_build_banks.py --smoke-test   # 5 items per bank
  python scripts/02_build_banks.py                # full run (resumable)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, CACHE, SR, get_logger, ProgressLog,
                   jsonl_append, jsonl_ids, load_audio)

log = get_logger("02_build_banks")

ASR_BANK   = ROOT / "itembanks" / "asr.jsonl"
KWS_BANK   = ROOT / "itembanks" / "kws.jsonl"
SAA_BANK   = ROOT / "itembanks" / "saa.jsonl"
PROGRESS   = ROOT / "checks" / "bank_progress.json"

RANDOM_SEED = 42

# Target counts (per proposal: ~100 ASR items, ~120 KWS items, ~90 SAA items)
ASR_N   = 100
CV_N    = 80
KWS_N   = 120
SAA_N   = 90
SMOKE_N = 5  # items per bank in smoke mode

# KWS target keywords from Google Speech Commands v2
KWS_TARGETS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]


# ── LibriSpeech → asr.jsonl ──────────────────────────────────────────────────
def build_asr_bank(prog: ProgressLog, n: int) -> None:
    key = "asr_bank"
    if prog.done(key):
        log.info("[skip] ASR bank already built.")
        return

    done_ids = jsonl_ids(ASR_BANK)
    items: list[dict] = []

    # Load speaker metadata for gender
    speakers_file = DATA / "speech_asr" / "LibriSpeech" / "SPEAKERS.TXT"
    speaker_gender = {}
    if speakers_file.exists():
        with open(speakers_file) as f:
            for line in f:
                if line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    spk_id = parts[0]
                    gender = "female" if parts[1] == "F" else "male" if parts[1] == "M" else "unknown"
                    speaker_gender[spk_id] = gender

    # Collect from test-clean and test-other
    for split in ["test-clean", "test-other"]:
        trans_files = sorted(
            (DATA / "speech_asr" / "LibriSpeech" / split).rglob("*.trans.txt")
        )
        for tf in trans_files:
            with open(tf) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    wav = tf.parent / f"{utt_id}.flac"
                    if not wav.exists():
                        continue
                    spk_id = utt_id.split("-")[0]
                    items.append({
                        "id": utt_id,
                        "wav": str(wav.relative_to(ROOT)),
                        "transcript": transcript,
                        "source": f"librispeech_{split}",
                        "accent": "american_english",
                        "gender": speaker_gender.get(spk_id, "unknown"),
                        "age": "unknown",
                    })

    # Balance across splits
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(items)
    for item in items[:n]:
        if item["id"] not in done_ids:
            jsonl_append(ASR_BANK, item)
            done_ids.add(item["id"])

    # Supplement with Common Voice (streamed) to get accent/gender diversity
    cv_target = CV_N if n == ASR_N else SMOKE_N
    _augment_asr_with_cv(done_ids, len(done_ids) + cv_target, prog)
    prog.mark(key)
    log.info(f"[done] ASR bank: {len(jsonl_ids(ASR_BANK))} items.")


def _augment_asr_with_cv(done_ids: set, target: int, prog: ProgressLog) -> None:
    """Stream Common Voice validation split to add accent/gender-labelled items."""
    existing = len(done_ids)
    if existing >= target:
        return
    needed = target - existing
    log.info(f"[CV] Streaming Common Voice to add {needed} diverse items...")
    try:
        from datasets import load_dataset
        cv = load_dataset(
            "fsicoli/common_voice_17_0",
            "en",
            split="validation",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE / "datasets"),
        )
        added = 0
        for ex in cv:
            if added >= needed:
                break
            
            accent = ex.get("accent") or ""
            gender = ex.get("gender") or ""
            if not accent or not gender or accent.lower() == "unknown" or gender.lower() == "unknown":
                continue
                
            utt_id = f"cv_{ex['client_id'][:8]}_{added}"
            if utt_id in done_ids:
                continue
            # Save the audio clip to disk so it's accessible during inference
            out = DATA / "speech_asr" / "common_voice" / f"{utt_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                import soundfile as sf
                audio = ex["audio"]
                sf.write(str(out), audio["array"], audio["sampling_rate"])
            jsonl_append(ASR_BANK, {
                "id": utt_id,
                "wav": str(out.relative_to(ROOT)),
                "transcript": ex.get("sentence", ""),
                "source": "common_voice_17",
                "accent": accent,
                "gender": gender,
                "age":    ex.get("age",    "unknown") or "unknown",
            })
            done_ids.add(utt_id)
            added += 1
        log.info(f"[CV] Added {added} Common Voice items.")
    except Exception as e:
        log.warning(f"[CV] Could not stream Common Voice: {e}. Skipping.")


# ── Speech Commands → kws.jsonl ──────────────────────────────────────────────
def build_kws_bank(prog: ProgressLog, n: int) -> None:
    key = "kws_bank"
    if prog.done(key):
        log.info("[skip] KWS bank already built.")
        return

    done_ids = jsonl_ids(KWS_BANK)
    kws_root = DATA / "speech_kws"
    rng = random.Random(RANDOM_SEED)

    items: list[dict] = []
    # Target keywords
    for kw in KWS_TARGETS:
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        rng.shuffle(wav_files)
        for w in wav_files[:n // len(KWS_TARGETS) + 2]:
            items.append({
                "id": f"kws_{kw}_{w.stem}",
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": True,
                "probe_id": None,
            })

    # Non-target (background / silence)
    for kw in ["_background_noise_", "silence"]:
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        rng.shuffle(wav_files)
        for w in wav_files[:n // 4]:
            items.append({
                "id": f"kws_nt_{kw}_{w.stem}",
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": False,
                "probe_id": None,
            })

    # Injection probes: held-out Speech Commands clips that *say* the target word.
    # These are used as background sounds in E2.  We fix 3 probes per target keyword.
    probe_ids: list[str] = []
    for kw in KWS_TARGETS[:4]:  # use first 4 keywords as injection targets
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        probes = wav_files[-3:] if len(wav_files) >= 3 else wav_files
        for w in probes:
            pid = f"probe_{kw}_{w.stem}"
            probe_ids.append(pid)
            items.append({
                "id": pid,
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": False,   # used as background, not foreground
                "probe_id": pid,
            })

    # Write probe IDs to prereg so they're frozen
    prereg_probe = ROOT / "prereg" / "injection_probe_ids.json"
    prereg_probe.parent.mkdir(parents=True, exist_ok=True)
    if not prereg_probe.exists():
        prereg_probe.write_text(json.dumps(probe_ids, indent=2))
        log.info(f"[prereg] Frozen {len(probe_ids)} injection-probe IDs.")

    rng.shuffle(items)
    for item in items[:n]:
        if item["id"] not in done_ids:
            jsonl_append(KWS_BANK, item)
            done_ids.add(item["id"])

    prog.mark(key)
    log.info(f"[done] KWS bank: {len(jsonl_ids(KWS_BANK))} items.")


# ── Speech Accent Archive → saa.jsonl ────────────────────────────────────────
def build_saa_bank(prog: ProgressLog, n: int) -> None:
    key = "saa_bank"
    if prog.done(key):
        log.info("[skip] SAA bank already built.")
        return

    dest_dir = DATA / "speech_saa"
    meta_csv = dest_dir / "speakers_all.csv"
    rec_dir  = dest_dir / "recordings"

    if not meta_csv.exists() or not rec_dir.exists():
        log.warning(
            "[SAA] data/speech_saa/ not populated. "
            "See data/speech_saa/HOW_TO_DOWNLOAD.txt. Skipping SAA bank."
        )
        prog.mark(key)
        return

    done_ids = jsonl_ids(SAA_BANK)
    rng = random.Random(RANDOM_SEED)

    with open(meta_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng.shuffle(rows)
    added = 0
    for row in rows:
        if added >= n:
            break
        speaker = row.get("filename", row.get("speakerid", ""))
        accent  = row.get("native_language", "unknown")
        gender  = row.get("sex", "unknown")
        # recordings are typically named <speaker>.mp3
        for ext in [".mp3", ".wav"]:
            wav = rec_dir / f"{speaker}{ext}"
            if wav.exists():
                uid = f"saa_{speaker}"
                if uid in done_ids:
                    break
                jsonl_append(SAA_BANK, {
                    "id": uid,
                    "wav": str(wav.relative_to(ROOT)),
                    "transcript": "Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
                    "accent": accent,
                    "gender": gender,
                })
                done_ids.add(uid)
                added += 1
                break

    prog.mark(key)
    log.info(f"[done] SAA bank: {len(jsonl_ids(SAA_BANK))} items.")


# ── Spoken SQuAD → sqa.jsonl ──────────────────────────────────────────────────
def build_sqa_bank(prog: ProgressLog, n: int) -> None:
    key = "sqa_bank"
    if prog.done(key):
        log.info("[skip] SQA bank already built.")
        return

    SQA_BANK = ROOT / "itembanks" / "sqa.jsonl"
    done_ids = jsonl_ids(SQA_BANK)
    
    passage_dir = DATA / "speech_sqa_passages"
    question_dir = DATA / "speech_sqa_questions"
    passage_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        import edge_tts
        import asyncio
        import soundfile as sf
        import io
    except ImportError as e:
        log.warning(f"[SQA] Missing dependencies ({e}). Run: pip install datasets edge-tts soundfile. Skipping.")
        return

    log.info("[SQA] Loading AudioLLMs/spoken_squad_test...")
    import datasets
    
    async def synthesize(text, out_path):
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(str(out_path))

    added = 0
    try:
        ds = load_dataset("AudioLLMs/spoken_squad_test", split="test", streaming=True)
        ds = ds.cast_column("context", datasets.Audio(decode=False))
        log.info(f"[SQA] Fetching first {n} items from streaming dataset...")
        items = list(ds.take(n))
    except Exception as e:
        log.error(f"[SQA] Failed to load/stream dataset: {e}")
        raise e

    for idx in range(n):
        uid = f"sqa_{idx}"
        if uid in done_ids:
            added += 1
            continue
            
        try:
            item = items[idx]
        except IndexError as e:
            log.error(f"[SQA] Index {idx} out of range in streamed items (total fetched: {len(items)}): {e}")
            raise e
            
        passage_text = "" # Spoken SQuAD HF test split doesn't contain passage transcript text
        question_text = item.get("instruction", "")
        answer_text = item.get("answer", "")
        audio_data = item.get("context", {}) or {}
            
        passage_wav = passage_dir / f"{uid}.wav"
        question_wav = question_dir / f"{uid}.wav"
        
        # Save or synthesize passage audio
        if not passage_wav.exists():
            if "array" in audio_data and "sampling_rate" in audio_data:
                sf.write(str(passage_wav), audio_data["array"], audio_data["sampling_rate"])
            elif "bytes" in audio_data and audio_data["bytes"]:
                with open(passage_wav, "wb") as f:
                    f.write(audio_data["bytes"])
            elif "path" in audio_data and audio_data["path"]:
                import shutil
                shutil.copy(audio_data["path"], passage_wav)
            else:
                # Synthesize passage since no audio is provided
                try:
                    asyncio.run(synthesize(passage_text, passage_wav))
                except Exception as e:
                    log.warning(f"[SQA] Failed to synthesize fallback passage for {uid}: {e}")
                    continue
                
        # Synthesize question audio
        if not question_wav.exists():
            try:
                asyncio.run(synthesize(question_text, question_wav))
            except Exception as e:
                log.warning(f"[SQA] Failed to synthesize question for {uid}: {e}")
                continue

        jsonl_append(SQA_BANK, {
            "id": uid,
            "wav": str(passage_wav.relative_to(ROOT)),
            "question_wav": str(question_wav.relative_to(ROOT)),
            "question": question_text,
            "answer": answer_text,
            "passage_text": passage_text,
            "source": "spoken_squad",
            "accent": "american_english",
            "gender": "unknown",
        })
        done_ids.add(uid)
        added += 1

    prog.mark(key)
    log.info(f"[done] SQA bank: {len(jsonl_ids(SQA_BANK))} items.")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 — Build foreground item banks")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Build tiny banks (5 items each) for a quick smoke test.")
    args = ap.parse_args()
    n = SMOKE_N if args.smoke_test else -1  # -1 means use full defaults

    asr_n = SMOKE_N if args.smoke_test else ASR_N
    kws_n = SMOKE_N if args.smoke_test else KWS_N
    saa_n = SMOKE_N if args.smoke_test else SAA_N
    sqa_n = SMOKE_N if args.smoke_test else ASR_N # SQA gets ~100 to match ASR

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    prog = ProgressLog(PROGRESS)
    # Reset smoke keys so full run can redo them properly
    if args.smoke_test:
        for k in ["asr_bank", "kws_bank", "saa_bank", "sqa_bank"]:
            prog._data.pop(k, None)

    build_asr_bank(prog, asr_n)
    build_kws_bank(prog, kws_n)
    build_saa_bank(prog, saa_n)
    build_sqa_bank(prog, sqa_n)

    log.info("=== Stage 2 complete. ===")


if __name__ == "__main__":
    main()
