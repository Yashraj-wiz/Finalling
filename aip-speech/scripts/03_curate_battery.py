#!/usr/bin/env python3
"""
03_curate_battery.py — Stage 3: curate backgrounds, build scrambled twins,
extract descriptors, freeze pre-registration.

Produces:
  data/bg/             ~20 curated mono 16 kHz WAVs (trimmed/looped, normalised)
  data/bg_scrambled/   8 phase-scrambled twins of the speech-like backgrounds
  descriptors/battery.parquet
  prereg/prereg.json

Usage:
  python scripts/03_curate_battery.py --smoke-test   # 3 backgrounds only
  python scripts/03_curate_battery.py                # full run (resumable)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, SR, SPEECH_LUFS, BG_LUFS, SNR_GRID,
                   get_logger, ProgressLog, load_audio, loudness_normalize)

log = get_logger("03_curate_battery")
PROGRESS = ROOT / "checks" / "battery_progress.json"
BG_DIR   = DATA / "bg"
BGS_DIR  = DATA / "bg_scrambled"
BATTERY  = ROOT / "descriptors" / "battery.parquet"
PREREG   = ROOT / "prereg" / "prereg.json"

# ── Battery definition (source → id, category) ───────────────────────────────
# Each entry: (bg_id, source_glob_pattern, category)
# source_glob_pattern is relative to DATA
BATTERY_SPEC: list[tuple[str, str, str]] = [
    # ESC-50 non-speech events (~12)
    # Category IDs verified against data/esc50/meta/esc50.csv
    ("esc_rain",       "esc50/**/audio/1-*-A-10.wav",  "non_speech"),  # rain          → ID 10
    ("esc_seawave",    "esc50/**/audio/1-*-A-11.wav",  "non_speech"),  # sea_waves     → ID 11
    ("esc_engine",     "esc50/**/audio/1-*-A-44.wav",  "non_speech"),  # engine        → ID 44
    ("esc_vacuum",     "esc50/**/audio/1-*-A-36.wav",  "non_speech"),  # vacuum_cleaner→ ID 36
    ("esc_footsteps",  "esc50/**/audio/1-*-A-25.wav",  "non_speech"),  # footsteps     → ID 25
    ("esc_fire",       "esc50/**/audio/1-*-A-12.wav",  "non_speech"),  # crackling_fire→ ID 12
    ("esc_helicopter", "esc50/**/audio/1-*-A-40.wav",  "non_speech"),  # helicopter    → ID 40
    ("esc_clock",      "esc50/**/audio/1-*-A-38.wav",  "non_speech"),  # clock_tick    → ID 38
    ("esc_keyboard",   "esc50/**/audio/1-*-A-32.wav",  "non_speech"),  # keyboard_typing→ ID 32
    ("esc_dog",        "esc50/**/audio/1-*-A-0.wav",   "non_speech"),  # dog           → ID 0
    ("esc_rooster",    "esc50/**/audio/1-*-A-1.wav",   "non_speech"),  # rooster       → ID 1
    ("esc_wind",       "esc50/**/audio/1-*-A-16.wav",  "non_speech"),  # wind          → ID 16
    # MS-SNSD speech-like environments (~4)
    ("snsd_cafeteria", "ms_snsd/noise_train/CafeTeria_1.wav",     "speech_like"),
    ("snsd_restaurant","ms_snsd/noise_train/Restaurant_1.wav",    "speech_like"),
    ("snsd_square",    "ms_snsd/noise_train/Square_1.wav",        "speech_like"),
    ("snsd_office",    "ms_snsd/noise_train/Office_1.wav",        "speech_like"),
    # NOISEX-92 babble
    ("noisex_babble",  "noisex/babble.wav",                        "speech_like"),
    # MUSAN music and hubbub
    ("musan_music",    "musan/**/music/fma/music-fma-0001.wav", "music"),
    ("musan_hubbub",   "musan/**/speech/librivox/speech-librivox-0001.*", "speech_like"),
    # Stationary (MS-SNSD AirConditioner)
    ("snsd_airconditioner", "ms_snsd/noise_train/AirConditioner_1.wav", "stationary"),
]

# Speech-like subset (for scrambled twins) — must match bg_ids above
SPEECH_LIKE_IDS = [
    "snsd_cafeteria", "snsd_restaurant", "snsd_square", "snsd_office",
    "noisex_babble", "musan_hubbub", "musan_music", "snsd_airconditioner",
]

CLIP_LEN_S = 30  # seconds to trim/loop each background to


# ── audio preparation ─────────────────────────────────────────────────────────
def _prepare_bg(src_path: Path, out_path: Path) -> np.ndarray | None:
    """Load, mono, resample, trim/loop to CLIP_LEN_S, loudness-normalise, save."""
    if not src_path.exists():
        log.warning(f"  [missing] {src_path}")
        return None
    x = load_audio(src_path)
    target_len = CLIP_LEN_S * SR
    if len(x) < target_len:
        x = np.tile(x, int(np.ceil(target_len / len(x))))
    x = x[:target_len]
    x = loudness_normalize(x, BG_LUFS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), x, SR)
    return x


# ── phase scrambling ──────────────────────────────────────────────────────────
def phase_scramble(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Same magnitude spectrum + energy, no temporal structure / linguistic content.
    From §2.3 of implementation plan.
    """
    X = np.fft.rfft(x)
    mag = np.abs(X)
    rng = np.random.default_rng(seed)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=mag.shape))
    return np.fft.irfft(mag * ph, n=len(x)).astype(np.float32)


# ── descriptor extraction ─────────────────────────────────────────────────────
def extract_descriptors(x: np.ndarray, bg_id: str) -> dict:
    """
    Compute all descriptors defined in §2.4 / Table in proposal §3.
    Returns a dict suitable for battery.parquet.
    """
    import pyloudnorm as pyln
    import librosa

    desc: dict = {"bg_id": bg_id}

    # loudness (LUFS)
    meter = pyln.Meter(SR)
    try:
        desc["loudness"] = float(meter.integrated_loudness(x))
    except Exception:
        desc["loudness"] = float("nan")

    # speech_likeness — P(speech) from webrtcvad or silero VAD
    desc["speech_likeness"] = _speech_likeness(x)

    # linguistic_content — Whisper word count × mean confidence on bg alone
    desc["linguistic_content"] = _linguistic_content(x)

    # mod_2to8Hz — temporal envelope modulation energy in 2–8 Hz (syllabic rate)
    desc["mod_2to8Hz"] = _mod_energy(x, f_lo=2.0, f_hi=8.0)

    # spectral_overlap — energy fraction in 300–3400 Hz (telephone band)
    desc["spectral_overlap"] = _spectral_overlap(x)

    # harmonicity (HNR)
    desc["harmonicity"] = _harmonicity(x)

    # stationarity = 1 / mean spectral flux
    desc["stationarity"] = _stationarity(x)

    # onset_density (onsets per second)
    onsets = librosa.onset.onset_detect(y=x, sr=SR, units="time")
    desc["onset_density"] = float(len(onsets) / (len(x) / SR))

    return desc


def _speech_likeness(x: np.ndarray) -> float:
    """Fraction of 30-ms frames classified as speech by energy-VAD proxy."""
    frame_len = int(0.03 * SR)
    hop = frame_len
    frames = [x[i:i+frame_len] for i in range(0, len(x)-frame_len, hop)]
    if not frames:
        return 0.0
    # Simple energy threshold: speech frames have RMS > 1% of max RMS
    rms = np.array([float(np.sqrt(np.mean(f**2))) for f in frames])
    thresh = 0.01 * float(rms.max()) if rms.max() > 0 else 0.0
    return float(np.mean(rms > thresh))


def _linguistic_content(x: np.ndarray) -> float:
    """
    Run Whisper-base on the background alone.
    Return confidence-weighted word count (proxy for linguistic content).
    """
    try:
        import whisper
        model = whisper.load_model("base", download_root=str(ROOT / "models" / "cache" / "whisper"))
        result = model.transcribe(x, language="en", fp16=False,
                                   word_timestamps=True)
        words = []
        for seg in result.get("segments", []):
            words.extend(seg.get("words", []))
        if not words:
            return 0.0
        return float(sum(abs(w.get("probability", 0.5)) for w in words))
    except Exception as e:
        log.warning(f"  [whisper] failed: {e}")
        return 0.0


def _mod_energy(x: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Modulation energy of the temporal envelope in [f_lo, f_hi] Hz."""
    from scipy.signal import butter, sosfilt, hilbert
    # Temporal envelope via Hilbert
    env = np.abs(hilbert(x))
    # Band-pass the envelope
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=SR, output="sos")
    filtered = sosfilt(sos, env)
    total = float(np.mean(env**2)) + 1e-9
    return float(np.mean(filtered**2) / total)


def _spectral_overlap(x: np.ndarray) -> float:
    """Fraction of spectral energy in 300–3400 Hz (telephone speech band)."""
    import librosa
    S = np.abs(librosa.stft(x))
    freqs = librosa.fft_frequencies(sr=SR)
    mask = (freqs >= 300) & (freqs <= 3400)
    total = float(np.sum(S**2)) + 1e-9
    return float(np.sum(S[mask, :]**2) / total)


def _harmonicity(x: np.ndarray) -> float:
    """
    Harmonic-to-noise ratio proxy: ratio of AC to DC power in autocorrelation.
    """
    import librosa
    f0s, voiced, _ = librosa.pyin(x, fmin=50, fmax=400, sr=SR)
    voiced_f0 = f0s[voiced & ~np.isnan(f0s)] if voiced is not None else np.array([])
    return float(np.mean(voiced_f0 > 0)) if len(voiced_f0) > 0 else 0.0


def _stationarity(x: np.ndarray) -> float:
    """1 / mean spectral flux (higher = more stationary)."""
    import librosa
    S = np.abs(librosa.stft(x))
    flux = np.sum(np.diff(S, axis=1)**2, axis=0)
    mean_flux = float(np.mean(flux)) + 1e-9
    return 1.0 / mean_flux


# ── curate battery ────────────────────────────────────────────────────────────
def curate_battery(prog: ProgressLog, smoke: bool) -> None:
    spec = BATTERY_SPEC[:3] if smoke else BATTERY_SPEC
    records: list[dict] = []

    for bg_id, glob_pattern, category in spec:
        key = f"curate_{bg_id}"
        out = BG_DIR / f"{bg_id}.wav"

        if prog.done(key):
            # Already processed — reload WAV from disk and recompute descriptors
            # so the parquet is always complete even on re-runs.
            if out.exists():
                log.info(f"[reload] {bg_id} already curated — reloading descriptors from disk.")
                x = load_audio(out)
                desc = extract_descriptors(x, bg_id)
                src_candidates = sorted(DATA.glob(glob_pattern))
                desc["category"] = category
                desc["source_file"] = str(src_candidates[0].relative_to(DATA)) if src_candidates else ""
                desc["wav"] = str(out.relative_to(ROOT))
                records.append(desc)
            else:
                log.warning(f"[skip] {bg_id} marked done but WAV missing at {out}. Re-run to fix.")
            continue

        src_candidates = sorted(DATA.glob(glob_pattern))
        if not src_candidates:
            log.warning(f"[missing src] {bg_id}: no files match data/{glob_pattern}")
            continue
        src = src_candidates[0]

        log.info(f"[curate] {bg_id} ← {src.name}")
        x = _prepare_bg(src, out)
        if x is None:
            continue

        log.info(f"[descriptors] {bg_id}")
        desc = extract_descriptors(x, bg_id)
        desc["category"] = category
        desc["source_file"] = str(src.relative_to(DATA))
        desc["wav"] = str(out.relative_to(ROOT))
        records.append(desc)
        prog.mark(key)

    # Build scrambled twins for speech-like backgrounds
    build_scrambled_twins(prog, smoke)

    # Save battery.parquet — always write even if records came from cache
    if records:
        import pandas as pd
        BATTERY.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.DataFrame(records).drop_duplicates("bg_id")
        combined.to_parquet(BATTERY, index=False)
        log.info(f"[battery] {len(combined)} backgrounds → {BATTERY}")
    else:
        log.warning("[battery] No records collected — battery.parquet not written.")

    freeze_prereg(smoke)



def build_scrambled_twins(prog: ProgressLog, smoke: bool) -> None:
    ids = SPEECH_LIKE_IDS[:2] if smoke else SPEECH_LIKE_IDS
    for bg_id in ids:
        key = f"scramble_{bg_id}"
        if prog.done(key):
            log.info(f"[skip] scramble for {bg_id} already done.")
            continue
        src = BG_DIR / f"{bg_id}.wav"
        if not src.exists():
            log.warning(f"[scramble] source missing: {src}")
            continue
        x = load_audio(src)
        xs = phase_scramble(x, seed=0)
        xs = loudness_normalize(xs, BG_LUFS)
        out = BGS_DIR / f"{bg_id}_scrambled.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), xs, SR)
        log.info(f"[scramble] {bg_id} → {out.name}")
        prog.mark(key)


# ── pre-registration ──────────────────────────────────────────────────────────
def freeze_prereg(smoke: bool) -> None:
    if PREREG.exists():
        log.info("[prereg] Already frozen. Skipping (nothing changes after first model run).")
        return
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    prereg = {
        "project": "AIP-Speech",
        "battery_ids": [s[0] for s in (BATTERY_SPEC[:3] if smoke else BATTERY_SPEC)],
        "speech_like_ids": SPEECH_LIKE_IDS[:2] if smoke else SPEECH_LIKE_IDS,
        "snr_grid_db": SNR_GRID,
        "scramble_method": "phase_scramble(seed=0): irfft(|rfft(x)| * exp(i*random_phase))",
        "scramble_backgrounds": SPEECH_LIKE_IDS[:2] if smoke else SPEECH_LIKE_IDS,
        "descriptor_definitions": {
            "speech_likeness":   "Fraction of 30ms frames above energy threshold (VAD proxy)",
            "linguistic_content":"Whisper-base confidence-weighted word count on background",
            "mod_2to8Hz":        "Temporal envelope modulation energy ratio in 2-8 Hz band",
            "spectral_overlap":  "Fraction of spectral energy in 300-3400 Hz",
            "harmonicity":       "Fraction of voiced frames from pyin F0 estimation",
            "stationarity":      "1 / mean spectral flux",
            "onset_density":     "Onset events per second",
            "loudness":          "Integrated LUFS (pyloudnorm)",
        },
        "significance_rule": "Paired permutation test p<0.05 (Bonferroni-corrected across backgrounds); "
                             "semantic claims require real>scrambled; energetic claims require SNR-monotone.",
        "metric_definitions": {
            "DWER":  "WER(noisy) - WER(clean), per item",
            "FAR":   "False-alarm rate on non-target KWS clips",
            "BIR":   "Fraction of inserted tokens matching background's own transcript vocabulary",
            "TIR":   "FAR specifically on injection-probe backgrounds",
            "DRI":   "(max_g DWER_g - min_g DWER_g) / mean DWER",
            "RER":   "effect_with_instruction / effect_without",
        },
        "smoke_mode": smoke,
    }
    PREREG.write_text(json.dumps(prereg, indent=2))
    log.info(f"[prereg] Frozen → {PREREG}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 — Curate battery, scrambles, descriptors")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Process only 3 backgrounds to verify the pipeline.")
    args = ap.parse_args()
    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    prog = ProgressLog(PROGRESS)
    curate_battery(prog, smoke=args.smoke_test)
    log.info("=== Stage 3 complete. ===")


if __name__ == "__main__":
    main()
