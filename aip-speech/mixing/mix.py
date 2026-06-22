"""
mixing/mix.py — on-the-fly stimulus mixer.

Three inspection mechanisms (§3.1 of proposal):
  1. materialize(row)        — regenerate any clip bit-for-bit, on demand.
  2. Persisted inspection sample → checks/inspection/  (written by Stage 3).
  3. diagnostics()           — numeric trace per stimulus → checks/mix_diagnostics.csv.

Nothing here writes audio except materialize() and the inspection sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow running from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import SR, SPEECH_LUFS, BG_LUFS, loudness_normalize, ROOT, csv_append

DIAG_CSV = ROOT / "checks" / "mix_diagnostics.csv"
_DIAG_FIELDS = [
    "stimulus_id", "speech_id", "background_id", "snr_db", "condition",
    "achieved_snr_db", "lufs", "clipped", "mix_wer",
]


# ─────────────────────────────────────────────────────────────────────────────
def mix(
    speech: np.ndarray,
    bg: np.ndarray | None,
    snr_db: float,
    seed: int,
    speech_lufs: float = SPEECH_LUFS,
) -> np.ndarray:
    """
    Mix speech + background at the requested SNR.
    Both are loudness-normalised before mixing.
    Returns float32 at SR. Clips to [-1,1] only if needed.
    """
    s = loudness_normalize(speech, speech_lufs)
    if bg is None:
        return s

    # Loop bg to at least speech length
    if len(bg) < len(s):
        bg = np.tile(bg, int(np.ceil(len(s) / len(bg))))

    rng = np.random.default_rng(seed)
    off = int(rng.integers(0, max(1, len(bg) - len(s))))
    b = bg[off : off + len(s)]

    ps = float(np.mean(s ** 2)) + 1e-9
    pb = float(np.mean(b ** 2)) + 1e-9
    b = b * np.sqrt(ps / (pb * (10 ** (snr_db / 10))))

    y = s + b
    peak = float(np.max(np.abs(y)))
    return (y / peak if peak > 1.0 else y).astype(np.float32)


def _achieved_snr(y: np.ndarray, s: np.ndarray) -> float:
    """Compute achieved SNR from mix y and clean speech s."""
    noise = y - s
    ps = float(np.mean(s ** 2)) + 1e-9
    pn = float(np.mean(noise ** 2)) + 1e-9
    return 10.0 * np.log10(ps / pn)


# ─────────────────────────────────────────────────────────────────────────────
def materialize(row: dict, out_wav: Path) -> None:
    """
    Regenerate the *exact* waveform for a manifest row and write it to disk.
    row must have: speech_path, bg_path (or None), snr_db, seed, condition.
    """
    import soundfile as sf
    sp = _load(row["speech_path"])
    bg = None if row.get("condition") == "clean" else _load(row["bg_path"])
    wav = mix(sp, bg, row["snr_db"], row["seed"])
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, SR)


def _load(path: str | Path) -> np.ndarray:
    from utils import load_audio
    return load_audio(path)


# ─────────────────────────────────────────────────────────────────────────────
def diagnostics(row: dict, y: np.ndarray, s_norm: np.ndarray) -> dict:
    """
    Compute numeric diagnostics for one stimulus without saving audio.
    Appends to DIAG_CSV and returns the dict.
    `mix_wer` is None unless explicitly computed (slow — inspection sample only).
    """
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    achieved = _achieved_snr(y, s_norm) if row.get("condition") != "clean" else float("nan")
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = float("nan")

    d = {
        "stimulus_id":   row.get("id", ""),
        "speech_id":     row.get("speech_id", ""),
        "background_id": row.get("background_id", ""),
        "snr_db":        row.get("snr_db", ""),
        "condition":     row.get("condition", ""),
        "achieved_snr_db": round(achieved, 3),
        "lufs":          round(lufs, 3),
        "clipped":       bool(float(np.max(np.abs(y))) >= 0.999),
        "mix_wer":       None,
    }
    csv_append(DIAG_CSV, d, fieldnames=_DIAG_FIELDS)
    return d
