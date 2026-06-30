"""
utils.py — shared helpers for AIP-Speech.
All scripts import from here to avoid duplication.
"""
from __future__ import annotations

import os, sys, json, csv, hashlib, logging
from pathlib import Path
from typing import Any

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # aip-speech/
DATA = ROOT / "data"
CACHE = ROOT / "models" / "cache"

# Redirect ALL HuggingFace / torch model downloads into the project directory.
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("TORCH_HOME", str(CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "datasets"))

SR = 16_000          # project-wide sample rate (Hz)
SPEECH_LUFS = -23.0  # speech loudness target
BG_LUFS = -23.0      # bg loudness target before SNR scaling
SNR_GRID = [10, 5, 0]

# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        # Force UTF-8 on the stream to avoid UnicodeEncodeError on Windows cp1252 consoles
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
                 if hasattr(sys.stdout, "buffer") else sys.stdout
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log

# ── JSONL helpers ─────────────────────────────────────────────────────────────
def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return [json.loads(l) for l in f if l.strip()]

def jsonl_ids(path: Path, key: str = "id") -> set:
    return {r[key] for r in jsonl_read(path) if key in r and r.get("raw") != "__ERROR__"}

# ── CSV helpers ───────────────────────────────────────────────────────────────
def csv_append(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = fieldnames or list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)

# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio(path: Path | str, sr: int = SR) -> np.ndarray:
    """Load mono float32 at project SR."""
    import soundfile as sf
    import librosa
    x, r = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if r != sr:
        x = librosa.resample(x.astype(np.float32), orig_sr=r, target_sr=sr)
    return x.astype(np.float32)

def loudness_normalize(x: np.ndarray, target_lufs: float) -> np.ndarray:
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    measured = meter.integrated_loudness(x)
    if np.isinf(measured):   # silence / near-silence
        return x
    return pyln.normalize.loudness(x, measured, target_lufs).astype(np.float32)

# ── Deterministic file-based progress tracker ─────────────────────────────────
class ProgressLog:
    """
    Tiny key-value store backed by a JSON file.
    Use to record completed IDs so a script can resume after interruption.
    """
    def __init__(self, path: Path):
        # Automatically redirect to a separate smoke progress file if running a smoke test
        # to avoid contaminating full runs.
        if "--smoke-test" in sys.argv:
            path = path.with_name(path.stem + "_smoke" + path.suffix)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = json.loads(path.read_text()) if path.exists() else {}

    def done(self, key: str) -> bool:
        return self._data.get(key) == "done"

    def mark(self, key: str) -> None:
        self._data[key] = "done"
        self.path.write_text(json.dumps(self._data, indent=2))

    def __len__(self) -> int:
        return len(self._data)

# ── Misc ──────────────────────────────────────────────────────────────────────
def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
