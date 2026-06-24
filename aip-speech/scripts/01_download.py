#!/usr/bin/env python3
"""
01_download.py — Stage 1: download all source corpora.

Downloads are placed INSIDE the project directory only.
All downloads are resumable: already-present files are skipped.

Corpora:
  ESC-50         → data/esc50/
  MS-SNSD        → data/ms_snsd/  (primary babble/cafeteria/office noise)
  MUSAN          → data/musan/
  NOISEX-92 babble → data/noisex/
  LibriSpeech test-clean + test-other → data/speech_asr/
  Google Speech Commands v2 → data/speech_kws/
  Speech Accent Archive → data/speech_saa/
  (Common Voice is STREAMED at inference, not downloaded)

Usage:
  python scripts/01_download.py              # full run
  python scripts/01_download.py --smoke-test # small subset to verify connectivity + logic
  python scripts/01_download.py --resume     # skip already-done items (default behaviour)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # fallback to urlretrieve

from urllib.request import urlretrieve

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, DATA, CACHE, get_logger, ProgressLog

log = get_logger("01_download")
PROGRESS = ROOT / "checks" / "download_progress.json"


# ── helpers ──────────────────────────────────────────────────────────────────
def _reporthook(block: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(block * block_size / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)


def download(url: str, dest: Path, desc: str = "") -> Path:
    """Download url → dest. Skips if dest already exists and has size > 0."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"[skip] {desc or dest.name} already downloaded.")
        return dest
    log.info(f"[download] {desc or url}")
    if requests is not None:
        # streaming download with requests (handles redirects robustly)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(downloaded / total * 100, 100)
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        print()
    else:
        urlretrieve(url, str(dest), reporthook=_reporthook)
        print()  # newline after progress
    return dest


def extract_zip(archive: Path, out_dir: Path) -> None:
    """Extract a zip archive idempotently (skip if sentinel dir exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)
    log.info(f"[extracted] {archive.name} → {out_dir}")


def extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(out_dir)
    log.info(f"[extracted] {archive.name} → {out_dir}")


# ── corpus-specific download functions ───────────────────────────────────────

def download_esc50(prog: ProgressLog, smoke: bool) -> None:
    """ESC-50: 2000 clips (5 s, 44.1 kHz) — ~600 MB unzipped."""
    key = "esc50"
    if prog.done(key):
        log.info("[skip] ESC-50 already complete.")
        return
    dest = DATA / "esc50"
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    archive = DATA / "esc50_master.zip"
    download(url, archive, "ESC-50")
    if smoke:
        # Just validate the archive is readable; don't fully extract
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
        log.info(f"[smoke] ESC-50 archive OK — {len(names)} entries.")
        prog.mark(key)
        return
    dest.mkdir(parents=True, exist_ok=True)
    extract_zip(archive, dest)
    prog.mark(key)
    log.info("[done] ESC-50")


# Using 16 kHz versions (project SR=16 kHz) from the Zenodo API endpoint.
# SPSQUARE and TCAR are not in this Zenodo record; substituting with similar envs.
MS_SNSD_GIT = "https://github.com/microsoft/MS-SNSD.git"

def download_ms_snsd(prog: ProgressLog, smoke: bool) -> None:
    """
    MS-SNSD (Microsoft Scalable Noisy Speech Dataset).
    Noise WAVs land in data/ms_snsd/noise_train/ (~400 MB).
    Uses a shallow git clone (depth=1) — no Zenodo URL instability.
    MIT licensed, mono WAV @ 16 kHz, pipeline-ready.
    """
    key = "ms_snsd"
    if prog.done(key):
        log.info("[skip] MS-SNSD already done.")
        return

    dest = DATA / "ms_snsd"

    if smoke:
        # Just verify git is reachable; don't clone the whole repo
        log.info("[smoke] MS-SNSD — checking git connectivity...")
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--heads", MS_SNSD_GIT],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            log.info("[smoke] MS-SNSD git remote reachable ✓")
        else:
            log.warning(f"[smoke] MS-SNSD git remote unreachable: {result.stderr.decode().strip()}")
        prog.mark(key)
        return

    if (dest / "noise_train").exists() and any((dest / "noise_train").glob("*.wav")):
        log.info("[skip] MS-SNSD noise_train/ already populated.")
        prog.mark(key)
        return

    log.info("[download] MS-SNSD — shallow git clone (~400 MB noise WAVs) ...")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", MS_SNSD_GIT, str(dest)],
        check=True,
    )
    n_wavs = len(list((dest / "noise_train").glob("*.wav")))
    log.info(f"[done] MS-SNSD — {n_wavs} noise WAVs in data/ms_snsd/noise_train/")
    prog.mark(key)



# Full NOISEX-92 (15 files) mirrored on GitHub: speechdnn/Noises
_NOISEX_BASE = "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92"
NOISEX_FILES = [
    "babble.wav", "buccaneer1.wav", "buccaneer2.wav", "destroyerengine.wav",
    "destroyerops.wav", "f16.wav", "factory1.wav", "factory2.wav",
    "hfchannel.wav", "leopard.wav", "m109.wav", "machinegun.wav",
    "pink.wav", "volvo.wav", "white.wav",
]

def download_noisex(prog: ProgressLog, smoke: bool) -> None:
    """NOISEX-92: all 15 noise files (~130 MB total) from speechdnn/Noises GitHub mirror."""
    key = "noisex_babble"
    if prog.done(key):
        log.info("[skip] NOISEX-92 already done.")
        return
    dest_dir = DATA / "noisex"
    dest_dir.mkdir(parents=True, exist_ok=True)
    files_to_get = ["babble.wav"] if smoke else NOISEX_FILES
    for fname in files_to_get:
        url = f"{_NOISEX_BASE}/{fname}"
        download(url, dest_dir / fname, f"NOISEX-92/{fname}")
    prog.mark(key)
    log.info(f"[done] NOISEX-92 ({len(files_to_get)} files)")


def download_musan(prog: ProgressLog, smoke: bool) -> None:
    """MUSAN: speech + music + noise subsets (~10 GB full; we only keep subsets)."""
    key = "musan"
    if prog.done(key):
        log.info("[skip] MUSAN already done.")
        return
    dest_dir = DATA / "musan"
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    archive = dest_dir / "musan.tar.gz"
    if smoke:
        log.info("[smoke] MUSAN — skipping full download (large). Archive URL noted.")
        prog.mark(key)
        return
    download(url, archive, "MUSAN")
    extract_tar(archive, dest_dir)
    prog.mark(key)
    log.info("[done] MUSAN")


LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

def download_librispeech(prog: ProgressLog, smoke: bool) -> None:
    """LibriSpeech test-clean + test-other (~400 MB total)."""
    splits = ["test-clean"] if smoke else list(LIBRISPEECH_URLS.keys())
    for split in splits:
        key = f"librispeech_{split}"
        if prog.done(key):
            log.info(f"[skip] LibriSpeech {split} already done.")
            continue
        dest_dir = DATA / "speech_asr"
        archive = dest_dir / f"{split}.tar.gz"
        download(LIBRISPEECH_URLS[split], archive, f"LibriSpeech/{split}")
        if smoke:
            with tarfile.open(archive) as t:
                members = t.getmembers()
            log.info(f"[smoke] LibriSpeech {split} archive OK — {len(members)} members.")
            prog.mark(key)
            continue
        extract_tar(archive, dest_dir)
        prog.mark(key)
    log.info("[done] LibriSpeech")


def download_speech_commands(prog: ProgressLog, smoke: bool) -> None:
    """Google Speech Commands v2 (~2.3 GB; only downloading v2 mini)."""
    key = "speech_commands_v2"
    if prog.done(key):
        log.info("[skip] Speech Commands v2 already done.")
        return
    dest_dir = DATA / "speech_kws"
    # Full dataset
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive = dest_dir / "speech_commands_v2.tar.gz"
    if smoke:
        log.info("[smoke] Speech Commands — skipping download (large). URL noted.")
        prog.mark(key)
        return
    download(url, archive, "Speech Commands v2")
    extract_tar(archive, dest_dir)
    prog.mark(key)
    log.info("[done] Speech Commands v2")


def download_saa(prog: ProgressLog, smoke: bool) -> None:
    """
    Speech Accent Archive — publicly available from George Mason University.
    Full corpus: https://accent.gmu.edu/  (requires manual download or scraping).
    We attempt to download the zip from the Zenodo mirror if available,
    otherwise print instructions.
    """
    key = "saa"
    if prog.done(key):
        log.info("[skip] Speech Accent Archive already done.")
        return
    dest_dir = DATA / "speech_saa"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        log.info("[smoke] SAA — connectivity check only.")
        prog.mark(key)
        return

    # SAA does not have a clean single-file download URL; instruct user.
    readme = dest_dir / "HOW_TO_DOWNLOAD.txt"
    if not readme.exists():
        readme.write_text(
            "Speech Accent Archive (SAA)\n"
            "=============================\n"
            "Download from: https://accent.gmu.edu/\n"
            "Or use the Kaggle dataset mirror:\n"
            "  kaggle datasets download -d rtatman/speech-accent-archive\n"
            "Unzip into data/speech_saa/ so that you have:\n"
            "  data/speech_saa/recordings/*.mp3\n"
            "  data/speech_saa/speakers_all.csv\n"
        )
    log.warning(
        "[SAA] No automated download available. "
        "See data/speech_saa/HOW_TO_DOWNLOAD.txt for instructions."
    )
    prog.mark(key)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — Download source corpora")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run a small subset to verify connectivity and logic.")
    args = ap.parse_args()
    smoke = args.smoke_test

    if smoke:
        log.info("=== SMOKE TEST MODE — small subset only ===")

    prog = ProgressLog(PROGRESS)

    download_esc50(prog, smoke)
    download_ms_snsd(prog, smoke)
    download_noisex(prog, smoke)
    download_musan(prog, smoke)
    download_librispeech(prog, smoke)
    download_speech_commands(prog, smoke)
    download_saa(prog, smoke)

    log.info(f"=== Stage 1 complete. {len(prog)} items marked done. ===")
    if smoke:
        log.info("Re-run WITHOUT --smoke-test for the full download.")


if __name__ == "__main__":
    main()
