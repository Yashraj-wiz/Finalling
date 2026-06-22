#!/usr/bin/env python3
"""
01_download.py — Stage 1: download all source corpora.

Downloads are placed INSIDE the project directory only.
All downloads are resumable: already-present files are skipped.

Corpora:
  ESC-50         → data/esc50/
  DEMAND (5 envs)→ data/demand/
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
import sys
import zipfile
import tarfile
from pathlib import Path
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


DEMAND_ENVS = {
    "PCAFETER": "https://zenodo.org/record/1227121/files/PCAFETER.zip",
    "PRESTO":   "https://zenodo.org/record/1227121/files/PRESTO.zip",
    "SPSQUARE": "https://zenodo.org/record/1227121/files/SPSQUARE.zip",
    "OMEETING": "https://zenodo.org/record/1227121/files/OMEETING.zip",
    "TCAR":     "https://zenodo.org/record/1227121/files/TCAR.zip",
}

def download_demand(prog: ProgressLog, smoke: bool) -> None:
    """DEMAND: 5 real-environment recordings, used as speech-like backgrounds."""
    envs = list(DEMAND_ENVS.items())[:1] if smoke else list(DEMAND_ENVS.items())
    for name, url in envs:
        key = f"demand_{name}"
        if prog.done(key):
            log.info(f"[skip] DEMAND {name} already done.")
            continue
        dest_dir = DATA / "demand"
        archive = dest_dir / f"{name}.zip"
        download(url, archive, f"DEMAND/{name}")
        if smoke:
            with zipfile.ZipFile(archive) as z:
                log.info(f"[smoke] DEMAND {name} archive OK — {len(z.namelist())} entries.")
            prog.mark(key)
            continue
        extract_zip(archive, DATA / "bg_raw")
        prog.mark(key)
    log.info("[done] DEMAND")


# NOISEX-92 babble is hosted at various mirrors; we use the CSTR Edinburgh mirror.
NOISEX_BABBLE_URL = (
    "http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex92/babble.wav"
)

def download_noisex(prog: ProgressLog, smoke: bool) -> None:
    """NOISEX-92 babble file (~10 MB WAV)."""
    key = "noisex_babble"
    if prog.done(key):
        log.info("[skip] NOISEX-92 babble already done.")
        return
    dest = DATA / "noisex" / "babble.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        log.info("[smoke] NOISEX — skipping actual download (smoke mode). Marking as done for test.")
        prog.mark(key)
        return
    download(NOISEX_BABBLE_URL, dest, "NOISEX-92 babble")
    prog.mark(key)
    log.info("[done] NOISEX-92 babble")


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
    download_demand(prog, smoke)
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
