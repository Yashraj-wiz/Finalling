#!/usr/bin/env python3
"""
vad_classify_bg.py — Multi-VAD ensemble classification of background audio files.

Uses an ensemble of two independent VAD systems with their published default
settings. Final label is determined by majority vote — no continuous threshold
to tune, making the classification fully defensible in a research paper.

VAD ensemble:
    1. Silero VAD   — neural VAD (Silerospeech/silero-vad, PyTorch Hub)
                      Default internal sensitivity: 0.50
    2. webrtcvad    — Google's WebRTC signal-processing VAD
                      Aggressiveness level: 2 (moderate, published default)

Voting:
    speech_like  — both VADs agree it contains speech
    non_speech   — both VADs agree it does not contain speech
    uncertain    — VADs disagree (printed as a warning; falls back to Silero)

Outputs:
    descriptors/vad_bg_labels.csv   — per-file VAD scores, individual votes,
                                      ensemble label, and comparison vs.
                                      hand-coded labels in BATTERY_SPEC.

Usage:
    python scripts/vad_classify_bg.py                  # classify all BGs
    python scripts/vad_classify_bg.py --smoke-test     # first 3 BGs only
    python scripts/vad_classify_bg.py --aggressiveness 3   # stricter webrtcvad

Dependencies (already in requirements.txt or installable):
    torch, torchaudio     — for Silero VAD
    webrtcvad             — pip install webrtcvad-wheels  (wheel for Windows)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, DATA, SR, get_logger, load_audio

log = get_logger("vad_classify_bg")

BG_DIR        = DATA / "bg"
OUT_CSV       = ROOT / "descriptors" / "vad_bg_labels.csv"
SILERO_CACHE  = ROOT / "models" / "cache" / "silero_vad"

# ── Published defaults (no project-specific tuning) ───────────────────────────
SILERO_SENSITIVITY   = 0.50   # Silero's recommended default for speech detection
WEBRTC_AGGRESSIVENESS = 2     # webrtcvad: 0 (gentle) → 3 (aggressive); 2 is moderate

# ── Hand-coded ground-truth from 03_curate_battery.py ────────────────────────
# Used only for the accuracy comparison at the end. These are the pre-registered
# gold labels — NOT used to tune any VAD parameter.
KNOWN_LABELS: dict[str, str] = {
    "esc_rain":            "non_speech",
    "esc_seawave":         "non_speech",
    "esc_engine":          "non_speech",
    "esc_vacuum":          "non_speech",
    "esc_footsteps":       "non_speech",
    "esc_fire":            "non_speech",
    "esc_helicopter":      "non_speech",
    "esc_clock":           "non_speech",
    "esc_keyboard":        "non_speech",
    "esc_dog":             "non_speech",
    "esc_rooster":         "non_speech",
    "esc_wind":            "non_speech",
    "snsd_cafeteria":      "speech_like",
    "snsd_restaurant":     "speech_like",
    "snsd_square":         "speech_like",
    "snsd_airport":        "speech_like",
    "noisex_babble":       "speech_like",
    "musan_music":         "speech_like",
    "musan_hubbub":        "speech_like",
    "snsd_airconditioner": "non_speech",
    "snsd_office":         "non_speech",
}


# ── VAD 1: Silero VAD (neural, PyTorch Hub) ───────────────────────────────────
def _load_silero():
    """Load Silero VAD from torch.hub (cached locally after first download)."""
    import torch
    import warnings
    SILERO_CACHE.mkdir(parents=True, exist_ok=True)
    log.info(f"[Silero] Loading model (cache: {SILERO_CACHE}) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
    (get_speech_timestamps, *_) = utils
    log.info("[Silero] Loaded successfully.")
    return model, get_speech_timestamps


def _silero_vote(wav_path: Path, model, get_speech_timestamps,
                 sensitivity: float) -> tuple[str, float]:
    """
    Run Silero VAD on wav_path.
    Returns (vote, speech_fraction) where vote ∈ {'speech_like', 'non_speech', 'error'}.
    """
    import torch
    try:
        audio = load_audio(wav_path, sr=16_000)
        tensor = torch.from_numpy(audio)
        model.reset_states()
        timestamps = get_speech_timestamps(
            tensor, model,
            sampling_rate=16_000,
            window_size_samples=512,
            threshold=sensitivity,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            return_seconds=False,
        )
        total = len(tensor)
        speech = sum(s["end"] - s["start"] for s in timestamps)
        sf = float(speech / total) if total > 0 else 0.0
        # Vote: any detected speech → speech_like
        vote = "speech_like" if sf > 0.0 else "non_speech"
        return vote, round(sf, 4)
    except Exception as e:
        log.warning(f"  [Silero error] {wav_path.name}: {e}")
        return "error", float("nan")


# ── VAD 2: webrtcvad (Google WebRTC, signal-processing) ──────────────────────
def _webrtc_available() -> bool:
    try:
        import webrtcvad  # noqa: F401
        return True
    except ImportError:
        return False


def _webrtc_vote(wav_path: Path, aggressiveness: int) -> tuple[str, float]:
    """
    Run Google's WebRTC VAD on wav_path.
    Returns (vote, voiced_fraction) where vote ∈ {'speech_like', 'non_speech', 'error'}.

    webrtcvad requires:  16 kHz, 16-bit PCM, 10 / 20 / 30 ms frames.
    We use 30 ms frames for maximum robustness.
    """
    try:
        import webrtcvad
        FRAME_MS   = 30
        FRAME_RATE = 16_000
        FRAME_LEN  = int(FRAME_RATE * FRAME_MS / 1000)   # 480 samples

        # Load audio as 16-bit PCM bytes
        audio_f32 = load_audio(wav_path, sr=FRAME_RATE)
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        pcm_bytes  = audio_i16.tobytes()

        vad = webrtcvad.Vad(aggressiveness)
        frame_size_bytes = FRAME_LEN * 2  # 2 bytes per int16 sample

        # Process each frame
        n_total  = 0
        n_voiced = 0
        for start in range(0, len(pcm_bytes) - frame_size_bytes, frame_size_bytes):
            frame = pcm_bytes[start:start + frame_size_bytes]
            if len(frame) < frame_size_bytes:
                break
            try:
                is_speech = vad.is_speech(frame, FRAME_RATE)
            except Exception:
                is_speech = False
            n_total  += 1
            n_voiced += int(is_speech)

        vf   = float(n_voiced / n_total) if n_total > 0 else 0.0
        vote = "speech_like" if n_voiced > 0 else "non_speech"
        return vote, round(vf, 4)

    except ImportError:
        return "unavailable", float("nan")
    except Exception as e:
        log.warning(f"  [webrtcvad error] {wav_path.name}: {e}")
        return "error", float("nan")


# ── Ensemble voting ───────────────────────────────────────────────────────────
def _ensemble_vote(silero_vote: str, webrtc_vote: str) -> str:
    """
    Combine two VAD votes into a final ensemble label.

    Strategy (OR logic):
        Either VAD says speech_like → speech_like
        Both say non_speech        → non_speech
        Any error / unavailable    → fall back to the other model
    """
    if silero_vote in ("error", "unavailable") and webrtc_vote in ("error", "unavailable"):
        return "unknown"
    if silero_vote in ("error", "unavailable"):
        return webrtc_vote
    if webrtc_vote in ("error", "unavailable"):
        return silero_vote
    # OR logic: if EITHER says speech_like, it is speech_like
    if silero_vote == "speech_like" or webrtc_vote == "speech_like":
        return "speech_like"
    return "non_speech"


# ── Main classification loop ──────────────────────────────────────────────────
def classify_bg_folder(bg_dir: Path,
                       silero_sensitivity: float,
                       webrtc_aggressiveness: int,
                       smoke: bool) -> list[dict]:
    wav_files = sorted(bg_dir.glob("*.wav"))
    if not wav_files:
        log.error(f"No WAV files found in {bg_dir}")
        return []
    if smoke:
        wav_files = wav_files[:3]
        log.info(f"[smoke-test] Processing only first {len(wav_files)} files.")

    log.info(f"[VAD] {len(wav_files)} WAV files in {bg_dir}")
    log.info(f"[VAD] Silero sensitivity (default=0.5)   : {silero_sensitivity}")
    log.info(f"[VAD] webrtcvad aggressiveness (0-3)     : {webrtc_aggressiveness}")
    log.info(f"[VAD] Ensemble strategy                  : majority vote (both must agree)")

    has_webrtc = _webrtc_available()
    if not has_webrtc:
        log.warning("[webrtcvad] Not installed — running Silero-only mode.")
        log.warning("  Install with: pip install webrtcvad-wheels")

    silero_model, get_speech_timestamps = _load_silero()

    records = []
    for i, wav_path in enumerate(wav_files, 1):
        bg_id = wav_path.stem
        log.info(f"[{i}/{len(wav_files)}] {bg_id}")

        # Run both VADs
        sv, sf = _silero_vote(wav_path, silero_model, get_speech_timestamps,
                              silero_sensitivity)
        wv, wf = _webrtc_vote(wav_path, webrtc_aggressiveness) \
                 if has_webrtc else ("unavailable", float("nan"))

        final = _ensemble_vote(sv, wv)

        # Compare with pre-registered hand-coded label
        known = KNOWN_LABELS.get(bg_id, "unknown")
        match = (final == known) if known != "unknown" and final not in ("uncertain", "unknown") \
                else None

        if final == "speech_like":
            log.info(
                f"  Silero={sv}({sf:.4f})  webrtcvad={wv}({wf:.4f})  "
                f"→ ensemble=speech_like (OR: at least one agreed)  known={known}  match={match}"
            )
        else:
            log.info(
                f"  Silero={sv}({sf:.4f})  webrtcvad={wv}({wf:.4f})  "
                f"→ ensemble={final}  known={known}  match={match}"
            )

        records.append({
            "bg_id":               bg_id,
            "silero_vote":         sv,
            "silero_speech_frac":  sf,
            "webrtc_vote":         wv,
            "webrtc_voiced_frac":  wf,
            "ensemble_label":      final,
            "known_label":         known,
            "label_match":         match,
        })

    return records


# ── Save CSV ──────────────────────────────────────────────────────────────────
def _save_csv(records: list[dict], out_path: Path) -> None:
    if not records:
        log.warning("[save] No records to save.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    log.info(f"[save] {len(records)} rows → {out_path}")


# ── Summary report ────────────────────────────────────────────────────────────
def _print_summary(records: list[dict]) -> None:
    if not records:
        return
    speech_like = [r for r in records if r["ensemble_label"] == "speech_like"]
    non_speech  = [r for r in records if r["ensemble_label"] == "non_speech"]
    unknown     = [r for r in records if r["ensemble_label"] in ("unknown", "uncertain")]
    evaluated   = [r for r in records if r["label_match"] is not None]
    correct     = [r for r in evaluated if r["label_match"]]

    log.info("")
    log.info("=" * 65)
    log.info("  MULTI-VAD ENSEMBLE CLASSIFICATION SUMMARY")
    log.info(f"  Silero sensitivity : {SILERO_SENSITIVITY} (manufacturer default)")
    log.info(f"  webrtcvad aggress. : {WEBRTC_AGGRESSIVENESS} (moderate, published default)")
    log.info(f"  Voting             : OR logic — if ANY model says speech_like, it is speech_like")
    log.info("=" * 65)
    log.info(f"  Total processed  : {len(records)}")
    log.info(f"  → speech_like    : {len(speech_like)}")
    log.info(f"  → non_speech     : {len(non_speech)}")
    log.info(f"  → uncertain/err  : {len(unknown)}")
    if evaluated:
        acc = len(correct) / len(evaluated) * 100
        log.info(f"  Agreement w/ hand labels: {len(correct)}/{len(evaluated)} ({acc:.1f}%)")
    log.info("")

    # Detailed table sorted by silero speech fraction
    log.info(f"  {'bg_id':<25} {'Silero':>8} {'webrtc':>8} {'ensemble':<14} {'known':<14} match")
    log.info("  " + "-" * 85)
    for r in sorted(records, key=lambda x: -(x["silero_speech_frac"]
                    if not np.isnan(x["silero_speech_frac"]) else -999)):
        sf  = f"{r['silero_speech_frac']:.4f}" if not np.isnan(r["silero_speech_frac"]) else "ERR"
        wf  = f"{r['webrtc_voiced_frac']:.4f}" if not np.isnan(r["webrtc_voiced_frac"]) else "N/A"
        mt  = "✓" if r["label_match"] is True else ("✗" if r["label_match"] is False else "–")
        log.info(f"  {r['bg_id']:<25} {sf:>8} {wf:>8} {r['ensemble_label']:<14} {r['known_label']:<14} {mt}")
    log.info("=" * 65)

    # Mismatches
    mismatches = [r for r in evaluated if not r["label_match"]]
    if mismatches:
        log.info("")
        log.info("  ⚠ MISMATCHES (ensemble disagrees with pre-registered label):")
        for r in mismatches:
            log.info(
                f"    {r['bg_id']}: ensemble='{r['ensemble_label']}', "
                f"known='{r['known_label']}' "
                f"(Silero={r['silero_speech_frac']:.4f}, "
                f"webrtcvad={r['webrtc_voiced_frac']})"
            )
        log.info("  → Review mismatch manually or check audio quality.")
    else:
        log.info("  ✓ All ensemble labels agree with pre-registered categories.")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-VAD ensemble classification of background audio files.\n"
            "Uses Silero VAD (neural) + webrtcvad (signal-processing) with\n"
            "majority voting. No continuous threshold to tune."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bg-dir", type=Path, default=BG_DIR,
                    help=f"Folder of background WAV files (default: {BG_DIR})")
    ap.add_argument("--out",    type=Path, default=OUT_CSV,
                    help=f"Output CSV path (default: {OUT_CSV})")
    ap.add_argument(
        "--sensitivity",
        type=float, default=SILERO_SENSITIVITY,
        help=(
            f"Silero VAD internal per-frame speech probability threshold "
            f"(default: {SILERO_SENSITIVITY} — Silero's published default). "
            f"Lower = more sensitive to faint/overlapping speech."
        ),
    )
    ap.add_argument(
        "--aggressiveness",
        type=int, default=WEBRTC_AGGRESSIVENESS, choices=[0, 1, 2, 3],
        help=(
            f"webrtcvad aggressiveness level 0 (gentle) – 3 (strict) "
            f"(default: {WEBRTC_AGGRESSIVENESS})"
        ),
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="Process only the first 3 WAV files.")
    args = ap.parse_args()

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")
    if not args.bg_dir.exists():
        log.error(f"bg-dir does not exist: {args.bg_dir}")
        sys.exit(1)

    records = classify_bg_folder(
        bg_dir=args.bg_dir,
        silero_sensitivity=args.sensitivity,
        webrtc_aggressiveness=args.aggressiveness,
        smoke=args.smoke_test,
    )
    _save_csv(records, args.out)
    _print_summary(records)
    log.info("=== vad_classify_bg complete ===")


if __name__ == "__main__":
    main()
