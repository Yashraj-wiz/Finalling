# CLAUDE.md — AIP-Speech Progress Log

> **Project:** AIP-Speech — Acoustic Interference Profiling for Speech-LLMs
> **Target venue:** IMPACT-SPEECH @ EMNLP 2026 (4-page short, 15 Jul 2026 AoE)
> **Compute:** Single 16 GB GPU, inference-only, ~10 days

---

## Stage Status

| Stage | Script | Status | Notes |
|-------|--------|--------|-------|
| 0 | `setup.sh` | ✅ written | Install deps via venv + pip |
| 1 | `scripts/01_download.py` | ✅ written | Download ESC-50, DEMAND, MUSAN, NOISEX, LibriSpeech, SpeechCommands, SAA; resumable |
| 2 | `scripts/02_build_banks.py` | ✅ written | Build ASR/KWS/SAA item banks; CV streamed |
| 3 | `scripts/03_curate_battery.py` | ✅ written | Curate ~20 backgrounds, scrambled twins, all descriptors, freeze prereg |
| 4 | `scripts/04_manipulation_checks.py` | ✅ written | bg_presence, wer_constancy, scramble_validity, determinism; materialise inspection sample |
| 5 | `scripts/05_inference.py` | ✅ written | All 5 model adapters; one-at-a-time GPU; row-by-row JSONL; resumable |
| 6 | `scoring/score_all.py` | ✅ written | WER/CER/FAR/Miss/BIR/TIR/DRI/RER scoring for all experiments |
| 7 | `scoring/analyse.py` | ✅ written | C-PROFILE OLS regression + bootstrap CI + VIF; C-FAIR permutation test; C-INJECT |
| 8 | `scoring/figures.py` | ✅ written | Figs 1–3 (main paper) + Figs A1–A5 (appendix) as PDF+PNG |

---

## Commands

### Environment setup
```bash
bash setup.sh
```

### Stage 1 — Download (smoke test)
```bash
python scripts/01_download.py --smoke-test
python scripts/01_download.py          # full run (resumable)
```

### Stage 2 — Build item banks (smoke test)
```bash
python scripts/02_build_banks.py --smoke-test
python scripts/02_build_banks.py
```

### Stage 3 — Curate battery (smoke test)
```bash
python scripts/03_curate_battery.py --smoke-test
python scripts/03_curate_battery.py
```

### Stage 4 — Manipulation checks (smoke test)
```bash
python scripts/04_manipulation_checks.py --smoke-test
python scripts/04_manipulation_checks.py
```

### Stage 5 — Inference (smoke test)
```bash
python scripts/05_inference.py --model qwen25_omni_3b --task asr --smoke-test
python scripts/05_inference.py --model qwen25_omni_3b --task asr   # one model/task
python scripts/05_inference.py --model all --task all               # full sweep
```

### Stage 6 — Scoring
```bash
python scoring/score_all.py --smoke-test
python scoring/score_all.py
```

### Stage 7 — Analysis
```bash
python scoring/analyse.py --smoke-test
python scoring/analyse.py
```

### Stage 8 — Figures
```bash
python scoring/figures.py
```

---

## Change Log

| Date | What |
|------|------|
| 2026-06-23 | **Full scaffold written.** All 8 stages scripted from proposal + implementation plan. `utils.py`, `mixing/mix.py`, `scripts/{01–05}`, `scoring/{score_all,analyse,figures}.py`, `setup.sh`, `.gitignore`, `CLAUDE.md`, `results.md` |

---

## Known Issues / Decisions

- All data, models, and caches are stored **inside `aip-speech/`** (never system-wide).
- `HF_HOME`, `TORCH_HOME` env vars are set in each script to `./models/cache`.
- Stimulus set is **never persisted** — use `mixing/mix.py::materialize()` to regenerate any clip.
- Common Voice is **streamed** (not downloaded) to stay under 3 GB.
- Day-6 gate: if E1 shows no descriptor law and no disparity signal, pivot to robustness-null framing.
