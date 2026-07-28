# Research Proposal (16 GB VRAM · 10-Day Feasible Edition)

## What Sounds Break a Speech-LLM's Ear — and For Whom?
### A descriptor-level, fairness-aware audit of Large Audio-Language Models under controlled background interference

**AIP-Speech — Acoustic Interference Profiling for Speech-LLMs.** Inference-only · **single 16 GB GPU** · ~10 days · **< 3 GB storage** · stimuli mixed on-the-fly but **fully inspectable and reproducible** (§3).

---

### Metadata

| Field | Value |
|---|---|
| Project | AIP-Speech — Acoustic Interference Profiling for Speech-LLMs |
| Researcher | Masaru — IvLabs |
| Venue | IMPACT-SPEECH @ EMNLP 2026 (archival short paper, 4 pp ACL; long-paper extensible) |
| Deadline | 15 July 2026 (AoE) — *re-confirm on the official CFP* |
| Compute | Inference-only, **single 16 GB GPU**, no training; target **≈ 73–85 k forward passes** |
| Storage | **< 3 GB** (source corpora + manifest + JSONL outputs; no persisted stimulus set) |
| Timeline | **10 days**, Day-6 go/no-go gate, robustness-null fallback |

> **No scene-mapping anywhere.** There is no "hospital/office/classroom." Every background is a **named, public, downloadable recording** characterized by **measurable acoustic descriptors**, and every foreground dataset is stated inline with the task that uses it (§4). The only construction step is **SNR mixing per a released manifest** — standard noise-robustness practice — with **no ad-hoc synthesis** of babble (§3.2).

---

## 1. The question, in one paragraph

Holding the spoken content fixed and mixing a real background recording behind it at a controlled signal-to-noise ratio, we ask: **which measurable properties of a background sound determine how badly a Speech-LLM fails, is that failure equitable across speakers, and is it driven by the sound's acoustic characteristics or demographic variations?** We answer with two objective tasks (ASR, keyword spotting), ~20 descriptor-annotated real backgrounds, five 16 GB-feasible models, and a scoped set of controls.

## 2. Hypotheses

| ID | Hypothesis | Settled by Experiment |
|---|---|---|
| **H1 — Profile law** | Degradation is predicted by background *descriptors* beyond SNR; **speech-likeness, linguistic content, 2–8 Hz (syllabic) modulation** dominate. | **E1** |
| **H2 — Content injection** | LALMs leak background content into outputs (ASR insertions; KWS false triggers), above the determinism floor, concentrated on linguistically-loaded backgrounds. | **E1** |
| **H3 — Disparate robustness** | The noise penalty differs across accent/gender; speech-like interference widens the gap more than non-speech noise. | **E2** |
| **H4 — Implicit (stretch)** | "Ignore background sound" reduces but does not eliminate the effect. | **E3** |
| **H0 — Robustness null** | Degradation is a pure function of SNR, equitable across subgroups, no injection above the floor. | all — a clean, publishable result |

## 3. Two design decisions you asked about

### 3.1 On-the-fly mixing is **not** opaque — three ways to see every intermediate step
We do not persist millions of mixed WAVs (that is the storage blow-up). But the pipeline stays fully inspectable:

1. **Deterministic regeneration on demand.** Every stimulus is a pure function of `(speech_id, background_id, snr, condition, seed)`. A one-line `materialize(row)` call re-creates the *identical* waveform any time you want to listen to, plot, or hand-check it. Nothing is lost — it is recomputed bit-for-bit.
2. **A persisted inspection sample.** A fixed ~60-clip sample spanning tasks × backgrounds × SNRs *is* written to `checks/inspection/` for manual listening and for all manipulation checks (background-presence, WER-constancy).
3. **A per-stimulus diagnostics log.** At mix time we log, *for every stimulus*, the numbers that say "what happened" — achieved SNR, pre/post loudness (LUFS), clipping flag, measured background-presence score, Whisper WER on the mix — to `checks/mix_diagnostics.csv`, **without** saving the audio. You get a complete numeric trace of the whole run plus the ability to regenerate any clip behind any row.

So the storage win costs you nothing in observability: mix-on-the-fly for the bulk, **regenerate-any-clip-exactly** for inspection, and a full diagnostics table for monitoring.

### 3.2 No synthesized babble — real, citable noise corpora instead
You were right that summing LibriSpeech speakers into "babble" is hard to reproduce and cite. The **interference battery (the manipulated variable, shared by all experiments)** is therefore built **only from named public recordings**:

| Battery slice | Source corpus (citable, downloadable) | Why |
|---|---|---|
| **Diverse non-speech events** (~12) | **ESC-50** (50 classes, 5 s, CC) — e.g., rain, sea waves, engine, vacuum, footsteps, fire, helicopter, clock-tick, keyboard, dog, rooster, wind | spans the non-speech descriptor space |
| **Real multi-talker babble / speech-like** (~4) | **MS-SNSD** environments CafeTeria (cafeteria), Restaurant (restaurant), Square (public square), Office (meeting); **NOISEX-92** *babble* (canteen, the canonical babble) | the crucial speech-like extreme, **recorded, not synthesized** |
| **Speech / hubbub / music anchors** (~3) | **MUSAN** speech (read-speech + hubbub) and music subsets | speech-like + music-with-vocals extremes |
| **Non-speech anchor** (~1) | **MUSAN** noise / MS-SNSD AirConditioner | broadband non-speech control |

**Injection probes** (a background that *says a specific word*, for the KWS trigger test) use a **fixed, published list of held-out Google Speech Commands clip IDs** mixed at a fixed SNR — fully reproducible from the released manifest, no synthesis. The released artifact includes exact source IDs + seeds, so every stimulus is regenerable by anyone.

**What we measure about each background (descriptors).** Every battery clip carries a pre-computed vector — all measurable, none hand-labelled for "scene":

| Descriptor | Extractor | Hypothesized role |
|---|---|---|
| speech_likeness | P(speech) from a VAD/speech classifier on the background alone | ↑ harm |
| linguistic_content | confidence-weighted word count from Whisper on the background alone | ↑ injection |
| mod_2to8Hz | temporal-envelope modulation energy, 2–8 Hz (syllabic rate) | ↑ masking |
| spectral_overlap | energy fraction in 300–3400 Hz | ↑ masking |
| harmonicity, stationarity, onset_density, loudness | HNR, inverse spectral flux, onsets/s, integrated LUFS | shape vs energy controls |

---

## 4. Experiments

Each experiment lists its tasks; each task states its **foreground dataset (exact portion) × the interference battery (§3.2) at the stated SNRs**, and its **metrics (standard + any custom one we introduce)**. SNR grid is `{+10, +5, 0}` dB throughout (reference +10; 0 dB is the stress point). The clean (background-free) rendering of each foreground item is the within-item paired baseline. Decoding is greedy/deterministic, so significance comes from **paired permutation tests across items** (a one-time determinism check, not a jitter-floor protocol, sets the sanity bar).

### E1 — Interference Profiling: *which sound properties break which task?* (C-PROFILE → H1, H2)

**Task 1.1 — ASR under background.**
- **Data:** LibriSpeech test-clean + test-other (~100 utterances, balanced) **×** full interference battery **×** SNR {+10,+5,0}; plus each utterance's clean rendering.
- **Metrics (standard):** WER, CER (`jiwer`); substitution/deletion/insertion split.
- **Analysis:** mixed-effects regression of ΔWER on descriptors (§6.1). No new metric needed.

**Task 1.2 — Keyword spotting / wake-word under background.**
- **Data:** Google Speech Commands v2 (~120 clips spanning target + non-target words) **×** full battery **×** SNR.
- **Metrics (standard):** Accuracy; **False-Alarm Rate** (FAR) on non-target backgrounds; Miss rate; **custom — Trigger-Injection Rate (TIR):** FAR specifically on injection-probe backgrounds (held-out Speech Commands clips uttering the target word, mixed behind the input).

*E1 output:* the descriptor law — a ranked, signed statement of which background properties drive failure, pooled across both tasks and all models.

### E2 — Disparate Robustness: *for whom is it worse?* (C-FAIR → H3) — the IMPACT-SPEECH spine

**Task 2.1 — ASR fairness, ecological (near-free).**
- **Data:** Common Voice v17 English (streamed; accent/gender/age-labelled), ~80 utterances chosen to span ≥4 accent groups × both genders **×** full battery **×** SNR. Because demographics are *baked into the E1-style ASR run*, this is largely a **post-hoc subgroup cut**, not a separate grid.
- **Metrics:** per-subgroup ΔWER; **custom — Robustness Gap** = max_g ΔWER_g − min_g ΔWER_g; **Disparate-Robustness Index (DRI)** = gap / mean Δ.

**Task 2.2 — ASR fairness, content-controlled.**
- **Data:** Speech Accent Archive (everyone reads the *same* "Please call Stella" paragraph), ~6 accent groups × balanced gender (~90 speakers) **×** {the 8 speech-like + 4 non-speech backgrounds} at **0 dB**. Same words across accents ⇒ a clean Δ comparison free of content confounds.
- **Metrics:** content-matched per-accent ΔWER; Robustness Gap; **descriptor × subgroup interaction** test (does speech-like widen the gap more than non-speech?). Clean-baseline gap overlaid so the *added* inequity from noise is explicit.

*E2 output:* the bias finding with an explanatory account of *which interference profiles* drive inequity.

### E3 — Steerability (stretch): *can you tell it to ignore the background?* (C-STEER → H4)

**Task 3.1 — ASR + KWS with a blinding instruction.**
- **Data:** a 1/3 subset of the E1 stimuli at 0 dB, re-run with the prepended instruction *"Ignore any background sounds; respond as if the audio were recorded in a silent room."*
- **Metrics:** **custom — Residual-Effect Ratio (RER)** = effect_with_instruction / effect_without (per task/model); instruction-compliance rate.

---

## 5. Models (validated for 16 GB)

Five core models spanning the encoder-coupling spectrum, all fitting 16 GB in **understanding-only mode** (speech-generation/talker heads disabled) with **4-bit weights** where needed. Load one at a time; free the GPU (`del model; torch.cuda.empty_cache()`) before the next.

| Model | Size / config for 16 GB | Coupling | Role |
|---|---|---|---|
| **Qwen2.5-Omni-3B** | ~3B, fp16 or 4-bit (Thinker only) | end-to-end | SOTA-family, comfortably fits |
| **Qwen2-Audio-7B** | 4-bit (~5–6 GB) | end-to-end (Whisper-init) | acoustically-sensitive exemplar |
| **Phi-4-multimodal** | ~5.6B, 4-bit/8-bit | LoRA adapter | strong LLM-decoder ASR, intermediate coupling |
| **Gemma 3n-E4B** | effective ~4B, edge-optimized | USM + Gemma | user-requested; fits easily; ≤30 s clips |
| *(swap/stretch)* MiniCPM-o 2.6 / Step-Audio 2 / Audio Flamingo 3 | 4-bit | — | extended set, reduced grid if ahead |
| *(optional API ~$10)* GPT-4o-Audio / Gemini-Flash | — | closed | reference points |

**16 GB practicalities:** batch size 1–2 for 7B (ASR generation is the bottleneck); KWS clips (~1 s, 1-token outputs) batch larger and dominate throughput; **dry-run a VRAM check** per model before committing it; avoid the full-omni Qwen2.5-Omni-7B (use the 3B or Thinker-only). If a 7B model OOMs even in 4-bit, drop it for a 3B/edge alternative — the roster has slack.

---

## 6. Analysis

### 6.1 C-PROFILE (E1)
Per task: `Δ ~ SNR + speech_likeness + linguistic_content + mod_2to8Hz + spectral_overlap + stationarity + onset_density + (1|item) + (1|model) + (1|background)` (`statsmodels`/`pymer4`). Standardized coefficients + bootstrap CIs + variance-inflation; dissociate collinear descriptors using the real anchor extremes (NOISEX babble vs MUSAN non-speech). **Money plot:** Δ vs speech-likeness and vs 2–8 Hz modulation.

### 6.2 C-FAIR (E2)
Robustness Gap + DRI per task; descriptor×subgroup interaction; clean-baseline gap overlaid.

### 6.3 Decision rule
An effect is **real** iff (a) it survives a paired permutation test (p<0.05, corrected across backgrounds) and (b) for energetic claims is **not monotone in SNR**. Disparity claims additionally require a significant subgroup/interaction term.

---

## 7. Figures and tables — main paper vs appendix

A 4-page short paper holds ~3 figures + ~2 tables.

**Main paper**

| Item | Type | Shows | From |
|---|---|---|---|
| **Fig 1** | Partial-dependence, 2 panels | ΔWER vs **speech-likeness** and vs **2–8 Hz modulation**, pooled over models | E1 / H1 |
| **Fig 2** | Grouped bars | **Disparate robustness**: ΔWER by accent/gender, speech-like vs non-speech, clean gap overlaid | E2 / H3 |
| **Table 1** | Summary | model × {clean, noisy@+10, Δ, worst-background Δ, injection rate} for ASR & KWS | headline |
| **Table 2** | Regression | standardized descriptor coefficients per task | E1 |

**Appendix**

| Item | Type | Shows |
|---|---|---|
| Fig A1 | Lines | **SNR dose–response** per background cluster (monotonicity) |
| Fig A2 | Stacked bars | **Error/injection taxonomy** (sub/del/ins/injection), clean vs noisy, per model |
| Fig A3 | Bars | **Steerability** RER per model (if E3 run) |
| Fig A4 | Scatter | **Battery map**: backgrounds in descriptor space, colored by harmfulness |
| Fig A5 | Bars | Architecture view: Δ vs encoder-coupling tier (descriptive) |
| Table A1 | Full | Per-background leaderboard (Δ per task) |
| Table A2 | Checks | Manipulation checks: background-presence, Whisper WER-constancy, determinism floor |
| Table A3 | Fairness | Full subgroup × background-type Δ with CIs |
| Table A4 | Prior art | Differentiation table (Appendix A) |

If injection is the most striking result, promote Fig A2 into the main paper and demote the architecture view.

---

## 8. Compute budget (16 GB-aware arithmetic)

| Block | Calc | Passes |
|---|---|---|
| E1 ASR | 100 × 20 bg × 3 SNR × 5 models | 30,000 |
| E1 ASR clean + text-oracle | 100 × 5 × 2 | 1,000 |
| E1 KWS | 120 × 20 × 3 × 5 | 36,000 |
| E1 KWS clean | 120 × 5 | 600 |
| E2 controlled (SAA) | 90 × 12 bg × 5 models @ 0 dB | 5,400 |
| **Core subtotal** | | **≈ 73.2 k** |
| *E3 steerability (stretch)* | 220 × 8 bg × 4 models @ 0 dB | +7,040 |
| *Stretch: 2 extended models, reduced ASR+KWS* | 220 × 12 bg × 2 | +5,280 |
| **With stretches** | | **≈ 85.5 k** |

**Wall-clock on 16 GB.** Smaller batches than 24 GB → assume ~1.5–2 s/clip un-batched for 7B; ~73.2 k × 1.75 s ≈ 35 GPU-hours core ≈ **~2 inference days** at ~16 h/day; KWS batching shortens this. Comfortable inside 10 days.

**Fallback knobs if a day slips:** 4 models · SNR {+10, 0} · 16 backgrounds · skip E3 · E2 ecological-cut only.

---

## 9. Timeline (10 days)

| Day | Work | Output |
|---|---|---|
| 1 | Env; download ESC-50, MS-SNSD, MUSAN subset, NOISEX babble, LibriSpeech + Speech Commands subsets, SAA; **stream** Common Voice; build foreground item banks | sources + JSONL banks |
| 2 | Curate ~20 backgrounds; extract descriptors; **freeze prereg**; write the on-the-fly **mixer + `materialize()` + diagnostics logger** | battery + descriptors + `prereg/` + mixer |
| 3 | 5 model adapters (16 GB, 4-bit, understanding-only); manipulation checks on the inspection sample; determinism check | adapters + `checks/` |
| 4–5 | Run **E1 (ASR + KWS)**, 5 models (the bulk) | `inference/` JSONL |
| 6 | **Go/no-go gate**; run **E2** SAA + text-oracle (+ E3 if ahead) | fairness |
| 7 | Stretch (E3 / extended models) **or** start scoring | — |
| 8 | Scoring (WER/FAR/TIR); **descriptor regression**; C-FAIR cuts | `results/` |
| 9 | Figures 1–2 (+appendix); disparity stats; CIs | plots |
| 10 | Write the 4-page draft; buffer | submission draft |

**Go/no-go (Day 6):** if E1 shows neither a descriptor law beyond SNR nor a disparity signal, pivot to the **robustness-null** framing rather than padding tasks.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| 7B model OOMs on 16 GB | 4-bit + understanding-only; per-model VRAM dry-run; 3B/edge swaps in the roster; free GPU between models. |
| Compute/time overrun | On-the-fly mixing (no I/O bottleneck) + KWS batching; fallback knobs; Day-6 gate. |
| Storage | Source corpora only (< 3 GB); stream Common Voice; never persist the stimulus set. |
| Reproducibility of backgrounds | All backgrounds are named public corpora; release exact source IDs + seeds; any clip regenerable via `materialize()`. |
| "On-the-fly = opaque" | Persisted inspection sample + deterministic regeneration + per-stimulus diagnostics CSV (§3.1). |
| Not novel vs RSA-Bench | Lead with descriptor law + fairness; cite/differentiate (Appendix A). |
| Descriptor collinearity | Report VIF; dissociate with NOISEX-babble vs MUSAN-non-speech; prefer partial-dependence to raw coefficients. |
| Model quirks (speech-out) | Robust adapters; defensive parsing; Whisper-transcribe speech-out; log parse failures. |

---

## 11. Expected outcomes

- **If H1–H3 hold:** a released **descriptor-annotated battery built from real public corpora**, a lightweight harness, a **generalizable law** of which sound properties break ASR/KWS, and a **disparate-robustness** finding with an explanatory account — a concrete deployment warning for voice assistants in noisy, multi-speaker settings. On-theme for IMPACT-SPEECH.
- **If H0 holds:** a rigorously-controlled demonstration that current LALMs treat background as undifferentiated energy and degrade equitably — a trustworthy guarantee plus a reusable benchmark.

**Minimum viable result:** E1 (ASR+KWS) + the C-PROFILE regression + the E2 ecological cut.

---

## Appendix A — Prior-art differentiation

| Prior work | Does | Leaves open (our wedge) |
|---|---|---|
| **RSA-Bench** (2601.10384) | 4 hand-built scenarios, source-count K=1–4, **fixed energy (no SNR sweep)**, 6 tasks; vocal-like > mechanical; denoising paradox | scenarios re-import event→scene mapping; **no descriptor law**, **no SNR dose-response**, **no fairness** |
| **Do LLM Decoders Listen Fairly?** (2604.21276) | LLM-decoder ASR bias across accent/gender, incl. under degradation | ASR only; generic noise; **no characterization of which backgrounds widen the gap**; no KWS/injection |
| **FairLENS** (2405.13166) | disparate ASR degradation under noise (law-enforcement) | classical ASR, not LALMs; coarse noise; single task |
| **VoiceBench** (2410.17196) | voice-assistant benchmark incl. a noise track | noise is one knob; no fairness/descriptor/injection |
| **WildSpeech-Bench** (2506.21875) | natural-conversation QA + background-speech/ESC-50 augmentation | robustness sub-track; no descriptor law/fairness/injection |
| **AudioTrust** (2505.16211) | trustworthiness incl. background-conversation/env-sound scenarios | coarse buckets; no per-descriptor law, no disparity, no SNR sweep |
| **VocalBench-DF** (2510.15406) | speech-LLM robustness to disfluency/overlap | speaker-side, not environmental-background characterization or fairness |
| **Object-hallucination / distractor in LALMs** (Kuan et al. 2024; 2025) | LALMs hallucinate absent sounds / distracted in audio-QA | audio-QA reasoning, not a cross-task background-robustness *failure taxonomy* tied to descriptors + demographics |

**Verdict:** "noise hurts LALMs" is occupied. The defensible contribution is the **descriptor law + disparate robustness + content-injection tracking**, with controls (SNR sweep, text-oracle) the closest competitor omits.

## Appendix B — Glossary

- **Interference battery** — the shared, manipulated variable: ~20 real background recordings (ESC-50, MS-SNSD, MUSAN, NOISEX) each with a descriptor vector.
- **Descriptor** — a measurable property of a background (speech-likeness, linguistic content, 2–8 Hz modulation, spectral overlap, stationarity, onset density, loudness).
- **TIR** — Trigger-Injection Rate (KWS): how much background content leaks into outputs.
- **Robustness Gap / DRI** — disparity in the *noise penalty* across demographic subgroups.
- **RER** — Residual-Effect Ratio: how much of the effect survives an "ignore background" instruction.
- **materialize(row)** — regenerate the exact waveform for any manifest row, on demand, for inspection.
- **On-the-fly mixing** — generating noisy waveforms in RAM at inference from seeds, never persisting them, while staying fully inspectable via the three mechanisms in §3.1.
