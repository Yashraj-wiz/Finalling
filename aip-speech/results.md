# results.md — AIP-Speech Experimental Results

> Updated automatically by `scoring/score_all.py` and `scoring/analyse.py`.

---

## Manipulation Checks (`checks/`)

| Check | Status | Notes |
|-------|--------|-------|
| Background presence | ⬜ pending | |
| WER constancy | ⬜ pending | |

| Determinism floor | ⬜ pending | |

---

## E1 — Interference Profiling

### T1.1 ASR (LibriSpeech)

| Model | Clean WER | SNR+10 WER | SNR+5 WER | SNR0 WER | Worst-bg ΔWER |
|-------|-----------|------------|-----------|----------|---------------|
| Qwen2.5-Omni-3B | — | — | — | — | — |
| Qwen2-Audio-7B | — | — | — | — | — |
| Phi-4-multimodal | — | — | — | — | — |
| Gemma-3n-E4B | — | — | — | — | — |

### T1.2 KWS (Google Speech Commands v2)

| Model | Clean Acc | FAR | Miss Rate | Worst-bg ΔACC |
|-------|-----------|-----|-----------|----------------|
| Qwen2.5-Omni-3B | — | — | — | — |
| Qwen2-Audio-7B | — | — | — | — |
| Phi-4-multimodal | — | — | — | — |
| Gemma-3n-E4B | — | — | — | — |

### T1.3 SQA (Spoken SQuAD)

| Model | Clean EM | Clean F1 | Clean Hallucination | Noisy EM | Noisy F1 | Noisy Hallucination |
|-------|----------|----------|---------------------|----------|----------|---------------------|
| Qwen2.5-Omni-3B | 0.690 | 0.701 | 0.350 | 0.655 | 0.682 | 0.350 |
| Qwen2-Audio-7B | 0.340 | 0.393 | 0.710 | 0.338 | 0.395 | 0.714 |
| Qwen2.5-Omni-7B | 0.100 | 0.163 | 1.000 | 0.078 | 0.132 | 1.000 |
| Phi-4-multimodal | failed | failed | failed | failed | failed | failed |
| Gemma-3n-E4B | failed | failed | failed | failed | failed | failed |

---

## E2 — Semantic vs Energetic

### T2.1 ASR: semantic gap @ 0 dB

| Background | ΔWER | Gap (semantic) |
|------------|------|----------------|
| PCAFETER | — | — |
| PRESTO | — | — |
| SPSQUARE | — | — |
| OMEETING | — | — |
| NOISEX-babble | — | — |

### T2.2 KWS: trigger injection

| Background | TIR | TIR (generic) |
|------------|------------|-----------------|---------------|
| Injection probes | — | — | — |

---

## E3 — Disparate Robustness

### T3.1 Ecological (Common Voice, accent/gender)

| Subgroup | Clean WER | ΔWER speech-like | ΔWER non-speech | Robustness Gap | DRI |
|----------|-----------|------------------|-----------------|----------------|-----|
| — | — | — | — | — | — |

### T3.2 Controlled (Speech Accent Archive)

| Accent Group | Clean WER | ΔWER @ 0 dB | Speech-like ΔWER | Non-speech ΔWER |
|--------------|-----------|-------------|------------------|-----------------|
| — | — | — | — | — |

---

## E4 — Steerability (stretch)

| Model | Effect w/o instruction | Effect w/ instruction | RER |
|-------|------------------------|----------------------|-----|
| — | — | — | — |

---

## C-PROFILE Regression (E1)

| Descriptor | ASR coeff (std) | KWS coeff (std) | p-value | VIF |
|------------|-----------------|-----------------|---------|-----|
| speech_likeness | — | — | — | — |
| linguistic_content | — | — | — | — |
| mod_2to8Hz | — | — | — | — |
| spectral_overlap | — | — | — | — |
| stationarity | — | — | — | — |
| onset_density | — | — | — | — |
| loudness | — | — | — | — |
| SNR | — | — | — | — |

---

## Day-6 Go/No-Go Decision

- [ ] Descriptor law found beyond SNR?
- [ ] Disparity signal found?
- [ ] Decision: **proceed** / **pivot to robustness-null**
