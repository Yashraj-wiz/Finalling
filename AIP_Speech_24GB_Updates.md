# AIP-Speech — Update Note for 24 GB VRAM

*Companion to `AIP_Speech_Proposal.md` (16 GB · 10-Day edition). This note lists only what changes when the budget goes from 16 GB → 24 GB. The science, datasets, hypotheses, experiments, analysis, and storage plan are all unchanged — the upgrade touches **precision, roster, batching, and wall-clock**, nothing on the data side.*

*Model facts below are current as of June 2026; re-confirm exact checkpoint sizes / licenses on each model card before freezing the roster (Day 1).*

---

## 0. TL;DR — what the extra 8 GB actually buys you

**The headline win is not "bigger models." It's higher precision on the models you already chose.**

1. **Drop 4-bit; run the 7–8B roster at 8-bit (or fp16 for a couple).** This is the single most defensible change. Your study measures *small WER/FAR deltas under interference*; 4-bit NF4 quantization perturbs exactly that signal, and a reviewer can ask "is your degradation an artifact of 4-bit?" At 24 GB you can pre-empt that question entirely.
2. **The full dense `Qwen2.5-Omni-7B` (Thinker) — which the 16 GB edition explicitly avoided — now fits.** You can run the 3B *and* 7B of the same family and read a clean within-family capacity effect.
3. **Promote stretch models into core** (Audio Flamingo 3, MiniCPM-o 2.6, Step-Audio 2) and **add two current-SOTA architecture anchors** (Granite-Speech-3.3-8B, Canary-Qwen-2.5B) to strengthen the encoder-coupling axis — *if* time, not VRAM, allows (see §6 scope warning).
4. **Real batching for ASR generation** (batch 4–8 on 8-bit 7B vs batch 1–2 before) roughly halves the core wall-clock, buying back time for E4 + a precision-sensitivity check.

**What 24 GB does *not* buy you:** the new 30B-class omni models are not a clean fit. `Qwen3-Omni-30B-A3B` is a Mixture-of-Experts with 30B *total* params (≈3B active) — MoE still holds **all** experts in VRAM, so even at 4-bit AWQ it's ~15–17 GB of weights before the encoder and KV cache, leaving little to no headroom at 24 GB. Treat it as a risky stretch (§3), not a core model. Don't redesign the study around it.

---

## 1. Precision: the one change that matters most

| | 16 GB edition | 24 GB edition |
|---|---|---|
| 7–8B weights | **4-bit (NF4), mandatory** | **8-bit default; fp16 for 1–2 key models** |
| Quantization as confound | present, unaddressed | removed (8-bit ≈ fp16 behavior) + explicitly tested |
| New control | — | **Precision-sensitivity check** (see below) |

**New manipulation check — add to §3.1 / Table A2.** On one representative model and a fixed ~40-clip slice, run **4-bit vs 8-bit vs fp16** at 0 dB and report ΔWER between precisions. Expected result: the *interference* effect is stable across precision while absolute WER shifts slightly. This is a cheap (~120 extra passes) one-time check that converts "why not just use 4-bit?" from a weakness into a stated, controlled result. It directly hardens H1–H3 against the "it's a quantization artifact" objection.

**Practical rule:** default everyone to **8-bit** (bnb int8 / AWQ-8 / GPTQ-8). Reserve **fp16** for your two most quantization-sensitive or most-cited models (e.g., Qwen2-Audio-7B, Phi-4-multimodal) so headline numbers are at full precision. Only fall back to 4-bit if a specific checkpoint won't fit even at 8-bit (rare at 24 GB).

---

## 2. VRAM budget at 24 GB (engineering estimates — dry-run per model on Day 3)

Approximate footprint for a dense 7–8B understanding-only audio-LM (weights + ~0.6–1.5B audio encoder + KV/activations for ~30 s clips). Use as a planning guide, not a guarantee — keep the proposal's existing `del model; torch.cuda.empty_cache()` between models and the per-model VRAM dry-run.

| Precision | 7–8B weights | + encoder + KV @ batch 1 | Practical batch on 24 GB | Verdict |
|---|---|---|---|---|
| **fp16** | ~14–16 GB | ~18–20 GB | 1–2 | fits, little headroom |
| **8-bit** | ~7–8 GB | ~11–13 GB | 4–8 | **comfortable — default** |
| **4-bit** | ~4–5 GB | ~7–9 GB | 8–16+ | only if needed |
| **30B-A3B MoE @ 4-bit AWQ** | ~15–17 GB | ~19–23 GB | 1 (no headroom) | **risky stretch only** |

Smaller models (Qwen2.5-Omni-3B, Gemma 3n-E4B, Canary-Qwen-2.5B) run fp16 with room to spare.

---

## 3. Revised model roster (replaces §5)

Core stays at **five models** — capacity for more is now budget-bound, not VRAM-bound, and adapter-debugging is the real time sink (see §6). Changes vs the 16 GB table are marked **[changed]** / **[new]**.

| Model | 24 GB config | Coupling | Role |
|---|---|---|---|
| **Qwen2.5-Omni-3B** | fp16 (Thinker only) | end-to-end | small-family anchor |
| **Qwen2.5-Omni-7B** **[new — was excluded at 16 GB]** | 8-bit (Thinker only) | end-to-end | within-family capacity contrast vs the 3B |
| **Qwen2-Audio-7B** **[changed]** | **fp16** (was 4-bit) | end-to-end (Whisper-init) | acoustically-sensitive exemplar, full precision |
| **Phi-4-multimodal** **[changed]** | **8-bit or fp16** (was 4-bit/8-bit) | LoRA adapter | strong LLM-decoder ASR, intermediate coupling |
| Gemma 3n-E4B | fp16, ≤30 s clips | USM + Gemma | edge anchor (keep or swap for a current-SOTA anchor below) |

**Optional roster strengtheners (add only if Day-6 gate is green and time allows):**

| Model | 24 GB config | Why add it |
|---|---|---|
| **Granite-Speech-3.3-8B** (IBM, Apache-2.0) **[new]** | 8-bit (~8 GB) or fp16 | Current top-tier **Conformer-encoder + LLM-decoder** ASR; a clean, citable point on the *high-coupling* end of your encoder-coupling axis. |
| **Canary-Qwen-2.5B** (NVIDIA) **[new]** | fp16 (trivial fit) | Tops the Open ASR leaderboard (SALM: FastConformer + Qwen LLM decoder). Distinct architecture; near-SOTA ASR reference. Note: ASR-centric — confirm it does your KWS task or use it ASR-only. |
| **Audio Flamingo 3** **[promoted from stretch]** | 8-bit | Open LALM with long-audio understanding; was already in your swap list. |
| MiniCPM-o 2.6 / Step-Audio 2 **[promoted from stretch]** | 8-bit | Extended-set diversity, now affordable at higher precision. |

**Risky stretch (attempt only with a Day-3 VRAM dry-run, batch 1, no Talker, with a fallback ready):**

| Model | Config | Caveat |
|---|---|---|
| **Qwen3-Omni-30B-A3B** (Instruct or Thinking, Apache-2.0) | 4-bit AWQ, batch 1 | MoE → all 30B weights resident; ~19–23 GB total. May OOM during generation on long clips. High-value *if* it fits (open-source SOTA on most audio benchmarks), but do not make it load-bearing. |
| **Qwen3.5-Omni-Light** (if released in a small dense tier) | check size first | Newest family (Mar 2026); only viable if the "Light" variant is genuinely small. Verify params on the model card before committing. |

**Closed-API reference points** (`GPT-4o-Audio` / `Gemini-Flash`, ~$10) are unchanged and orthogonal to local VRAM.

**Practicalities (replaces the §5 "16 GB practicalities" paragraph):** batch 4–8 for 8-bit 7B ASR generation (was 1–2); KWS clips batch much larger and dominate throughput. Still dry-run a VRAM check per model before committing. The full-omni `Qwen2.5-Omni-7B` is now usable in Thinker-only mode — you no longer need the 3B-only workaround, though keeping both gives you the capacity contrast for free.

---

## 4. Compute budget update (revises §8)

Pass counts are **unchanged** (the grid didn't grow) — only wall-clock improves, because you can batch.

- **16 GB assumption:** ~1.5–2 s/clip un-batched for 7B → ~82 k × 1.75 s ≈ **40 GPU-h core**.
- **24 GB, 8-bit + batch 4–8 ASR:** effective throughput roughly **1.5–2× faster** on the ASR-generation bottleneck; KWS already batched. Core ≈ **~20–30 GPU-h ≈ 1.5–2 inference days** at ~16 h/day.

**What to do with the reclaimed ~1–1.5 days (in priority order):**
1. The **precision-sensitivity check** (§1) — highest marginal credibility, lowest cost.
2. **E4 steerability** moved from stretch → core (it's a clean, reviewer-friendly result).
3. *Then* optionally one extended model from §3 — but read §6 first.

Keep the existing **fallback knobs** (4 models · SNR {+10, 0} · 16 backgrounds · skip E4) — they still apply if a day slips.

---

## 5. Section-by-section edit checklist (literal changes to the main proposal)

| Location | Change |
|---|---|
| **Title + §6 header** | "16 GB VRAM" → "24 GB VRAM"; "16 GB-feasible models" → "24 GB-feasible models". |
| **Metadata → Compute row** | "single 16 GB GPU" → "single 24 GB GPU"; note **8-bit default, fp16 for key models** (was "4-bit weights where needed"). |
| **§3.1 / Table A2** | Add the **precision-sensitivity check** (4-bit vs 8-bit vs fp16) to manipulation checks. |
| **§5 (Models)** | Replace table + practicalities with §3 above; add `Qwen2.5-Omni-7B`; change 4-bit → 8-bit/fp16. |
| **§8 (Compute)** | Update wall-clock arithmetic per §4; pass counts stay. Replace "Smaller batches than 24 GB" sentence with the 8-bit batching numbers. |
| **§9 (Timeline)** | Day 3: "5 model adapters (16 GB, 4-bit…)" → "8-bit/fp16; run precision-sensitivity check on the inspection slice". Day 6/7: E4 promoted to core if ahead. |
| **§10 (Risks)** | See §7 below — relax the 4-bit-OOM risk, add the MoE-fit risk. |
| **Figure A5** (encoder-coupling view) | Now better populated if you add Granite-Speech (high coupling) — worth a sentence; the architecture axis is stronger with the new anchors. |

---

## 6. Scope discipline — read before adding models

The 16 GB edition's own Day-3 risk was **adapter debugging**, not VRAM. That hasn't changed. **Every model you add costs adapter-wrangling, defensive output-parsing, and a VRAM dry-run — and risks the 10-day timeline far more than precision does.**

Recommended discipline:
- **Spend the upgrade on precision and batching first** (free credibility + free time). These are pure wins.
- Treat the §3 "strengtheners" as **Day-6-gate-gated**: add at most one or two, and only if E1 is already producing the descriptor law on the core five.
- Do **not** build the study around `Qwen3-Omni-30B`. If it loads, run it as a bonus high-capacity datapoint; if it OOMs, drop it with zero impact on the minimum viable result.

The minimum viable result (E1 + C-PROFILE regression + E3 ecological cut) is unchanged and still runs on the core five.

---

## 7. Updated risks (revises §10)

| Risk | 24 GB status |
|---|---|
| ~~7B model OOMs on 16 GB~~ | **Largely retired.** 7–8B fit at 8-bit/fp16 with headroom; 4-bit only as a last resort. Keep per-model VRAM dry-run. |
| **Quantization confound** *(new, now addressable)* | Run 8-bit/fp16; add the precision-sensitivity check (§1) so degradation can't be dismissed as a 4-bit artifact. |
| **30B MoE doesn't fit** *(new)* | Expected and acceptable. It's a stretch only; 4-bit AWQ + batch 1 + Talker disabled, with a ready fallback. Never load-bearing. |
| **Scope creep from "free" VRAM** *(new)* | Cap core at five models; gate additions behind Day 6. Adapter debugging — not memory — is the timeline risk. |
| Time/compute overrun | *Reduced.* Batching cuts core wall-clock ~1.5–2×; fallback knobs and Day-6 gate unchanged. |
| Storage | **Unchanged** (< 3 GB; data side untouched by the VRAM upgrade). |

---

## 8. One-line summary

> Spend the extra 8 GB on **precision (8-bit/fp16) and batching**, not on chasing 30B MoE models: it makes your degradation measurements quantization-robust, adds a clean precision control, lets the full dense Qwen2.5-Omni-7B and current-SOTA anchors (Granite-Speech, Canary-Qwen) into the roster, and buys back ~a day of wall-clock — all without touching the data design or the 10-day plan.
