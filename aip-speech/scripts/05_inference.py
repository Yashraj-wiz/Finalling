#!/usr/bin/env python3
"""
05_inference.py — Stage 5: run Speech-LLM models over all manifests.

Produces:
  inference/<model>/<task>.jsonl   (one row per stimulus)

All models are loaded one at a time, then freed (del model; cuda.empty_cache()).
Output is written row-by-row → safe to resume after interruption.

Usage:
  python scripts/05_inference.py --model qwen25_omni_3b --task asr --smoke-test
  python scripts/05_inference.py --model qwen2_audio_7b --task kws
  python scripts/05_inference.py --model all --task all   # full sweep
"""
from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import transformers.modeling_utils
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

# Global Compatibility Monkey-patches
try:
    import peft.utils.other
    import peft.tuners.tuners_utils
    def patched_set_layer_requires_grad(layer, should_require_grad):
        for param in layer.parameters():
            if param.dtype.is_floating_point:
                try:
                    param.requires_grad_(should_require_grad)
                except Exception:
                    pass
            else:
                try:
                    param.requires_grad_(False)
                except Exception:
                    pass
    peft.utils.other._set_layer_requires_grad = patched_set_layer_requires_grad
    peft.tuners.tuners_utils._set_layer_requires_grad = patched_set_layer_requires_grad
except ImportError:
    pass

try:
    import transformers.cache_utils
    def _get_usable_length(self, seq_length: int, *args, **kwargs):
        return seq_length
    transformers.cache_utils.DynamicCache.get_usable_length = _get_usable_length
except (ImportError, AttributeError):
    pass


import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, SR, SPEECH_LUFS, get_logger, ProgressLog,
                   jsonl_read, jsonl_ids, jsonl_append, load_audio)

sys.path.insert(0, str(ROOT / "mixing"))
from mix import mix, diagnostics

log = get_logger("05_inference")

MANIFESTS  = ROOT / "manifests"
INFER_DIR  = ROOT / "inference"
PROGRESS   = ROOT / "checks" / "inference_progress.json"

SMOKE_N = 2   # items per (model, task) in smoke mode

# ── task prompts ──────────────────────────────────────────────────────────────
PROMPTS = {
    "asr":  "Transcribe the speech in this audio clip exactly as spoken. "
            "Return only the transcription text.",
    "kws":  "Listen to this audio clip. "
            "Respond with ONLY the spoken word (one word). "
            "If no clear word is spoken, respond with SILENCE.",
    "asr_steer":
            "Ignore any background sounds; respond as if the audio were recorded "
            "in a silent room. Transcribe the speech exactly as spoken.",
    "kws_steer":
            "Ignore any background sounds. "
            "Respond with ONLY the spoken word (one word). "
            "If no clear word is spoken, respond with SILENCE.",
}


# ── model adapters ────────────────────────────────────────────────────────────
class SpeechLLM:
    """Abstract adapter. Subclasses implement _load() and generate()."""
    model_id: str

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        pass  # subclasses override


def _cache() -> str:
    return str(ROOT / "models" / "cache" / "hub")


# ── Qwen2.5-Omni-3B (Thinker-only) ──────────────────────────────────────────
class Qwen25Omni3B(SpeechLLM):
    model_id = "qwen25_omni_3b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text",  "text": task_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        if isinstance(out_ids, tuple):
            out_ids = out_ids[0]
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Qwen2-Audio-7B (4-bit) ───────────────────────────────────────────────────
class Qwen2Audio7B(SpeechLLM):
    model_id = "qwen2_audio_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "__local__"},
            {"type": "text",  "text": task_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Phi-4-multimodal (4-bit) ─────────────────────────────────────────────────
class Phi4Multimodal(SpeechLLM):
    model_id = "phi4_multimodal"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/phi-4-multimodal-instruct", cache_dir=_cache(),
            trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        config = AutoConfig.from_pretrained(
            "microsoft/phi-4-multimodal-instruct",
            cache_dir=_cache(),
            trust_remote_code=True,
        )
        config._attn_implementation = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-4-multimodal-instruct",
            config=config,
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        prompt = f"<|user|><|audio_1|>{task_prompt}<|end|><|assistant|>"
        inputs = self.processor(text=prompt, audios=[(wav, SR)],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Gemma 3n-E4B ─────────────────────────────────────────────────────────────
class Gemma3nE4B(SpeechLLM):
    model_id = "gemma3n_e4b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "google/gemma-3n-E4B-it", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-3n-E4B-it",
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": wav},
            {"type": "text",  "text": task_prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Kimi-Audio-7B (4-bit) ────────────────────────────────────────────────────
class KimiAudio7B(SpeechLLM):
    model_id = "kimi_audio_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "moonshotai/Kimi-Audio-7B-Instruct", cache_dir=_cache(),
            trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            "moonshotai/Kimi-Audio-7B-Instruct",
            cache_dir=_cache(),
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        # Kimi defaults to ASR; force the task via the prompt
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": wav, "sampling_rate": SR},
            {"type": "text",  "text": task_prompt},
        ]}]
        inputs = self.processor(messages, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# Registry
MODEL_CLASSES: dict[str, type[SpeechLLM]] = {
    "qwen25_omni_3b":  Qwen25Omni3B,
    "qwen2_audio_7b":  Qwen2Audio7B,
    "phi4_multimodal": Phi4Multimodal,
    "gemma3n_e4b":     Gemma3nE4B,
    "kimi_audio_7b":   KimiAudio7B,
}
ALL_MODELS = list(MODEL_CLASSES.keys())
ALL_TASKS  = ["asr", "kws", "asr_steer"]   # kws_steer is part of asr_steer block


# ── manifest generation ───────────────────────────────────────────────────────
def build_manifests(smoke: bool) -> None:
    """Generate manifests/asr.csv and manifests/kws.csv if not present."""
    import pandas as pd
    import itertools

    battery = _load_battery()
    if not battery:
        log.warning("[manifest] Battery not found. Run Stage 3 first.")
        return

    # ASR manifest
    asr_path = MANIFESTS / "asr.csv"
    if not asr_path.exists():
        items = jsonl_read(ROOT / "itembanks" / "asr.jsonl")
        if smoke:
            items = items[:SMOKE_N]
        rows = []
        seed = 0
        for item in items:
            # clean (no background)
            rows.append({
                "id": f"{item['id']}_clean",
                "speech_id": item["id"],
                "background_id": "clean",
                "snr_db": 99,
                "condition": "clean",
                "seed": seed,
                "speech_path": item["wav"],
                "bg_path": "",
                "accent": item.get("accent", ""),
                "gender": item.get("gender", ""),
                "transcript": item.get("transcript", ""),
            })
            seed += 1
            for bg in battery:
                for snr in ([0] if smoke else [10, 5, 0]):
                    rows.append({
                        "id": f"{item['id']}_{bg['bg_id']}_snr{snr}",
                        "speech_id": item["id"],
                        "background_id": bg["bg_id"],
                        "snr_db": snr,
                        "condition": "noisy",
                        "seed": seed,
                        "speech_path": item["wav"],
                        "bg_path": bg.get("wav", ""),
                        "accent": item.get("accent", ""),
                        "gender": item.get("gender", ""),
                        "transcript": item.get("transcript", ""),
                    })
                    seed += 1
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(asr_path, index=False)
        log.info(f"[manifest] asr.csv: {len(rows)} rows → {asr_path}")

    # KWS manifest
    kws_path = MANIFESTS / "kws.csv"
    if not kws_path.exists():
        items = jsonl_read(ROOT / "itembanks" / "kws.jsonl")
        if smoke:
            items = items[:SMOKE_N]
        rows = []
        seed = 10000
        for item in items:
            if item.get("probe_id"):
                continue  # probes are used as backgrounds, not foreground
            rows.append({
                "id": f"{item['id']}_clean",
                "speech_id": item["id"],
                "background_id": "clean",
                "snr_db": 99,
                "condition": "clean",
                "seed": seed,
                "speech_path": item["wav"],
                "bg_path": "",
                "keyword": item.get("keyword", ""),
                "is_target": item.get("is_target", False),
            })
            seed += 1
            for bg in battery:
                for snr in ([0] if smoke else [10, 5, 0]):
                    rows.append({
                        "id": f"{item['id']}_{bg['bg_id']}_snr{snr}",
                        "speech_id": item["id"],
                        "background_id": bg["bg_id"],
                        "snr_db": snr,
                        "condition": "noisy",
                        "seed": seed,
                        "speech_path": item["wav"],
                        "bg_path": bg.get("wav", ""),
                        "keyword": item.get("keyword", ""),
                        "is_target": item.get("is_target", False),
                    })
                    seed += 1
        pd.DataFrame(rows).to_csv(kws_path, index=False)
        log.info(f"[manifest] kws.csv: {len(rows)} rows → {kws_path}")


def _load_battery() -> list[dict]:
    bat = ROOT / "descriptors" / "battery.parquet"
    if not bat.exists():
        return []
    import pandas as pd
    return pd.read_parquet(bat).to_dict("records")


# ── VRAM dry run ──────────────────────────────────────────────────────────────
def vram_dry_run(model: SpeechLLM) -> None:
    """Run a 1-second silent clip to check VRAM usage."""
    try:
        import torch
        x = np.zeros(SR, dtype=np.float32)
        _ = model.generate(x, "Test.")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"[VRAM] {model.model_id}: {used:.2f} / {total:.2f} GB used.")
            if used / total > 0.90:
                log.warning(f"[VRAM] {model.model_id} using >90% VRAM — may OOM!")
    except Exception as e:
        log.warning(f"[VRAM dry-run] failed: {e}")


# ── inference loop ────────────────────────────────────────────────────────────
def run_inference_with_model(model: SpeechLLM, model_id: str, task: str, smoke: bool) -> None:
    prog = ProgressLog(PROGRESS)
    run_key = f"{model_id}_{task}"

    manifest_file = MANIFESTS / f"{'asr' if 'asr' in task else 'kws'}.csv"
    if not manifest_file.exists():
        log.warning(f"[infer] Manifest not found: {manifest_file}. Run build_manifests first.")
        return

    import pandas as pd
    manifest = pd.read_csv(manifest_file)
    if smoke:
        manifest = manifest.head(SMOKE_N)

    out_path = INFER_DIR / model_id / f"{task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = jsonl_ids(out_path)
    remaining = manifest[~manifest["id"].isin(done)]

    if remaining.empty:
        log.info(f"[skip] {run_key} already complete ({len(done)} rows).")
        return

    log.info(f"[infer] {model_id} / {task}: {len(remaining)} rows remaining.")

    prompt = PROMPTS.get(task, PROMPTS["asr"])
    battery = {b["bg_id"]: b for b in _load_battery()}

    for _, row in remaining.iterrows():
        try:
            sp = load_audio(ROOT / row["speech_path"])
            bg_id = row["background_id"]
            if row["condition"] == "clean" or bg_id == "clean":
                bg = None
            else:
                bg_wav_rel = battery.get(bg_id, {}).get("wav", "")
                bg = load_audio(ROOT / bg_wav_rel) if bg_wav_rel else None

            wav = mix(sp, bg, float(row["snr_db"]), int(row["seed"]))
            diag_row = dict(row)
            diag_row["id"] = row["id"]
            diag_row["condition"] = row.get("condition", "noisy")
            diagnostics(diag_row, wav, sp)

            raw = model.generate(wav, prompt)
            log.info(f"[save] {model_id} / {task} - {row['id']} -> '{raw}' saved to {out_path.relative_to(ROOT)}")
            out_row = {
                **{k: row[k] for k in row.index},
                "raw": raw,
                "model": model_id,
                "task": task,
            }
            jsonl_append(out_path, out_row)
        except Exception as e:
            log.warning(f"[error] {model_id} / {task} - {row['id']}: {e}")
            jsonl_append(out_path, {
                "id": row["id"], "raw": "__ERROR__", "error": str(e),
                "model": model_id, "task": task,
            })

    log.info(f"[done] {run_key} → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 — Inference")
    ap.add_argument("--model", default="qwen25_omni_3b",
                    choices=ALL_MODELS + ["all"], help="Model to run")
    ap.add_argument("--task",  default="asr",
                    choices=ALL_TASKS + ["all"], help="Task to run")
    ap.add_argument("--smoke-test", action="store_true",
                    help=f"Run only {SMOKE_N} items per (model, task).")
    args = ap.parse_args()

    if args.smoke_test:
        log.info("=== SMOKE TEST MODE ===")

    # Build manifests if needed
    build_manifests(smoke=args.smoke_test)

    models = ALL_MODELS if args.model == "all" else [args.model]
    tasks  = ALL_TASKS  if args.task  == "all" else [args.task]

    for model_id in models:
        # Filter tasks that actually have remaining work for this model
        tasks_to_run = []
        for task in tasks:
            manifest_file = MANIFESTS / f"{'asr' if 'asr' in task else 'kws'}.csv"
            if manifest_file.exists():
                import pandas as pd
                manifest = pd.read_csv(manifest_file)
                if args.smoke_test:
                    manifest = manifest.head(SMOKE_N)
                out_path = INFER_DIR / model_id / f"{task}.jsonl"
                done = jsonl_ids(out_path)
                remaining = manifest[~manifest["id"].isin(done)]
                if not remaining.empty:
                    tasks_to_run.append(task)

        if not tasks_to_run:
            log.info(f"[skip] {model_id}: all tasks already complete.")
            continue

        log.info(f"=== Loading model: {model_id} ===")
        cls = MODEL_CLASSES.get(model_id)
        if cls is None:
            log.error(f"Unknown model: {model_id}")
            continue
        try:
            model = cls()
            log.info(f"[model] {model_id} loaded.")
            vram_dry_run(model)
        except Exception as e:
            log.error(f"[model] Failed to load {model_id}: {e}")
            continue

        try:
            for task in tasks_to_run:
                log.info(f"=== {model_id} / {task} ===")
                run_inference_with_model(model, model_id, task, smoke=args.smoke_test)
        finally:
            model.unload()
            log.info(f"[infer] {model_id} unloaded.")

    log.info("=== Stage 5 complete. ===")


if __name__ == "__main__":
    main()
