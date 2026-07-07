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
    "asr_steer_p1":
            "You are an automatic speech recognition (ASR) system. "
            "The recording may contain background noise, music, environmental sounds, or other speakers. "
            "Transcribe ONLY the speech spoken by the primary foreground speaker. "
            "Ignore every background sound and every background voice. "
            "Return only the transcription.",
    "asr_steer_p2":
            "Focus exclusively on the foreground speaker who is closest to the microphone. "
            "Treat every other sound, including background conversations, television, music, traffic, and environmental noises, as irrelevant interference. "
            "Do not transcribe any words that are not spoken by the foreground speaker. "
            "Return only the foreground speaker's transcript.",
    "asr_steer_p3":
            "Your task is to transcribe the intended foreground speaker only. "
            "Never include: "
            "- Background conversations "
            "- Speech from other people "
            "- Television or radio audio "
            "- Music lyrics "
            "- Announcements "
            "- Environmental sounds "
            "- Guessed or inferred words "
            "If a word is unclear because of background interference, omit it rather than guessing. "
            "Output only the transcription.",
    "asr_steer_p4":
            "You are a professional automatic speech recognition engine designed for extremely noisy real-world environments. "
            "Your objective is to produce the same transcript that would have been obtained if the recording had been captured in a completely silent room. "
            "Ignore every background sound regardless of its loudness. "
            "Only transcribe speech produced by the intended foreground speaker. "
            "Return only the transcript.",
    "asr_steer_p5":
            "You are a highly reliable automatic speech recognition (ASR) system designed to accurately transcribe speech in noisy real-world environments. "
            "The provided audio may contain background conversations, multiple speakers, music, television, radio, announcements, traffic, machinery, environmental sounds, or other acoustic interference. "
            "Your task is to identify the intended foreground speaker (the speaker closest to the microphone or the dominant primary speaker) and transcribe ONLY that speaker's speech. "
            "Ignore all background voices, overlapping conversations, music, environmental noises, sound effects, and any speech that does not belong to the intended foreground speaker. "
            "Do not transcribe words originating from background speakers, even if they are clearly audible. Do not infer, hallucinate, or guess words that are masked by interference. If a foreground word is unintelligible due to noise, omit it rather than guessing. "
            "Internally determine which speaker is the primary foreground speaker before producing the transcription, and mentally separate foreground speech from all competing sounds. "
            "Your objective is to produce the same transcript that would have been obtained if the foreground speaker had been recorded in a completely silent environment. "
            "Output only the final transcript without explanations, notes, confidence scores, or any additional text.",
    "kws_steer":
            "Ignore any background sounds. "
            "Respond with ONLY the spoken word (one word). "
            "If no clear word is spoken, respond with SILENCE.",
    "kws_steer_p1":
            "You are an automatic keyword spotting (KWS) system. "
            "The recording may contain background noise, music, environmental sounds, or other speakers. "
            "Respond with ONLY the single keyword spoken by the primary foreground speaker. "
            "Ignore every background sound and every background voice. "
            "If no foreground keyword is spoken, respond with SILENCE.",
    "kws_steer_p2":
            "Focus exclusively on the foreground speaker who is closest to the microphone. "
            "Treat every other sound, including background conversations, television, music, traffic, and environmental noises, as irrelevant interference. "
            "Respond with ONLY the single keyword spoken by the foreground speaker. "
            "If no clear foreground keyword is spoken, respond with SILENCE.",
    "kws_steer_p3":
            "Your task is to identify the single keyword spoken by the intended foreground speaker. "
            "Never respond with words from: "
            "- Background conversations "
            "- Speech from other people "
            "- Television or radio audio "
            "- Guessed or inferred words "
            "If a keyword is unclear because of background interference, respond with SILENCE. "
            "Output only the single keyword.",
    "kws_steer_p4":
            "You are a professional keyword spotting engine designed for extremely noisy real-world environments. "
            "Your objective is to detect the same keyword that would have been detected if the recording had been captured in a completely silent room. "
            "Ignore every background sound regardless of its loudness. "
            "Only respond to the keyword spoken by the intended foreground speaker. "
            "Return only the keyword, or SILENCE if none is detected.",
    "kws_steer_p5":
            "You are a highly reliable keyword spotting (KWS) system designed for noisy real-world environments. "
            "The provided audio may contain background conversations, multiple speakers, music, television, radio, announcements, traffic, machinery, environmental sounds, or other acoustic interference. "
            "Identify the intended foreground speaker and respond with ONLY the single keyword spoken by that speaker. "
            "Ignore all background voices, overlapping conversations, music, and environmental noises. "
            "If a foreground keyword is unintelligible due to noise, respond with SILENCE rather than guessing. "
            "Output only the final keyword without explanations, notes, or any additional text.",
    "saa":  "Transcribe the speech in this audio clip exactly as spoken. "
            "Return only the transcription text.",
}


# ── model adapters ────────────────────────────────────────────────────────────
# Max audio length to prevent KV-cache OOM on 16 GB GPU
MAX_AUDIO_SAMPLES = SR * 30   # 30 seconds

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


def _clip_audio(wav: np.ndarray) -> np.ndarray:
    """Truncate audio to MAX_AUDIO_SAMPLES to avoid KV-cache OOM."""
    if len(wav) > MAX_AUDIO_SAMPLES:
        log.warning(f"[clip] Truncating audio from {len(wav)/SR:.1f}s to {MAX_AUDIO_SAMPLES/SR:.0f}s")
        return wav[:MAX_AUDIO_SAMPLES]
    return wav


# ── Qwen2.5-Omni-3B (Thinker-only) ──────────────────────────────────────────
class Qwen25Omni3B(SpeechLLM):
    model_id = "qwen25_omni_3b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B", cache_dir=_cache(), trust_remote_code=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            cache_dir=_cache(),
            torch_dtype=torch.float16,
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
        wav = _clip_audio(wav)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, return_audio=False)
        if isinstance(out_ids, tuple):
            out_ids = out_ids[0]
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Qwen2.5-Omni-7B (Thinker-only, 8-bit) ────────────────────────────────────
class Qwen25Omni7B(SpeechLLM):
    model_id = "qwen25_omni_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
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
        wav = _clip_audio(wav)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, return_audio=False)
        if isinstance(out_ids, tuple):
            out_ids = out_ids[0]
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


# ── Qwen2-Audio-7B (8-bit) ───────────────────────────────────────────────────
class Qwen2Audio7B(SpeechLLM):
    model_id = "qwen2_audio_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
        wav = _clip_audio(wav)
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
        import torch, json, os
        from transformers import AutoProcessor, AutoModelForCausalLM
        from huggingface_hub import snapshot_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL_ID = "microsoft/phi-4-multimodal-instruct"

        # Download model files to cache (skips download if already cached)
        model_dir = snapshot_download(
            MODEL_ID,
            cache_dir=_cache(),
        )

        # Directly patch config.json: phi4's own __init__ reads _attn_implementation
        # from the config object which is populated from this file. The baked-in value
        # is "flash_attention_2" which phi4's code explicitly rejects.
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            cfg_json = json.load(f)
        if cfg_json.get("_attn_implementation") not in (None, "eager", "sdpa"):
            cfg_json["_attn_implementation"] = "eager"
            cfg_json.pop("_attn_implementation_autoset", None)
            with open(config_path, "w") as f:
                json.dump(cfg_json, f, indent=2)

        # Step 2.5: Patch speech_conformer_encoder.py to avoid meta tensor item() crash
        # When transformers loads the model, it creates tensors on the meta device.
        # This breaks in_length = torch.tensor(feat_in) -> int(out_length).
        # Forcing device='cpu' fixes it.
        encoder_path = os.path.join(model_dir, "speech_conformer_encoder.py")
        if os.path.exists(encoder_path):
            with open(encoder_path, "r") as f:
                encoder_code = f.read()
            if "torch.tensor(feat_in, dtype=torch.float)" in encoder_code:
                encoder_code = encoder_code.replace(
                    "torch.tensor(feat_in, dtype=torch.float)",
                    "torch.tensor(feat_in, dtype=torch.float, device='cpu')"
                )
                with open(encoder_path, "w") as f:
                    f.write(encoder_code)

        # Load using the MODEL ID so the trust_remote_code module loader 
        # correctly resolves the symlinked .py files in the cache volume.
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir=_cache(), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=_cache(),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # accelerate auto-enables this; phi4's custom
                                      # __init__ calls .item() which breaks meta tensors
        ).to(self.device)
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": f"<|audio_1|>{task_prompt}"}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        wav = _clip_audio(wav)
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
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "google/gemma-3n-E4B-it", cache_dir=_cache(), trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-3n-E4B-it",
            cache_dir=_cache(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": _clip_audio(wav)},
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
        import sys
        
        kimi_path = ROOT / "Kimi-Audio"
        if str(kimi_path) not in sys.path:
            sys.path.append(str(kimi_path))
            
        from kimia_infer.api.kimia import KimiAudio
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KimiAudio(
            model_path="moonshotai/Kimi-Audio-7B-Instruct",
            load_detokenizer=True,
        )

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import soundfile as sf
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
            
        try:
            wav = _clip_audio(wav)
            sf.write(temp_wav, wav, SR)
            
            messages = [
                {"role": "user", "message_type": "audio", "content": temp_wav},
                {"role": "user", "message_type": "text", "content": task_prompt}
            ]
            
            sampling_params = {
                "text_temperature": 0.0,
                "text_top_k": 1,
            }
            
            _, text = self.model.generate(messages, **sampling_params, output_type="text")
            return str(text).strip()
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def unload(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# Registry
MODEL_CLASSES: dict[str, type[SpeechLLM]] = {
    "qwen25_omni_3b":  Qwen25Omni3B,
    "qwen25_omni_7b":  Qwen25Omni7B,
    "qwen2_audio_7b":  Qwen2Audio7B,
    "phi4_multimodal": Phi4Multimodal,
    "gemma3n_e4b":     Gemma3nE4B,
    "kimi_audio_7b":   KimiAudio7B,
}
ALL_MODELS = list(MODEL_CLASSES.keys())
ALL_TASKS  = [
    "asr", "kws", "saa",
    "asr_steer", "asr_steer_p1", "asr_steer_p2", "asr_steer_p3", "asr_steer_p4", "asr_steer_p5",
    "kws_steer", "kws_steer_p1", "kws_steer_p2", "kws_steer_p3", "kws_steer_p4", "kws_steer_p5",
]


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

    # SAA manifest
    saa_path = MANIFESTS / "saa.csv"
    if not saa_path.exists():
        items = jsonl_read(ROOT / "itembanks" / "saa.jsonl")
        if smoke:
            items = items[:SMOKE_N]
        rows = []
        seed = 20000
        
        saa_bgs = [
            "snsd_cafeteria", "snsd_restaurant", "snsd_square", "snsd_office",
            "noisex_babble", "musan_hubbub", "musan_music", "snsd_airconditioner", # 8 speech-like
            "esc_rain", "esc_engine", "esc_dog", "esc_keyboard" # 4 non-speech
        ]
        
        for item in items:
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
                if bg["bg_id"] not in saa_bgs:
                    continue
                # only 0 dB for SAA
                for snr in [0]:
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
        pd.DataFrame(rows).to_csv(saa_path, index=False)
        log.info(f"[manifest] saa.csv: {len(rows)} rows → {saa_path}")



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

    # Resolve manifest: strip steer/pN suffixes to get base task name (asr or kws)
    manifest_name = task
    for suffix in ["_steer_p1", "_steer_p2", "_steer_p3", "_steer_p4", "_steer_p5", "_steer"]:
        if task.endswith(suffix):
            manifest_name = task[: task.rfind(suffix)]
            break
    manifest_file = MANIFESTS / f"{manifest_name}.csv"
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

    total_remaining = len(remaining)
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        try:
            sp_path_str = str(row["speech_path"]).replace("\\", "/")
            sp = load_audio(ROOT / sp_path_str)
            bg_id = row["background_id"]
            if row["condition"] == "clean" or bg_id == "clean":
                bg = None
            else:
                bg_wav_rel = battery.get(bg_id, {}).get("wav", "")
                if bg_wav_rel:
                    bg_path_str = str(bg_wav_rel).replace("\\", "/")
                    bg = load_audio(ROOT / bg_path_str)
                else:
                    bg = None

            wav = mix(sp, bg, float(row["snr_db"]), int(row["seed"]))
            diag_row = dict(row)
            diag_row["id"] = row["id"]
            diag_row["condition"] = row.get("condition", "noisy")
            diagnostics(diag_row, wav, sp)

            raw = model.generate(wav, prompt)
            log.info(f"[save] {model_id} / {task} ({i}/{total_remaining}) - {row['id']} -> '{raw}' saved to {out_path.relative_to(ROOT)}")
            out_row = {
                **{k: row[k] for k in row.index},
                "raw": raw,
                "model": model_id,
                "task": task,
            }
            jsonl_append(out_path, out_row)
        except Exception as e:
            import traceback
            log.warning(f"[error] {model_id} / {task} - {row['id']}: {e}\n{traceback.format_exc()}")
            jsonl_append(out_path, {
                "id": row["id"], "raw": "__ERROR__", "error": str(e),
                "model": model_id, "task": task,
            })
            # Clear CUDA cache after OOM so subsequent rows don't cascade-fail
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                log.info(f"[recovery] Cleared CUDA cache after OOM on {row['id']}")

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
            base_task = "kws" if task.startswith("kws") else "asr"
            manifest_file = MANIFESTS / f"{base_task}.csv"
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
            import traceback
            log.error(f"[model] Failed to load {model_id}: {e}")
            log.error(traceback.format_exc())
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
