import modal
import subprocess
import os
import sys

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({"FORCE_BUILD": "v8"})
    .apt_install("git", "build-essential", "ffmpeg")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1")
    .pip_install(
        "timm", "transformers==4.48.2", "accelerate", "bitsandbytes", 
        "soundfile", "librosa", "pyloudnorm", "numpy", "scipy", 
        "openai-whisper", "jiwer", "pandas", "pyarrow", "backoff", 
        "datasets", "huggingface_hub", "statsmodels", "matplotlib", 
        "seaborn", "tqdm", "requests", "peft==0.11.1", "loguru", "omegaconf", 
        "conformer", "diffusers", "torchdyn", "decord", "blobfile", 
        "deepspeed", "easydict", "fire", "hyperpyyaml", "immutabledict", 
        "sacrebleu", "jsonlines", "validators", "sty", "colorama", 
        "ujson", "cairosvg", "wget", "gdown", "sentencepiece", "edge-tts"
    )
    .run_commands("CC=gcc CXX=g++ pip install flash-attn --no-build-isolation")
)

LOCAL_AIP_SPEECH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Core repository code (exclude data/results)
image = image.add_local_dir(
    LOCAL_AIP_SPEECH_DIR,
    remote_path="/workspace",
    ignore=[
        ".venv", ".git", "inference", "inference_1", "inference_backup",
        "models", "data", "data_subset", "checks", "__pycache__", "results"
    ]
)

# Mount battery descriptors
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "descriptors"),
    remote_path="/workspace/descriptors"
)

# Mount background noise wavs
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "data", "bg"),
    remote_path="/workspace/data/bg"
)

# Mount itembanks (including kws_steer.jsonl)
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "itembanks"),
    remote_path="/workspace/itembanks"
)

# Mount KWS data directory
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "data", "speech_kws"),
    remote_path="/workspace/data/speech_kws"
)

# Mount manifests
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "manifests"),
    remote_path="/workspace/manifests"
)

app = modal.App("aip-kws-steer-pipeline")

cache_volume = modal.Volume.from_name("aip-models-cache", create_if_missing=True)
kws_steer_inference_nfs = modal.NetworkFileSystem.from_name("aip-kws-steer-nfs", create_if_missing=True)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}) if os.environ.get("HF_TOKEN") else None
secrets = [hf_secret] if hf_secret else []

@app.function(
    image=image,
    gpu="L4", 
    timeout=86400, 
    volumes={
        "/workspace/models/cache/hub": cache_volume,
    },
    network_file_systems={
        "/workspace/kws_steer_output": kws_steer_inference_nfs,
    },
    secrets=secrets,
)
def run_kws_steer_inference(model: str = "all", smoke_test: bool = False):
    import time
    import shutil
    os.chdir("/workspace")
    
    # Symlink results and inference directories to output volume
    os.makedirs("/workspace/kws_steer_output/inference", exist_ok=True)
    os.makedirs("/workspace/kws_steer_output/results", exist_ok=True)
    os.makedirs("/workspace/kws_steer_output/manifests", exist_ok=True)
    
    if not os.path.exists("/workspace/inference"):
        os.symlink("/workspace/kws_steer_output/inference", "/workspace/inference")
    if not os.path.exists("/workspace/inference_1"):
        os.symlink("/workspace/kws_steer_output/inference", "/workspace/inference_1")
    if not os.path.exists("/workspace/results"):
        os.symlink("/workspace/kws_steer_output/results", "/workspace/results")
    
    # Remove stale kws_steer.csv on volume so 05_inference builds full 6100-item manifest from itembanks
    for p in ["/workspace/manifests/kws_steer.csv", "/workspace/kws_steer_output/manifests/kws_steer.csv"]:
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
    if not os.path.exists("/workspace/manifests"):
        os.symlink("/workspace/kws_steer_output/manifests", "/workspace/manifests")

    # Run Inference for KWS STEERABILITY task ONLY
    print("\n--- RUNNING KWS STEERABILITY INFERENCE ---")
    cmd_infer = ["python", "scripts/05_inference.py", "--model", model, "--task", "kws_steer_p5"]
    if smoke_test:
        cmd_infer.append("--smoke-test")
        
    proc2 = subprocess.Popen(cmd_infer)
    proc2.wait()
    if proc2.returncode != 0:
        raise RuntimeError("Failed KWS STEERABILITY inference")
        
    # Score KWS STEERABILITY task remotely
    print("\n--- SCORING KWS STEERABILITY METRICS ---")
    cmd_score = ["python", "scripts/06_score_metrics.py"]
    if smoke_test:
        cmd_score.append("--smoke-test")
        
    proc3 = subprocess.run(cmd_score, capture_output=False)
    if proc3.returncode != 0:
        raise RuntimeError("Failed scoring metrics")
        
    print("\nKWS STEERABILITY INFERENCE & EVALUATION COMPLETED SUCCESSFULLY!")

@app.local_entrypoint()
def main(model: str = "all", smoke_test: bool = False):
    import time
    import threading
    
    local_inference_dir = os.path.join(LOCAL_AIP_SPEECH_DIR, "inference_1")
    local_sync_all_dir = os.path.join(LOCAL_AIP_SPEECH_DIR, "inference_sync_all_1784759190")
    local_results_dir = os.path.join(LOCAL_AIP_SPEECH_DIR, "results")
    
    print(f"Starting KWS STEERABILITY Inference on Modal. model={model}, smoke_test={smoke_test}")
    
    def sync_volume():
        print(f"Starting periodic volume sync (every 5 mins)...")
        os.makedirs(local_inference_dir, exist_ok=True)
        os.makedirs(local_sync_all_dir, exist_ok=True)
        os.makedirs(local_results_dir, exist_ok=True)
        while True:
            time.sleep(300)
            try:
                temp_sync = os.path.join(LOCAL_AIP_SPEECH_DIR, "temp_kws_steer_sync")
                os.makedirs(temp_sync, exist_ok=True)
                subprocess.run(["modal", "nfs", "get", "aip-kws-steer-nfs", "inference", temp_sync, "--force"], check=False)
                
                import shutil
                if os.path.exists(temp_sync):
                    shutil.copytree(temp_sync, local_inference_dir, dirs_exist_ok=True)
                    shutil.copytree(temp_sync, local_sync_all_dir, dirs_exist_ok=True)
                
                shutil.rmtree(temp_sync, ignore_errors=True)
                print("[Local Sync] Data successfully pulled from Modal NFS.")
            except Exception as e:
                print(f"[Local Sync Warning] Sync failed: {e}")

    sync_thread = threading.Thread(target=sync_volume, daemon=True)
    sync_thread.start()
    
    try:
        run_kws_steer_inference.remote(model, smoke_test)
    finally:
        print("\nRun finished. Executing final sync of inference and results...")
        temp_sync = os.path.join(LOCAL_AIP_SPEECH_DIR, "temp_kws_steer_sync")
        os.makedirs(temp_sync, exist_ok=True)
        subprocess.run(["modal", "nfs", "get", "aip-kws-steer-nfs", "inference", temp_sync, "--force"], check=False)
        import shutil
        if os.path.exists(temp_sync):
            shutil.copytree(temp_sync, local_inference_dir, dirs_exist_ok=True)
            shutil.copytree(temp_sync, local_sync_all_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_sync, ignore_errors=True)
        print("[Final Sync] Complete!")
