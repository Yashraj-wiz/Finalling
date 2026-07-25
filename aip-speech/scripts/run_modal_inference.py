import modal
import subprocess
import os

# Define the environment with PyTorch and required dependencies from requirements.txt
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({"FORCE_BUILD": "phi4-fix-v3"})
    .apt_install("git", "build-essential", "ffmpeg")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1")
    .pip_install(
        "timm",
        "transformers==4.48.2",
        "accelerate",
        "bitsandbytes",
        "soundfile",
        "librosa",
        "pyloudnorm",
        "numpy",
        "scipy",
        "openai-whisper",
        "jiwer",
        "pandas",
        "pyarrow",
        "backoff",
        "datasets",
        "huggingface_hub",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "tqdm",
        "requests",
        "peft==0.11.1",
        "loguru",
        "omegaconf",
        "conformer",
        "diffusers",
        "torchdyn",
        "decord",
        "blobfile",
        "deepspeed",
        "easydict",
        "fire",
        "hyperpyyaml",
        "immutabledict",
        "sacrebleu",
        "jsonlines",
        "validators",
        "sty",
        "colorama",
        "ujson",
        "cairosvg",
        "wget",
        "gdown",
        "sentencepiece"
    )
    .run_commands("CC=gcc CXX=g++ pip install flash-attn --no-build-isolation")
)

# Define the local mount path
LOCAL_AIP_SPEECH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

image = image.add_local_dir(
    LOCAL_AIP_SPEECH_DIR,
    remote_path="/workspace",
    ignore=[
        ".venv",
        ".git",
        "inference",
        "inference_backup",
        "models",
        "data",
        "data_subset",
        "checks",
        "__pycache__"
    ]
)

# Mount only the curated 49MB data subset required for the active manifests
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "data_subset", "data"),
    remote_path="/workspace/data"
)

app = modal.App("aip-speech-inference")

# Define persistent volumes
# One for caching model weights so they are not downloaded every run
cache_volume = modal.Volume.from_name("aip-models-cache", create_if_missing=True)
# One for storing inference results to be synced locally
inference_volume = modal.Volume.from_name("aip-inference-out", create_if_missing=True)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}) if os.environ.get("HF_TOKEN") else None
secrets = [hf_secret] if hf_secret else []

@app.function(
    image=image,
    gpu="L4", 
    timeout=86400, # 24 hours
    volumes={
        "/workspace/models/cache/hub": cache_volume,
        "/workspace/inference": inference_volume,
    },
    secrets=secrets,
)
def run_inference(smoke_test: bool = False, model: str = "all", task: str = "all"):
    import sys
    import time
    
    # Change working directory to the workspace
    os.chdir("/workspace")
    
    # Build the command string
    cmd = ["python", "scripts/05_inference.py", "--model", model, "--task", task]
    if smoke_test:
        cmd.append("--smoke-test")
        
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the underlying script and stream output
    proc = subprocess.Popen(cmd)
    
    # Periodically commit changes on the inference volume to ensure they are available for syncing
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        try:
            inference_volume.commit()
        except Exception as e:
            print(f"Volume commit failed: {e}")
        time.sleep(300)  # commit every 5 minutes
    
    # Final commit
    inference_volume.commit()
    
    if proc.returncode != 0:
        print(f"Error: inference script failed with exit code {proc.returncode}", file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        print("Inference completed successfully!")

@app.local_entrypoint()
def main(smoke_test: bool = False, model: str = "all", task: str = "all"):
    import time
    import threading
    import subprocess
    
    timestamp = int(time.time())
    # Create a unique folder for this run's sync to avoid touching or overwriting other models' local data
    sync_dir = f"inference_sync_{model}_{timestamp}"
    print(f"Starting Modal run on L4 GPU. smoke_test={smoke_test}, model={model}, task={task}")
    print(f"Local sync directory for this run: {sync_dir}/")
    
    def sync_volume():
        print(f"Starting periodic volume sync to local ./{sync_dir} directory (every 5 mins)...")
        os.makedirs(sync_dir, exist_ok=True)
        # We only download the directory of the model being run to avoid pulling everything
        remote_path = f"/{model}" if model != "all" else "/"
        while True:
            time.sleep(300)
            print(f"Syncing {remote_path} from Modal volume to local {sync_dir}...")
            try:
                subprocess.run(["modal", "volume", "get", "aip-inference-out", remote_path, f"{sync_dir}/", "--force"], check=False)
                print("Sync complete.")
            except Exception as e:
                print(f"Sync failed: {e}")
                
    # Start the sync thread as a daemon so it exits when main exits
    sync_thread = threading.Thread(target=sync_volume, daemon=True)
    sync_thread.start()
    
    # Run the modal function (this will block until it finishes)
    try:
        run_inference.remote(smoke_test, model, task)
    finally:
        # One final sync after it finishes or errors
        print("Run finished or interrupted. Final sync...")
        remote_path = f"/{model}" if model != "all" else "/"
        subprocess.run(["modal", "volume", "get", "aip-inference-out", remote_path, f"{sync_dir}/", "--force"], check=False)
