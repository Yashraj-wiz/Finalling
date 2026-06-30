import modal
import subprocess
import os

# Define the environment with PyTorch and required dependencies from requirements.txt
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "torchaudio",
        "torchvision",
        "timm",
        "transformers>=4.49.0",
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
        "peft",
    )
    .apt_install("ffmpeg")
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
    
    # Change working directory to the workspace
    os.chdir("/workspace")
    
    # Build the command string
    cmd = ["python", "scripts/05_inference.py", "--model", model, "--task", task]
    if smoke_test:
        cmd.append("--smoke-test")
        
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the underlying script and stream output
    result = subprocess.run(cmd, check=False)
    
    # Commit changes on the inference volume to ensure they are available for syncing
    inference_volume.commit()
    
    if result.returncode != 0:
        print(f"Error: inference script failed with exit code {result.returncode}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    else:
        print("Inference completed successfully!")

@app.local_entrypoint()
def main(smoke_test: bool = False, model: str = "all", task: str = "all"):
    print(f"Starting Modal run on L4 GPU. smoke_test={smoke_test}, model={model}, task={task}")
    run_inference.remote(smoke_test, model, task)
