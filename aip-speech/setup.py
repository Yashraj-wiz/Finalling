#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=== AIP-Speech environment setup ===")
    
    # 1. Check if running in a virtual environment
    # sys.prefix != sys.base_prefix is the standard way to check if venv is active
    is_venv = sys.prefix != sys.base_prefix or os.environ.get('VIRTUAL_ENV')
    if not is_venv:
        print("[WARNING] You do not appear to be running inside a virtual environment.")
        print("It is highly recommended to create and activate a virtual environment first:")
        print("  Windows: python -m venv .venv && .\\.venv\\Scripts\\activate")
        print("  Linux/Mac: python3 -m venv .venv && source .venv/bin/activate")
        choice = input("Do you want to continue installation in the global environment? (y/N): ")
        if choice.lower() not in ('y', 'yes'):
            print("Aborting.")
            sys.exit(1)

    root_dir = Path(__file__).resolve().parent
    requirements_file = root_dir / "requirements.txt"

    # 2. Upgrade pip and install dependencies
    print("\n--- Installing Dependencies ---")
    try:
        print("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        if requirements_file.exists():
            print(f"Installing packages from {requirements_file.name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        else:
            print("[ERROR] requirements.txt not found!")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        sys.exit(1)

    # 3. Create necessary folders
    print("\n--- Creating Directories ---")
    dirs_to_create = [
        "data/bg", "data/bg_raw", 
        "data/speech_asr", "data/speech_kws", "data/speech_saa", 
        "data/esc50", "data/ms_snsd", "data/musan", "data/noisex",
        "descriptors", "itembanks", "prereg", "manifests", 
        "inference", "scoring", "results", "checks/inspection",
        "models/cache"
    ]
    
    for d in dirs_to_create:
        dir_path = root_dir / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {d}")

    print("\n=== Setup Completed Successfully! ===")
    print(f"Hugging Face & Torch cache directory: {root_dir / 'models' / 'cache'}")

if __name__ == "__main__":
    main()
