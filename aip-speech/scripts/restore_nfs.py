import modal
import os
import shutil

app = modal.App("nfs-fix")
nfs = modal.NetworkFileSystem.from_name("aip-sqa-out-nfs", create_if_missing=True)

good_file_dir = modal.Image.debian_slim().add_local_dir("inference_sync_all_1784759190/qwen25_omni_7b", remote_path="/good")

@app.function(image=good_file_dir, network_file_systems={"/workspace/inference": nfs})
def restore():
    os.makedirs("/workspace/inference/qwen25_omni_7b", exist_ok=True)
    shutil.copy2("/good/sqa.jsonl", "/workspace/inference/qwen25_omni_7b/sqa.jsonl")
    print("RESTORED GOOD RESULTS!")

@app.local_entrypoint()
def main():
    restore.remote()
