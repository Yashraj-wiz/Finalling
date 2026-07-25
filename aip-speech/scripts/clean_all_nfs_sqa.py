import modal

app = modal.App("clean-nfs-sqa-all")
nfs = modal.NetworkFileSystem.from_name("aip-sqa-out-nfs")

@app.function(
    image=modal.Image.debian_slim(),
    network_file_systems={"/workspace/sqa_output": nfs}
)
def clean_nfs():
    import os, json
    
    inf_dir = "/workspace/sqa_output/inference"
    for model_dir in os.listdir(inf_dir):
        sqa_file = os.path.join(inf_dir, model_dir, "sqa.jsonl")
        if os.path.exists(sqa_file):
            valid_rows = []
            bad_cnt = 0
            with open(sqa_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        valid_rows.append(json.loads(line, strict=False))
                    except Exception as e:
                        bad_cnt += 1
            print(f"[NFS Clean] {model_dir}/sqa.jsonl: {len(valid_rows)} valid rows, {bad_cnt} bad rows removed.")
            with open(sqa_file, "w", encoding="utf-8") as f:
                for r in valid_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

@app.local_entrypoint()
def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    clean_nfs.remote()
