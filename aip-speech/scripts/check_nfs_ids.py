import modal

app = modal.App("check-done-sqa")
nfs = modal.NetworkFileSystem.from_name("aip-sqa-out-nfs")

@app.function(
    image=modal.Image.debian_slim().pip_install("pandas"),
    network_file_systems={"/workspace/sqa_output": nfs}
)
def check_sqa():
    import pandas as pd, json, os
    
    out_p = "/workspace/sqa_output/inference/qwen25_omni_3b/sqa.jsonl"
    rows = []
    if os.path.exists(out_p):
        with open(out_p, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try: rows.append(json.loads(l, strict=False))
                    except: pass
    
    df_done = pd.DataFrame(rows)
    print(f"Total rows in Modal NFS qwen25_omni_3b/sqa.jsonl: {len(df_done)}")
    if not df_done.empty and "id" in df_done.columns:
        print(f"Unique 'id' count in sqa.jsonl: {df_done['id'].nunique()}")

@app.local_entrypoint()
def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    check_sqa.remote()
