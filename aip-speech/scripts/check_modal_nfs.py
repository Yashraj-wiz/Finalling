import modal

app = modal.App("check-nfs-sqa")
nfs = modal.NetworkFileSystem.from_name("aip-sqa-out-nfs")

@app.function(network_file_systems={"/workspace/sqa_output": nfs})
def check_sqa():
    import os, json
    for root, dirs, files in os.walk("/workspace/sqa_output"):
        for f in files:
            fp = os.path.join(root, f)
            try:
                rows = sum(1 for l in open(fp, "r", encoding="utf-8") if l.strip())
                print(f"  {fp}: {os.path.getsize(fp)} bytes, {rows} rows")
            except:
                print(f"  {fp}: {os.path.getsize(fp)} bytes")

@app.local_entrypoint()
def main():
    check_sqa.remote()
