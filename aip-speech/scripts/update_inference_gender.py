import json
from pathlib import Path

def main():
    root_dir = Path(r"g:\IvLabs\Impact_Speech\aip-speech")
    asr_bank = root_dir / "itembanks" / "asr.jsonl"
    inference_1_dir = root_dir / "inference_1"
    
    if not asr_bank.exists():
        print(f"Error: {asr_bank} does not exist.")
        return
        
    if not inference_1_dir.exists():
        print(f"Error: {inference_1_dir} does not exist.")
        return

    # 1. Load ASR bank and build gender mapping
    speech_gender_map = {}
    with open(asr_bank, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            speech_id = item.get("id")
            gender = item.get("gender")
            if speech_id and gender and gender != "unknown":
                # Map to female_feminine or male_masculine
                if gender.lower() == "female":
                    mapped_gender = "female_feminine"
                else:
                    mapped_gender = "male_masculine"
                speech_gender_map[speech_id] = mapped_gender

    print(f"Loaded {len(speech_gender_map)} gender mappings from {asr_bank.name}")

    # 2. Iterate through all .jsonl files in inference_1 and update
    for model_dir in inference_1_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        for jsonl_file in model_dir.glob("*.jsonl"):
            print(f"Processing {jsonl_file.relative_to(root_dir)}...")
            updated_rows = []
            updates_count = 0
            
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    
                    speech_id = row.get("speech_id")
                    current_gender = row.get("gender")
                    
                    # If it's a LibriSpeech item (unknown gender initially) and we have a mapping for it
                    if speech_id in speech_gender_map and current_gender == "unknown":
                        row["gender"] = speech_gender_map[speech_id]
                        updates_count += 1
                        
                    updated_rows.append(row)
            
            if updates_count > 0:
                # Rewrite the file with updated rows
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for row in updated_rows:
                        f.write(json.dumps(row) + "\n")
                print(f"  -> Updated {updates_count} rows in {jsonl_file.name}")
            else:
                print(f"  -> No updates needed.")

if __name__ == "__main__":
    main()
