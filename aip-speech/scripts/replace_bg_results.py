import json
import os
from pathlib import Path

def main():
    root_dir = Path(r"g:\IvLabs\Impact_Speech\aip-speech")
    inference_dir = root_dir / "inference"
    inference_1_dir = root_dir / "inference_1"
    
    if not inference_1_dir.exists():
        print(f"Error: {inference_1_dir} does not exist.")
        return

    # Iterate over specific model folders in inference_1
    for model_name in ["gemma3n_e4b"]:
        model_dir = inference_1_dir / model_name
        if not model_dir.is_dir():
            continue
        
        # Process all .jsonl files in the model directory
        for inf_1_file in model_dir.glob("*.jsonl"):
            task_file = inf_1_file.name
            inf_file = inference_dir / model_name / task_file
            
            airport_rows = []
            
            # Read snsd_airport rows from the source inference directory if it exists
            if inf_file.exists():
                with open(inf_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            if obj.get("background_id") == "snsd_airport":
                                airport_rows.append(line)
                        except json.JSONDecodeError:
                            pass
            
            new_lines = []
            office_removed = 0
            
            # Read the target inference_1 file and remove snsd_office rows
            with open(inf_1_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        if obj.get("background_id") == "snsd_office":
                            office_removed += 1
                            continue  # skip this row
                        new_lines.append(line)
                    except json.JSONDecodeError:
                        new_lines.append(line)
                        
            # Append the extracted snsd_airport rows
            new_lines.extend(airport_rows)
            
            # Write the updated content back to inference_1
            with open(inf_1_file, 'w', encoding='utf-8') as f:
                for line in new_lines:
                    f.write(line)
            
            print(f"[{model_name}/{task_file}] Removed {office_removed} 'snsd_office' rows, added {len(airport_rows)} 'snsd_airport' rows.")

if __name__ == "__main__":
    main()
