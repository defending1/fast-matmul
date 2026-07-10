import os
import sys
import json

def get_next_run_number(csv_dir):
    max_x = 0
    if os.path.exists(csv_dir):
        for name in os.listdir(csv_dir):
            if os.path.isdir(os.path.join(csv_dir, name)):
                if name.startswith("run"):
                    try:
                        x = int(name[3:])
                        if x > max_x:
                            max_x = x
                    except ValueError:
                        pass
    return max_x + 1

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(project_root, "generated", "csv")
    log_dir = os.path.join(project_root, "generated", "log")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("RUN_ID")
    
    if job_id:
        mapping_path = os.path.join(csv_dir, "run_mapping.json")
        mapping = {}
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
            except Exception:
                pass
                
        if job_id in mapping:
            run_folder = mapping[job_id]
        else:
            next_x = get_next_run_number(csv_dir)
            run_folder = f"run{next_x}"
            mapping[job_id] = run_folder
            try:
                with open(mapping_path, "w") as f:
                    json.dump(mapping, f, indent=4)
            except Exception:
                pass
    else:
        next_x = get_next_run_number(csv_dir)
        run_folder = f"run{next_x}"
        
    os.makedirs(os.path.join(csv_dir, run_folder, "rust"), exist_ok=True)
    os.makedirs(os.path.join(csv_dir, run_folder, "c"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, run_folder), exist_ok=True)
    
    print(run_folder)

if __name__ == "__main__":
    main()
