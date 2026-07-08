#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""Merge C++ benchmark results from MATLAB-like arrays into consolidated CSV files.

This script parses all text files in benchmarks/generated/benchmarks_*.txt,
extracts the timing data, and writes the consolidated output in both
long (tidy) and wide (pivoted) CSV formats.
"""

import os
import glob
import re
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    generated_dir = os.path.join(project_root, "benchmarks", "generated")
    
    # Pattern to find all benchmark files
    pattern = os.path.join(generated_dir, "benchmarks_*.txt")
    filepaths = glob.glob(pattern)
    
    # Filter out any files that do not fit the criteria
    filepaths = [f for f in filepaths if os.path.basename(f) != "benchmarks.txt"]
    
    if not filepaths:
        print(f"No C++ benchmark output files found matching '{pattern}'.")
        return
        
    print(f"Found {len(filepaths)} files to parse.")
    
    data_rows = []
    
    # Regex to parse MATLAB-like array lines:
    # e.g., STRASSEN_1 = [ 64 64 64 1 0.756113 ; ];
    array_pattern = re.compile(r'([A-Za-z0-9_]+)\s*=\s*\[([^\]]*)\]\s*;')
    
    for filepath in filepaths:
        basename = os.path.basename(filepath)
        stem = basename[:-4] # strip .txt
        
        # Parse mode and job_id from filename
        # e.g., benchmarks_seq_12345 -> mode='seq', job_id='12345'
        # e.g., benchmarks_dfs -> mode='dfs', job_id=None
        mode = "unknown"
        job_id = None
        if stem.startswith("benchmarks_"):
            rest = stem[len("benchmarks_"):]
            parts = rest.split("_")
            if parts[-1].isdigit():
                job_id = parts[-1]
                mode = "_".join(parts[:-1])
            else:
                job_id = None
                mode = rest
                
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
            
        # Find all array definitions
        matches = array_pattern.findall(content)
        for var_name, array_content in matches:
            # Parse algorithm and level from variable name (e.g. STRASSEN_1 -> STRASSEN, 1)
            var_parts = var_name.rsplit("_", 1)
            if len(var_parts) == 2 and var_parts[1].isdigit():
                alg_name = var_parts[0]
                level_from_var = int(var_parts[1])
            else:
                alg_name = var_name
                level_from_var = 0
                
            # Parse the individual data points in the array
            # Splits by semicolon
            runs = array_content.split(";")
            for run in runs:
                run = run.strip()
                if not run:
                    continue
                tokens = run.split()
                if len(tokens) == 5:
                    try:
                        m = int(tokens[0])
                        k = int(tokens[1])
                        n = int(tokens[2])
                        level = int(tokens[3])
                        time = float(tokens[4])
                        
                        data_rows.append({
                            "m": m,
                            "k": k,
                            "n": n,
                            "algorithm": alg_name,
                            "recursion_level": level,
                            "mode": mode,
                            "time": time,
                            "job_id": job_id
                        })
                    except ValueError as ve:
                        print(f"Warning: Failed to parse tokens in {basename}: {tokens} ({ve})")
                        
    if not data_rows:
        print("No valid C++ benchmark data rows found. Exiting.")
        return
        
    df_long = pd.DataFrame(data_rows)
    
    # Sort for cleanliness
    df_long = df_long.sort_values(by=["m", "k", "n", "algorithm", "recursion_level", "mode"]).reset_index(drop=True)
    
    # Write the long/tidy format CSV
    long_csv_path = os.path.join(generated_dir, "benchmarks_merged.csv")
    df_long.to_csv(long_csv_path, index=False)
    print(f"Successfully wrote long-format results to: {long_csv_path}")
    
    # Pivot to wide format
    # Index is matrix dimensions, columns are algorithm-level-mode combinations
    try:
        df_wide = df_long.pivot_table(
            index=["m", "k", "n"],
            columns=["algorithm", "recursion_level", "mode"],
            values="time"
        )
        
        # Flatten MultiIndex columns
        flat_cols = []
        for col in df_wide.columns:
            alg, level, mode = col
            col_name = f"{alg}_L{int(level)}_{mode}"
            flat_cols.append(col_name)
            
        df_wide.columns = flat_cols
        df_wide = df_wide.reset_index()
        
        # Write the wide format CSV
        wide_csv_path = os.path.join(generated_dir, "benchmarks_merged_wide.csv")
        df_wide.to_csv(wide_csv_path, index=False)
        print(f"Successfully wrote wide-format results to: {wide_csv_path}")
    except Exception as e:
        print(f"Warning: Could not pivot to wide format: {e}")

if __name__ == "__main__":
    main()
