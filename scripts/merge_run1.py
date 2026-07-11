#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
# ]
# ///
import os
import glob
import re
import pandas as pd
import numpy as np
import subprocess

def parse_job_id(filename):
    # Extracts the job ID (integer) from a filename like benchmark_results_28583.csv or benchmarks_seq_28695.txt
    match = re.search(r'_(\d+)\.(csv|txt)$', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_rust_results(input_dir, output_results_path, output_base_path):
    print("Merging Rust results...")
    
    # Gather all CSV files
    all_csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter files
    base_files = [f for f in all_csvs if "_base_" in os.path.basename(f)]
    strassen_files = [f for f in all_csvs if "_base_" not in os.path.basename(f) and "benchmark_results" in os.path.basename(f)]
    
    # Sort by Job ID (Follow job order)
    base_files = sorted(base_files, key=lambda f: parse_job_id(os.path.basename(f)))
    strassen_files = sorted(strassen_files, key=lambda f: parse_job_id(os.path.basename(f)))
    
    print(f"Found {len(base_files)} base CSV files.")
    print(f"Found {len(strassen_files)} Strassen CSV files.")
    
    strassen_merged = False
    # 2. Merge Strassen results
    if strassen_files:
        dfs = []
        for f in strassen_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")
                
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            
            group_keys = ['size', 'base_choice', 'recursion_level', 'size_cutoff']
            columns_order = [
                'size', 'base_choice', 'recursion_level', 'size_cutoff',
                'mkl_seq', 'mkl_par', 'faer_seq', 'faer_par',
                'strassen_seq', 'strassen_dfs', 'strassen_bfs', 'strassen_hybrid'
            ]
            
            temp_df = merged_df.copy()
            fill_cols = [k for k in group_keys if k != 'size']
            for col in fill_cols:
                if col in temp_df.columns:
                    temp_df[col] = temp_df[col].fillna(-1.0)
                    
            final_df = temp_df.groupby(group_keys, as_index=False).first()
            
            for col in fill_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].replace(-1.0, np.nan)
                    
            final_df = final_df.sort_values(by=group_keys).reset_index(drop=True)
            
            existing_cols = [c for c in columns_order if c in final_df.columns]
            extra_cols = [c for c in final_df.columns if c not in columns_order]
            final_df = final_df[existing_cols + extra_cols]
            
            os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
            final_df.to_csv(output_results_path, index=False)
            print(f"Wrote merged Strassen results to {output_results_path}")
            strassen_merged = True
            
    # 3. Merge Base results
    if base_files:
        dfs = []
        for f in base_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")
                
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            
            base_group_keys = ['size']
            base_columns_order = ['size', 'mkl_seq', 'mkl_par', 'faer_seq', 'faer_par']
            
            final_df = merged_df.groupby(base_group_keys, as_index=False).first()
            final_df = final_df.sort_values(by=base_group_keys).reset_index(drop=True)
            
            existing_cols = [c for c in base_columns_order if c in final_df.columns]
            extra_cols = [c for c in final_df.columns if c not in base_columns_order]
            final_df = final_df[existing_cols + extra_cols]
            
            os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
            final_df.to_csv(output_base_path, index=False)
            print(f"Wrote merged base results to {output_base_path}")
            
    return strassen_merged

def merge_c_results(input_dir, output_txt_path):
    print("Merging C results...")
    
    # Gather and sort C output files by Job ID
    all_txts = glob.glob(os.path.join(input_dir, "*.txt"))
    all_txts = sorted(all_txts, key=lambda f: parse_job_id(os.path.basename(f)))
    
    print(f"Found {len(all_txts)} C benchmark text files.")
    
    # Store runs grouped by variable name
    # e.g., "MKL_0" -> list of "2 2 2 0 0.002509"
    var_runs = {}
    
    # Regex to parse MATLAB-like array lines:
    array_pattern = re.compile(r'([A-Za-z0-9_]+)\s*=\s*\[([^\]]*)\]\s*;')
    
    for filepath in all_txts:
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
            
        matches = array_pattern.findall(content)
        for var_name, array_content in matches:
            if var_name not in var_runs:
                var_runs[var_name] = []
            
            # Split by semicolon
            runs = array_content.split(";")
            for run in runs:
                run = run.strip()
                if run:
                    var_runs[var_name].append(run)
                    
    # Format the combined text
    sorted_vars = sorted(var_runs.keys())
    
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        for var_name in sorted_vars:
            runs = var_runs[var_name]
            # format runs: "  run1 ;  run2 ; ... ;"
            formatted_runs = " ;  ".join(runs)
            if formatted_runs:
                formatted_runs += " ;"
            f.write(f"{var_name} = [ {formatted_runs} ];\n\n\n")
            
    print(f"Wrote merged C results to {output_txt_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define directories
    rust_input_dir = os.path.join(project_root, "generated", "csv", "run1", "rust")
    c_input_dir = os.path.join(project_root, "generated", "csv", "run1", "c")
    
    # run1 output paths
    rust_output_results_run1 = os.path.join(project_root, "generated", "csv", "run1", "benchmark_results.csv")
    rust_output_base_run1 = os.path.join(project_root, "generated", "csv", "run1", "benchmark_results_base.csv")
    
    # standard output paths (so standard plot script outputs to generated/plots)
    rust_output_results_std = os.path.join(project_root, "generated", "csv", "benchmark_results.csv")
    rust_output_base_std = os.path.join(project_root, "generated", "csv", "benchmark_results_base.csv")
    
    c_output_txt = os.path.join(project_root, "benchmarks", "generated", "benchmarks_seq.txt")
    c_output_txt_copy = os.path.join(project_root, "generated", "csv", "run1", "benchmarks_seq.txt")
    
    # 1. Merge Rust results into run1/
    strassen_merged = merge_rust_results(rust_input_dir, rust_output_results_run1, rust_output_base_run1)
    
    # Also write them to standard generated/csv/ for general plotting compatibility
    merge_rust_results(rust_input_dir, rust_output_results_std, rust_output_base_std)
    
    # 2. Merge C results
    merge_c_results(c_input_dir, c_output_txt)
    merge_c_results(c_input_dir, c_output_txt_copy)
    
    # 3. Run plots
    if strassen_merged:
        plot_script = os.path.join(project_root, "python", "plot.py")
        if os.path.exists(plot_script):
            # Run plot script on the standard location to generate plots in generated/plots/
            print(f"Generating plots in generated/plots/ using '{plot_script}' on standard CSV...")
            try:
                res = subprocess.run(["uv", "run", plot_script, rust_output_results_std], check=True)
                if res.returncode == 0:
                    print("Standard plots generated successfully in generated/plots/!")
            except Exception as e:
                print(f"Error running plot script on standard CSV: {e}")
                
            # Also run plot script on run1 location to generate plots in generated/csv/run1/
            print(f"Generating plots in generated/csv/run1/ using '{plot_script}' on run1 CSV...")
            try:
                res = subprocess.run(["uv", "run", plot_script, rust_output_results_run1], check=True)
                if res.returncode == 0:
                    print("Run1 plots generated successfully in generated/csv/run1/!")
            except Exception as e:
                print(f"Error running plot script on run1 CSV: {e}")

if __name__ == "__main__":
    main()
