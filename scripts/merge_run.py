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
import argparse
import pandas as pd
import numpy as np
import subprocess

def parse_job_id(filename):
    # Extracts the job ID (integer) from a filename like benchmark_results_28583.csv or benchmarks_bfs_28859.txt
    match = re.search(r'_(\d+)\.(csv|txt)$', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_rust_results(input_dir, output_results_path, output_base_path):
    print(f"Merging Rust results from {input_dir}...")
    
    # Gather all CSV files
    all_csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter files
    base_files = [f for f in all_csvs if "_base_" in os.path.basename(f)]
    strassen_files = [f for f in all_csvs if "_base_" not in os.path.basename(f) and "benchmark_results" in os.path.basename(f)]
    
    # Sort by Job ID
    base_files = sorted(base_files, key=lambda f: parse_job_id(os.path.basename(f)))
    strassen_files = sorted(strassen_files, key=lambda f: parse_job_id(os.path.basename(f)))
    
    print(f"  Found {len(base_files)} base CSV files.")
    print(f"  Found {len(strassen_files)} Strassen CSV files.")
    
    strassen_merged = False
    # 1. Merge Strassen results
    if strassen_files:
        dfs = []
        for f in strassen_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {f}: {e}")
                
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
            print(f"  Wrote merged Strassen results to {output_results_path}")
            strassen_merged = True
            
    # 2. Merge Base results
    if base_files:
        dfs = []
        for f in base_files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {f}: {e}")
                
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
            print(f"  Wrote merged base results to {output_base_path}")
            
    return strassen_merged

def merge_c_results(input_dir, output_dir_std, output_dir_run, run_type):
    print(f"Merging C results from {input_dir} (type={run_type})...")
    
    all_txts = glob.glob(os.path.join(input_dir, "*.txt"))
    if not all_txts:
        print("  No C benchmark text files found.")
        return
        
    array_pattern = re.compile(r'([A-Za-z0-9_]+)\s*=\s*\[([^\]]*)\]\s*;')
    
    if run_type == "seq":
        # Group everything into benchmarks_seq.txt
        all_txts = sorted(all_txts, key=lambda f: parse_job_id(os.path.basename(f)))
        var_runs = {}
        for filepath in all_txts:
            try:
                with open(filepath, "r") as f:
                    content = f.read()
            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")
                continue
                
            matches = array_pattern.findall(content)
            for var_name, array_content in matches:
                if var_name not in var_runs:
                    var_runs[var_name] = []
                runs = array_content.split(";")
                for run in runs:
                    run = run.strip()
                    if run:
                        var_runs[var_name].append(run)
                        
        for base_out_dir in [output_dir_std, output_dir_run]:
            os.makedirs(base_out_dir, exist_ok=True)
            output_txt_path = os.path.join(base_out_dir, "benchmarks_seq.txt")
            sorted_vars = sorted(var_runs.keys())
            with open(output_txt_path, "w") as f:
                for var_name in sorted_vars:
                    runs = var_runs[var_name]
                    formatted_runs = " ;  ".join(runs)
                    if formatted_runs:
                        formatted_runs += " ;"
                    f.write(f"{var_name} = [ {formatted_runs} ];\n\n\n")
            print(f"  Wrote merged sequential C results to {output_txt_path}")
            
    else: # par
        # Partition by parallel modes: bfs, dfs, hybrid
        modes = ['dfs', 'bfs', 'hybrid']
        for mode in modes:
            mode_files = [f for f in all_txts if f"benchmarks_{mode}_" in os.path.basename(f)]
            mode_files = sorted(mode_files, key=lambda f: parse_job_id(os.path.basename(f)))
            
            if not mode_files:
                continue
                
            print(f"  Found {len(mode_files)} C benchmark files for parallel mode '{mode}'.")
            var_runs = {}
            for filepath in mode_files:
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                except Exception as e:
                    print(f"    Warning: Could not read {filepath}: {e}")
                    continue
                    
                matches = array_pattern.findall(content)
                for var_name, array_content in matches:
                    if var_name not in var_runs:
                        var_runs[var_name] = []
                    runs = array_content.split(";")
                    for run in runs:
                        run = run.strip()
                        if run:
                            var_runs[var_name].append(run)
                            
            for base_out_dir in [output_dir_std, output_dir_run]:
                os.makedirs(base_out_dir, exist_ok=True)
                output_txt_path = os.path.join(base_out_dir, f"benchmarks_{mode}.txt")
                sorted_vars = sorted(var_runs.keys())
                with open(output_txt_path, "w") as f:
                    for var_name in sorted_vars:
                        runs = var_runs[var_name]
                        formatted_runs = " ;  ".join(runs)
                        if formatted_runs:
                            formatted_runs += " ;"
                        f.write(f"{var_name} = [ {formatted_runs} ];\n\n\n")
                print(f"  Wrote merged parallel C results for {mode} to {output_txt_path}")

def merge_sequential(project_root):
    print("\n=== Merging Sequential Results ===")
    rust_input_dir = os.path.join(project_root, "generated", "csv", "run_seq", "rust")
    c_input_dir = os.path.join(project_root, "generated", "csv", "run_seq", "c")
    
    rust_output_results_run_seq = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results.csv")
    rust_output_base_run_seq = os.path.join(project_root, "generated", "csv", "run_seq", "benchmark_results_base.csv")
    
    rust_output_results_std = os.path.join(project_root, "generated", "csv", "benchmark_results.csv")
    rust_output_base_std = os.path.join(project_root, "generated", "csv", "benchmark_results_base.csv")
    
    c_output_dir_std = os.path.join(project_root, "benchmarks", "generated")
    c_output_dir_run_seq = os.path.join(project_root, "generated", "csv", "run_seq")
    
    strassen_merged = merge_rust_results(rust_input_dir, rust_output_results_run_seq, rust_output_base_run_seq)
    merge_rust_results(rust_input_dir, rust_output_results_std, rust_output_base_std)
    merge_c_results(c_input_dir, c_output_dir_std, c_output_dir_run_seq, "seq")
    
    if strassen_merged:
        plot_script = os.path.join(project_root, "python", "plot.py")
        if os.path.exists(plot_script):
            print(f"Generating plots in generated/plots/ using '{plot_script}' on standard CSV...")
            subprocess.run(["uv", "run", plot_script, rust_output_results_std, "--mode", "sequential"], check=False)
            print(f"Generating plots in generated/csv/run_seq/ using '{plot_script}' on run_seq CSV...")
            subprocess.run(["uv", "run", plot_script, rust_output_results_run_seq, "--mode", "sequential"], check=False)
            
        grid_plot_script = os.path.join(project_root, "python", "plot_grid.py")
        if os.path.exists(grid_plot_script):
            print(f"Generating sequential grid plots using '{grid_plot_script}'...")
            subprocess.run(["uv", "run", grid_plot_script, "--mode", "sequential"], check=False)

def merge_parallel(project_root):
    print("\n=== Merging Parallel Results ===")
    rust_input_dir = os.path.join(project_root, "generated", "csv", "run_par", "rust")
    c_input_dir = os.path.join(project_root, "generated", "csv", "run_par", "c")
    
    rust_output_results_run_par = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results.csv")
    rust_output_base_run_par = os.path.join(project_root, "generated", "csv", "run_par", "benchmark_results_base.csv")
    
    rust_output_results_std = os.path.join(project_root, "generated", "csv", "benchmark_results.csv")
    rust_output_base_std = os.path.join(project_root, "generated", "csv", "benchmark_results_base.csv")
    
    c_output_dir_std = os.path.join(project_root, "benchmarks", "generated")
    c_output_dir_run_par = os.path.join(project_root, "generated", "csv", "run_par")
    
    strassen_merged = merge_rust_results(rust_input_dir, rust_output_results_run_par, rust_output_base_run_par)
    merge_rust_results(rust_input_dir, rust_output_results_std, rust_output_base_std)
    merge_c_results(c_input_dir, c_output_dir_std, c_output_dir_run_par, "par")
    
    if strassen_merged:
        plot_script = os.path.join(project_root, "python", "plot.py")
        if os.path.exists(plot_script):
            print(f"Generating plots in generated/plots/ using '{plot_script}' on standard CSV...")
            subprocess.run(["uv", "run", plot_script, rust_output_results_std, "--mode", "parallel"], check=False)
            print(f"Generating plots in generated/csv/run_par/ using '{plot_script}' on run_par CSV...")
            subprocess.run(["uv", "run", plot_script, rust_output_results_run_par, "--mode", "parallel"], check=False)
            
        grid_plot_script = os.path.join(project_root, "python", "plot_grid.py")
        if os.path.exists(grid_plot_script):
            print(f"Generating parallel grid plots using '{grid_plot_script}'...")
            subprocess.run(["uv", "run", grid_plot_script, "--mode", "parallel"], check=False)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Unified script to merge sequential and parallel benchmark results.")
    parser.add_argument(
        "mode", 
        choices=["seq", "par", "both"], 
        nargs="?", 
        default="both",
        help="Merge mode: 'seq' (sequential), 'par' (parallel), or 'both' (merge both; default)"
    )
    args = parser.parse_args()
    
    if args.mode in ("seq", "both"):
        merge_sequential(project_root)
        
    if args.mode in ("par", "both"):
        merge_parallel(project_root)
        
    # Always update the base comparison plot if we merged sequential/parallel
    plot_base_comp = os.path.join(project_root, "python", "plot_mkl_faer_only.py")
    if os.path.exists(plot_base_comp):
        print(f"\nUpdating side-by-side base comparison plot using '{plot_base_comp}'...")
        subprocess.run(["uv", "run", plot_base_comp], check=False)

    # Always update the Ballard comparison plot if we merged sequential/parallel
    grid_plot_script = os.path.join(project_root, "python", "plot_grid.py")
    if os.path.exists(grid_plot_script):
        print(f"\nUpdating Ballard comparison plot using '{grid_plot_script}'...")
        subprocess.run(["uv", "run", grid_plot_script, "--mode", "compare_ballard"], check=False)

if __name__ == "__main__":
    main()
