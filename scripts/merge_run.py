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
import sys

def parse_job_id(filename):
    """Extracts the job ID (integer) from a filename like benchmark_results_28583.csv or benchmarks_bfs_28859.txt"""
    match = re.search(r'_(\d+)\.(csv|txt)$', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_rust_results(input_dir):
    """Merges Rust benchmark CSV files (Strassen and base files) from input_dir and returns the dataframes."""
    print(f"Merging Rust results from {input_dir}...")
    
    # Gather all CSV files
    all_csvs = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter files
    base_files = [
        f for f in all_csvs 
        if "_base_" in os.path.basename(f)
        and "benchmark_results_base_merged.csv" != os.path.basename(f)
        and "benchmark_results_base.csv" != os.path.basename(f)
    ]
    strassen_files = [
        f for f in all_csvs 
        if "_base_" not in os.path.basename(f) 
        and "benchmark_results" in os.path.basename(f)
        and "benchmark_results_levels" not in os.path.basename(f)
        and "benchmark_results_cutoff" not in os.path.basename(f)
        and os.path.basename(f) != "benchmark_results.csv"
    ]
    
    # Sort by Job ID
    base_files = sorted(base_files, key=lambda f: parse_job_id(os.path.basename(f)))
    strassen_files = sorted(strassen_files, key=lambda f: parse_job_id(os.path.basename(f)))
    
    print(f"  Found {len(base_files)} base CSV files.")
    print(f"  Found {len(strassen_files)} Strassen CSV files.")
    
    levels_df = None
    cutoff_df = None
    
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
            final_strassen_df = final_df[existing_cols + extra_cols]
            
            # Split into levels and cutoff dataframes
            levels_df = final_strassen_df[final_strassen_df['recursion_level'].notna()]
            cutoff_df = final_strassen_df[final_strassen_df['size_cutoff'].notna()]
            
    final_base_df = None
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
            final_base_df = final_df[existing_cols + extra_cols]
            
    return levels_df, cutoff_df, final_base_df

def merge_c_results(input_dir):
    """Merges C benchmark text files from input_dir grouped by mode prefix."""
    print(f"Merging C results from {input_dir}...")
    
    all_txts = glob.glob(os.path.join(input_dir, "*.txt"))
    if not all_txts:
        print("  No C benchmark text files found.")
        return {}
        
    array_pattern = re.compile(r'([A-Za-z0-9_]+)\s*=\s*\[([^\]]*)\]\s*;')
    
    # Group C files by prefix extracted from name
    grouped_files = {}
    for f in all_txts:
        basename = os.path.basename(f)
        match = re.match(r'benchmarks_([a-zA-Z0-9_]+)_\d+\.txt$', basename)
        if match:
            mode = match.group(1)
        else:
            mode = "unknown"
        if mode not in grouped_files:
            grouped_files[mode] = []
        grouped_files[mode].append(f)
        
    merged_results = {}
    
    for mode, files in grouped_files.items():
        sorted_files = sorted(files, key=lambda f: parse_job_id(os.path.basename(f)))
        print(f"  Found {len(sorted_files)} C benchmark files for mode '{mode}'.")
        
        var_runs = {}
        for filepath in sorted_files:
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
                        
        merged_results[mode] = var_runs
        
    return merged_results

def format_c_merged_content(var_runs):
    """Formats in-memory runs into MATLAB array string syntax."""
    content = ""
    sorted_vars = sorted(var_runs.keys())
    for var_name in sorted_vars:
        runs = var_runs[var_name]
        formatted_runs = " ;  ".join(runs)
        if formatted_runs:
            formatted_runs += " ;"
        content += f"{var_name} = [ {formatted_runs} ];\n\n\n"
    return content

def merge_run_dir(run_dir_path, project_root):
    """Processes a single run directory, merging its Rust and C subdirectories, and returns whether it is a parallel run."""
    run_dir_path = os.path.abspath(run_dir_path)
    run_dir_name = os.path.basename(run_dir_path)
    print(f"\n=== Merging Run Directory: {run_dir_name} ===")
    
    # 1. Locate inputs
    rust_input_dir = os.path.join(run_dir_path, "rust")
    if not os.path.exists(rust_input_dir) or not any(f.endswith(".csv") for f in os.listdir(rust_input_dir)):
        rust_input_dir = run_dir_path
        
    c_input_dir = os.path.join(run_dir_path, "c")
    
    # 2. Merge Rust Results
    levels_df, cutoff_df, base_df = merge_rust_results(rust_input_dir)
    
    levels_merged = False
    if levels_df is not None and not levels_df.empty:
        for out_dir in [run_dir_path, os.path.join(project_root, "generated", "csv")]:
            os.makedirs(out_dir, exist_ok=True)
            levels_df.to_csv(os.path.join(out_dir, "benchmark_results_levels.csv"), index=False)
        print(f"  Wrote merged levels results.")
        levels_merged = True
        
    cutoff_merged = False
    if cutoff_df is not None and not cutoff_df.empty:
        for out_dir in [run_dir_path, os.path.join(project_root, "generated", "csv")]:
            os.makedirs(out_dir, exist_ok=True)
            cutoff_df.to_csv(os.path.join(out_dir, "benchmark_results_cutoff.csv"), index=False)
        print(f"  Wrote merged cutoff results.")
        cutoff_merged = True
        
    if base_df is not None:
        for out_dir in [run_dir_path, os.path.join(project_root, "generated", "csv")]:
            os.makedirs(out_dir, exist_ok=True)
            base_df.to_csv(os.path.join(out_dir, "benchmark_results_base.csv"), index=False)
        print(f"  Wrote merged base results.")
        
    # 3. Merge C Results
    if os.path.exists(c_input_dir):
        c_merged = merge_c_results(c_input_dir)
        for mode, var_runs in c_merged.items():
            formatted_content = format_c_merged_content(var_runs)
            output_dirs = [
                run_dir_path,
                os.path.join(project_root, "benchmarks", "generated")
            ]
            for out_dir in output_dirs:
                os.makedirs(out_dir, exist_ok=True)
                output_txt_path = os.path.join(out_dir, f"benchmarks_{mode}.txt")
                with open(output_txt_path, "w") as f:
                    f.write(formatted_content)
                print(f"  Wrote merged C results for mode '{mode}' to {output_txt_path}")
                
    # 4. Auto-detect sequential vs parallel
    is_parallel = "par" in run_dir_name.lower()
    mode_str = "parallel" if is_parallel else "sequential"
    
    # 5. Run plots
    if levels_merged or cutoff_merged:
            
        grid_plot_script = os.path.join(project_root, "python", "plot_grid.py")
        if os.path.exists(grid_plot_script):
            print(f"Generating {mode_str} grid plots using '{grid_plot_script}'...")
            grid_cmd = ["uv", "run", grid_plot_script, "--mode", mode_str]
            if is_parallel:
                grid_cmd.extend(["--par-dir", run_dir_name])
            subprocess.run(grid_cmd, check=False)
            
            if is_parallel:
                print(f"Generating parallel cutoff grid plots using '{grid_plot_script}'...")
                grid_cmd_cutoff = ["uv", "run", grid_plot_script, "--mode", "cutoff_grid"]
                grid_cmd_cutoff.extend(["--par-dir", run_dir_name])
                subprocess.run(grid_cmd_cutoff, check=False)
            
    return is_parallel

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description="Unified script to merge sequential and parallel benchmark results.")
    parser.add_argument(
        "run_dirs", 
        nargs="*",
        help="Run directories to merge (e.g., generated/csv/run_seq) or legacy modes ('seq', 'par', 'par2', 'both')"
    )
    args = parser.parse_args()
    
    if not args.run_dirs:
        args.run_dirs = ["both"]
        
    target_dirs = []
    
    def resolve_folder(name):
        if os.path.exists(name):
            return name
        csv_under = os.path.join(project_root, "generated", "csv", name)
        if os.path.exists(csv_under):
            return csv_under
        return None
        
    for item in args.run_dirs:
        if item == "both":
            target_dirs.extend([
                os.path.join(project_root, "generated", "csv", "run_seq"),
                os.path.join(project_root, "generated", "csv", "run_par")
            ])
        elif item == "seq":
            target_dirs.append(os.path.join(project_root, "generated", "csv", "run_seq"))
        elif item == "par":
            target_dirs.append(os.path.join(project_root, "generated", "csv", "run_par"))
        elif item == "par2":
            target_dirs.append(os.path.join(project_root, "generated", "csv", "run_par2"))
        else:
            resolved = resolve_folder(item)
            if resolved:
                target_dirs.append(resolved)
            else:
                print(f"Error: Could not find run directory '{item}'")
                sys.exit(1)
                
    last_par_dir = None
    for run_dir in target_dirs:
        is_par = merge_run_dir(run_dir, project_root)
        if is_par:
            last_par_dir = os.path.basename(run_dir)
            
    # Always update the base comparison plot
    plot_base_comp = os.path.join(project_root, "python", "plot_mkl_faer_only.py")
    if os.path.exists(plot_base_comp):
        print(f"\nUpdating side-by-side base comparison plot using '{plot_base_comp}'...")
        cmd = ["uv", "run", plot_base_comp]
        if last_par_dir:
            cmd.extend(["--par-dir", last_par_dir])
        subprocess.run(cmd, check=False)

    # Always update the Ballard comparison plot
    grid_plot_script = os.path.join(project_root, "python", "plot_grid.py")
    if os.path.exists(grid_plot_script):
        print(f"\nUpdating Ballard comparison plot using '{grid_plot_script}'...")
        cmd = ["uv", "run", grid_plot_script, "--mode", "compare_ballard"]
        if last_par_dir:
            cmd.extend(["--par-dir", last_par_dir])
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
