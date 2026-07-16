#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""Merge results from multiple individual benchmark CSV files into a consolidated CSV file

and run the plot script.
"""

import glob
import os
import sys
import pandas as pd
import subprocess

def merge_csvs(csv_files, output_path, group_keys, columns_order):
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            
    if not dfs:
        return False
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Fill NaN for group keys temporarily
    temp_df = merged_df.copy()
    fill_cols = [k for k in group_keys if k != 'size']
    for col in fill_cols:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].fillna(-1.0)
            
    final_df = temp_df.groupby(group_keys, as_index=False).first()
    
    # Restore -1.0 back to NaN
    import numpy as np
    for col in fill_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].replace(-1.0, np.nan)
            
    final_df = final_df.sort_values(by=group_keys).reset_index(drop=True)
    
    existing_cols = [c for c in columns_order if c in final_df.columns]
    extra_cols = [c for c in final_df.columns if c not in columns_order]
    final_df = final_df[existing_cols + extra_cols]
    
    final_df.to_csv(output_path, index=False)
    print(f"Successfully wrote merged results to: {output_path}")
    return True

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_dir = os.path.join(project_root, "generated", "csv")
    
    # 1. Merge Strassen results
    pattern = os.path.join(csv_dir, "benchmark_results_*.csv")
    csv_files = glob.glob(pattern)
    csv_files = [
        f for f in csv_files 
        if os.path.basename(f) not in ("benchmark_results_levels.csv", "benchmark_results_cutoff.csv") 
        and "_base_" not in os.path.basename(f)
        and "benchmark_results_base.csv" != os.path.basename(f)
        and os.path.basename(f) != "benchmark_results.csv"
    ]
    
    levels_merged = False
    cutoff_merged = False
    
    if csv_files:
        dfs = []
        for f in csv_files:
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
            
            # Fill NaN for group keys temporarily
            temp_df = merged_df.copy()
            fill_cols = [k for k in group_keys if k != 'size']
            for col in fill_cols:
                if col in temp_df.columns:
                    temp_df[col] = temp_df[col].fillna(-1.0)
                    
            final_df = temp_df.groupby(group_keys, as_index=False).first()
            
            import numpy as np
            for col in fill_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].replace(-1.0, np.nan)
                    
            final_df = final_df.sort_values(by=group_keys).reset_index(drop=True)
            
            existing_cols = [c for c in columns_order if c in final_df.columns]
            extra_cols = [c for c in final_df.columns if c not in columns_order]
            final_df = final_df[existing_cols + extra_cols]
            
            levels_df = final_df[final_df['recursion_level'].notna()]
            cutoff_df = final_df[final_df['size_cutoff'].notna()]
            
            if not levels_df.empty:
                levels_output_path = os.path.join(csv_dir, "benchmark_results_levels.csv")
                levels_df.to_csv(levels_output_path, index=False)
                print(f"Successfully wrote merged levels results to: {levels_output_path}")
                levels_merged = True
                        
            if not cutoff_df.empty:
                cutoff_output_path = os.path.join(csv_dir, "benchmark_results_cutoff.csv")
                cutoff_df.to_csv(cutoff_output_path, index=False)
                print(f"Successfully wrote merged cutoff results to: {cutoff_output_path}")
                cutoff_merged = True

    # 2. Merge Base results
    base_pattern = os.path.join(csv_dir, "benchmark_results_base_*.csv")
    base_csv_files = glob.glob(base_pattern)
    base_merged_filename = "benchmark_results_base_merged.csv"
    base_csv_files = [
        f for f in base_csv_files 
        if os.path.basename(f) != base_merged_filename
        and os.path.basename(f) != "benchmark_results_base.csv"
    ]
    
    if not base_csv_files:
        default_base_csv = os.path.join(csv_dir, "benchmark_results_base.csv")
        if os.path.exists(default_base_csv):
            base_csv_files = [default_base_csv]
            
    if base_csv_files:
        base_output_path = os.path.join(csv_dir, "benchmark_results_base.csv")
        base_group_keys = ['size']
        base_columns_order = ['size', 'mkl_seq', 'mkl_par', 'faer_seq', 'faer_par']
        merge_csvs(base_csv_files, base_output_path, base_group_keys, base_columns_order)

if __name__ == "__main__":
    main()
