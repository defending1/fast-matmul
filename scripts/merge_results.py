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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_dir = os.path.join(project_root, "generated", "csv")
    
    # Pattern to find all job CSV files
    pattern = os.path.join(csv_dir, "benchmark_results_*.csv")
    csv_files = glob.glob(pattern)
    
    # Filter out the merged file itself if it matches the pattern
    merged_filename = "benchmark_results_merged.csv"
    csv_files = [f for f in csv_files if os.path.basename(f) != merged_filename]
    
    if not csv_files:
        print(f"No individual benchmark CSV files found matching '{pattern}'.")
        # Try checking if the default benchmark_results.csv is present
        default_csv = os.path.join(csv_dir, "benchmark_results.csv")
        if os.path.exists(default_csv):
            print(f"Using default results: {default_csv}")
            csv_files = [default_csv]
        else:
            print("No CSV files to merge. Exiting.")
            sys.exit(0)
            
    print(f"Found {len(csv_files)} CSV files to merge.")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            
    if not dfs:
        print("No valid dataframes loaded. Exiting.")
        sys.exit(1)
        
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # We want to combine rows that have the same key: ['size', 'base_choice', 'recursion_level', 'size_cutoff']
    # Group by keys, taking the first non-null value for each column.
    # To handle NaN values in grouping keys across various pandas versions, we temporarily fill them.
    group_keys = ['size', 'base_choice', 'recursion_level', 'size_cutoff']
    
    temp_df = merged_df.copy()
    temp_df['recursion_level'] = temp_df['recursion_level'].fillna(-1.0)
    temp_df['size_cutoff'] = temp_df['size_cutoff'].fillna(-1.0)
    
    # Group and aggregate taking the first non-null value
    final_df = temp_df.groupby(['size', 'base_choice', 'recursion_level', 'size_cutoff'], as_index=False).first()
    
    # Restore -1.0 back to NaN
    import numpy as np
    final_df['recursion_level'] = final_df['recursion_level'].replace(-1.0, np.nan)
    final_df['size_cutoff'] = final_df['size_cutoff'].replace(-1.0, np.nan)
    
    # Sort for cleanliness
    # Sort size ascending, base_choice descending (so 'faer' comes before 'dgemm' or vice versa, standard sort),
    # recursion_level ascending, size_cutoff ascending.
    final_df = final_df.sort_values(by=group_keys).reset_index(drop=True)
    
    # Ensure correct column ordering
    columns_order = [
        'size', 'base_choice', 'recursion_level', 'size_cutoff',
        'mkl_seq', 'mkl_par', 'faer_seq', 'faer_par',
        'strassen_seq', 'strassen_dfs', 'strassen_bfs', 'strassen_hybrid'
    ]
    # Keep only columns that exist, and append any extra ones
    existing_cols = [c for c in columns_order if c in final_df.columns]
    extra_cols = [c for c in final_df.columns if c not in columns_order]
    final_df = final_df[existing_cols + extra_cols]
    
    # Save the merged results
    output_path = os.path.join(csv_dir, merged_filename)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully wrote merged results to: {output_path}")
    
    # Run the plot script on the merged results
    plot_script = os.path.join(project_root, "python", "plot.py")
    if os.path.exists(plot_script):
        print(f"Generating plots from merged CSV using '{plot_script}'...")
        try:
            # We use 'uv run python/plot.py <csv_path>' as specified in the rules
            res = subprocess.run(["uv", "run", plot_script, output_path], check=True)
            if res.returncode == 0:
                print("Plots generated successfully!")
        except Exception as e:
            print(f"Error running plot script: {e}")
    else:
        print(f"Warning: Plot script not found at {plot_script}. Skipping plot generation.")

if __name__ == "__main__":
    main()
