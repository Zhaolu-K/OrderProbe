# -*- coding: utf-8 -*-
"""
S_Rob Calculator - Overall Robustness Score

Computes the final robustness score S_Rob as the harmonic mean of sequential
and structural robustness components.

Formula: S_Rob = 2·S_seq·S_struct / (S_seq + S_struct)

This harmonic mean aggregation penalizes imbalance, requiring models to
demonstrate robustness in both internal ordering (S_seq) and syntactic
generalization (S_struct).

Process:
1. Load S_seq results from sequential robustness calculation
2. Load S_struct results from structural robustness calculation
3. Compute harmonic mean for each model
4. Output final robustness assessment
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(r"results/robustness")
OUTPUT_DIR = Path(r"results/robustness")

def calculate_s_rob(s_seq: float, s_struct: float) -> float:
    """Calculate S_Rob using harmonic mean of S_seq and S_struct."""
    if pd.isna(s_seq) or pd.isna(s_struct):
        return np.nan

    # Avoid division by zero
    denominator = s_seq + s_struct
    if denominator == 0:
        return np.nan

    # Harmonic mean: 2·S_seq·S_struct / (S_seq + S_struct)
    s_rob = (2 * s_seq * s_struct) / denominator
    return s_rob

def main():
    """Main function."""
    print("S_Rob Calculator - Overall Robustness Score")
    print("Formula: S_Rob = 2·S_seq·S_struct / (S_seq + S_struct)")

    # File paths
    sseq_file = RESULTS_DIR / "sseq_results.csv"
    sstruct_file = RESULTS_DIR / "s_struct_results.csv"

    if not sseq_file.exists() or not sstruct_file.exists():
        print("S_seq or S_struct result files not found.")
        print("Please run sseq_calculator.py and simple_sstruct.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_sseq = pd.read_csv(sseq_file, encoding='utf-8-sig')
    df_sstruct = pd.read_csv(sstruct_file, encoding='utf-8-sig')

    # Merge on Model column
    df_combined = pd.merge(df_sseq, df_sstruct, on='Model', how='outer')

    results = []
    for _, row in df_combined.iterrows():
        model = row['Model']
        s_seq = row['S_seq'] if 'S_seq' in row and not pd.isna(row['S_seq']) else np.nan
        s_struct = row['S_struct'] if 'S_struct' in row and not pd.isna(row['S_struct']) else np.nan

        s_rob = calculate_s_rob(s_seq, s_struct)

        results.append({
            'Model': model,
            'S_seq': s_seq,
            'S_struct': s_struct,
            'S_Rob': round(s_rob, 6) if not np.isnan(s_rob) else np.nan
        })

        if not np.isnan(s_rob):
            print(f"{model}: S_Rob = 2×{s_seq:.6f}×{s_struct:.6f} / ({s_seq:.6f} + {s_struct:.6f}) = {s_rob:.6f}")

    # Save results
    if results:
        output_file = OUTPUT_DIR / "srob_results.csv"
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Final robustness results saved to: {output_file}")

if __name__ == '__main__':
    main()
