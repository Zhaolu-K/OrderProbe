# -*- coding: utf-8 -*-
"""
S_seq Calculator - Sequential Robustness

Computes S_seq (Sequential Robustness) using MDA and MDR values.
S_seq measures resilience against internal ordering permutations.

Formula: S_seq = 1 - (α·MDR + β·MDA)
where α = β = 0.5 (equal weights for average stability and peak resilience)

Process:
1. Load MDA and MDR results
2. For each model, compute S_seq = 1 - (0.5·MDR + 0.5·MDA)
3. Output results with equal weighting of degradation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(r"results/robustness")
OUTPUT_DIR = Path(r"results/robustness")

# Weights for S_seq calculation
ALPHA = 0.5  # Weight for MDR (average degradation)
BETA = 0.5   # Weight for MDA (maximum degradation)

def calculate_s_seq(mdr_value: float, mda_value: float) -> float:
    """Calculate S_seq using MDA and MDR."""
    if pd.isna(mdr_value) or pd.isna(mda_value):
        return np.nan

    s_seq = 1 - (ALPHA * mdr_value + BETA * mda_value)
    return s_seq

def main():
    """Main function."""
    print("S_seq Calculator - Sequential Robustness")
    print(f"Formula: S_seq = 1 - ({ALPHA}·MDR + {BETA}·MDA)")

    # File paths
    mdr_file = RESULTS_DIR / "mdr_results.csv"
    mda_file = RESULTS_DIR / "mda_results.csv"

    if not mdr_file.exists() or not mda_file.exists():
        print("MDR or MDA result files not found. Please run mdr_calculator.py and mda_calculator.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_mdr = pd.read_csv(mdr_file, encoding='utf-8-sig')
    df_mda = pd.read_csv(mda_file, encoding='utf-8-sig')

    # Merge on Model column
    df_combined = pd.merge(df_mdr, df_mda, on='Model', how='outer')

    results = []
    for _, row in df_combined.iterrows():
        model = row['Model']
        mdr = row['MDR'] if 'MDR' in row and not pd.isna(row['MDR']) else np.nan
        mda = row['MDA'] if 'MDA' in row and not pd.isna(row['MDA']) else np.nan

        s_seq = calculate_s_seq(mdr, mda)

        results.append({
            'Model': model,
            'MDR': mdr,
            'MDA': mda,
            'S_seq': round(s_seq, 6) if not np.isnan(s_seq) else np.nan
        })

        if not np.isnan(s_seq):
            print(f"{model}: S_seq = 1 - ({ALPHA}×{mdr:.6f} + {BETA}×{mda:.6f}) = {s_seq:.6f}")

    # Save results
    if results:
        output_file = OUTPUT_DIR / "sseq_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
