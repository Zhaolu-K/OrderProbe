# -*- coding: utf-8 -*-
"""
S_Cons Calculator - Structural Consistency

Computes S_Cons (Structural Consistency) by combining E_perf and R_sens.
S_Cons measures overall structural stability through multiplicative aggregation.

Formula: S_Cons = (1 - E_perf) × (1 - R_sens)
Where E_perf is the performance deviation and R_sens is the rigidity sensitivity.

Process:
1. Read E_perf results from e_perf_results.csv
2. Read R_sens results from r_sens_results.csv
3. For each model, compute S_Cons = (1 - E_perf) × (1 - R_sens)
4. Save combined results
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(r"results/consistency")


def calculate_structural_consistency(e_perf: float, r_sens: float) -> float:
    """Calculate S_Cons from E_perf and R_sens values."""
    if pd.isna(e_perf) or pd.isna(r_sens):
        return np.nan

    # S_Cons = (1 - E_perf) × (1 - R_sens)
    s_cons = (1 - e_perf) * (1 - r_sens)
    return s_cons


def main():
    """Main function."""
    print("=" * 60)
    print("S_Cons Calculator - Structural Consistency")
    print("=" * 60)

    # Check results directory
    if not RESULTS_DIR.exists():
        print(f"[ERROR] Results directory does not exist: {RESULTS_DIR}")
        return

    # File paths
    e_perf_file = RESULTS_DIR / "e_perf_results.csv"
    r_sens_file = RESULTS_DIR / "r_sens_results.csv"

    if not e_perf_file.exists():
        print(f"[ERROR] E_perf results file not found: {e_perf_file}")
        print("[INFO] Please run e_perf_calculator.py first")
        return

    if not r_sens_file.exists():
        print(f"[ERROR] R_sens results file not found: {r_sens_file}")
        print("[INFO] Please run r_sens_calculator.py first")
        return

    # Read E_perf results
    try:
        df_e_perf = pd.read_csv(e_perf_file, encoding='utf-8-sig')
        e_perf_dict = dict(zip(df_e_perf['Model Name'], df_e_perf['E_perf']))
    except Exception as e:
        print(f"[ERROR] Failed to read E_perf results: {e}")
        return

    # Read R_sens results
    try:
        df_r_sens = pd.read_csv(r_sens_file, encoding='utf-8-sig')
        r_sens_dict = dict(zip(df_r_sens['Model Name'], df_r_sens['R_sens']))
    except Exception as e:
        print(f"[ERROR] Failed to read R_sens results: {e}")
        return

    # Find common models
    common_models = sorted(set(e_perf_dict.keys()) & set(r_sens_dict.keys()))

    if not common_models:
        print("[ERROR] No common models found between E_perf and R_sens results")
        return

    results = []

    # Calculate S_Cons for each model
    print("Calculating Structural Consistency...")
    print("-" * 40)

    for model_name in common_models:
        e_perf = e_perf_dict[model_name]
        r_sens = r_sens_dict[model_name]

        s_cons = calculate_structural_consistency(e_perf, r_sens)

        if not pd.isna(s_cons):
            results.append({
                'Model Name': model_name,
                'E_perf': e_perf,
                'R_sens': r_sens,
                'S_Cons': s_cons
            })
            print("25s" ".4f" ".4f" ".4f")
        else:
            print(f"[WARN] Failed to calculate S_Cons for {model_name}")

    if not results:
        print("[ERROR] No valid S_Cons values calculated")
        return

    # Save results
    output_file = RESULTS_DIR / "s_cons_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False, engine='openpyxl')

    print(f"\n[INFO] Results saved to: {output_file}")
    print(f"[INFO] Processed {len(results)} models successfully")

    # Print detailed summary
    print("\nStructural Consistency Summary:")
    print("=" * 60)
    print("25s")
    print("-" * 60)
    for result in results:
        print("25s")


if __name__ == "__main__":
    main()
