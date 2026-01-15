# -*- coding: utf-8 -*-
"""
R_sens Calculator - Rigidity Sensitivity

Computes R_sens (Rigidity Sensitivity) as part of structural consistency assessment.
R_sens measures the maximum gap between recognition capability and realized performance across all permutations.

Formula: R_sens = max_{p ∈ P} (S_max - S_mean)
Where P is the set of all permutations, S_max is the maximum semantic accuracy score
among explanations generated for permutation p, and S_mean is the average semantic accuracy score.

Process:
1. Read S_Acc scores from mean and max result files for each arrangement
2. For each arrangement, calculate the gap between max and mean scores
3. Find the maximum gap across all arrangements
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
BASELINE_MEAN_DIR = Path(r"data/baseline_mean")
BASELINE_MAX_DIR = Path(r"data/baseline_max")
OUTPUT_DIR = Path(r"results/consistency")


def calculate_rigidity_sensitivity(mean_file: Path, max_file: Path) -> float:
    """Calculate R_sens (Rigidity Sensitivity) for a single file pair."""
    # Read Excel files
    try:
        mean_excel = pd.ExcelFile(mean_file, engine='openpyxl')
        max_excel = pd.ExcelFile(max_file, engine='openpyxl')
        mean_sheets = [s for s in mean_excel.sheet_names if s != 'summary']
        max_sheets = [s for s in max_excel.sheet_names if s != 'summary']
        mean_excel.close()
        max_excel.close()
    except Exception as e:
        print(f"[ERROR] Failed to read Excel files: {e}")
        return np.nan

    # Find common arrangement sheets
    common_sheets = sorted(set(mean_sheets) & set(max_sheets))

    if not common_sheets:
        print(f"[WARN] No common arrangement sheets found between {mean_file.name} and {max_file.name}")
        return np.nan

    arrangement_gaps = []

    # Process each arrangement
    for sheet_name in common_sheets:
        try:
            # Read mean sheet (CSV format)
            mean_file_path = Path(mean_file)
            mean_csv = mean_file_path.parent / f"{mean_file_path.stem}_{sheet_name}.csv"
            df_mean = pd.read_csv(mean_csv, encoding='utf-8-sig')

            # Read max sheet (CSV format)
            max_file_path = Path(max_file)
            max_csv = max_file_path.parent / f"{max_file_path.stem}_{sheet_name}.csv"
            df_max = pd.read_csv(max_csv, encoding='utf-8-sig')

            # Find S_Acc columns
            s_acc_col_mean = None
            s_acc_col_max = None

            for col in df_mean.columns:
                if col.endswith('_S_Acc'):
                    s_acc_col_mean = col
                    break

            for col in df_max.columns:
                if col.endswith('_S_Acc'):
                    s_acc_col_max = col
                    break

            if s_acc_col_mean is None or s_acc_col_max is None:
                print(f"[WARN] S_Acc column not found in sheet {sheet_name}")
                continue

            # Get S_Acc values
            mean_values = df_mean[s_acc_col_mean].replace([np.inf, -np.inf], np.nan).dropna()
            max_values = df_max[s_acc_col_max].replace([np.inf, -np.inf], np.nan).dropna()

            if len(mean_values) == 0 or len(max_values) == 0:
                print(f"[WARN] No valid S_Acc values in sheet {sheet_name}")
                continue

            # Calculate S_mean and S_max for this arrangement
            s_mean = mean_values.mean()  # Average of mean scores
            s_max = max_values.max()     # Maximum of max scores

            if not (pd.isna(s_mean) or pd.isna(s_max)):
                # Calculate gap for this arrangement: S_max - S_mean
                gap = s_max - s_mean
                arrangement_gaps.append(gap)

        except Exception as e:
            print(f"[WARN] Error processing sheet {sheet_name}: {e}")
            continue

    if not arrangement_gaps:
        print(f"[WARN] No valid arrangement gaps calculated")
        return np.nan

    # R_sens = maximum of (S_max - S_mean) across all arrangements
    r_sens = max(arrangement_gaps)
    return r_sens


def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    return filename.replace('.csv', '')


def main():
    """Main function."""
    print("=" * 60)
    print("R_sens Calculator - Rigidity Sensitivity")
    print("=" * 60)

    # Check directories
    if not BASELINE_MEAN_DIR.exists():
        print(f"[ERROR] Mean directory does not exist: {BASELINE_MEAN_DIR}")
        return

    if not BASELINE_MAX_DIR.exists():
        print(f"[ERROR] Max directory does not exist: {BASELINE_MAX_DIR}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all Excel files from mean directory
    mean_files = list(BASELINE_MEAN_DIR.glob("*.csv"))

    if not mean_files:
        print(f"[ERROR] No Excel files found in mean directory: {BASELINE_MEAN_DIR}")
        return

    results = []

    # Process each file
    for mean_file in sorted(mean_files):
        model_name = extract_model_name(mean_file.name)

        # Find corresponding max file
        max_file = BASELINE_MAX_DIR / mean_file.name
        if not max_file.exists():
            print(f"[WARN] Max file not found for {mean_file.name}, skipping")
            continue

        print(f"Processing {model_name}...")

        # Calculate R_sens
        r_sens = calculate_rigidity_sensitivity(mean_file, max_file)

        if not pd.isna(r_sens):
            results.append({
                'Model Name': model_name,
                'R_sens': r_sens
            })
            print(".4f")
        else:
            print(f"[WARN] Failed to calculate R_sens for {model_name}")

    if not results:
        print("[ERROR] No valid results calculated")
        return

    # Save results
    output_file = OUTPUT_DIR / "r_sens_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False, engine='openpyxl')

    print(f"\n[INFO] Results saved to: {output_file}")
    print(f"[INFO] Processed {len(results)} models successfully")

    # Print summary
    print("\nR_sens Summary:")
    print("-" * 40)
    for result in results:
        print("25s")


if __name__ == "__main__":
    main()
