# -*- coding: utf-8 -*-
"""
S_struct Calculator - Paper Standard Format

Formula: S_struct = 1 - Normalize(σ(μ₁, μ₂, ..., μ₆))
where μ_k = (1/|D_k|) * Σ_{x ∈ D_k} S_Acc^mean(x)

Process:
1. Detect language from filename
2. Parse structure mapping for detected language
3. Read average results to get S_Acc^mean for each idiom
4. Calculate μ_k for each structure k
5. Compute standard deviation and normalize
6. Return final S_struct score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Config
BASELINE_MEAN_DIR = Path(r"data/baseline_mean")
STRUCTURE_FILE = Path(r"data/structure_definitions.txt")
OUTPUT_DIR = Path(r"results")

# Structures and languages
STRUCTURES = ['verb_object', 'parallel', 'complement', 'serial', 'modifier_head', 'subject_predicate']
LANG_KEYWORDS = {'chinese': ['chinese'], 'traditional': ['traditional'], 'japanese': ['japanese'], 'korean': ['korean']}

def parse_structures():
    """Parse structure definition file."""
    structure_map = {}
    current_lang = None

    with open(STRUCTURE_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Detect language
        lang_match = None
        for lang_name in LANG_KEYWORDS.keys():
            if line == lang_name or line == lang_name + '：':
                lang_match = lang_name
                break

        if lang_match:
            current_lang = lang_match
            if current_lang not in structure_map:
                structure_map[current_lang] = {}
            i += 1
            continue

        # Detect structure
        if current_lang:
            for struct in STRUCTURES:
                if struct in line:
                    row_numbers = []
                    if '：' in line:
                        parts = re.split('：', line, 1)
                        if len(parts) > 1:
                            matches = re.findall(r'line (\d+)', parts[1])
                            if matches:
                                row_numbers = [int(m) for m in matches]

                    if not row_numbers and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line:
                            matches = re.findall(r'line (\d+)', next_line)
                            if matches:
                                row_numbers = [int(m) for m in matches]
                                i += 1

                    if row_numbers:
                        structure_map[current_lang][struct] = row_numbers
                    break
        i += 1

    return structure_map

def detect_language(filename: str) -> str:
    """Detect language from filename."""
    for lang, keywords in LANG_KEYWORDS.items():
        for keyword in keywords:
            if keyword in filename:
                return lang
    return 'chinese'

def calculate_s_struct(mean_file: Path, structure_map: dict) -> float:
    """Calculate S_struct using paper formula."""
    # Detect language
    lang = detect_language(mean_file.name)
    if lang not in structure_map:
        return np.nan

    lang_structure_map = structure_map[lang]

    # Read average results (CSV format)
    mean_file_path = Path(mean_file)
    summary_csv = mean_file_path.parent / f"{mean_file_path.stem}_summary.csv"
    df_avg = pd.read_csv(summary_csv, encoding='utf-8-sig')

    # Find S_Acc column
    s_acc_col = None
    for col in df_avg.columns:
        if 'average_S_Acc' in col:
            s_acc_col = col
            break

    if s_acc_col is None:
        return np.nan

    s_acc_values = df_avg[s_acc_col].replace([np.inf, -np.inf], np.nan)

    # Calculate μ_k for each structure
    structure_means = {}
    for struct in STRUCTURES:
        if struct not in lang_structure_map:
            continue

        row_indices = []
        for row_num in lang_structure_map[struct]:
            pandas_idx = row_num - 2
            if 0 <= pandas_idx < len(df_avg):
                row_indices.append(pandas_idx)

        if len(row_indices) == 0:
            continue

        struct_scores = s_acc_values.iloc[row_indices]
        valid_scores = struct_scores.dropna()
        if len(valid_scores) > 0:
            structure_means[struct] = valid_scores.mean()

    # Calculate standard deviation
    valid_means = list(structure_means.values())
    if len(valid_means) < 2:
        return np.nan

    std_dev = np.std(valid_means, ddof=0)

    # Normalize
    mean_range = max(valid_means) - min(valid_means)
    if mean_range > 0:
        normalized_std = std_dev / mean_range
    else:
        normalized_std = 0.0

    # S_struct = 1 - normalized_std
    s_struct = 1.0 - min(1.0, max(0.0, normalized_std))
    return s_struct

def main():
    """Main function."""
    print("S_struct Calculator - Paper Standard Format")

    if not BASELINE_MEAN_DIR.exists() or not STRUCTURE_FILE.exists():
        print("Required files not found")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse structures
    structure_map = parse_structures()

    # Process all files
    mean_files = list(BASELINE_MEAN_DIR.glob("*.csv"))
    results = []

    for mean_file in sorted(mean_files):
        try:
            s_struct = calculate_s_struct(mean_file, structure_map)
            lang = detect_language(mean_file.name)
            model_name = Path(mean_file).stem

            results.append({
                'Model': model_name,
                'Language': lang,
                'S_struct': round(s_struct, 6) if not np.isnan(s_struct) else np.nan
            })

            print(f"{model_name}: S_struct = {s_struct:.6f}")

        except Exception as e:
            print(f"Error: {mean_file.name} - {e}")
            continue

    # Save results
    if results:
        output_file = OUTPUT_DIR / "s_struct_results.csv"
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
