# -*- coding: utf-8 -*-
"""
S_Info Calculator - Information Density

Computes S_Info (Information Density) using ROUGE precision and brevity penalty.
S_Info rewards high information content per token by penalizing verbose explanations.

Formula: S_Info = BP × P_ROUGE
Where:
- P_ROUGE: ROUGE precision score
- BP: Brevity penalty factor

Process:
1. Load model predictions and reference explanations
2. Compute ROUGE precision for each prediction-reference pair
3. Apply brevity penalty based on length ratio
4. Aggregate across idioms and arrangements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import math
import re

# Configuration
BASELINE_MEAN_DIR = Path(r"data/baseline_mean")
ORIGINAL_DATA_DIR = Path(r"data/reference_explanations")
OUTPUT_DIR = Path(r"results/information_density")

# ROUGE Configuration
MAX_N_GRAM = 4  # Maximum n-gram order for ROUGE
BETA = 1.0      # F-beta score parameter (standard F1)


def load_rouge_library():
    """Load ROUGE evaluation library."""
    try:
        from rouge import Rouge
        return Rouge()
    except ImportError:
        print("[WARN] ROUGE library not available. Using simplified implementation.")
        return None


def clean_text(text: str) -> str:
    """Clean and normalize text for evaluation."""
    if pd.isna(text):
        return ""
    # Remove brackets and normalize whitespace
    text = re.sub(r'[\[\]]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for ROUGE evaluation."""
    if not text:
        return []

    # Simple sentence splitting (can be enhanced)
    sentences = re.split(r'[。！？.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_rouge_precision(pred: str, ref: str, rouge=None) -> float:
    """
    Calculate ROUGE precision score.

    Args:
        pred: Generated prediction text
        ref: Reference text
        rouge: ROUGE evaluation object (optional)

    Returns:
        ROUGE precision score (0-1)
    """
    if not pred or not ref:
        return 0.0

    try:
        if rouge is not None:
            # Use ROUGE library if available
            scores = rouge.get_scores(pred, ref, avg=True)
            return scores['rouge-l']['p']  # Precision of ROUGE-L
        else:
            # Simplified character-level ROUGE implementation
            pred_chars = set(pred)
            ref_chars = set(ref)

            if not ref_chars:
                return 0.0

            # Character-level precision: intersection / prediction length
            intersection = len(pred_chars & ref_chars)
            precision = intersection / len(pred_chars) if pred_chars else 0.0

            return float(precision)

    except Exception as e:
        print(f"[WARN] ROUGE calculation failed: {e}")
        return 0.0


def calculate_brevity_penalty(pred: str, ref: str) -> float:
    """
    Calculate Brevity Penalty (BP) to penalize verbose explanations.

    BP imposes penalty on outputs that exceed reference length without
    proportional semantic gain.

    Formula:
    BP = 1 if len(pred) > len(ref)  (no penalty for longer predictions)
         exp(1 - len(ref)/len(pred)) if len(pred) <= len(ref)

    Args:
        pred: Generated prediction text
        ref: Reference text

    Returns:
        Brevity penalty factor (0-1)
    """
    if not pred or not ref:
        return 0.0

    c = len(pred)  # Candidate (prediction) length
    r = len(ref)   # Reference length

    if c == 0:
        return 0.0

    if c > r:
        # No penalty for longer predictions (they may need more words to explain)
        return 1.0
    else:
        # Exponential penalty for shorter predictions
        # As prediction gets much shorter than reference, penalty increases
        return float(math.exp(1 - r / c))


def calculate_information_density(pred: str, ref: str, rouge=None) -> Tuple[float, float, float]:
    """
    Calculate Information Density components.

    Args:
        pred: Generated prediction text
        ref: Reference text
        rouge: ROUGE evaluation object (optional)

    Returns:
        Tuple of (P_ROUGE, BP, S_Info)
    """
    if not pred or not ref:
        return 0.0, 0.0, 0.0

    # Calculate ROUGE precision
    p_rouge = calculate_rouge_precision(pred, ref, rouge)

    # Calculate brevity penalty
    bp = calculate_brevity_penalty(pred, ref)

    # Calculate information density: S_Info = BP × P_ROUGE
    s_info = bp * p_rouge

    return p_rouge, bp, s_info


def process_sheet(df_sheet: pd.DataFrame, df_reference: pd.DataFrame,
                  rouge=None) -> pd.DataFrame:
    """Process a single sheet to calculate information density."""
    results = []

    # Create reference mapping
    ref_dict = {}
    if 'idiom' in df_reference.columns and 'explanation' in df_reference.columns:
        for _, row in df_reference.iterrows():
            idiom = str(row['idiom']).strip()
            explanation = str(row['explanation']).strip()
            if idiom and explanation:
                ref_dict[idiom] = explanation

    # Find prediction column
    pred_col = None
    for col in df_sheet.columns:
        if 'Prediction' in col:
            pred_col = col
            break

    if pred_col is None:
        print("[WARN] No prediction column found")
        return pd.DataFrame()

    # Process each idiom
    for idx, row in df_sheet.iterrows():
        idiom = str(row.get('Idiom', f'Idiom_{idx}')).strip()
        prediction = str(row[pred_col]).strip()

        # Get reference explanation
        reference = ref_dict.get(idiom)
        if not reference:
            print(f"[WARN] No reference found for idiom: {idiom}")
            p_rouge, bp, s_info = 0.0, 0.0, 0.0
        else:
            # Calculate information density
            p_rouge, bp, s_info = calculate_information_density(prediction, reference, rouge)

        results.append({
            'Idiom': idiom,
            'Reference': reference or '',
            'Prediction': prediction,
            'P_ROUGE': p_rouge,
            'BP': bp,
            'S_Info': s_info
        })

    return pd.DataFrame(results)


def process_file(mean_file: Path, rouge=None) -> dict:
    """Process a single Excel file."""
    try:
        # Read Excel file
        excel = pd.ExcelFile(mean_file, engine='openpyxl')
        sheet_names = [s for s in excel.sheet_names if s != 'summary']

        if not sheet_names:
            print(f"[WARN] No valid sheets found in {mean_file.name}")
            return {}

        # Load reference data
        reference_file = get_reference_file(mean_file)
        if not reference_file.exists():
            print(f"[WARN] Reference file not found: {reference_file}")
            df_reference = pd.DataFrame()
        else:
            df_reference = pd.read_csv(reference_file, encoding='utf-8-sig')

        sheet_results = {}

        # Process each arrangement sheet
        for sheet_name in sheet_names:
            try:
                # For CSV format, construct the corresponding CSV file path
                mean_file_path = Path(mean_file)
                csv_file = mean_file_path.parent / f"{mean_file_path.stem}_{sheet_name}.csv"
                df_sheet = pd.read_csv(csv_file, encoding='utf-8-sig')
                df_result = process_sheet(df_sheet, df_reference, rouge)

                if not df_result.empty:
                    # Calculate average S_Info for this sheet
                    valid_scores = df_result['S_Info'].dropna()
                    if len(valid_scores) > 0:
                        sheet_avg = valid_scores.mean()
                    else:
                        sheet_avg = np.nan

                    sheet_results[sheet_name] = {
                        'data': df_result,
                        'average': sheet_avg
                    }

            except Exception as e:
                print(f"[WARN] Failed to process sheet {sheet_name}: {e}")
                continue

        # Calculate overall average across all sheets
        sheet_averages = [result['average'] for result in sheet_results.values() if not pd.isna(result['average'])]
        overall_average = np.mean(sheet_averages) if sheet_averages else np.nan

        return {
            'sheet_results': sheet_results,
            'overall_average': overall_average
        }

    except Exception as e:
        print(f"[ERROR] Failed to process file {mean_file.name}: {e}")
        return {}


def get_reference_file(mean_file: Path) -> Path:
    """Get corresponding reference file based on language detection."""
    filename = mean_file.name.lower()

    if 'chinese' in filename:
        return ORIGINAL_DATA_DIR / "chinese_explanations.csv"
    elif 'traditional' in filename:
        return ORIGINAL_DATA_DIR / "traditional_chinese_explanations.csv"
    elif 'japanese' in filename:
        return ORIGINAL_DATA_DIR / "japanese_explanations.csv"
    elif 'korean' in filename:
        return ORIGINAL_DATA_DIR / "korean_explanations.csv"
    else:
        # Default to Chinese
        return ORIGINAL_DATA_DIR / "chinese_explanations.csv"


def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    return filename.replace('.csv', '')


def main():
    """Main function."""
    print("=" * 60)
    print("S_Info Calculator - Information Density")
    print("=" * 60)

    # Load ROUGE library (optional)
    rouge = load_rouge_library()

    # Check directories
    if not BASELINE_MEAN_DIR.exists():
        print(f"[ERROR] Mean directory does not exist: {BASELINE_MEAN_DIR}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all Excel files
    mean_files = list(BASELINE_MEAN_DIR.glob("*.csv"))

    if not mean_files:
        print(f"[ERROR] No Excel files found in: {BASELINE_MEAN_DIR}")
        return

    results = []

    # Process each file
    for mean_file in sorted(mean_files):
        model_name = extract_model_name(mean_file.name)
        print(f"\nProcessing {model_name}...")

        # Calculate information density
        file_result = process_file(mean_file, rouge)

        if file_result and not pd.isna(file_result.get('overall_average')):
            results.append({
                'Model Name': model_name,
                'S_Info': file_result['overall_average']
            })

            # Save detailed results
            output_file = OUTPUT_DIR / f"{model_name}_s_info.csv"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for sheet_name, sheet_result in file_result['sheet_results'].items():
                    summary_data.append({
                        'Sheet': sheet_name,
                        'Average_S_Info': sheet_result['average']
                    })

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Individual sheet results
                for sheet_name, sheet_result in file_result['sheet_results'].items():
                    sheet_result['data'].to_excel(writer, sheet_name=sheet_name, index=False)

            print(".4f")
        else:
            print(f"[WARN] Failed to calculate S_Info for {model_name}")

    if not results:
        print("[ERROR] No valid results calculated")
        return

    # Save overall results
    output_file = OUTPUT_DIR / "s_info_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False, engine='openpyxl')

    print(f"\n[INFO] Results saved to: {output_file}")
    print(f"[INFO] Processed {len(results)} models successfully")

    # Print summary
    print("\nInformation Density Summary:")
    print("-" * 40)
    for result in results:
        print("25s")


if __name__ == "__main__":
    main()
