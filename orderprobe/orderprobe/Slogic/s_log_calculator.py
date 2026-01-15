# -*- coding: utf-8 -*-
"""
S_Log Calculator - Logical Validity

Computes S_Log (Logical Validity) using Natural Language Inference (NLI) model.
S_Log measures the probability that a generated explanation logically entails the canonical definition.

Formula: S_Log = P_NLI(e ⇒ r)
Where e is the generated explanation and r is the reference definition.

Process:
1. Load NLI model for entailment prediction
2. For each idiom, compute entailment probability between prediction and reference
3. Aggregate across multiple references and arrangements
4. Output logical validity scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
BASELINE_MEAN_DIR = Path(r"data/baseline_mean")
ORIGINAL_DATA_DIR = Path(r"data/reference_explanations")
OUTPUT_DIR = Path(r"results/logic")

# NLI Model Configuration
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # Multi-lingual NLI model
MAX_LEN = 512
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cache for NLI predictions
_NLI_CACHE = {}


def load_nli_model(model_name: str = NLI_MODEL_NAME):
    """Load NLI model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[ERROR] Failed to load NLI model {model_name}: {e}")
        return None, None


def get_label_indices(model) -> tuple:
    """Get indices for Contradiction, Neutral, Entailment labels."""
    id2label = getattr(model.config, "id2label", None)
    label2id = getattr(model.config, "label2id", None)

    def norm(s):
        return str(s).strip().lower().replace('-', '').replace('_', '')

    # Try label2id first
    if isinstance(label2id, dict) and len(label2id) >= 3:
        cand = {norm(k): v for k, v in label2id.items()}
        c = cand.get("contradiction", cand.get("contra", -1))
        n = cand.get("neutral", -1)
        e = cand.get("entailment", cand.get("entail", -1))
        if min(c, n, e) >= 0:
            return int(c), int(n), int(e)

    # Try id2label
    if isinstance(id2label, dict) and len(id2label) >= 3:
        rev = {norm(v): int(i) for i, v in id2label.items()}
        c = rev.get("contradiction", rev.get("contra", -1))
        n = rev.get("neutral", -1)
        e = rev.get("entailment", rev.get("entail", -1))
        if min(c, n, e) >= 0:
            return int(c), int(n), int(e)

    # Default order [Contradiction, Neutral, Entailment]
    return 0, 1, 2


def compute_nli_probability(premise: str, hypothesis: str, tokenizer, model) -> float:
    """
    Compute entailment probability using NLI model.

    Args:
        premise: Reference definition (r)
        hypothesis: Generated explanation (e)
        tokenizer: NLI tokenizer
        model: NLI model

    Returns:
        Entailment probability P_NLI(e ⇒ r)
    """
    if tokenizer is None or model is None:
        return np.nan

    # Create cache key
    key = (premise, hypothesis)
    if key in _NLI_CACHE:
        return _NLI_CACHE[key]

    try:
        with torch.no_grad():
            # Tokenize input
            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding=True
            ).to(DEVICE)

            # Get model predictions
            outputs = model(**inputs)
            logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # Get label indices
            c_idx, n_idx, e_idx = get_label_indices(model)

            if max(c_idx, n_idx, e_idx) >= len(probs):
                return np.nan

            # Return entailment probability
            entailment_prob = float(probs[e_idx])

            # Cache result
            _NLI_CACHE[key] = entailment_prob
            return entailment_prob

    except Exception as e:
        print(f"[WARN] NLI computation failed: {e}")
        return np.nan


def compute_logical_validity_for_sheet(df_sheet: pd.DataFrame, df_reference: pd.DataFrame,
                                       tokenizer, model) -> pd.DataFrame:
    """Compute logical validity for a single sheet."""
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
            s_log = np.nan
        else:
            # Compute logical validity: P_NLI(e ⇒ r)
            s_log = compute_nli_probability(reference, prediction, tokenizer, model)

        results.append({
            'Idiom': idiom,
            'Reference': reference or '',
            'Prediction': prediction,
            'S_Log': s_log
        })

    return pd.DataFrame(results)


def compute_logical_validity_for_file(mean_file: Path, tokenizer, model) -> dict:
    """Compute logical validity for a single Excel file."""
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
                df_result = compute_logical_validity_for_sheet(df_sheet, df_reference, tokenizer, model)

                if not df_result.empty:
                    # Calculate average S_Log for this sheet
                    valid_scores = df_result['S_Log'].dropna()
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
    print("S_Log Calculator - Logical Validity")
    print("=" * 60)

    # Load NLI model
    print("Loading NLI model...")
    tokenizer, model = load_nli_model()
    if tokenizer is None or model is None:
        print("[ERROR] Failed to load NLI model. Exiting.")
        return

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

        # Compute logical validity
        file_result = compute_logical_validity_for_file(mean_file, tokenizer, model)

        if file_result and not pd.isna(file_result.get('overall_average')):
            results.append({
                'Model Name': model_name,
                'S_Log': file_result['overall_average']
            })

            # Save detailed results
            output_file = OUTPUT_DIR / f"{model_name}_s_log.csv"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for sheet_name, sheet_result in file_result['sheet_results'].items():
                    summary_data.append({
                        'Sheet': sheet_name,
                        'Average_S_Log': sheet_result['average']
                    })

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Individual sheet results
                for sheet_name, sheet_result in file_result['sheet_results'].items():
                    sheet_result['data'].to_excel(writer, sheet_name=sheet_name, index=False)

            print(".4f")
        else:
            print(f"[WARN] Failed to calculate S_Log for {model_name}")

    if not results:
        print("[ERROR] No valid results calculated")
        return

    # Save overall results
    output_file = OUTPUT_DIR / "s_log_results.csv"
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False, engine='openpyxl')

    print(f"\n[INFO] Results saved to: {output_file}")
    print(f"[INFO] Processed {len(results)} models successfully")

    # Print summary
    print("\nLogical Validity Summary:")
    print("-" * 40)
    for result in results:
        print("25s")


if __name__ == "__main__":
    main()
