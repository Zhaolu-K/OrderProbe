# Logical Validity ($S_{\mathrm{Logic}}$) Calculator

This module implements the Logical Validity ($S_{\mathrm{Logic}}$) evaluation metric using Natural Language Inference (NLI) models. $S_{\mathrm{Logic}}$ measures whether generated idiom explanations logically entail their canonical definitions.

## Overview

Logical validity ensures that high-scoring explanations are not only semantically similar but also logically consistent with the ground truth. This prevents models from achieving high scores through keyword matching while conveying contradictory meanings.

**Formula**:
$$S_{\text{Logic}} = P_{\text{NLI}}(e \Rightarrow r)$$
Plain text:
```
S_Log = P_NLI(e ⇒ r)
```

Where:
- `e`: Generated explanation (hypothesis)
- `r`: Canonical reference definition (premise)
- `P_NLI(e ⇒ r)`: Probability that e logically entails r according to NLI model

## Module Structure

```
Slogic/
├── s_log_calculator.py    # Main logical validity calculator
└── README.md              # This documentation
```

## Technical Implementation

### NLI Model
- **Model**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Type**: Multi-lingual NLI classifier
- **Labels**: Contradiction, Neutral, Entailment
- **Output**: Entailment probability (0-1)

### Processing Pipeline

1. **Input Loading**: Read model predictions and reference explanations
2. **Language Detection**: Automatically select appropriate reference file
3. **NLI Computation**: Calculate P_NLI(e ⇒ r) for each prediction-reference pair
4. **Aggregation**: Average across idioms and arrangements
5. **Output**: Generate detailed and summary results

## Usage

### Environment Setup

```bash
# Ensure data directories exist
mkdir -p data/baseline_mean
mkdir -p data/reference_explanations
mkdir -p results/logic
```

### Running the Calculator

```bash
cd demo/Slogic
python s_log_calculator.py
```

### Input Requirements

#### Model Output Files (`data/baseline_mean/`)
- **Format**: CSV (.csv)
- **Sheets**: Each arrangement as separate sheet (arrangement1, arrangement2, etc.)
- **Required Columns**:
  - `Idiom`: Idiom identifier
  - `Prediction`: Model generated explanation

#### Reference Files (`data/reference_explanations/`)
- **Format**: CSV (.csv)
- **Required Columns**:
  - `idiom`: Idiom name
  - `explanation`: Canonical explanation

**Supported Languages**:
- `chinese_explanations.csv`
- `traditional_chinese_explanations.csv`
- `japanese_explanations.csv`
- `korean_explanations.csv`

## Output Format

### Overall Results (`results/logic/s_log_results.csv`)
```
Model Name    S_Log
Model_A       0.785
Model_B       0.692
```

### Detailed Results (`results/logic/{model_name}_s_log.csv`)

**Summary Sheet**:
```
Sheet          Average_S_Log
arrangement1          0.812
arrangement2          0.745
Overall        0.778
```

**Individual Sheets** (one per arrangement):
```
Idiom      Reference                    Prediction                    S_Log
idiom1      [standard explanation...]                [model prediction...]                0.834
idiom2      [standard explanation...]                [model prediction...]                0.756
```

## Score Interpretation

### $S_{\mathrm{Logic}}$ Range: [0, 1]
- **0.0**: Complete logical contradiction
- **0.5**: Neutral logical relationship
- **0.8+**: Strong logical entailment
- **1.0**: Perfect logical entailment

### Use Cases
 - **High** $S_{\mathrm{Logic}}$ **+ High** $S_{\mathrm{Acc}}$: Reliable explanation
 - **High** $S_{\mathrm{Acc}}$ **+ Low** $S_{\mathrm{Logic}}$: Semantically similar but logically inconsistent
- **Low S_Log**: Indicates potential factual errors or contradictions

## Technical Details

### NLI Computation
```python
def compute_nli_probability(premise, hypothesis, tokenizer, model):
    # Tokenize: [CLS] premise [SEP] hypothesis [SEP]
    inputs = tokenizer(premise, hypothesis, ...)
    outputs = model(inputs)

    # Extract entailment probability
    probs = softmax(outputs.logits)
    return probs[entailment_index]
```

### Caching Mechanism
- **Purpose**: Avoid redundant NLI computations
- **Implementation**: Dictionary cache with (premise, hypothesis) keys
- **Benefit**: Significant speedup for repeated evaluations

### Error Handling
- **Missing References**: Returns NaN with warning
- **Model Failures**: Graceful fallback with NaN scores
- **Invalid Inputs**: Comprehensive input validation

## Dependencies

```txt
transformers>=4.21.0
torch>=1.12.0
pandas>=1.5.0
openpyxl>=3.0.0
numpy>=1.21.0
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Set DEVICE='cpu' or reduce batch processing
   ```

2. **Missing Reference Data**
   ```
   Error: Reference file not found
   Solution: Ensure reference_explanations/ contains appropriate .csv files
   ```

3. **NLI Model Loading Failure**
   ```
   Error: Failed to load NLI model
   Solution: Check internet connection and model permissions
   ```

### Performance Optimization

- **GPU Usage**: Automatic CUDA detection for acceleration
- **Batch Processing**: Efficient handling of multiple predictions
- **Caching**: Prevents redundant NLI computations
- **Memory Management**: Optimized for large datasets

## Implementation Notes

- **Multi-lingual Support**: Single NLI model handles all supported languages
- **Probabilistic Output**: Direct entailment probability (not binary classification)
- **Reference Aggregation**: Automatic matching between predictions and references
- **Robust Evaluation**: Handles missing data and edge cases gracefully

## References

This implementation follows the ACL 2026 paper methodology for logical validity assessment using NLI models. The approach ensures that semantic similarity measures are complemented by logical consistency checks.

---

For integration with other metrics ($S_{\mathrm{Acc}}$, $S_{\mathrm{Cons}}$, robustness measures), combine $S_{\mathrm{Logic}}$ with complementary evaluation approaches for comprehensive idiom explanation assessment.
