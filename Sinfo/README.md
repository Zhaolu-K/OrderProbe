# Information Density ($S_{\mathrm{Info}}$) Calculator

This module implements the Information Density (`S_Info`) evaluation metric that rewards concise, high-information explanations by penalizing verbose outputs that lack proportional semantic gain.

## Overview

Information Density addresses the "knowledge dumping" problem where models generate excessively verbose explanations to maximize keyword coverage. $S_{\mathrm{Info}}$ encourages models to prioritize precise reasoning over redundant generation.

**Formula**:
$$S_{\text{Info}} = \text{BP} \cdot P_{\text{ROUGE}}$$

Where:
- **P_ROUGE**: ROUGE precision score (semantic overlap)
- **BP**: Brevity Penalty factor (length-based penalty)

## Module Structure

```
Sinfo/
├── s_info_calculator.py    # Main information density calculator
└── README.md              # This documentation
```

## Components

### 1. ROUGE Precision (P_ROUGE)

**Purpose**: Measures semantic overlap between prediction and reference

**Implementation**:
- Uses ROUGE-L precision score
- Falls back to character-level similarity if ROUGE library unavailable
- Higher scores indicate better semantic coverage

### 2. Brevity Penalty (BP)

**Purpose**: Penalizes explanations that exceed reference length without proportional semantic gain

**Formula**:
$$
\mathrm{BP} =
\begin{cases}
1, & \text{if } c > r,\\[4pt]
\exp(1 - r/c), & \text{if } c \le r.
\end{cases}
$$
Display form:
$$
\mathrm{BP} = 
\begin{cases}
1 & \text{if } c > r,\\[4pt]
\exp(1 - r/c) & \text{if } c \le r.
\end{cases}
$$

Where:
- `c`: Prediction length (candidate)
- `r`: Reference length
- Penalty increases as prediction becomes much shorter than reference

**Rationale**:
- **Longer predictions**: No penalty (may need more words to explain complex concepts)
- **Shorter predictions**: Exponential penalty (may lack necessary detail)

## Usage

### Environment Setup

```bash
# Ensure data directories exist
mkdir -p data/baseline_mean
mkdir -p data/reference_explanations
mkdir -p results/information_density
```

### Running the Calculator

```bash
cd demo/Sinfo
python s_info_calculator.py
```

### Input Requirements

#### Model Output Files (`data/baseline_mean/`)
- **Format**: CSV (.csv)
- **Sheets**: Each arrangement as separate sheet
- **Required Columns**:
  - `Idiom`: Idiom identifier
  - `Prediction`: Model generated explanation

#### Reference Files (`data/reference_explanations/`)
- **Format**: CSV (.csv)
- **Required Columns**:
  - `idiom`: Idiom name
  - `explanation`: Canonical explanation

## Output Format

### Overall Results (`results/information_density/s_info_results.csv`)
```
Model Name    S_Info
Model_A       0.723
Model_B       0.689
```

### Detailed Results (`results/information_density/{model_name}_s_info.csv`)

**Summary Sheet**:
```
Sheet          Average_S_Info
permutation1   0.756
permutation2   0.698
Overall        0.727
```

**Individual Sheets** (one per arrangement):
```
Idiom      Reference                    Prediction                    P_ROUGE    BP      S_Info
idiom1     [reference explanation...]    [model prediction...]       0.834      0.945   0.788
idiom2     [reference explanation...]    [model prediction...]       0.712      0.876   0.624
```

## Technical Details

### ROUGE Implementation

#### Primary Method (with rouge library):
```python
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(prediction, reference, avg=True)
p_rouge = scores['rouge-l']['p']  # ROUGE-L precision
```

#### Fallback Method (character-level):
```python
pred_chars = set(prediction)
ref_chars = set(reference)
intersection = len(pred_chars & ref_chars)
p_rouge = intersection / len(pred_chars) if pred_chars else 0.0
```

### Brevity Penalty Calculation

```python
def calculate_brevity_penalty(pred, ref):
    c = len(pred)  # Candidate length
    r = len(ref)   # Reference length

    if c > r:
        return 1.0  # No penalty for longer predictions
    else:
        return math.exp(1 - r/c)  # Penalty for shorter predictions
```

### Information Density Aggregation

```python
s_info = bp * p_rouge  # Element-wise multiplication
# Then aggregated across idioms and arrangements
```

## Score Interpretation

### $S_{\mathrm{Info}}$ Range: [0, 1]
- **0.0**: No semantic overlap or extreme brevity penalty
- **0.5**: Moderate information density
- **0.8+**: High information density (concise and informative)
- **1.0**: Perfect semantic overlap with optimal length

### Component Analysis
- **High P_ROUGE + High BP**: Well-balanced explanation
- **High P_ROUGE + Low BP**: Verbose but semantically rich
- **Low P_ROUGE + High BP**: Concise but semantically poor
- **Low P_ROUGE + Low BP**: Poor overall quality

### Use Cases
- **Encourages Conciseness**: Penalizes unnecessary verbosity
- **Rewards Efficiency**: Higher scores for information-rich explanations
- **Balances Trade-offs**: Semantic quality vs. explanation length

## Dependencies

```txt
pandas>=1.5.0
numpy>=1.21.0
openpyxl>=3.0.0
rouge>=1.0.0        # Optional, fallback available
```

## Troubleshooting

### Common Issues

1. **ROUGE Library Not Available**
   ```
   Solution: Install rouge library or use character-level fallback
   pip install rouge
   ```

2. **Missing Reference Data**
   ```
   Error: Reference file not found
   Solution: Ensure reference_explanations/ contains appropriate .csv files
   ```

3. **Empty Predictions**
   ```
   Result: NaN scores
   Solution: Check input data quality and preprocessing
   ```

### Performance Optimization

- **ROUGE Library**: Significantly faster than character-level fallback
- **Batch Processing**: Efficient handling of multiple predictions
- **Memory Usage**: Minimal memory footprint for large datasets

## Implementation Notes

- **Multi-lingual Support**: Automatic language detection and reference selection
- **Robust Evaluation**: Handles missing data and edge cases gracefully
- **Flexible Metrics**: Supports both library-based and fallback ROUGE implementations
- **Aggregation Options**: Configurable aggregation across multiple references

## References

This implementation follows the ACL 2026 paper methodology for Information Density evaluation. The brevity penalty mechanism effectively counters verbose generation strategies while maintaining semantic evaluation quality.

---

For integration with other metrics ($S_{\mathrm{Acc}}$, $S_{\mathrm{Log}}$, $S_{\mathrm{Cons}}$), combine $S_{\mathrm{Info}}$ with complementary evaluation approaches for comprehensive idiom explanation assessment.
