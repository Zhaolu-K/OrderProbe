# Semantic Accuracy ($S_{\mathrm{Acc}}$) Calculator

This module implements the Semantic Accuracy ($S_{\mathrm{Acc}}$) evaluation framework for idiom explanation quality assessment. $S_{\mathrm{Acc}}$ is a hierarchical three-layer hybrid metric that integrates multiple similarity measures to evaluate the semantic quality of generated idiom explanations.

## Overview

$S_{\mathrm{Acc}}$ evaluates idiom explanations using a weighted ensemble of three complementary approaches:

**Formula**:

$$
S_{\mathrm{Acc}}^{\mathrm{mean}} = w_1 \cdot S'_{\mathrm{ce}} + w_2 \cdot \left(\frac{S'_{\mathrm{bert}} + S'_{\mathrm{sts}} + S'_{\mathrm{cos}}}{3}\right) + w_3 \cdot S_{f\beta}
$$

Where:
- `w₁ = 0.5`: Cross-Encoder weight (primary evaluator)
- `w₂ = 0.3`: Representation ensemble weight
- `w₃ = 0.2`: F_β score weight (lexical safeguard)

## Module Structure

```
Sacc/
├── sacc_chinese.py           # Chinese idiom evaluation
├── sacc_traditional_chinese.py  # Traditional Chinese evaluation
├── sacc_japanese.py          # Japanese idiom evaluation
├── sacc_korean.py            # Korean idiom evaluation
└── README.md                 # This documentation
```

## Components

### 1. Cross-Encoder

**Weight**: 0.5 (Primary evaluator)

**Implementation**:
- Model: `BAAI/bge-reranker-base`
- Function: `cross_encoder_over_refs`
- Purpose: Full self-attention over concatenated input for detecting subtle semantic entailments

**Process**:
1. Concatenate prediction and each reference: `[CLS] pred [SEP] ref [SEP]`
2. Compute cross-attention scores
3. Aggregate across multiple references (max or mean)

### 2. Representation Ensemble

**Weight**: 0.3 (Semantic baseline)

**Components**:
- **BERTScore ($S'_{\mathrm{bert}}$)**: BERT-based precision/recall/F1 similarity
- **STS ($S'_{\mathrm{sts}}$)**: SentenceTransformer embedding cosine similarity
- **Lexical Cosine ($S'_{\mathrm{cos}}$)**: Token-based cosine similarity

**Implementation**:
```python
rep_val = (BERTScore + STS_cosine + lexical_cosine) / 3
```

### 3. F_β Score

**Weight**: 0.2 (Lexical safeguard)

**Implementation**:
- β = 2 (reduces over-generation penalty)
- Focuses on content words only
- Penalizes factual discrepancies and hallucinations

## Language-Specific Configurations

### Chinese - `sacc_chinese.py`
```python
EMBED_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-base'
BERTSCORE_LANG = 'zh'
STOPWORDS: Chinese function words
NEGATIONS: Chinese negation words
```

### Traditional Chinese - `sacc_traditional_chinese.py`
- Same configuration as Chinese
- Handles Traditional Chinese character set

### Japanese - `sacc_japanese.py`
```python
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-base'
BERTSCORE_LANG = 'ja'
STOPWORDS: Japanese stopwords
```

### Korean - `sacc_korean.py`
```python
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-base'
BERTSCORE_LANG = 'ko'
STOPWORDS: Korean stopwords
```

## Usage

### Environment Setup

```bash
# Required environment variables
export MODEL_FILE="path/to/model_output.csv"    # Input: model predictions
export OUTPUT_FILE="path/to/results.csv"       # Output: S_Acc results
export REF_AGG_MODE="max"                       # "max" or "mean" aggregation
export ORIGINAL_FILE="data/reference_explanations/chinese_explanations.csv"  # Reference data
```

### Running Evaluation

```bash
cd demo/Sacc

# Chinese idioms
python sacc_chinese.py

# Traditional Chinese idioms
python sacc_traditional_chinese.py

# Japanese idioms
python sacc_japanese.py

# Korean idioms
python sacc_korean.py
```

## Input Format

### Model Output File (MODEL_FILE)
**Format**: CSV (.csv)

**Required Sheets**: Each sheet represents a different arrangement
```
arrangement1/
├── idiom: Idiom name
├── Reference: Ground truth explanation
├── Prediction: Model generated explanation
└── [Other columns...]
```

### Reference File (ORIGINAL_FILE)
**Format**: CSV (.csv)

**Structure**:
```
Sheet: Main
├── idiom: Idiom names
├── explanation: Ground truth explanations
└── [Other metadata...]
```

## Output Format

### Results File (OUTPUT_FILE)
**Format**: CSV (.csv)

**Sheets**: One sheet per input arrangement + summary sheet

**Columns**:
```
idiom: Idiom name
Reference: Ground truth explanation
Prediction: Model prediction
Cross_Encoder: S'_ce score
BERTScore: S'_bert score
STS: S'_sts score
Lexical_Cosine: S'_cos score
Representation: S'_rep ensemble score
F_Beta: S_fβ score
S_Acc: Final S_Acc score
Polarity_Conflict: Polarity analysis result
```

## Configuration Parameters

### Core Parameters
```python
BATCH_SIZE_EMB = 64          # Embedding batch size
USE_BERTSCORE = True         # Enable/disable BERTScore
MAX_LEN = 512               # Maximum sequence length
BETA_F = 2.0               # F_β parameter
```

### Aggregation Modes
- `REF_AGG_MODE = "max"`: Take maximum score across references
- `REF_AGG_MODE = "mean"`: Take average score across references

### Polarity Analysis
```python
POLARITY_PENALTY = 0.5      # Penalty for polarity conflicts
POLARITY_MODE = "all"       # Analysis mode
POLARITY_RATIO = 0.5        # Threshold ratio
```

## Technical Details

### Cross-Encoder Implementation
```python
def cross_encoder_over_refs(model, pred, refs):
    pairs = [[pred, ref] for ref in refs]
    scores = model.predict(pairs)
    return max(scores) if REF_AGG_MODE == "max" else mean(scores)
```

### Representation Ensemble
```python
def representation_ensemble(bert_val, sts_val, lex_val):
    vals = [v for v in [bert_val, sts_val, lex_val] if not np.isnan(v)]
    return np.mean(vals) if vals else np.nan
```

### F_β Score Calculation
```python
def content_fbeta_with_polarity(pred, refs, idiom):
    # Content word extraction
    pred_tokens = tokenize_content(pred)
    ref_tokens = [tokenize_content(r) for r in refs]

    # $\mathrm{F}_\beta$ calculation with β=2
    # Polarity conflict detection
    # Return F_β score and polarity analysis
```

## Score Interpretation

### $S_{\mathrm{Acc}}$ Range: [0, 1]
- **0.0**: No semantic similarity
- **0.5**: Moderate semantic understanding
- **0.8+**: Good semantic quality
- **1.0**: Perfect semantic match

### Component Contributions
- **Cross-Encoder (50%)**: Primary semantic entailment detection
- **Representation (30%)**: Robust semantic baseline across methods
- **$\mathrm{F}_\beta$ (20%)**: Lexical accuracy safeguard

## Dependencies

- **Core**: pandas, numpy, torch
- **NLP**: transformers, sentence-transformers, jieba
- **Metrics**: bert-score
- **Utils**: tqdm, scikit-learn

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce BATCH_SIZE_EMB or use CPU
   ```

2. **Missing Reference File**
   ```
   Error: FileNotFoundError
   Solution: Set ORIGINAL_FILE environment variable
   ```

3. **Empty Results**
   ```
   Check: Input file format and column names
   Check: Reference file contains expected idioms
   ```

### Performance Optimization

- Use GPU for faster processing
- Adjust batch sizes based on available memory
- Use REF_AGG_MODE="max" for faster evaluation
- Pre-tokenize references for repeated evaluations

## References

This implementation is based on the ACL 2026 paper methodology for comprehensive idiom evaluation. The hierarchical approach ensures both deep semantic understanding and lexical precision.

## Version Notes

- **Current**: Multi-language support with optimized models
- **Features**: Polarity analysis, content word focus, multi-reference aggregation
- **Optimization**: Batch processing, GPU acceleration, memory efficiency

---

For questions about specific language implementations or customization needs, refer to the individual script docstrings or open an issue.

