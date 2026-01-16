# OrderProbe: Deterministic Structural Reconstruction Benchmark

**How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction**

This repository implements **OrderProbe**, the comprehensive benchmark introduced in the ACL 2026 paper for evaluating large language models' (LLMs) structural reconstruction capabilities. **OrderProbe focuses on a critical but underexplored question: Can LLMs recognize and reconstruct correct internal structure when input order is scrambled?** OrderProbe addresses the fundamental ambiguity of sentence-level restoration by using fixed four-character expressions in Chinese, Japanese, and Korean, which have unique canonical forms enabling exact-match evaluation.

## Key Contributions

- **Core Research Question**: Can LLMs recognize and reconstruct correct internal structure when input order is scrambled?
- **Deterministic Evaluation**: Addresses sentence-level restoration ambiguity using fixed four-character expressions with unique canonical forms
- **Multi-lingual Benchmark**: 3,543 curated samples across Chinese (Simplified/Traditional), Japanese, and Korean scripts
- **Diagnostic Framework**: Six-dimensional analysis beyond surface accuracy rankings
- **Structural Dissociation**: Reveals semantic-structure gap in LLMs' capabilities
- **Cross-Script Analysis**: Logographic vs. phonogrammatic script performance comparison

## Core Findings

- **Structure Recognition is Hard**: When internal order is scrambled, LLMs struggle to recognize the correct structure - zero-shot recovery frequently below 35% even for frontier systems
- **Semantic ≠ Structural Understanding**: Models can recall meaning but fail at structural planning, revealing a fundamental dissociation between semantic competence and structural awareness
- **Script Matters for Structure**: Logographic scripts (Chinese/Japanese) provide stronger local anchors than phonogrammatic Korean Hangul, affecting reconstruction difficulty
- **CoT Helps But Limited**: Chain-of-thought prompting improves structural reconstruction but gains are highly model-dependent

## Project Structure

```
demo/
├── Sacc/              # Semantic Accuracy (S_Acc^mean)
│   ├── sacc_chinese.py           # Chinese idiom evaluation
│   ├── sacc_traditional_chinese.py  # Traditional Chinese evaluation
│   ├── sacc_japanese.py          # Japanese idiom evaluation
│   ├── sacc_korean.py            # Korean idiom evaluation
│   └── README.md
├── Scons/             # Structural Consistency (S_Cons)
│   ├── e_perf_calculator.py      # Performance Deviation (E_perf)
│   ├── r_sens_calculator.py      # Rigidity Sensitivity (R_sens)
│   ├── s_cons_calculator.py      # Structural Consistency (S_Cons)
│   └── README.md
├── Slogic/            # Logical Validity (S_Log)
│   ├── s_log_calculator.py       # Logical Validity (S_Log)
│   └── README.md
├── Sinfo/             # Information Density (S_Info)
│   ├── s_info_calculator.py      # Information Density (S_Info)
│   └── README.md
├── Srobust/           # Robustness Metrics (S_Rob)
│   ├── mdr_calculator.py         # Mean Degradation Relative (MDR)
│   ├── mda_calculator.py         # Mean Degradation Absolute (MDA)
│   ├── sseq_calculator.py        # Sequential Robustness (S_seq)
│   ├── srob_calculator.py        # Composite Robustness (S_Rob)
│   ├── simple_sstruct.py         # Structural Robustness (S_struct)
│   └── README.md
└── README.md          # This file
```

## Dataset Statistics

OrderProbe comprises **3,543 curated four-character expressions** across:

- **Script Typologies**: Simplified Chinese, Traditional Chinese, Japanese, Korean
- **Syntactic Categories**: 6 structural patterns (Parallel, Coordinate, Subject-Predicate, Verb-Object, etc.)
- **Evaluation Scope**: 23 non-identity permutations per expression = **81,489** perturbed inputs

**Core Evaluation Paradigm**:
- **Input**: Scrambled four-character sequences (internal structure destroyed)
- **Task**: Reconstruct the unique canonical order
- **Challenge**: Test whether LLMs can recognize correct internal structure from disordered constituents

**Key Design Features**:
- **Deterministic Evaluation**: Unique canonical forms enable exact-match scoring
- **Structure Recognition Test**: Focuses on internal order reconstruction rather than semantic understanding
- **Cross-Script Comparison**: Logographic (Chinese/Japanese) vs. phonogrammatic (Korean) scripts
- **Structural Perturbation**: Internal reordering preserves all lexical content while destroying structure
- **Multi-Source Curation**: Expert filtering from linguistic dictionaries and repositories

## Core Metrics

### Primary Metric: Recovery Rate

**Global structural integrity indicator - exact match of canonical order**

```
Recovery = (1/N) × Σ I(ŷᵢ = yᵢ)
```

Measures the ability to reconstruct the canonical four-character sequence from scrambled constituents.

### Diagnostic Metrics

#### 1. Semantic Fidelity (S_Acc^mean)

Evaluates explanation quality using a tiered hybrid metric integrating cross-encoder relevance, multilingual embedding similarity, and lexical safeguards:

**Formula**:
```
S_Acc^mean = w₁ × S'_ce + w₂ × (S'_bert + S'_sts + S'_cos)/3 + w₃ × S_fβ
```

**Weights**: w₁=0.5 (Cross-Encoder), w₂=0.3 (Representation Ensemble), w₃=0.2 (F_β safeguard)

#### 3. Logical Validity (S_Log)

Detects fluent but contradictory definitions using entailment probability:

**Formula**:
```
S_Log = P_NLI(e ⇒ r)
```

Uses multilingual NLI classifier to ensure explanations logically entail reference meanings.

#### 4. Structural Consistency (S_Cons)

Quantifies invariance to internal structural shuffles:

**Formula**:
```
E_perf = (1/|P|) × Σ_{p ∈ P} (S_max - S_mean)
R_sens = max_{p ∈ P} (S_max - S_mean)
S_Cons = (1 - E_perf) × (1 - R_sens)
```

Penalizes both average capability loss and localized brittleness across permutations.

#### 5. Robustness (S_Rob)

Combines sequential and structural robustness dimensions:

**Sequential Robustness**:
```
S_seq = 1 - (α × MDR + β × MDA), α=β=0.5
```

**Structural Robustness**:
```
μ_k = (1/|D_k|) × Σ_{x ∈ D_k} S_Acc^mean(x)
S_struct = 1 - Normalize(σ(μ₁, μ₂, ..., μ₆))
```

**Composite Robustness**:
```
S_Rob = (2 × S_seq × S_struct) / (S_seq + S_struct)
```

#### 6. Information Density (S_Info)

Rewards concise explanations by penalizing verbosity:

**Formula**:
```
S_Info = BP × P_ROUGE
```

Counters "knowledge dumping" with brevity penalty and ROUGE precision.
- **E_perf**: Average performance deviation across arrangements
- **R_sens**: Maximum rigidity sensitivity across arrangements
- **S_Cons**: Composite structural consistency score

### 5. Robustness Metrics

Assesses model resilience under various perturbations:

#### Sequential Robustness (S_seq)
```
MDR = mean((Original_Score - S_Acc) / Original_Score)
MDA = max(Original_Score - S_Acc)
S_seq = 1 - (0.5 × MDR + 0.5 × MDA)
```

#### Structural Robustness (S_struct)
```
μ_k = average(S_Acc^mean) over idioms in structure k
σ = standard deviation of {μ₁, μ₂, ..., μ₆}
S_struct = 1 - Normalize(σ)
```

#### Composite Robustness (S_Rob)
```
S_Rob = (2 × S_seq × S_struct) / (S_seq + S_struct)
```

## Installation

### Requirements

**Python Version**: 3.8 or higher

**System Dependencies**:
```bash
# For BERTScore (optional)
pip install torch torchvision torchaudio

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zhaolu-K/OrderProbe.git
   cd OrderProbe
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install for development**:
   ```bash
   pip install -e ".[dev]"
   ```

### Verify Installation

```bash
python -c "import orderprobe; print('OrderProbe installed successfully!')"
```

## Usage

### Basic Usage

```python
# Example: Evaluate semantic accuracy for Chinese idioms
from orderprobe.Sacc import sacc_chinese

# Set environment variables
import os
os.environ['MODEL_FILE'] = 'path/to/your/model_output.csv'
os.environ['ORIGINAL_FILE'] = 'path/to/reference_explanations.csv'

# Run evaluation
sacc_chinese.main()
```

### Command Line Usage

```bash
# Evaluate semantic accuracy
python -m orderprobe.Sacc.sacc_chinese

# Evaluate information density
python -m orderprobe.Sinfo.s_info_calculator

# Evaluate logical validity
python -m orderprobe.Slogic.s_log_calculator
```

### Advanced Usage

#### Custom Configuration

```bash
# Set custom parameters
export MODEL_FILE="data/model_output.csv"
export ORIGINAL_FILE="data/reference_explanations.csv"
export OUTPUT_FILE="results/custom_results.csv"
export REF_AGG_MODE="max"  # or "mean"
```

#### Batch Processing

```bash
# Process multiple language variants
for lang in chinese traditional_chinese japanese korean; do
    export ORIGINAL_FILE="data/${lang}_explanations.csv"
    export OUTPUT_FILE="results/${lang}_results.csv"
    python -m orderprobe.Sacc.sacc_${lang//_/-}
done
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Evaluations

#### 1. Semantic Accuracy Evaluation

```bash
cd demo/Sacc
# Set environment variables
export MODEL_FILE="path/to/model_output.csv"
export OUTPUT_FILE="path/to/results.csv"
python sacc_chinese.py  # or other language versions
```

#### 2. Information Density Evaluation

```bash
cd demo/Sinformation
python s_info_calculator.py
```

#### 3. Logical Validity Evaluation

```bash
cd demo/Slogic
python s_log_calculator.py
```

#### 4. Structural Consistency Evaluation

```bash
cd demo/Scons
# Run in sequence
python e_perf_calculator.py
python r_sens_calculator.py
python s_cons_calculator.py
```

#### 4. Robustness Evaluation

```bash
cd demo/Srobust
# Run in sequence
python mdr_calculator.py
python mda_calculator.py
python sseq_calculator.py
python srob_calculator.py
```

## Data Format Requirements

### Input Files

#### Model Output Files
- **Format**: CSV (.csv)
- **Sheets**: Each sheet represents an arrangement (e.g., "arrangement1", "arrangement2", etc.)
- **Columns**: Must contain idiom predictions and evaluation scores

#### Reference Files
- **Location**: `data/reference_explanations/`
- **Format**: Excel files with ground truth idiom explanations
- **Naming**: `{language}_explanations.csv`

### Output Files

All calculators generate Excel files with results in the `results/` directory:
- `sacc_*.csv`: Semantic accuracy scores
- `s_info_results.csv`: Information density scores
- `s_log_results.csv`: Logical validity scores
- `e_perf_results.csv`: Performance deviation scores
- `r_sens_results.csv`: Rigidity sensitivity scores
- `s_cons_results.csv`: Structural consistency scores
- `mdr_results.csv`, `mda_results.csv`, etc.: Robustness metrics

## Configuration

### Environment Variables

```bash
# Sacc Configuration
MODEL_FILE=path/to/input.csv    # Input: model predictions
OUTPUT_FILE=path/to/output.csv  # Output: S_Acc results
REF_AGG_MODE=max                # "max" or "mean" aggregation
ORIGINAL_FILE=path/to/reference.csv  # Reference data

# General Configuration
DATA_DIR=data/
RESULTS_DIR=results/
```

### Model Configuration

Pre-configured models for different languages:
- **Embedding**: BAAI/bge-small-zh-v1.5 (Chinese), paraphrase-multilingual-MiniLM-L12-v2 (others)
- **Cross-Encoder**: BAAI/bge-reranker-base
- **NLI**: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multi-lingual)
- **BERTScore**: Language-specific models

## Metric Interpretation

### Score Ranges
- **S_Acc**: [0, 1], higher is better semantic quality
- **S_Info**: [0, 1], higher indicates better information density (concise and informative)
- **S_Log**: [0, 1], higher indicates better logical entailment
- **S_Cons**: [0, 1], higher indicates better structural stability
- **S_seq**: [0, 1], higher indicates better sequential robustness
- **S_struct**: [0, 1], higher indicates better structural robustness
- **S_Rob**: [0, 1], harmonic mean of sequential and structural robustness

### Performance Guidelines
- **S_Acc > 0.8**: Excellent semantic understanding
- **S_Info > 0.75**: Good information density (concise and informative)
- **S_Log > 0.7**: Good logical consistency
- **S_Cons > 0.9**: High structural stability
- **S_Rob > 0.85**: Good overall robustness

### Combined Assessment
For comprehensive evaluation, consider all metrics together:
- **High S_Acc + High S_Info + High S_Log**: Reliable, concise, and logically consistent explanations
- **High S_Acc + Low S_Info**: Semantically good but verbose or redundant
- **High S_Acc + Low S_Log**: Semantically similar but potentially contradictory
- **High S_Cons**: Consistent performance across structural variations
- **High S_Rob**: Robust against various perturbations

## Technical Details

### Dependencies

- **Core**: pandas, numpy, torch
- **NLP**: transformers, sentence-transformers, jieba
- **Metrics**: bert-score
- **UI**: tqdm

### Hardware Requirements

- **GPU**: Recommended for large-scale evaluation (CUDA support)
- **RAM**: 8GB+ for batch processing
- **Storage**: 10GB+ for models and data

### Performance Optimization

- Batch processing for embedding calculations
- GPU acceleration for transformer models
- Memory-efficient data loading
- Progress bars for long-running tasks

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{he2026orderprobe,
  title={How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction},
  author={He, Yingjie and Kang, Zhaolu and Jiang, Kehan and Zhang, Qianyuan and Qian, Jiachen and Meng, Chunlei and Feng, Yujie and Wang, Yuan and Dou, Jiabao and Wu, Aming and Zheng, Leqi and Zhao, Pengxiang and Liu, Jiaxin and Zhang, Zeyu and Wang, Lei and Wang, Guansu and Zhan, Qishi and He, Xiaomin and Zhang, Meisheng and Ni, Jianyuan},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  pages={1--20}
}
```

## Contributing

This is an open-source implementation. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For questions or issues:
- Check individual module READMEs for detailed usage
- Review the paper for theoretical background
- Open an issue on GitHub
- Check [examples/](examples/) for usage demonstrations

## Project Structure Details

```
demo/
├── __init__.py              # Package initialization
├── README.md                # Main documentation
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
├── .gitignore              # Git ignore rules
├── examples/               # Usage examples
│   └── basic_usage.py
├── Sacc/                   # Semantic Accuracy
├── Slogic/                 # Logical Validity
├── Sinfo/                  # Information Density
├── Scons/                  # Structural Consistency
└── Srobust/                # Robustness Metrics
```

## Open Source Readiness

This repository is fully prepared for open source release:

- ** Complete Documentation**: Comprehensive READMEs for all modules
- ** Proper Packaging**: setuptools configuration with dependencies
- ** License**: MIT license for academic and commercial use
- ** Code Quality**: PEP8 compliant, type hints, comprehensive error handling
- ** Internationalization**: Full English documentation and comments
- ** Examples**: Basic usage demonstrations
- ** Contributing**: Clear contribution guidelines
- ** Security**: No hardcoded credentials or sensitive data

---

**Academic Note**: This implementation provides the complete OrderProbe benchmark and diagnostic evaluation framework as described in the ACL 2026 paper "How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction". OrderProbe addresses a fundamental question in LLM evaluation: **Can models recognize and reconstruct correct internal structure when input order is scrambled?** All metrics are implemented according to their original mathematical formulations, ensuring scientific reproducibility and enabling researchers to replicate the key findings on LLMs' structural reconstruction capabilities.
