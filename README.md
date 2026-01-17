# OrderProbe: Deterministic Structural Reconstruction Benchmark

**How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction**

📄 [**arXiv:2601.08626**](http://arxiv.org/abs/2601.08626) | [**PDF**](http://arxiv.org/pdf/2601.08626.pdf)

This repository contains **OrderProbe**, a new benchmark from our ACL 2026 paper for testing how well large language models handle word order. **The main question we're asking is: Can AI models figure out the correct word order when we scramble it?** We solve the problem of multiple correct answers by using short four-character phrases from Chinese, Japanese, and Korean that have only one right way to arrange them.

## Key Contributions

- 🎯 **Core Research Question**: Can AI models figure out the correct word order when we scramble it?
- 🔍 **Deterministic Evaluation**: Solves the problem of multiple correct answers by using short phrases that have only one right arrangement
- 🌐 **Multi-lingual Benchmark**: 3,543 curated samples across Chinese (Simplified/Traditional), Japanese, and Korean scripts
- 📊 **Diagnostic Framework**: Detailed analysis covering six different aspects, not just basic accuracy
- 🔬 **Structural Dissociation**: Shows the gap between understanding meaning and understanding structure
- 📈 **Cross-Script Analysis**: Comparison of different writing systems (like Chinese vs. Korean)

## 🔬 Core Findings

- ⚠️ **Structure Recognition is Hard**: When we scramble the internal order of words, AI models have trouble figuring out the correct structure - even the best models often get less than 35% right without any hints
- 🧠 **Meaning vs. Structure**: Models can understand what words mean but struggle to put them in the right order, showing that these are different skills
- 🌏 **Writing Systems Matter**: Languages like Chinese and Japanese give models better clues for structure than Korean, which affects how hard the task is
- 💭 **Thinking Step-by-Step Helps**: Asking models to explain their thinking improves results, but the benefits vary a lot between different models

## Project Structure

```
orderprobe/
├── data/              # Benchmark datasets and linguistic structures
│   ├── *.csv                      # Base idiom expressions
│   ├── Full permutation in *.csv  # Scrambled variants
│   ├── Structure/                 # Linguistic structure classifications
│   │   ├── */*.csv               # 6-category structure labels per language
│   └── README.md                 # Dataset documentation
├── examples/          # Usage examples and demos
│   └── basic_usage.py            # Basic usage demonstration
├── Sacc/              # Semantic Accuracy (`S_Acc^mean`)
│   ├── sacc_chinese.py           # Chinese idiom evaluation
│   ├── sacc_traditional_chinese.py  # Traditional Chinese evaluation
│   ├── sacc_japanese.py          # Japanese idiom evaluation
│   ├── sacc_korean.py            # Korean idiom evaluation
│   └── README.md
├── Scons/             # Structural Consistency (`S_Cons`)
│   ├── e_perf_calculator.py      # Performance Deviation (`E_perf`)
│   ├── r_sens_calculator.py      # Rigidity Sensitivity (`R_sens`)
│   ├── s_cons_calculator.py      # Structural Consistency (`S_Cons`)
│   └── README.md
├── Slogic/            # Logical Validity (`S_Log`)
│   ├── s_log_calculator.py       # Logical Validity (`S_Log`)
│   └── README.md
├── Sinfo/             # Information Density (`S_Info`)
│   ├── s_info_calculator.py      # Information Density (`S_Info`)
│   └── README.md
├── Srobust/           # Robustness Metrics (`S_Rob`)
│   ├── mdr_calculator.py         # Mean Degradation Relative (`MDR`)
│   ├── mda_calculator.py         # Mean Degradation Absolute (`MDA`)
│   ├── sseq_calculator.py        # Sequential Robustness (`S_seq`)
│   ├── srob_calculator.py        # Composite Robustness (`S_Rob`)
│   ├── simple_sstruct.py         # Structural Robustness (`S_struct`)
│   └── README.md
└── README.md          # This file
```

## 📊 Dataset Statistics

OrderProbe includes **3,543 carefully selected four-character phrases** from:

- **Languages**: Simplified Chinese, Traditional Chinese, Japanese, Korean
- **Phrase Types**: 6 different kinds of word arrangements
- **Test Cases**: For each phrase, we create 23 different scrambled versions = **81,489** total test examples

**How the Test Works**:
- **What we give the model**: A four-character phrase with words in the wrong order
- **What the model must do**: Put the words back in the correct order
- **The challenge**: Can the model figure out the right structure from scrambled pieces?

**Why This Works**:
- **Clear right answers**: Each phrase has only one correct arrangement
- **Structure-focused**: Tests word order skills, not just meaning understanding
- **Cross-Language Comparison**: How different languages (like Chinese vs. Korean) affect the task difficulty
- **Structural Perturbation**: Internal reordering preserves all lexical content while destroying structure
- **Multi-Source Curation**: Expert filtering from linguistic dictionaries and repositories

## 🎯 Core Metrics

### Primary Metric: Recovery Rate

**Global structural integrity indicator - exact match of canonical order**

$$\text{Recovery} = \frac{1}{N} \times \sum I(\hat{y}_i = y_i)$$

Measures the ability to reconstruct the canonical four-character sequence from scrambled constituents.

### Diagnostic Metrics

#### 1. Semantic Fidelity (`S_Acc^mean`)

Evaluates explanation quality using a tiered hybrid metric integrating cross-encoder relevance, multilingual embedding similarity, and lexical safeguards:

**Formula**:
$$S_{\text{Acc}}^{\text{mean}} = w_1 \times S'_{\text{ce}} + w_2 \times \frac{S'_{\text{bert}} + S'_{\text{sts}} + S'_{\text{cos}}}{3} + w_3 \times S_{f\beta}$$

**Weights**: w₁=0.5 (Cross-Encoder), w₂=0.3 (Representation Ensemble), w₃=0.2 (F_β safeguard)

#### 3. Logical Validity (`S_Log`)

Detects fluent but contradictory definitions using entailment probability:

**Formula**:
$$S_{\text{Log}} = P_{\text{NLI}}(e \Rightarrow r)$$

Uses multilingual NLI classifier to ensure explanations logically entail reference meanings.

#### 4. Structural Consistency (`S_Cons`)

Quantifies invariance to internal structural shuffles:

**Formula**:
$$E_{\text{perf}} = \frac{1}{|P|} \times \sum_{p \in P} (S_{\text{max}} - S_{\text{mean}})$$
$$R_{\text{sens}} = \max_{p \in P} (S_{\text{max}} - S_{\text{mean}})$$
$$S_{\text{Cons}} = (1 - E_{\text{perf}}) \times (1 - R_{\text{sens}})$$

Penalizes both average capability loss and localized brittleness across permutations.

#### 5. Robustness (`S_Rob`)

Combines sequential and structural robustness dimensions:

**Sequential Robustness**:
$$S_{\text{seq}} = 1 - (\alpha \times \text{MDR} + \beta \times \text{MDA}), \quad \alpha=\beta=0.5$$

**Structural Robustness**:
$$\mu_k = \frac{1}{|D_k|} \times \sum_{x \in D_k} S_{\text{Acc}}^{\text{mean}}(x)$$
$$S_{\text{struct}} = 1 - \text{Normalize}(\sigma(\mu_1, \mu_2, \dots, \mu_6))$$

**Composite Robustness**:
$$S_{\text{Rob}} = \frac{2 \times S_{\text{seq}} \times S_{\text{struct}}}{S_{\text{seq}} + S_{\text{struct}}}$$

Where:
- **MDR** (Mean Degradation Relative): $$\text{MDR} = \text{mean}\left(\frac{\text{Original\_Score} - S_{\text{Acc}}}{\text{Original\_Score}}\right)$$
- **MDA** (Mean Degradation Absolute): $$\text{MDA} = \max(\text{Original\_Score} - S_{\text{Acc}})$$
 - **MDR** (Mean Degradation Relative): $$\mathrm{MDR} = \mathrm{mean}\!\left(\frac{\mathrm{OriginalScore} - S_{\mathrm{Acc}}}{\mathrm{OriginalScore}}\right)$$
 - **MDA** (Mean Degradation Absolute): $$\mathrm{MDA} = \max(\mathrm{OriginalScore} - S_{\mathrm{Acc}})$$
- **σ**: Standard deviation of mean scores across 6 linguistic structures

#### 6. Information Density (`S_Info`)

Rewards concise explanations by penalizing verbosity:

**Formula**:
$$S_{\text{Info}} = \text{BP} \times P_{\text{ROUGE}}$$

Counters "knowledge dumping" with brevity penalty and ROUGE precision.

## 🛠️ Installation

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

## 📖 Usage

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

## 🚀 Quick Start

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

## 📋 Data Format Requirements

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

## ⚙️ Configuration

### Environment Variables

```bash
# Sacc Configuration
MODEL_FILE=path/to/input.csv    # Input: model predictions
OUTPUT_FILE=path/to/output.csv  # Output: `S_Acc` results
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

## 📈 Metric Interpretation

### Score Ranges
 - **`S_Acc`**: [0, 1], higher is better semantic quality
 - **`S_Info`**: [0, 1], higher indicates better information density (concise and informative)
 - **`S_Log`**: [0, 1], higher indicates better logical entailment
 - **`S_Cons`**: [0, 1], higher indicates better structural stability
 - **`S_seq`**: [0, 1], higher indicates better sequential robustness
 - **`S_struct`**: [0, 1], higher indicates better structural robustness
 - **`S_Rob`**: [0, 1], harmonic mean of sequential and structural robustness

### Performance Guidelines
 - **`S_Acc` > 0.8**: Excellent semantic understanding
 - **`S_Info` > 0.75**: Good information density (concise and informative)
 - **`S_Log` > 0.7**: Good logical consistency
 - **`S_Cons` > 0.9**: High structural stability
 - **`S_Rob` > 0.85**: Good overall robustness

### Combined Assessment
For comprehensive evaluation, consider all metrics together:
 - **High `S_Acc` + High `S_Info` + High `S_Log`**: Reliable, concise, and logically consistent explanations
 - **High `S_Acc` + Low `S_Info`**: Semantically good but verbose or redundant
 - **High `S_Acc` + Low `S_Log`**: Semantically similar but potentially contradictory
 - **High `S_Cons`**: Consistent performance across structural variations
 - **High `S_Rob`**: Robust against various perturbations

## 🔧 Technical Details

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

## 📚 Citation

If you use this framework in your research, please cite:

**Preprint**: [arXiv:2601.08626](http://arxiv.org/abs/2601.08626)

```bibtex
@inproceedings{he2026orderprobe,
  title={How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction},
  author={He, Yingjie and Kang, Zhaolu and Jiang, Kehan and Zhang, Qianyuan and Qian, Jiachen and Meng, Chunlei and Feng, Yujie and Wang, Yuan and Dou, Jiabao and Wu, Aming and Zheng, Leqi and Zhao, Pengxiang and Liu, Jiaxin and Zhang, Zeyu and Wang, Lei and Wang, Guansu and Zhan, Qishi and He, Xiaomin and Zhang, Meisheng and Ni, Jianyuan},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  pages={1--20},
  archivePrefix={arXiv},
  eprint={2601.08626},
  primaryClass={cs.CL}
}
```

## 🤝 Contributing

This is an open-source implementation. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or issues:
- Check individual module READMEs for detailed usage
- Review the paper for theoretical background
- Open an issue on GitHub
- Check [examples/](examples/) for usage demonstrations

## 🏗️ Project Structure Details

```
orderprobe/
├── __init__.py              # Package initialization
├── README.md                # Main documentation
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── requirements.txt         # Python dependencies
├── setup.py                 # Package installation
├── .gitignore              # Git ignore rules
├── data/                    # Benchmark datasets
│   ├── *.csv               # Base idiom expressions (4 languages)
│   ├── Full permutation in *.csv  # Scrambled variants (81,489 samples)
│   ├── Structure/          # Linguistic structure classifications
│   │   ├── */*.csv        # 6-category structure labels per language
│   └── README.md           # Dataset documentation
├── examples/               # Usage examples
│   └── basic_usage.py
├── Sacc/                   # Semantic Accuracy (`S_Acc^mean`)
├── Slogic/                 # Logical Validity (`S_Log`)
├── Sinfo/                  # Information Density (`S_Info`)
├── Scons/                  # Structural Consistency (`S_Cons`)
└── Srobust/                # Robustness Metrics (`S_Rob`)
```

## ✅ Open Source Readiness

This repository is fully prepared for open source release:

- 📖 **Complete Documentation**: Comprehensive READMEs for all modules
- 📦 **Proper Packaging**: setuptools configuration with dependencies
- 📄 **License**: MIT license for academic and commercial use
- ✨ **Code Quality**: PEP8 compliant, type hints, comprehensive error handling
- 🌍 **Internationalization**: Full English documentation and comments
- 🚀 **Examples**: Basic usage demonstrations
- 🤝 **Contributing**: Clear contribution guidelines
- 🔒 **Security**: No hardcoded credentials or sensitive data

---

**Note**: This code provides the complete OrderProbe benchmark from our ACL 2026 paper. All the evaluation methods follow the exact formulas from the paper, so other researchers can reproduce our results and test their own models.
