# OrderProbe Dataset

This directory contains the core dataset for OrderProbe: A benchmark for evaluating LLMs' structural reconstruction capabilities.

## 📁 Dataset Overview

OrderProbe evaluates whether large language models can recognize and reconstruct correct internal structure when input order is scrambled. The dataset focuses on four-character idioms (Chengyu) in Chinese, Japanese, and Korean, which have unique canonical forms enabling exact-match evaluation.

## 📊 Data Files

### Core Expression Files
These files contain the base idioms with their semantic explanations:

- **`Simplified Chinese.csv`** - Simplified Chinese idioms and explanations
- **`Traditional Chinese.csv`** - Traditional Chinese idioms and explanations
- **`Japanese.csv`** - Japanese four-character expressions
- **`Korean.csv`** - Korean four-character expressions

**Format**: `idiom,explanation1||explanation2||explanation3`

### Full Permutation Files
These files contain all possible scrambled variants for each idiom:

- **`Full permutation in Simplified Chinese.csv`**
- **`Full permutation in Traditional Chinese.csv`**
- **`Full permutation in Japanese.csv`**
- **`Full permutation in Korean.csv`**

**Format**: First column is the canonical form, followed by 23 scrambled variants.

## 🔧 Data Structure

### Base Expression Format
```
idiom1,explanation1||explanation2||explanation3
idiom2,explanation1||explanation2
```

### Permutation Format
```
canonical_form,permutation1,permutation2,permutation3,...,permutation23
idiom1,variant1,variant2,...
idiom2,variant1,variant2,...
```

## 📈 Dataset Statistics

- **Total Expressions**: 3,543 curated four-character expressions
- **Languages**: 4 (Simplified Chinese, Traditional Chinese, Japanese, Korean)
- **Syntactic Categories**: 6 structural patterns
- **Total Evaluation Samples**: 81,489 (3,543 × 23 permutations)

## 🎯 Evaluation Paradigm

1. **Input**: Scrambled four-character sequence
2. **Task**: Reconstruct the canonical order
3. **Challenge**: Recognize correct internal structure from disordered constituents

## 📖 Usage

### Loading Base Data
```python
import pandas as pd

# Load expressions with explanations
df = pd.read_csv('Simplified Chinese.csv', header=None, names=['idiom', 'explanations'])
print(df.head())
```

### Loading Permutations
```python
# Load all permutations for evaluation
df_perms = pd.read_csv('Full permutation in Simplified Chinese.csv')
canonical = df_perms['原始词语']  # First column contains canonical forms
permutations = df_perms.iloc[:, 1:]  # Remaining columns contain scrambled variants
```

### Integration with OrderProbe
```python
# Set data paths
os.environ['ORIGINAL_FILE'] = 'data/Simplified Chinese.csv'
os.environ['MODEL_FILE'] = 'data/Full permutation in Simplified Chinese.csv'

# Run evaluation
python -m orderprobe.Sacc.sacc_chinese
```

## 🔍 Data Curation Process

1. **Multi-source Collection**: Aggregated from linguistic dictionaries and repositories
2. **Expert Filtering**: Removed non-standard, ambiguous, or modern slang expressions
3. **Semantic Annotation**: Dictionary definitions augmented with paraphrastic variants
4. **Permutation Generation**: All non-identity permutations (23 variants per expression)

## 📋 File Details

| File | Language | Expressions | Purpose |
|------|----------|-------------|---------|
| Simplified Chinese.csv | zh-CN | ~880 | Base expressions + explanations |
| Traditional Chinese.csv | zh-TW | ~880 | Base expressions + explanations |
| Japanese.csv | ja | ~880 | Base expressions + explanations |
| Korean.csv | ko | ~880 | Base expressions + explanations |
| Full permutation in *.csv | All | 3,543 | Scrambled variants for evaluation |

## 🎓 Research Applications

- **Structural Reconstruction**: Test LLMs' ability to reorder scrambled sequences
- **Cross-lingual Comparison**: Compare performance across logographic vs. phonogrammatic scripts
- **Semantic vs. Structural**: Investigate dissociation between meaning recall and structure planning

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{he2026orderprobe,
  title={How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction},
  author={He, Yingjie and Kang, Zhaolu and Jiang, Kehan and Zhang, Qianyuan and Qian, Jiachen and Meng, Chunlei and Feng, Yujie and Wang, Yuan and Dou, Jiabao and Wu, Aming and Zheng, Leqi and Zhao, Pengxiang and Liu, Jiaxin and Zhang, Zeyu and Wang, Lei and Wang, Guansu and Zhan, Qishi and He, Xiaomin and Zhang, Meisheng and Ni, Jianyuan},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  pages={1--20}
}
```

## ⚖️ License

This dataset is released under the MIT License. See the main project LICENSE file for details.

---

**Note**: This dataset supports OrderProbe's mission to understand LLMs' structural awareness capabilities. The expressions are carefully curated to ensure linguistic validity and evaluation reliability.
