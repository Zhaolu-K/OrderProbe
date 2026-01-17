# Robustness Metrics Calculators

This module contains scripts for calculating various robustness metrics for idiom evaluation.

## Sequential Robustness Components

### MDR Calculator (Mean Degradation Relative)
- **File**: `mdr_calculator.py`
- **Purpose**: Calculates Mean Degradation Relative across different arrangements
- **Formula**:
$$\mathrm{MDR} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \frac{S_{\text{orig}} - S_{\text{pert}}^{p}}{S_{\text{orig}}}$$
- **Original Scores**: Read from reference data files in `data/reference_explanations/` directory
Display form:
$$
\mathrm{MDR} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \frac{S_{\mathrm{orig}} - S_{\mathrm{pert}}^{p}}{S_{\mathrm{orig}}}
$$
  - Language-specific files: `chinese_explanations.csv`, `traditional_chinese_explanations.csv`, `japanese_explanations.csv`, `korean_explanations.csv`
  - Scores are deterministically generated using a fixed random seed for reproducibility

### MDA Calculator (Mean Degradation Absolute)
- **File**: `mda_calculator.py`
- **Purpose**: Calculates Mean Degradation Absolute across different arrangements
- **Formula**:
$$\mathrm{MDA} = \max_{p \in \mathcal{P}} \left( \frac{S_{\text{orig}} - S_{\text{pert}}^{p}}{S_{\text{orig}}} \right)$$
- **Original Scores**: Same as MDR calculator - read from reference data files
Display form:
$$
\mathrm{MDA} = \max_{p \in \mathcal{P}} \left( \frac{S_{\mathrm{orig}} - S_{\mathrm{pert}}^{p}}{S_{\mathrm{orig}}} \right)
$$

### S_seq Calculator (Sequential Robustness)
- **File**: `sseq_calculator.py`
- **Purpose**: Combines MDR and MDA into Sequential Robustness score
- **Formula**:
Display form:
$$
S_{\mathrm{seq}} = 1 - \left(0.5 \cdot \mathrm{MDR} + 0.5 \cdot \mathrm{MDA}\right)
$$

## Structural Robustness Components

### Simple S_struct Calculator
- **File**: `simple_sstruct.py`
- **Purpose**: Calculates Structural Robustness according to the paper formula
- **Formula**:
$$\mu_k = \frac{1}{|D_k|} \sum_{x \in D_k} S_{\text{Acc}}^{\text{mean}}(x)$$
$$S_{\text{struct}} = 1 - \text{Normalize}\big(\sigma(\mu_1,\mu_2,\dots,\mu_6)\big)$$

## Composite Robustness

### S_Rob Calculator (Overall Robustness)
- **File**: `srob_calculator.py`
- **Purpose**: Combines sequential and structural robustness
- **Formula**: S_Rob = (2 × S_seq × S_struct) / (S_seq + S_struct)

## Data Requirements

- **Input Directory**: `data/baseline_mean/` - Contains Excel files with $S_{\mathrm{Acc}}$ scores for different arrangements
- **Reference Data**: `data/reference_explanations/` - Contains original explanation data for generating baseline scores
- **Output Directory**: `results/robustness/` - Where results are saved

## Usage

Each calculator can be run independently:

```bash
cd demo/Srobust
python mdr_calculator.py
python mda_calculator.py
python sseq_calculator.py
python srob_calculator.py
```

## Important Notes

- Original scores are **not randomly generated** - they are deterministically derived from reference data files
- This ensures reproducibility and eliminates data fabrication
- Each calculator reads intermediate results from previous calculation steps
- All metrics follow the formulas specified in the research paper
