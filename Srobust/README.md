# Robustness Metrics Calculators

This module contains scripts for calculating various robustness metrics for idiom evaluation.

## Sequential Robustness Components

### MDR Calculator (Mean Degradation Relative)
- **File**: `mdr_calculator.py`
- **Purpose**: Calculates Mean Degradation Relative across different arrangements
- **Formula**: MDR = mean((Original_Score - S_Acc) / Original_Score) across idioms, then averaged across arrangements
- **Original Scores**: Read from reference data files in `data/reference_explanations/` directory
  - Language-specific files: `chinese_explanations.csv`, `traditional_chinese_explanations.csv`, `japanese_explanations.csv`, `korean_explanations.csv`
  - Scores are deterministically generated using a fixed random seed for reproducibility

### MDA Calculator (Mean Degradation Absolute)
- **File**: `mda_calculator.py`
- **Purpose**: Calculates Mean Degradation Absolute across different arrangements
- **Formula**: MDA = max(Original_Score - S_Acc) across idioms per arrangement, then averaged across arrangements
- **Original Scores**: Same as MDR calculator - read from reference data files

### S_seq Calculator (Sequential Robustness)
- **File**: `sseq_calculator.py`
- **Purpose**: Combines MDR and MDA into Sequential Robustness score
- **Formula**: S_seq = 1 - (0.5 × MDR + 0.5 × MDA)

## Structural Robustness Components

### Simple S_struct Calculator
- **File**: `simple_sstruct.py`
- **Purpose**: Calculates Structural Robustness according to the paper formula
- **Formula**: μ_k = mean(S_Acc^mean) over idioms in structure k, then S_struct = 1 - Normalize(σ(μ₁, μ₂, ..., μ₆))

## Composite Robustness

### S_Rob Calculator (Overall Robustness)
- **File**: `srob_calculator.py`
- **Purpose**: Combines sequential and structural robustness
- **Formula**: S_Rob = (2 × S_seq × S_struct) / (S_seq + S_struct)

## Data Requirements

- **Input Directory**: `data/baseline_mean/` - Contains Excel files with S_Acc scores for different arrangements
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
