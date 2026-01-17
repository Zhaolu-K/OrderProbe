# Structural Consistency Calculators

This module contains scripts for calculating Structural Consistency (S_Cons) metrics, which measure a model's stability under different structural configurations.

## Components

### Performance Deviation 
- **File**: `e_perf_calculator.py`
- **Purpose**: Measures average gap between recognition capability and realized performance
- **Formula**:
$$E_{\text{perf}} = \frac{1}{|P|} \sum_{p \in P} (S_{\text{max}} - S_{\text{mean}})$$
- **Where**:
  - `P` is the set of all permutations
  - `S_max` is the maximum semantic accuracy score for permutation `p`
  - `S_mean` is the average semantic accuracy score for permutation `p`
Display form:

$$
E_{\mathrm{perf}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \left(S_{\max} - S_{\mathrm{mean}}\right)
$$

### Rigidity Sensitivity 
- **File**: `r_sens_calculator.py`
- **Purpose**: Measures the maximum gap between capability and performance across permutations
- **Formula**:
$$R_{\text{sens}} = \max_{p \in P} (S_{\text{max}} - S_{\text{mean}})$$
Display form:

$$
R_{\mathrm{sens}} = \max_{p \in \mathcal{P}} \left(S_{\max} - S_{\mathrm{mean}}\right)
$$

### Structural Consistency ($S_{\mathrm{Cons}}$)
- **File**: `s_cons_calculator.py`
- **Purpose**: Combines E_perf and R_sens into overall consistency score
- **Formula**:
$$S_{\text{Cons}} = (1 - E_{\text{perf}}) \cdot (1 - R_{\text{sens}})$$
- **Interpretation**: Higher values indicate better structural stability
Display form:

$$
S_{\mathrm{Cons}} = (1 - E_{\mathrm{perf}})\cdot(1 - R_{\mathrm{sens}})
$$

## Data Requirements

-- **Mean Results**: `data/baseline_mean/` - Excel files with average $S_{\mathrm{Acc}}$ scores per arrangement
-- **Max Results**: `data/baseline_max/` - Excel files with maximum $S_{\mathrm{Acc}}$ scores per arrangement
- **Output Directory**: `results/consistency/` - Where intermediate and final results are saved

## Processing Pipeline

1. **Run E_perf Calculator**:
   ```bash
   cd demo/Scons
   python e_perf_calculator.py
   ```
   - Processes each model's mean/max file pair
   - Calculates average capability-performance gap across arrangements
   - Outputs `e_perf_results.csv`

2. **Run R_sens Calculator**:
   ```bash
   python r_sens_calculator.py
   ```
   - Finds maximum capability-performance gap across arrangements
   - Outputs `r_sens_results.csv`

3. **Run S_Cons Calculator**:
   ```bash
   python s_cons_calculator.py
   ```
   - Combines E_perf and R_sens results
   - Computes final structural consistency scores
   - Outputs `s_cons_results.csv`

## File Format Requirements

### Input Excel Files
- **Sheets**: Each arrangement should have its own sheet (e.g., "arrangement1", "arrangement2", etc.)
- **Columns**: Must contain $$S_{\mathrm{Acc}}$$
 columns ending with $$S_{\mathrm{Acc}}$$

- **Data**: Numeric values representing semantic accuracy scores

### Example Structure
```
Model.csv
├── arrangement1
│   ├── idiom1_S_Acc: 0.85
│   ├── idiom2_S_Acc: 0.72
│   └── ...
├── arrangement2
│   ├── idiom1_S_Acc: 0.88
│   ├── idiom2_S_Acc: 0.79
│   └── ...
└── summary (excluded from calculations)
```

## Output Files

- `e_perf_results.csv`: Performance deviation scores by model
- `r_sens_results.csv`: Rigidity sensitivity scores by model
- `s_cons_results.csv`: Combined structural consistency scores

## Important Notes

- **Data Integrity**: Scores are read from separate mean and max result files
- **Gap Calculation**: For each arrangement, `$S_{\mathrm{max}} - S_{\mathrm{mean}}$` represents internal volatility
-- **Multiplicative Penalty**: $S_{\mathrm{Cons}}$ uses multiplicative aggregation to penalize deficiencies in either dimension
- **Robustness**: Missing data in individual arrangements is handled gracefully
- **Reproducibility**: Results are deterministic given the same input data

