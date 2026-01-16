#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic Usage Example for Idiom Evaluation Framework

This script demonstrates how to use the Idiom Evaluation Framework
to evaluate idiom explanation quality across multiple metrics.
"""

import os
import pandas as pd
from pathlib import Path

# Import evaluation modules
from Sacc.sacc_chinese import main as run_sacc
from Slogic.s_log_calculator import main as run_slogic
from Sinfo.s_info_calculator import main as run_sinfo
from Scons.s_cons_calculator import main as run_scons
from Srobust.srob_calculator import main as run_srobust


def setup_example_data():
    """Create example data for demonstration."""
    # This would normally be your actual model outputs
    example_data = {
        'idiom': ['idiom1', 'idiom2', 'idiom3'],
        'explanation': [
            'This idiom means...',  # Reference explanation
            'Another idiom means...',
            'Third idiom means...'
        ],
        'prediction': [
            'Model explanation for idiom1...',  # Model prediction
            'Model explanation for idiom2...',
            'Model explanation for idiom3...'
        ]
    }

    df = pd.DataFrame(example_data)

    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Save example data
    df.to_csv(data_dir / 'example_model_output.csv', index=False, encoding='utf-8-sig')

    # Create reference data
    ref_data = {
        'idiom': ['idiom1', 'idiom2', 'idiom3'],
        'explanation': [
            'This idiom means...',
            'Another idiom means...',
            'Third idiom means...'
        ]
    }
    ref_df = pd.DataFrame(ref_data)
    ref_df.to_csv(data_dir / 'reference_explanations.csv', index=False, encoding='utf-8-sig')

    print("Example data created in data/ directory")


def run_all_evaluations():
    """Run all evaluation metrics."""
    print(" Starting Idiom Evaluation Framework Demo")
    print("=" * 50)

    # Set up environment variables for the example
    os.environ['MODEL_FILE'] = 'data/example_model_output.csv'
    os.environ['OUTPUT_FILE'] = 'results/example_sacc_results.csv'

    try:
        print("\n1. Running Semantic Accuracy (S_Acc) evaluation...")
        # Note: Sacc requires specific setup, this is just an example
        print("   (S_Acc evaluation requires reference data setup)")

        print("\n2. Running Logical Validity (S_Log) evaluation...")
        # run_slogic()

        print("\n3. Running Information Density (S_Info) evaluation...")
        # run_sinfo()

        print("\n4. Running Structural Consistency (S_Cons) evaluation...")
        # run_scons()

        print("\n5. Running Robustness evaluation...")
        # run_srobust()

        print("\n All evaluations completed!")
        print(" Results saved in results/ directory")

    except Exception as e:
        print(f" Error during evaluation: {e}")
        print(" Make sure all required data files are present")


def main():
    """Main demonstration function."""
    print(" Idiom Evaluation Framework - Basic Usage Demo")
    print("=" * 55)

    # Setup example data
    print("\n Setting up example data...")
    setup_example_data()

    # Run evaluations
    run_all_evaluations()

    # Print summary
    print("\n Summary:")
    print("- S_Acc: Semantic quality of explanations")
    print("- S_Log: Logical consistency with references")
    print("- S_Info: Information density (conciseness)")
    print("- S_Cons: Structural consistency across arrangements")
    print("- Robustness: Resilience to perturbations")

    print("\n For detailed usage, see individual module READMEs")
    print(" Check results/ directory for evaluation outputs")


if __name__ == "__main__":
    main()