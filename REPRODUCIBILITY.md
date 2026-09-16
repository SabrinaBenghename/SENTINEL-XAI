# SENTINEL-XAI Reproducibility

Python version: 3.14.3

Platform: Windows-11-10.0.26200-SP0

## Installation

python -m pip install -r requirements.txt

## Frozen Models

results/models/experiment_031_regime_balanced_subsystem_iforest.joblib

results/models/experiment_037_random_forest_diagnoser.joblib

## Phase Validations

python experiments/experiment_005_final_nominal_validation.py

python experiments/experiment_012_phase2_validation.py

python experiments/experiment_015_phase3_validation.py

python experiments/experiment_023_phase4_final_validation.py

python experiments/experiment_032_phase5_final_validation.py

python experiments/experiment_035_phase6_final_validation.py

python experiments/experiment_040_phase7_final_validation.py

python experiments/experiment_045_phase8_final_validation.py

## Independent Monte Carlo Evaluation

python experiments/experiment_046_monte_carlo_scenario_design.py

python experiments/experiment_047_monte_carlo_generation.py

python experiments/experiment_048_monte_carlo_generalization.py

python experiments/experiment_049_monte_carlo_failure_analysis.py

python experiments/experiment_050_confidence_gate_independent_validation.py

python experiments/experiment_051_monte_carlo_xai_robustness.py

python experiments/experiment_052_phase9_final_validation.py

## Final Research Outputs

python experiments/experiment_053_final_research_summary.py

python experiments/experiment_054_final_architecture_figures.py

## Independent-Test Rules

Do not retrain using the Monte Carlo test scenarios.

Do not alter the frozen 0.70 confidence threshold.

Do not use hidden simulation truth variables as operational classifier inputs.

Do not tune the independent test set until desired metrics are obtained.
