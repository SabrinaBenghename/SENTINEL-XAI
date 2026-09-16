# SENTINEL-XAI

Hybrid Explainable AI for Spacecraft Health Monitoring and Fault Diagnosis.

## Research Goal

SENTINEL-XAI investigates whether spacecraft subsystem faults can be detected and diagnosed more effectively by combining engineering rules, physics-based residual monitoring, machine learning, supervised diagnosis, confidence-aware decision logic, and explainable AI.

## Architecture

![Architecture](results/figures/experiment_054_sentinel_xai_architecture.png)

The system follows this pipeline:

Spacecraft telemetry -> rules + physics + ML -> hybrid fusion -> Random Forest diagnosis -> confidence gate -> SHAP explanation -> health decision.

## Fault Classes

- Solar-array degradation
- Battery degradation
- Thermal anomaly
- Reaction-wheel degradation
- Battery-voltage sensor drift
- Telemetry dropout

The diagnosis model also includes the nominal class.

## Main Detection Results

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Rule-Based | 1.0000 | 0.5634 | 0.7207 |
| Model-Based | 1.0000 | 0.6488 | 0.7870 |
| Isolation Forest | 0.9174 | 0.1448 | 0.2502 |
| Hybrid | 1.0000 | 0.7487 | 0.8563 |

## Fault Diagnosis

Development results:

- Accuracy: 0.9332
- Balanced accuracy: 0.9447
- Macro F1: 0.9419

Independent Monte Carlo results:

- Accuracy: 0.8527
- Balanced accuracy: 0.8541
- Macro F1: 0.9003

![Diagnosis](results/figures/experiment_053_diagnosis_generalization.png)

## Confidence-Aware Diagnosis

Frozen confidence threshold: 0.70

Independent results:

- Coverage: 0.7371
- Accepted accuracy: 0.9783
- Error-capture rate: 0.8913

The Monte Carlo test set was not used to retune the confidence threshold.

## Explainability

Development:

- SHAP top-5 probability drop: 0.7026
- Random-5 probability drop: 0.0145

Independent Monte Carlo:

- SHAP top-5 probability drop: 0.6305
- Random-5 probability drop: 0.0148
- SHAP beat random ablation in 120/120 scenarios
- Mean within-class Jaccard: 0.6876
- Consensus overlap: 0.8400

![XAI](results/figures/experiment_053_xai_generalization.png)

## Independent Monte Carlo Evaluation

120 randomized scenarios were generated.

No retraining and no threshold retuning were performed.

Hybrid detector:

- Precision: 0.9970
- Recall: 0.7100
- F1: 0.8294

Fault diagnosis:

- Accuracy: 0.8527
- Macro F1: 0.9003

Scenario success:

- 120/120 scenarios

Mean correct-diagnosis delay:

- 19.45 minutes

Median correct-diagnosis delay:

- 12.55 minutes

## Research Contributions

![Contributions](results/figures/experiment_054_research_contributions.png)

1. Hybrid detection
2. Fault diagnosis
3. Explainable AI
4. Independent Monte Carlo validation

## Research Evolution

![Evolution](results/figures/experiment_054_research_evolution.png)

## Limitations

The project uses simulated spacecraft telemetry.

The frozen confidence threshold did not fully reproduce the original development calibration on unseen Monte Carlo conditions.

Battery degradation remained one of the more difficult fault modes.

Future work includes real spacecraft telemetry, hardware-in-the-loop testing, simultaneous faults, probability calibration, broader operating regimes, and domain adaptation.

## Installation

Run:

python -m pip install -r requirements.txt

## Scientific Scope

The results apply to the simulated spacecraft models and fault scenarios used in this project.

They do not represent flight qualification or certification.
