from pathlib import Path
import importlib.metadata
import platform

import pandas as pd


root = Path(__file__).resolve().parents[1]
tables = root / "results" / "tables"
reports = root / "results" / "reports"

tables.mkdir(parents=True, exist_ok=True)
reports.mkdir(parents=True, exist_ok=True)


important_files = [
    "src/simulation/power_model.py",
    "src/simulation/thermal_model.py",
    "src/simulation/reaction_wheel_model.py",
    "src/simulation/sensor_model.py",
    "src/simulation/spacecraft_simulator.py",
    "src/faults/fault_manager.py",
    "src/faults/sensor_faults.py",
    "src/detection/rule_based_detector.py",
    "src/detection/model_based_detector_refined.py",
    "src/detection/hybrid_fusion.py",
    "src/estimation/battery_ekf.py",
    "src/diagnosis/diagnostic_features.py",
    "src/diagnosis/random_forest_diagnoser.py",
    "src/diagnosis/confidence_aware_diagnoser.py",
    "results/models/experiment_031_regime_balanced_subsystem_iforest.joblib",
    "results/models/experiment_037_random_forest_diagnoser.joblib",
    "data/nominal/nominal_spacecraft_telemetry.csv",
    "data/nominal/nominal_spacecraft_telemetry_24h.csv",
    "results/tables/experiment_035_phase6_final_comparison.csv",
    "results/tables/experiment_040_phase7_final_summary.csv",
    "results/tables/experiment_045_phase8_final_summary.csv",
    "results/tables/experiment_046_monte_carlo_scenarios.csv",
    "results/tables/experiment_047_monte_carlo_manifest.csv",
    "results/tables/experiment_048_detection_method_metrics.csv",
    "results/tables/experiment_048_per_fault_robustness.csv",
    "results/tables/experiment_050_confidence_gate_summary.csv",
    "results/tables/experiment_051_monte_carlo_xai_samples.csv",
    "results/tables/experiment_051_xai_stability.csv",
    "results/tables/experiment_052_phase9_final_summary.csv",
    "results/tables/experiment_053_master_research_results.csv",
    "results/figures/experiment_053_final_detection_comparison.png",
    "results/figures/experiment_053_diagnosis_generalization.png",
    "results/figures/experiment_053_xai_generalization.png",
    "results/figures/experiment_054_sentinel_xai_architecture.png",
    "results/figures/experiment_054_research_evolution.png",
    "results/figures/experiment_054_research_contributions.png",
]


packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "joblib",
    "shap",
]


def get_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    rows = []

    for relative in important_files:
        path = root / relative

        rows.append(
            {
                "relative_path": relative,
                "exists": path.exists(),
                "size_bytes": (
                    path.stat().st_size
                    if path.exists() and path.is_file()
                    else 0
                ),
            }
        )

    manifest = pd.DataFrame(rows)

    manifest_file = tables / "experiment_055_repository_manifest.csv"
    manifest.to_csv(manifest_file, index=False)

    versions = {}

    for package in packages:
        versions[package] = get_version(package)

    requirements_lines = []

    for package, version in versions.items():
        if version is None:
            requirements_lines.append(package)
        else:
            requirements_lines.append(f"{package}=={version}")

    requirements_file = root / "requirements.txt"

    requirements_file.write_text(
        "\n".join(requirements_lines) + "\n",
        encoding="utf-8",
    )

    readme_file = root / "README.md"

    readme_text = """# SENTINEL-XAI

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
"""

    readme_file.write_text(
        readme_text,
        encoding="utf-8",
    )

    reproducibility_file = root / "REPRODUCIBILITY.md"

    reproducibility_text = f"""# SENTINEL-XAI Reproducibility

Python version: {platform.python_version()}

Platform: {platform.platform()}

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
"""

    reproducibility_file.write_text(
        reproducibility_text,
        encoding="utf-8",
    )

    missing = manifest[~manifest["exists"]]

    report_file = reports / "experiment_055_packaging_summary.txt"

    report_lines = [
        "SENTINEL-XAI - EXPERIMENT 055",
        "RESEARCH PACKAGING SUMMARY",
        "",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        f"Files checked: {len(manifest)}",
        f"Files present: {int(manifest['exists'].sum())}",
        f"Files missing: {len(missing)}",
        "",
        "Package versions:",
    ]

    for package, version in versions.items():
        report_lines.append(
            f"{package}: {version}"
        )

    if len(missing) == 0:
        report_lines.extend(
            [
                "",
                "RESULT: ALL IMPORTANT FILES FOUND",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "Missing files:",
            ]
        )

        for relative in missing["relative_path"]:
            report_lines.append(relative)

    report_file.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("SENTINEL-XAI - EXPERIMENT 055")
    print("README + REPRODUCIBILITY PACKAGING")
    print("=" * 90)
    print()

    print(f"Python: {platform.python_version()}")
    print()

    for package, version in versions.items():
        print(f"{package}: {version}")

    print()
    print(f"Important files checked: {len(manifest)}")
    print(f"Files present: {int(manifest['exists'].sum())}")
    print(f"Files missing: {len(missing)}")

    if len(missing) > 0:
        print()
        print("Missing files:")

        for relative in missing["relative_path"]:
            print(relative)

    print()
    print("Created:")
    print(readme_file)
    print(reproducibility_file)
    print(requirements_file)
    print(manifest_file)
    print(report_file)
    print()

    if len(missing) == 0:
        print("RESULT: RESEARCH PACKAGING PASSED")
    else:
        print("RESULT: PACKAGING CREATED WITH MISSING FILES")

    print("=" * 90)


if __name__ == "__main__":
    main()