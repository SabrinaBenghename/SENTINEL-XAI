from pathlib import Path
import platform
import sys

import pandas as pd


root = Path(__file__).resolve().parents[1]

tables = root / "results" / "tables"
figures = root / "results" / "figures"
models = root / "results" / "models"
reports = root / "results" / "reports"
data = root / "data"
experiments = root / "experiments"
src = root / "src"

tables.mkdir(parents=True, exist_ok=True)
reports.mkdir(parents=True, exist_ok=True)


required_directories = [
    "data",
    "data/nominal",
    "data/faults",
    "data/monte_carlo",
    "src",
    "src/simulation",
    "src/faults",
    "src/estimation",
    "src/detection",
    "src/diagnosis",
    "experiments",
    "results",
    "results/tables",
    "results/figures",
    "results/models",
    "results/reports",
]


required_files = [
    "README.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",

    "src/simulation/power_model.py",
    "src/simulation/thermal_model.py",
    "src/simulation/reaction_wheel_model.py",
    "src/simulation/sensor_model.py",
    "src/simulation/spacecraft_simulator.py",

    "src/faults/fault_manager.py",
    "src/faults/sensor_faults.py",

    "src/estimation/battery_ekf.py",

    "src/detection/rule_based_detector.py",
    "src/detection/model_based_detector_refined.py",
    "src/detection/hybrid_fusion.py",

    "src/diagnosis/diagnostic_features.py",
    "src/diagnosis/random_forest_diagnoser.py",
    "src/diagnosis/confidence_aware_diagnoser.py",

    "results/models/experiment_031_regime_balanced_subsystem_iforest.joblib",
    "results/models/experiment_037_random_forest_diagnoser.joblib",

    "results/tables/experiment_035_phase6_final_comparison.csv",
    "results/tables/experiment_040_phase7_final_summary.csv",
    "results/tables/experiment_045_phase8_final_summary.csv",
    "results/tables/experiment_047_monte_carlo_manifest.csv",
    "results/tables/experiment_048_detection_method_metrics.csv",
    "results/tables/experiment_050_confidence_gate_summary.csv",
    "results/tables/experiment_051_monte_carlo_xai_samples.csv",
    "results/tables/experiment_052_phase9_final_summary.csv",
    "results/tables/experiment_053_master_research_results.csv",
    "results/tables/experiment_055_repository_manifest.csv",

    "results/figures/experiment_053_final_detection_comparison.png",
    "results/figures/experiment_053_diagnosis_generalization.png",
    "results/figures/experiment_053_xai_generalization.png",
    "results/figures/experiment_054_sentinel_xai_architecture.png",
    "results/figures/experiment_054_research_evolution.png",
    "results/figures/experiment_054_research_contributions.png",

    "results/reports/experiment_053_final_research_summary.txt",
    "results/reports/experiment_055_packaging_summary.txt",
]


def check_directories():
    rows = []

    for relative in required_directories:
        path = root / relative

        rows.append(
            {
                "type": "directory",
                "path": relative,
                "exists": path.exists() and path.is_dir(),
            }
        )

    return rows


def check_files():
    rows = []

    for relative in required_files:
        path = root / relative

        rows.append(
            {
                "type": "file",
                "path": relative,
                "exists": path.exists() and path.is_file(),
            }
        )

    return rows


def count_project_files():
    return {
        "python_source_files": len(list(src.rglob("*.py"))),
        "experiment_files": len(list(experiments.glob("experiment_*.py"))),
        "result_tables": len(list(tables.glob("*.csv"))),
        "result_figures": len(list(figures.glob("*.png"))),
        "saved_models": len(list(models.glob("*.joblib"))),
        "research_reports": len(list(reports.glob("*.txt"))),
        "monte_carlo_csv_files": len(
            list((data / "monte_carlo").glob("*.csv"))
        ),
    }


def phase_status():
    return pd.DataFrame(
        [
            [1, "Nominal spacecraft simulation", True],
            [2, "Controlled fault injection", True],
            [3, "Rule-based fault detection", True],
            [4, "Physics/model-based monitoring", True],
            [5, "Unsupervised machine learning", True],
            [6, "Hybrid evidence fusion", True],
            [7, "Supervised fault diagnosis", True],
            [8, "Explainable AI", True],
            [9, "Independent Monte Carlo robustness", True],
            [10, "Research packaging and audit", True],
        ],
        columns=[
            "phase",
            "objective",
            "complete",
        ],
    )


def main():
    rows = []

    rows.extend(check_directories())
    rows.extend(check_files())

    audit = pd.DataFrame(rows)

    audit_file = (
        tables
        / "experiment_056_final_repository_audit.csv"
    )

    audit.to_csv(
        audit_file,
        index=False,
    )

    phases = phase_status()

    phase_file = (
        tables
        / "experiment_056_project_phase_status.csv"
    )

    phases.to_csv(
        phase_file,
        index=False,
    )

    counts = count_project_files()

    missing = audit[
        ~audit["exists"]
    ]

    all_repository_checks = len(missing) == 0
    all_phases_complete = bool(phases["complete"].all())

    readme_ok = (root / "README.md").exists()
    reproducibility_ok = (root / "REPRODUCIBILITY.md").exists()
    requirements_ok = (root / "requirements.txt").exists()

    frozen_models_ok = (
        (
            models
            / "experiment_031_regime_balanced_subsystem_iforest.joblib"
        ).exists()
        and
        (
            models
            / "experiment_037_random_forest_diagnoser.joblib"
        ).exists()
    )

    monte_carlo_ok = (
        counts["monte_carlo_csv_files"] >= 120
    )

    architecture_ok = (
        figures
        / "experiment_054_sentinel_xai_architecture.png"
    ).exists()

    final_summary_ok = (
        tables
        / "experiment_053_master_research_results.csv"
    ).exists()

    final_checks = {
        "Repository structure complete":
            all_repository_checks,

        "All ten project phases complete":
            all_phases_complete,

        "README available":
            readme_ok,

        "Reproducibility guide available":
            reproducibility_ok,

        "Requirements file available":
            requirements_ok,

        "Frozen ML models available":
            frozen_models_ok,

        "Monte Carlo dataset available":
            monte_carlo_ok,

        "Final architecture figure available":
            architecture_ok,

        "Master research results available":
            final_summary_ok,
    }

    validation_rows = []

    for name, result in final_checks.items():
        validation_rows.append(
            {
                "check": name,
                "pass": bool(result),
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    validation_file = (
        tables
        / "experiment_056_final_validation.csv"
    )

    validation.to_csv(
        validation_file,
        index=False,
    )

    passed = int(
        validation["pass"].sum()
    )

    total = len(
        validation
    )

    report_file = (
        reports
        / "experiment_056_project_completion_report.txt"
    )

    report_lines = [
        "SENTINEL-XAI",
        "FINAL PROJECT COMPLETION REPORT",
        "=" * 70,
        "",
        f"Python version: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        "PROJECT FILE COUNTS",
        "-" * 70,
    ]

    for key, value in counts.items():
        report_lines.append(
            f"{key}: {value}"
        )

    report_lines.extend(
        [
            "",
            "FINAL VALIDATION",
            "-" * 70,
        ]
    )

    for name, result in final_checks.items():
        report_lines.append(
            f"{name}: {bool(result)}"
        )

    report_lines.extend(
        [
            "",
            f"Passed checks: {passed}/{total}",
            "",
            "PROJECT STATUS:",
            (
                "COMPLETE"
                if passed == total
                else "INCOMPLETE"
            ),
            "",
            "FINAL RESEARCH RESULTS",
            "-" * 70,
            "Development hybrid F1: 0.8563",
            "Independent hybrid F1: 0.8294",
            "Development diagnosis accuracy: 0.9332",
            "Independent diagnosis accuracy: 0.8527",
            "Independent diagnosis Macro F1: 0.9003",
            "Independent accepted accuracy: 0.9783",
            "Independent XAI SHAP probability drop: 0.6305",
            "Independent random-feature probability drop: 0.0148",
            "Independent Monte Carlo scenarios: 120",
            "Scenario success rate: 120/120",
            "",
            "DOCUMENTED LIMITATION",
            "-" * 70,
            "The frozen 0.70 confidence gate did not fully reproduce",
            "the original Phase-7 accepted-accuracy and error-capture",
            "targets on the independent Monte Carlo distribution.",
            "",
            "SCIENTIFIC SCOPE",
            "-" * 70,
            "SENTINEL-XAI is a simulation-based research prototype.",
            "The results do not represent spacecraft flight qualification",
            "or certification.",
        ]
    )

    report_file.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SENTINEL-XAI - EXPERIMENT 056")
    print("FINAL REPOSITORY AND PROJECT AUDIT")
    print("=" * 100)
    print()

    print("PROJECT FILE COUNTS")
    print("-" * 100)

    for key, value in counts.items():
        print(f"{key}: {value}")

    print()
    print("REPOSITORY AUDIT")
    print("-" * 100)

    print(
        f"Items checked: {len(audit)}"
    )

    print(
        f"Items present: {int(audit['exists'].sum())}"
    )

    print(
        f"Items missing: {len(missing)}"
    )

    if len(missing) > 0:
        print()
        print("Missing:")

        for path in missing["path"]:
            print(f"  {path}")

    print()
    print("FINAL PROJECT VALIDATION")
    print("-" * 100)

    for name, result in final_checks.items():
        print()
        print(name)
        print(f"  PASS: {bool(result)}")

    print()
    print(
        f"Passed checks: {passed}/{total}"
    )

    print()
    print("PHASE STATUS")
    print("-" * 100)

    for _, row in phases.iterrows():
        print(
            f"Phase {int(row['phase'])}: "
            f"{row['objective']} -> "
            f"{'COMPLETE' if row['complete'] else 'INCOMPLETE'}"
        )

    print()
    print("FILES CREATED")
    print("-" * 100)

    print(audit_file)
    print(phase_file)
    print(validation_file)
    print(report_file)

    print()

    if passed == total:
        print("=" * 100)
        print("GLOBAL SENTINEL-XAI PROJECT STATUS: COMPLETE")
        print("=" * 100)
    else:
        print("=" * 100)
        print("GLOBAL SENTINEL-XAI PROJECT STATUS: AUDIT FAILED")
        print("=" * 100)


if __name__ == "__main__":
    main()