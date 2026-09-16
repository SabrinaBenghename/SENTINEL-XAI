from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


def main():

    tables_dir = (
        project_root
        / "results"
        / "tables"
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # LOAD PHASE-7 RESULTS
    # ======================================================

    diagnosis_dataset = load_csv(
        tables_dir
        / "experiment_036_diagnosis_feature_dataset.csv"
    )

    rf_metrics = load_csv(
        tables_dir
        / "experiment_037_global_metrics.csv"
    ).iloc[0]

    rf_per_class = load_csv(
        tables_dir
        / "experiment_037_per_class_metrics.csv"
    )

    comparison = load_csv(
        tables_dir
        / "experiment_038_phase6_vs_phase7_predictions.csv"
    )

    selective_predictions = load_csv(
        tables_dir
        / "experiment_039_confidence_aware_predictions.csv"
    )

    selective_per_class = load_csv(
        tables_dir
        / "experiment_039_per_class_selective_metrics.csv"
    )

    threshold_analysis = load_csv(
        tables_dir
        / "experiment_039_threshold_analysis.csv"
    )

    # ======================================================
    # PHASE 6 vs PHASE 7
    # ======================================================

    phase6_accuracy = float(
        comparison[
            "phase6_correct"
        ].mean()
    )

    phase7_accuracy = float(
        comparison[
            "phase7_correct"
        ].mean()
    )

    accuracy_improvement = (
        phase7_accuracy
        - phase6_accuracy
    )

    # ======================================================
    # RAW CLASSIFIER RESULTS
    # ======================================================

    balanced_accuracy = float(
        rf_metrics[
            "balanced_accuracy"
        ]
    )

    macro_f1 = float(
        rf_metrics[
            "macro_f1"
        ]
    )

    minimum_class_recall = float(
        rf_per_class[
            "recall"
        ].min()
    )

    # ======================================================
    # CONFIDENCE-AWARE RESULTS
    # ======================================================

    accepted_mask = (
        selective_predictions[
            "accepted"
        ]
        .astype(bool)
    )

    raw_correct = (
        selective_predictions[
            "raw_correct"
        ]
        .astype(bool)
    )

    accepted_count = int(
        accepted_mask.sum()
    )

    uncertain_count = int(
        (~accepted_mask).sum()
    )

    total_samples = len(
        selective_predictions
    )

    coverage = (
        accepted_count
        /
        total_samples
    )

    accepted_correct = int(
        (
            accepted_mask
            &
            raw_correct
        ).sum()
    )

    accepted_errors = int(
        (
            accepted_mask
            &
            (~raw_correct)
        ).sum()
    )

    total_errors = int(
        (~raw_correct).sum()
    )

    rejected_errors = int(
        (
            (~accepted_mask)
            &
            (~raw_correct)
        ).sum()
    )

    accepted_accuracy = (
        accepted_correct
        /
        accepted_count
        if accepted_count > 0
        else 0.0
    )

    error_capture_rate = (
        rejected_errors
        /
        total_errors
        if total_errors > 0
        else 1.0
    )

    # ======================================================
    # SELECTED THRESHOLD ROW
    # ======================================================

    selected_threshold_row = (
        threshold_analysis[
            np.isclose(
                threshold_analysis[
                    "threshold"
                ],
                0.70,
            )
        ]
        .iloc[0]
    )

    selected_threshold = float(
        selected_threshold_row[
            "threshold"
        ]
    )

    # ======================================================
    # PER-CLASS SELECTIVE CHECKS
    # ======================================================

    minimum_selective_coverage = float(
        selective_per_class[
            "coverage"
        ].min()
    )

    every_class_has_accepted_samples = bool(
        (
            selective_per_class[
                "accepted"
            ]
            >
            0
        ).all()
    )

    # ======================================================
    # TARGET LEAKAGE RE-CHECK
    # ======================================================

    metadata_columns = {
        "source_dataset",
        "time_h",
        "target_label",
    }

    feature_columns = [
        column
        for column
        in diagnosis_dataset.columns
        if column
        not in metadata_columns
    ]

    forbidden_tokens = [
        "_true",
        "fault_label",
        "fault_active",
        "fault_severity",
        "effective_severity",
        "requested_severity",
        "fault_progress",
    ]

    forbidden_features = [
        feature
        for feature
        in feature_columns
        if any(
            token
            in feature.lower()
            for token
            in forbidden_tokens
        )
    ]

    target_leakage_free = (
        len(
            forbidden_features
        )
        == 0
        and
        "target_label"
        not in feature_columns
        and
        "time_h"
        not in feature_columns
        and
        "source_dataset"
        not in feature_columns
    )

    # ======================================================
    # VALIDATION CHECKS
    # ======================================================

    checks = [

        {
            "check":
                "Diagnosis dataset has no target leakage",

            "value":
                int(
                    target_leakage_free
                ),

            "criterion":
                "== 1",

            "pass":
                target_leakage_free,
        },

        {
            "check":
                "Phase 7 accuracy improves over Phase 6",

            "value":
                phase7_accuracy,

            "criterion":
                f"> {phase6_accuracy:.4f}",

            "pass":
                phase7_accuracy
                >
                phase6_accuracy,
        },

        {
            "check":
                "Random Forest accuracy",

            "value":
                phase7_accuracy,

            "criterion":
                ">= 0.90",

            "pass":
                phase7_accuracy
                >=
                0.90,
        },

        {
            "check":
                "Balanced accuracy",

            "value":
                balanced_accuracy,

            "criterion":
                ">= 0.90",

            "pass":
                balanced_accuracy
                >=
                0.90,
        },

        {
            "check":
                "Macro F1",

            "value":
                macro_f1,

            "criterion":
                ">= 0.90",

            "pass":
                macro_f1
                >=
                0.90,
        },

        {
            "check":
                "Every class has useful raw recall",

            "value":
                minimum_class_recall,

            "criterion":
                ">= 0.75",

            "pass":
                minimum_class_recall
                >=
                0.75,
        },

        {
            "check":
                "Accepted diagnosis accuracy",

            "value":
                accepted_accuracy,

            "criterion":
                ">= 0.99",

            "pass":
                accepted_accuracy
                >=
                0.99,
        },

        {
            "check":
                "Confidence layer captures classifier errors",

            "value":
                error_capture_rate,

            "criterion":
                ">= 0.95",

            "pass":
                error_capture_rate
                >=
                0.95,
        },

        {
            "check":
                "Selective diagnosis coverage",

            "value":
                coverage,

            "criterion":
                ">= 0.60",

            "pass":
                coverage
                >=
                0.60,
        },

        {
            "check":
                "Every class retains accepted diagnoses",

            "value":
                int(
                    every_class_has_accepted_samples
                ),

            "criterion":
                "== 1",

            "pass":
                every_class_has_accepted_samples,
        },
    ]

    checks_df = pd.DataFrame(
        checks
    )

    global_validation = bool(
        checks_df[
            "pass"
        ].all()
    )

    # ======================================================
    # FINAL SUMMARY TABLE
    # ======================================================

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "Phase 6 classification accuracy",

                "value":
                    phase6_accuracy,
            },

            {
                "metric":
                    "Phase 7 raw classification accuracy",

                "value":
                    phase7_accuracy,
            },

            {
                "metric":
                    "Phase 7 balanced accuracy",

                "value":
                    balanced_accuracy,
            },

            {
                "metric":
                    "Phase 7 macro F1",

                "value":
                    macro_f1,
            },

            {
                "metric":
                    "Selective diagnosis coverage",

                "value":
                    coverage,
            },

            {
                "metric":
                    "Accepted diagnosis accuracy",

                "value":
                    accepted_accuracy,
            },

            {
                "metric":
                    "Error capture rate",

                "value":
                    error_capture_rate,
            },
        ]
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    validation_file = (
        tables_dir
        / "experiment_040_phase7_validation.csv"
    )

    summary_file = (
        tables_dir
        / "experiment_040_phase7_final_summary.csv"
    )

    checks_df.to_csv(
        validation_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ======================================================
    # FINAL FIGURE
    # ======================================================

    figure_file = (
        figures_dir
        / "experiment_040_phase7_final_summary.png"
    )

    names = [
        "Phase 6\nAccuracy",
        "Phase 7\nRaw Accuracy",
        "Phase 7\nMacro F1",
        "Selective\nCoverage",
        "Accepted\nAccuracy",
    ]

    values = [
        phase6_accuracy,
        phase7_accuracy,
        macro_f1,
        coverage,
        accepted_accuracy,
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        names,
        values,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Final Phase-7 Diagnosis Summary"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 88)

    print(
        "SENTINEL-XAI - EXPERIMENT 040"
    )

    print(
        "PHASE 7 FINAL DIAGNOSIS VALIDATION"
    )

    print("=" * 88)

    print()

    print(
        "Raw diagnosis performance:"
    )

    print()

    print(
        f"Phase 6 classification accuracy: "
        f"{phase6_accuracy:.4f}"
    )

    print(
        f"Phase 7 classification accuracy: "
        f"{phase7_accuracy:.4f}"
    )

    print(
        f"Absolute improvement: "
        f"{accuracy_improvement:+.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print()

    print(
        "Confidence-aware diagnosis:"
    )

    print()

    print(
        f"Confidence threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Accepted diagnoses: "
        f"{accepted_count}"
    )

    print(
        f"Uncertain diagnoses: "
        f"{uncertain_count}"
    )

    print(
        f"Coverage: "
        f"{coverage:.4f}"
    )

    print(
        f"Accepted accuracy: "
        f"{accepted_accuracy:.4f}"
    )

    print(
        f"Accepted errors: "
        f"{accepted_errors}"
    )

    print(
        f"Errors rejected as uncertain: "
        f"{rejected_errors}/{total_errors}"
    )

    print(
        f"Error-capture rate: "
        f"{error_capture_rate:.4f}"
    )

    print()

    print("-" * 88)

    print(
        "FINAL PHASE-7 VALIDATION CHECKS"
    )

    print("-" * 88)

    for _, row in checks_df.iterrows():

        print()

        print(
            row["check"]
        )

        print(
            f"  Value: "
            f"{row['value']}"
        )

        print(
            f"  Criterion: "
            f"{row['criterion']}"
        )

        print(
            f"  PASS: "
            f"{bool(row['pass'])}"
        )

    print()

    print("-" * 88)

    print(
        f"Passed checks: "
        f"{int(checks_df['pass'].sum())}"
        f"/"
        f"{len(checks_df)}"
    )

    print(
        f"GLOBAL PHASE 7 VALIDATION: "
        f"{global_validation}"
    )

    print("-" * 88)

    print()

    print(
        "Development note:"
    )

    print(
        "The 0.70 confidence threshold was selected "
        "using Phase-7 development results."
    )

    print(
        "Its independent validation remains reserved "
        "for fresh Phase-9 Monte Carlo scenarios."
    )

    print()

    print(
        "Validation table saved to:"
    )

    print(
        validation_file
    )

    print()

    print(
        "Final summary saved to:"
    )

    print(
        summary_file
    )

    print()

    print(
        "Figure saved to:"
    )

    print(
        figure_file
    )

    print("=" * 88)


if __name__ == "__main__":
    main()