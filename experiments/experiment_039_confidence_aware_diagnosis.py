from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    FAULT_CLASSES,
)

from src.diagnosis.confidence_aware_diagnoser import (
    ConfidenceAwareDiagnoser,
)


# ==========================================================
# BLOCKED SPLIT
# ==========================================================

def blocked_split(
    df,
    train_fraction=0.70,
):

    train_parts = []
    test_parts = []

    classes = [
        "nominal",
        *FAULT_CLASSES,
    ]

    for class_name in classes:

        subset = (
            df[
                df["target_label"]
                ==
                class_name
            ]
            .sort_values(
                "time_h"
            )
            .reset_index(
                drop=True
            )
        )

        split_index = int(
            len(subset)
            * train_fraction
        )

        train_parts.append(
            subset.iloc[
                :split_index
            ].copy()
        )

        test_parts.append(
            subset.iloc[
                split_index:
            ].copy()
        )

    return (
        pd.concat(
            train_parts,
            ignore_index=True,
        ),
        pd.concat(
            test_parts,
            ignore_index=True,
        ),
    )


# ==========================================================
# SAFE DIVISION
# ==========================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD DATASET
    # ======================================================

    dataset_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_036_diagnosis_feature_dataset.csv"
    )

    diagnosis_df = pd.read_csv(
        dataset_file
    )

    _, test_df = blocked_split(
        diagnosis_df,
        train_fraction=0.70,
    )

    test_x = test_df[
        DIAGNOSTIC_FEATURES
    ]

    test_y = test_df[
        "target_label"
    ].to_numpy()

    # ======================================================
    # LOAD FROZEN RANDOM FOREST
    # ======================================================

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_037_random_forest_diagnoser.joblib"
    )

    diagnoser = joblib.load(
        model_file
    )

    # ======================================================
    # CONFIDENCE-AWARE WRAPPER
    #
    # Development threshold selected from Experiment 038.
    # Independent validation will occur in Phase 9.
    # ======================================================

    confidence_threshold = 0.70

    confidence_diagnoser = (
        ConfidenceAwareDiagnoser(
            diagnoser=diagnoser,
            confidence_threshold=(
                confidence_threshold
            ),
        )
    )

    prediction = (
        confidence_diagnoser.predict(
            test_x
        )
    )

    raw_labels = (
        prediction.raw_labels
    )

    final_labels = (
        prediction.final_labels
    )

    confidence = (
        prediction.confidence
    )

    accepted = (
        prediction.accepted
    )

    # ======================================================
    # RAW CLASSIFIER PERFORMANCE
    # ======================================================

    raw_correct = (
        raw_labels
        ==
        test_y
    )

    raw_accuracy = float(
        np.mean(
            raw_correct
        )
    )

    total_errors = int(
        (~raw_correct).sum()
    )

    # ======================================================
    # SELECTIVE / CONFIDENCE-AWARE PERFORMANCE
    # ======================================================

    accepted_count = int(
        accepted.sum()
    )

    uncertain_count = int(
        (~accepted).sum()
    )

    coverage = safe_divide(
        accepted_count,
        len(test_y),
    )

    accepted_correct = (
        raw_correct
        &
        accepted
    )

    accepted_errors = (
        (~raw_correct)
        &
        accepted
    )

    rejected_errors = (
        (~raw_correct)
        &
        (~accepted)
    )

    rejected_correct = (
        raw_correct
        &
        (~accepted)
    )

    accepted_accuracy = safe_divide(
        int(
            accepted_correct.sum()
        ),
        accepted_count,
    )

    error_capture_rate = safe_divide(
        int(
            rejected_errors.sum()
        ),
        total_errors,
    )

    total_correct = int(
        raw_correct.sum()
    )

    correct_retention_rate = safe_divide(
        int(
            accepted_correct.sum()
        ),
        total_correct,
    )

    # ======================================================
    # PER-CLASS COVERAGE / ACCURACY
    # ======================================================

    classes = [
        "nominal",
        *FAULT_CLASSES,
    ]

    per_class_rows = []

    for class_name in classes:

        class_mask = (
            test_y
            ==
            class_name
        )

        class_count = int(
            class_mask.sum()
        )

        class_accepted = (
            class_mask
            &
            accepted
        )

        class_uncertain = (
            class_mask
            &
            (~accepted)
        )

        accepted_in_class = int(
            class_accepted.sum()
        )

        correct_and_accepted = int(
            (
                class_accepted
                &
                raw_correct
            ).sum()
        )

        per_class_rows.append(
            {
                "class":
                    class_name,

                "samples":
                    class_count,

                "accepted":
                    accepted_in_class,

                "uncertain":
                    int(
                        class_uncertain.sum()
                    ),

                "coverage":
                    safe_divide(
                        accepted_in_class,
                        class_count,
                    ),

                "accepted_accuracy":
                    safe_divide(
                        correct_and_accepted,
                        accepted_in_class,
                    ),
            }
        )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    # ======================================================
    # THRESHOLD SWEEP
    #
    # Development analysis only.
    # ======================================================

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    threshold_rows = []

    for threshold in thresholds:

        threshold_accept = (
            confidence
            >=
            threshold
        )

        count = int(
            threshold_accept.sum()
        )

        correct = int(
            (
                threshold_accept
                &
                raw_correct
            ).sum()
        )

        errors = int(
            (
                threshold_accept
                &
                (~raw_correct)
            ).sum()
        )

        threshold_rows.append(
            {
                "threshold":
                    threshold,

                "accepted_samples":
                    count,

                "coverage":
                    safe_divide(
                        count,
                        len(test_y),
                    ),

                "accepted_accuracy":
                    safe_divide(
                        correct,
                        count,
                    ),

                "accepted_errors":
                    errors,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # ======================================================
    # PREDICTION TABLE
    # ======================================================

    prediction_df = pd.DataFrame(
        {
            "source_dataset":
                test_df[
                    "source_dataset"
                ].values,

            "time_h":
                test_df[
                    "time_h"
                ].values,

            "true_label":
                test_y,

            "raw_prediction":
                raw_labels,

            "confidence":
                confidence,

            "accepted":
                accepted,

            "final_diagnosis":
                final_labels,

            "raw_correct":
                raw_correct,
        }
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

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

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        tables_dir
        / "experiment_039_confidence_aware_predictions.csv"
    )

    per_class_file = (
        tables_dir
        / "experiment_039_per_class_selective_metrics.csv"
    )

    threshold_file = (
        tables_dir
        / "experiment_039_threshold_analysis.csv"
    )

    prediction_df.to_csv(
        predictions_file,
        index=False,
    )

    per_class_df.to_csv(
        per_class_file,
        index=False,
    )

    threshold_df.to_csv(
        threshold_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PER-CLASS COVERAGE
    # ======================================================

    coverage_figure = (
        figures_dir
        / "experiment_039_per_class_coverage.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        per_class_df[
            "class"
        ],
        per_class_df[
            "coverage"
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Diagnostic Class"
    )

    plt.ylabel(
        "Accepted Diagnosis Coverage"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Confidence-Aware Diagnosis Coverage"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        coverage_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — THRESHOLD TRADE-OFF
    # ======================================================

    tradeoff_figure = (
        figures_dir
        / "experiment_039_confidence_tradeoff.png"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        threshold_df[
            "threshold"
        ],
        threshold_df[
            "coverage"
        ],
        marker="o",
        label="Coverage",
    )

    plt.plot(
        threshold_df[
            "threshold"
        ],
        threshold_df[
            "accepted_accuracy"
        ],
        marker="o",
        label="Accepted accuracy",
    )

    plt.axvline(
        confidence_threshold,
        linestyle="--",
        label="Selected development threshold",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Confidence / Coverage Trade-off"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        tradeoff_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 88)

    print(
        "SENTINEL-XAI - EXPERIMENT 039"
    )

    print(
        "CONFIDENCE-AWARE FAULT DIAGNOSIS"
    )

    print("=" * 88)

    print()

    print(
        f"Test samples: "
        f"{len(test_y)}"
    )

    print(
        f"Raw diagnosis accuracy: "
        f"{raw_accuracy:.4f}"
    )

    print(
        f"Development confidence threshold: "
        f"{confidence_threshold:.2f}"
    )

    print()

    print("-" * 88)
    print(
        "SELECTIVE DIAGNOSIS PERFORMANCE"
    )
    print("-" * 88)

    print()

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
        f"Accuracy among accepted diagnoses: "
        f"{accepted_accuracy:.4f}"
    )

    print()

    print(
        f"Original classifier errors: "
        f"{total_errors}"
    )

    print(
        f"Errors still accepted: "
        f"{int(accepted_errors.sum())}"
    )

    print(
        f"Errors rejected as uncertain: "
        f"{int(rejected_errors.sum())}"
    )

    print(
        f"Error-capture rate: "
        f"{error_capture_rate:.4f}"
    )

    print()

    print(
        f"Correct diagnoses retained: "
        f"{int(accepted_correct.sum())}"
        f"/"
        f"{total_correct}"
    )

    print(
        f"Correct-retention rate: "
        f"{correct_retention_rate:.4f}"
    )

    print()

    print("-" * 88)
    print(
        "PER-CLASS SELECTIVE RESULTS"
    )
    print("-" * 88)

    for _, row in (
        per_class_df.iterrows()
    ):

        print()

        print(
            row["class"]
        )

        print(
            f"  Coverage: "
            f"{row['coverage']:.4f}"
        )

        print(
            f"  Accepted accuracy: "
            f"{row['accepted_accuracy']:.4f}"
        )

        print(
            f"  Uncertain samples: "
            f"{int(row['uncertain'])}"
        )

    print()

    print("-" * 88)
    print(
        "THRESHOLD DEVELOPMENT ANALYSIS"
    )
    print("-" * 88)

    print()

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print()

    print("=" * 88)

    if (
        accepted_accuracy
        >=
        0.99
        and
        error_capture_rate
        >=
        0.95
    ):

        print(
            "RESULT: CONFIDENCE-AWARE "
            "DIAGNOSIS PASSED"
        )

    else:

        print(
            "RESULT: CONFIDENCE-AWARE "
            "DIAGNOSIS REQUIRES REVIEW"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The 0.70 confidence threshold is "
        "development-derived."
    )

    print(
        "Independent threshold validation is "
        "reserved for Phase 9 Monte Carlo testing."
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        predictions_file
    )

    print(
        per_class_file
    )

    print(
        threshold_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        coverage_figure
    )

    print(
        tradeoff_figure
    )

    print("=" * 88)


if __name__ == "__main__":
    main()