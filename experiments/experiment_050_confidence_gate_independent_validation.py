from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

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


# ==========================================================
# FROZEN PHASE-7 CONFIGURATION
# ==========================================================

FROZEN_CONFIDENCE_THRESHOLD = 0.70


# These criteria were already used in Phase 7.
# They are NOT chosen from the Monte Carlo results.

DEVELOPMENT_MIN_ACCEPTED_ACCURACY = 0.99

DEVELOPMENT_MIN_ERROR_CAPTURE_RATE = 0.95

DEVELOPMENT_MIN_COVERAGE = 0.60


FAULT_TYPES = [
    "solar_array_degradation",
    "battery_degradation",
    "thermal_anomaly",
    "reaction_wheel_degradation",
    "battery_voltage_sensor_drift",
    "telemetry_dropout",
]


# ==========================================================
# HELPERS
# ==========================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return float(
        numerator
        /
        denominator
    )


def load_csv(
    filename,
):

    path = (
        tables_dir
        /
        filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def as_bool(
    values,
):

    series = pd.Series(
        values
    )

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series
            .astype(bool)
            .to_numpy()
        )

    if pd.api.types.is_numeric_dtype(
        series
    ):

        return (
            series
            .fillna(0)
            .astype(float)
            .ne(0)
            .to_numpy()
        )

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "t",
            ]
        )
        .to_numpy()
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD INDEPENDENT MONTE CARLO DIAGNOSIS RESULTS
    # ======================================================

    predictions = load_csv(
        "experiment_048_diagnosis_predictions.csv"
    )

    if len(
        predictions
    ) == 0:

        raise RuntimeError(
            "Experiment-048 diagnosis table is empty."
        )

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    required_columns = [
        "scenario_id",
        "fault_type",
        "time_h",
        "raw_prediction",
        "confidence",
        "accepted",
        "raw_correct",
    ]

    missing_columns = [
        column

        for column in required_columns

        if column
        not in
        predictions.columns
    ]

    if missing_columns:

        raise ValueError(
            "Experiment-048 diagnosis table is "
            "missing required columns:\n"
            f"{missing_columns}"
        )

    # ======================================================
    # NORMALIZE DATA
    # ======================================================

    confidence = pd.to_numeric(
        predictions[
            "confidence"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    raw_correct = as_bool(
        predictions[
            "raw_correct"
        ]
    )

    saved_accepted = as_bool(
        predictions[
            "accepted"
        ]
    )

    # ======================================================
    # RECOMPUTE FROZEN GATE
    #
    # This verifies Experiment 048 really used 0.70.
    # ======================================================

    frozen_accepted = (
        confidence
        >=
        FROZEN_CONFIDENCE_THRESHOLD
    )

    gate_consistency = bool(
        np.array_equal(
            saved_accepted,
            frozen_accepted,
        )
    )

    accepted = frozen_accepted

    # ======================================================
    # GLOBAL COUNTS
    # ======================================================

    total_samples = int(
        len(
            predictions
        )
    )

    total_correct = int(
        raw_correct.sum()
    )

    total_errors = int(
        (
            ~raw_correct
        ).sum()
    )

    accepted_samples = int(
        accepted.sum()
    )

    uncertain_samples = int(
        (
            ~accepted
        ).sum()
    )

    accepted_correct = int(
        (
            accepted
            &
            raw_correct
        ).sum()
    )

    accepted_errors = int(
        (
            accepted
            &
            ~raw_correct
        ).sum()
    )

    rejected_errors = int(
        (
            ~accepted
            &
            ~raw_correct
        ).sum()
    )

    rejected_correct = int(
        (
            ~accepted
            &
            raw_correct
        ).sum()
    )

    # ======================================================
    # GLOBAL CONFIDENCE-GATE METRICS
    # ======================================================

    raw_accuracy = safe_divide(
        total_correct,
        total_samples,
    )

    coverage = safe_divide(
        accepted_samples,
        total_samples,
    )

    accepted_accuracy = safe_divide(
        accepted_correct,
        accepted_samples,
    )

    uncertain_fraction = safe_divide(
        uncertain_samples,
        total_samples,
    )

    error_capture_rate = safe_divide(
        rejected_errors,
        total_errors,
    )

    correct_retention_rate = safe_divide(
        accepted_correct,
        total_correct,
    )

    uncertain_accuracy = safe_divide(
        rejected_correct,
        uncertain_samples,
    )

    # ======================================================
    # CONFIDENCE SEPARATION
    # ======================================================

    if total_correct > 0:

        mean_correct_confidence = float(
            np.mean(
                confidence[
                    raw_correct
                ]
            )
        )

        median_correct_confidence = float(
            np.median(
                confidence[
                    raw_correct
                ]
            )
        )

    else:

        mean_correct_confidence = np.nan
        median_correct_confidence = np.nan

    if total_errors > 0:

        mean_error_confidence = float(
            np.mean(
                confidence[
                    ~raw_correct
                ]
            )
        )

        median_error_confidence = float(
            np.median(
                confidence[
                    ~raw_correct
                ]
            )
        )

        maximum_error_confidence = float(
            np.max(
                confidence[
                    ~raw_correct
                ]
            )
        )

    else:

        mean_error_confidence = np.nan
        median_error_confidence = np.nan
        maximum_error_confidence = np.nan

    # ======================================================
    # PER-FAULT INDEPENDENT RESULTS
    # ======================================================

    per_fault_rows = []

    for fault_type in FAULT_TYPES:

        mask = (
            predictions[
                "fault_type"
            ].astype(str)
            ==
            fault_type
        ).to_numpy()

        subset_confidence = (
            confidence[
                mask
            ]
        )

        subset_correct = (
            raw_correct[
                mask
            ]
        )

        subset_accepted = (
            accepted[
                mask
            ]
        )

        samples = int(
            mask.sum()
        )

        correct_count = int(
            subset_correct.sum()
        )

        error_count = int(
            (
                ~subset_correct
            ).sum()
        )

        accepted_count = int(
            subset_accepted.sum()
        )

        accepted_correct_count = int(
            (
                subset_accepted
                &
                subset_correct
            ).sum()
        )

        accepted_error_count = int(
            (
                subset_accepted
                &
                ~subset_correct
            ).sum()
        )

        rejected_error_count = int(
            (
                ~subset_accepted
                &
                ~subset_correct
            ).sum()
        )

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "samples":
                    samples,

                "raw_accuracy":
                    safe_divide(
                        correct_count,
                        samples,
                    ),

                "mean_confidence":
                    float(
                        np.mean(
                            subset_confidence
                        )
                    ),

                "coverage":
                    safe_divide(
                        accepted_count,
                        samples,
                    ),

                "accepted_samples":
                    accepted_count,

                "accepted_accuracy":
                    safe_divide(
                        accepted_correct_count,
                        accepted_count,
                    ),

                "accepted_errors":
                    accepted_error_count,

                "error_capture_rate":
                    safe_divide(
                        rejected_error_count,
                        error_count,
                    ),

                "correct_retention_rate":
                    safe_divide(
                        accepted_correct_count,
                        correct_count,
                    ),
            }
        )

    per_fault_df = pd.DataFrame(
        per_fault_rows
    )

    # ======================================================
    # DESCRIPTIVE THRESHOLD SENSITIVITY
    #
    # IMPORTANT:
    # This does NOT select a new threshold.
    # 0.70 remains frozen.
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

        threshold_accepted = (
            confidence
            >=
            threshold
        )

        threshold_accepted_count = int(
            threshold_accepted.sum()
        )

        threshold_correct = int(
            (
                threshold_accepted
                &
                raw_correct
            ).sum()
        )

        threshold_errors = int(
            (
                threshold_accepted
                &
                ~raw_correct
            ).sum()
        )

        threshold_rejected_errors = int(
            (
                ~threshold_accepted
                &
                ~raw_correct
            ).sum()
        )

        threshold_rows.append(
            {
                "threshold":
                    threshold,

                "frozen_threshold":
                    bool(
                        np.isclose(
                            threshold,
                            FROZEN_CONFIDENCE_THRESHOLD,
                        )
                    ),

                "accepted_samples":
                    threshold_accepted_count,

                "coverage":
                    safe_divide(
                        threshold_accepted_count,
                        total_samples,
                    ),

                "accepted_accuracy":
                    safe_divide(
                        threshold_correct,
                        threshold_accepted_count,
                    ),

                "accepted_errors":
                    threshold_errors,

                "error_capture_rate":
                    safe_divide(
                        threshold_rejected_errors,
                        total_errors,
                    ),

                "correct_retention_rate":
                    safe_divide(
                        threshold_correct,
                        total_correct,
                    ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # ======================================================
    # CONFIDENCE BINS
    # ======================================================

    bin_edges = [
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.000001,
    ]

    bin_labels = [
        "<0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    confidence_bins = pd.cut(
        confidence,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    confidence_bin_rows = []

    for label in bin_labels:

        mask = (
            confidence_bins
            ==
            label
        )

        count = int(
            mask.sum()
        )

        if count > 0:

            bin_accuracy = float(
                raw_correct[
                    mask
                ].mean()
            )

            mean_confidence = float(
                confidence[
                    mask
                ].mean()
            )

        else:

            bin_accuracy = np.nan
            mean_confidence = np.nan

        confidence_bin_rows.append(
            {
                "confidence_bin":
                    label,

                "samples":
                    count,

                "mean_confidence":
                    mean_confidence,

                "observed_accuracy":
                    bin_accuracy,
            }
        )

    confidence_bins_df = pd.DataFrame(
        confidence_bin_rows
    )

    # ======================================================
    # ORIGINAL PHASE-7 CRITERIA
    #
    # These were specified before the Monte Carlo test.
    # ======================================================

    validation_checks = {

        "Frozen threshold reproduced exactly":
            gate_consistency,

        "Confidence threshold unchanged at 0.70":
            bool(
                FROZEN_CONFIDENCE_THRESHOLD
                ==
                0.70
            ),

        "All six fault classes represented":
            bool(
                set(
                    predictions[
                        "fault_type"
                    ].astype(str)
                )
                ==
                set(
                    FAULT_TYPES
                )
            ),

        "Accepted diagnosis accuracy":
            bool(
                accepted_accuracy
                >=
                DEVELOPMENT_MIN_ACCEPTED_ACCURACY
            ),

        "Error-capture rate":
            bool(
                error_capture_rate
                >=
                DEVELOPMENT_MIN_ERROR_CAPTURE_RATE
            ),

        "Selective diagnosis coverage":
            bool(
                coverage
                >=
                DEVELOPMENT_MIN_COVERAGE
            ),
    }

    # ======================================================
    # SAVE TABLES
    # ======================================================

    summary_file = (
        tables_dir
        /
        "experiment_050_confidence_gate_summary.csv"
    )

    per_fault_file = (
        tables_dir
        /
        "experiment_050_per_fault_confidence.csv"
    )

    threshold_file = (
        tables_dir
        /
        "experiment_050_threshold_sensitivity.csv"
    )

    bins_file = (
        tables_dir
        /
        "experiment_050_confidence_bins.csv"
    )

    validation_file = (
        tables_dir
        /
        "experiment_050_validation_checks.csv"
    )

    summary_df = pd.DataFrame(
        [
            {
                "threshold":
                    FROZEN_CONFIDENCE_THRESHOLD,

                "samples":
                    total_samples,

                "raw_accuracy":
                    raw_accuracy,

                "accepted_samples":
                    accepted_samples,

                "uncertain_samples":
                    uncertain_samples,

                "coverage":
                    coverage,

                "accepted_accuracy":
                    accepted_accuracy,

                "accepted_errors":
                    accepted_errors,

                "total_classifier_errors":
                    total_errors,

                "errors_rejected":
                    rejected_errors,

                "error_capture_rate":
                    error_capture_rate,

                "correct_retention_rate":
                    correct_retention_rate,

                "uncertain_accuracy":
                    uncertain_accuracy,

                "mean_confidence_correct":
                    mean_correct_confidence,

                "mean_confidence_wrong":
                    mean_error_confidence,

                "maximum_error_confidence":
                    maximum_error_confidence,
            }
        ]
    )

    validation_df = pd.DataFrame(
        [
            {
                "check":
                    name,

                "pass":
                    result,
            }

            for name, result
            in validation_checks.items()
        ]
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    threshold_df.to_csv(
        threshold_file,
        index=False,
    )

    confidence_bins_df.to_csv(
        bins_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — THRESHOLD SENSITIVITY
    # ======================================================

    threshold_figure = (
        figures_dir
        /
        "experiment_050_confidence_threshold_sensitivity.png"
    )

    plt.figure(
        figsize=(9, 6)
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

    plt.plot(
        threshold_df[
            "threshold"
        ],
        threshold_df[
            "error_capture_rate"
        ],
        marker="o",
        label="Error-capture rate",
    )

    plt.axvline(
        FROZEN_CONFIDENCE_THRESHOLD,
        linestyle="--",
        label="Frozen 0.70 threshold",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Independent Confidence-Gate Validation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        threshold_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — PER-FAULT SELECTIVE PERFORMANCE
    # ======================================================

    per_fault_figure = (
        figures_dir
        /
        "experiment_050_per_fault_confidence_gate.png"
    )

    x = np.arange(
        len(
            per_fault_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        per_fault_df[
            "coverage"
        ],
        width=width,
        label="Coverage",
    )

    plt.bar(
        x + width / 2,
        per_fault_df[
            "accepted_accuracy"
        ],
        width=width,
        label="Accepted accuracy",
    )

    plt.xticks(
        x,
        per_fault_df[
            "fault_type"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Frozen 0.70 Gate by Fault Type"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        per_fault_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — CORRECT vs WRONG CONFIDENCE
    # ======================================================

    confidence_figure = (
        figures_dir
        /
        "experiment_050_correct_wrong_confidence.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        confidence[
            raw_correct
        ],
        bins=20,
        alpha=0.6,
        label="Correct predictions",
    )

    plt.hist(
        confidence[
            ~raw_correct
        ],
        bins=20,
        alpha=0.6,
        label="Wrong predictions",
    )

    plt.axvline(
        FROZEN_CONFIDENCE_THRESHOLD,
        linestyle="--",
        label="Frozen 0.70 threshold",
    )

    plt.xlabel(
        "Random Forest Confidence"
    )

    plt.ylabel(
        "Samples"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Confidence Separation on Unseen Monte Carlo Data"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        confidence_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 104)

    print(
        "SENTINEL-XAI - EXPERIMENT 050"
    )

    print(
        "INDEPENDENT CONFIDENCE-GATE VALIDATION"
    )

    print("=" * 104)

    print()

    print(
        "Evaluation data: Experiment-048 "
        "fresh Monte Carlo diagnosis predictions"
    )

    print(
        "Model retraining: NO"
    )

    print(
        "Confidence threshold retuning: NO"
    )

    print(
        f"Frozen threshold: "
        f"{FROZEN_CONFIDENCE_THRESHOLD:.2f}"
    )

    print()

    print("-" * 104)

    print(
        "GLOBAL SELECTIVE-DIAGNOSIS PERFORMANCE"
    )

    print("-" * 104)

    print()

    print(
        f"Total active-fault samples: "
        f"{total_samples}"
    )

    print(
        f"Raw diagnosis accuracy: "
        f"{raw_accuracy:.4f}"
    )

    print()

    print(
        f"Accepted diagnoses: "
        f"{accepted_samples}"
    )

    print(
        f"Uncertain diagnoses: "
        f"{uncertain_samples}"
    )

    print(
        f"Coverage: "
        f"{coverage:.4f}"
    )

    print(
        f"Accepted accuracy: "
        f"{accepted_accuracy:.4f}"
    )

    print()

    print(
        f"Original classifier errors: "
        f"{total_errors}"
    )

    print(
        f"Errors accepted: "
        f"{accepted_errors}"
    )

    print(
        f"Errors rejected as uncertain: "
        f"{rejected_errors}"
    )

    print(
        f"Error-capture rate: "
        f"{error_capture_rate:.4f}"
    )

    print()

    print(
        f"Correct diagnoses retained: "
        f"{accepted_correct}/"
        f"{total_correct}"
    )

    print(
        f"Correct-retention rate: "
        f"{correct_retention_rate:.4f}"
    )

    print()

    print(
        f"Mean confidence when correct: "
        f"{mean_correct_confidence:.4f}"
    )

    print(
        f"Mean confidence when wrong: "
        f"{mean_error_confidence:.4f}"
    )

    print(
        f"Median confidence when correct: "
        f"{median_correct_confidence:.4f}"
    )

    print(
        f"Median confidence when wrong: "
        f"{median_error_confidence:.4f}"
    )

    print(
        f"Maximum confidence among errors: "
        f"{maximum_error_confidence:.4f}"
    )

    print()

    print("-" * 104)

    print(
        "PER-FAULT FROZEN-GATE PERFORMANCE"
    )

    print("-" * 104)

    for _, row in (
        per_fault_df.iterrows()
    ):

        print()

        print(
            row[
                "fault_type"
            ]
        )

        print(
            f"  Raw accuracy: "
            f"{row['raw_accuracy']:.4f}"
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
            f"  Error-capture rate: "
            f"{row['error_capture_rate']:.4f}"
        )

        print(
            f"  Correct-retention rate: "
            f"{row['correct_retention_rate']:.4f}"
        )

    print()

    print("-" * 104)

    print(
        "DESCRIPTIVE THRESHOLD SENSITIVITY"
    )

    print("-" * 104)

    print()

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The table above is descriptive only."
    )

    print(
        "No alternative threshold is selected "
        "from the Monte Carlo test set."
    )

    print()

    print("-" * 104)

    print(
        "INDEPENDENT VALIDATION AGAINST "
        "PHASE-7 DEVELOPMENT CRITERIA"
    )

    print("-" * 104)

    passed = 0

    for name, result in (
        validation_checks.items()
    ):

        if (
            name
            ==
            "Accepted diagnosis accuracy"
        ):

            value_text = (
                f"{accepted_accuracy:.4f} "
                f"(criterion >= "
                f"{DEVELOPMENT_MIN_ACCEPTED_ACCURACY:.2f})"
            )

        elif (
            name
            ==
            "Error-capture rate"
        ):

            value_text = (
                f"{error_capture_rate:.4f} "
                f"(criterion >= "
                f"{DEVELOPMENT_MIN_ERROR_CAPTURE_RATE:.2f})"
            )

        elif (
            name
            ==
            "Selective diagnosis coverage"
        ):

            value_text = (
                f"{coverage:.4f} "
                f"(criterion >= "
                f"{DEVELOPMENT_MIN_COVERAGE:.2f})"
            )

        else:

            value_text = str(
                bool(
                    result
                )
            )

        print()

        print(
            name
        )

        print(
            f"  Value: {value_text}"
        )

        print(
            f"  PASS: {bool(result)}"
        )

        if result:
            passed += 1

    print()

    print(
        f"Passed checks: "
        f"{passed}/"
        f"{len(validation_checks)}"
    )

    print()

    if passed == len(
        validation_checks
    ):

        print(
            "RESULT: FROZEN CONFIDENCE GATE "
            "FULLY SATISFIES THE ORIGINAL "
            "PHASE-7 CRITERIA"
        )

    else:

        print(
            "RESULT: FROZEN CONFIDENCE GATE "
            "DOES NOT FULLY SATISFY ALL "
            "ORIGINAL PHASE-7 CRITERIA "
            "ON THE INDEPENDENT MONTE CARLO SET"
        )

    print()

    print(
        "A failed criterion is a valid robustness "
        "result and will not trigger test-set retuning."
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        summary_file
    )

    print(
        per_fault_file
    )

    print(
        threshold_file
    )

    print(
        bins_file
    )

    print(
        validation_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        threshold_figure
    )

    print(
        per_fault_figure
    )

    print(
        confidence_figure
    )

    print("=" * 104)


if __name__ == "__main__":
    main()