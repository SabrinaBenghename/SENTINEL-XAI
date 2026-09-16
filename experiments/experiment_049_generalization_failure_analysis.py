from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# PROJECT
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


FAULT_TYPES = [
    "solar_array_degradation",
    "battery_degradation",
    "thermal_anomaly",
    "reaction_wheel_degradation",
    "battery_voltage_sensor_drift",
    "telemetry_dropout",
]


ALL_LABELS = [
    "nominal",
    *FAULT_TYPES,
]


# ==========================================================
# HELPERS
# ==========================================================

def load_csv(
    filename,
):

    path = (
        tables_dir
        / filename
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Required Experiment-048 table missing:\n{path}"
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
            ]
        )
        .to_numpy()
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD EXPERIMENT 048
    # ======================================================

    diagnosis = load_csv(
        "experiment_048_diagnosis_predictions.csv"
    )

    detector = load_csv(
        "experiment_048_detector_predictions.csv"
    )

    scenarios = load_csv(
        "experiment_048_scenario_metrics.csv"
    )

    # ======================================================
    # NORMALIZE BOOLEAN COLUMNS
    # ======================================================

    diagnosis[
        "accepted_bool"
    ] = as_bool(
        diagnosis[
            "accepted"
        ]
    )

    diagnosis[
        "raw_correct_bool"
    ] = as_bool(
        diagnosis[
            "raw_correct"
        ]
    )

    diagnosis[
        "operational_correct_bool"
    ] = as_bool(
        diagnosis[
            "operational_correct"
        ]
    )

    detector[
        "fault_active_bool"
    ] = as_bool(
        detector[
            "fault_active"
        ]
    )

    detector[
        "hybrid_detected_bool"
    ] = as_bool(
        detector[
            "hybrid_detected"
        ]
    )

    scenarios[
        "scenario_success_bool"
    ] = as_bool(
        scenarios[
            "scenario_correctly_diagnosed"
        ]
    )

    # ======================================================
    # RAW PHASE-7 CONFUSION MATRIX
    # ======================================================

    observed_predictions = list(
        pd.unique(
            diagnosis[
                "raw_prediction"
            ]
            .astype(str)
        )
    )

    prediction_labels = [

        label

        for label in ALL_LABELS

        if label
        in
        observed_predictions

    ] + [

        label

        for label in observed_predictions

        if label
        not in
        ALL_LABELS
    ]

    diagnosis_cm = (
        pd.crosstab(
            diagnosis[
                "fault_type"
            ],
            diagnosis[
                "raw_prediction"
            ],
        )
        .reindex(
            index=FAULT_TYPES,
            columns=prediction_labels,
            fill_value=0,
        )
    )

    # ======================================================
    # PHASE-6 HYBRID ACTIVE-FAULT CONFUSION
    # ======================================================

    active_detector = (
        detector[
            detector[
                "fault_active_bool"
            ]
        ]
        .copy()
    )

    hybrid_predictions = list(
        pd.unique(
            active_detector[
                "hybrid_label"
            ]
            .astype(str)
        )
    )

    hybrid_labels = [

        label

        for label in ALL_LABELS

        if label
        in
        hybrid_predictions

    ] + [

        label

        for label in hybrid_predictions

        if label
        not in
        ALL_LABELS
    ]

    hybrid_cm = (
        pd.crosstab(
            active_detector[
                "fault_type"
            ],
            active_detector[
                "hybrid_label"
            ],
        )
        .reindex(
            index=FAULT_TYPES,
            columns=hybrid_labels,
            fill_value=0,
        )
    )

    # ======================================================
    # PER-FAULT ERROR DIAGNOSIS
    # ======================================================

    fault_rows = []

    for fault_type in FAULT_TYPES:

        subset = (
            diagnosis[
                diagnosis[
                    "fault_type"
                ]
                ==
                fault_type
            ]
            .copy()
        )

        predictions = (
            subset[
                "raw_prediction"
            ]
            .astype(str)
        )

        counts = (
            predictions
            .value_counts()
        )

        dominant_prediction = (
            counts.index[0]
        )

        dominant_fraction = float(
            counts.iloc[0]
            /
            len(
                subset
            )
        )

        correct = (
            subset[
                "raw_correct_bool"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        accepted = (
            subset[
                "accepted_bool"
            ]
            .to_numpy(
                dtype=bool
            )
        )

        confidence = (
            subset[
                "confidence"
            ]
            .to_numpy(
                dtype=float
            )
        )

        nominal_fraction = float(
            (
                predictions
                ==
                "nominal"
            ).mean()
        )

        accepted_correct_fraction = float(
            (
                accepted
                &
                correct
            ).mean()
        )

        accepted_wrong_fraction = float(
            (
                accepted
                &
                ~correct
            ).mean()
        )

        if correct.any():

            mean_conf_correct = float(
                confidence[
                    correct
                ].mean()
            )

        else:

            mean_conf_correct = np.nan

        if (
            ~correct
        ).any():

            mean_conf_wrong = float(
                confidence[
                    ~correct
                ].mean()
            )

        else:

            mean_conf_wrong = np.nan

        detector_subset = (
            active_detector[
                active_detector[
                    "fault_type"
                ]
                ==
                fault_type
            ]
        )

        hybrid_nominal_fraction = float(
            (
                detector_subset[
                    "hybrid_label"
                ]
                ==
                "nominal"
            ).mean()
        )

        hybrid_alarm_fraction = float(
            detector_subset[
                "hybrid_detected_bool"
            ].mean()
        )

        fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "samples":
                    len(
                        subset
                    ),

                "raw_accuracy":
                    float(
                        correct.mean()
                    ),

                "dominant_raw_prediction":
                    dominant_prediction,

                "dominant_prediction_fraction":
                    dominant_fraction,

                "raw_nominal_prediction_fraction":
                    nominal_fraction,

                "accepted_correct_fraction":
                    accepted_correct_fraction,

                "accepted_wrong_fraction":
                    accepted_wrong_fraction,

                "mean_confidence_when_correct":
                    mean_conf_correct,

                "mean_confidence_when_wrong":
                    mean_conf_wrong,

                "hybrid_alarm_fraction":
                    hybrid_alarm_fraction,

                "hybrid_nominal_fraction":
                    hybrid_nominal_fraction,
            }
        )

    failure_summary = pd.DataFrame(
        fault_rows
    )

    # ======================================================
    # SEVERITY ANALYSIS BY FAULT
    # ======================================================

    scenarios[
        "severity_bin"
    ] = pd.cut(
        scenarios[
            "severity"
        ],
        bins=[
            0.20,
            0.40,
            0.60,
            0.800001,
        ],
        labels=[
            "low_0.20_0.40",
            "medium_0.40_0.60",
            "high_0.60_0.80",
        ],
        include_lowest=True,
    )

    severity_by_fault = (
        scenarios
        .groupby(
            [
                "fault_type",
                "severity_bin",
            ],
            observed=False,
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            hybrid_detection_recall=(
                "hybrid_detection_recall",
                "mean",
            ),

            raw_diagnosis_accuracy=(
                "raw_diagnosis_accuracy",
                "mean",
            ),

            confidence_coverage=(
                "selective_coverage",
                "mean",
            ),

            end_to_end_correct_fraction=(
                "end_to_end_correct_fraction",
                "mean",
            ),

            scenario_success_rate=(
                "scenario_success_bool",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # PROFILE ANALYSIS BY FAULT
    # ======================================================

    profile_by_fault = (
        scenarios
        .groupby(
            [
                "fault_type",
                "profile",
            ]
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            hybrid_detection_recall=(
                "hybrid_detection_recall",
                "mean",
            ),

            raw_diagnosis_accuracy=(
                "raw_diagnosis_accuracy",
                "mean",
            ),

            end_to_end_correct_fraction=(
                "end_to_end_correct_fraction",
                "mean",
            ),

            scenario_success_rate=(
                "scenario_success_bool",
                "mean",
            ),

            mean_delay_min=(
                "diagnosis_delay_min",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # CRITICAL FAILURE SCENARIOS
    #
    # Focus on the two classes that collapsed in Exp. 048.
    # ======================================================

    critical = (
        scenarios[
            scenarios[
                "fault_type"
            ]
            .isin(
                [
                    "solar_array_degradation",
                    "reaction_wheel_degradation",
                ]
            )
        ]
        .copy()
    )

    critical = (
        critical[
            [
                "scenario_id",
                "fault_type",
                "severity",
                "profile",
                "noise_scale",
                "fault_start_time_h",
                "hybrid_detection_recall",
                "raw_diagnosis_accuracy",
                "selective_coverage",
                "end_to_end_correct_fraction",
                "scenario_success_bool",
            ]
        ]
        .sort_values(
            [
                "fault_type",
                "severity",
            ]
        )
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    diagnosis_cm_file = (
        tables_dir
        / "experiment_049_diagnosis_confusion_matrix.csv"
    )

    hybrid_cm_file = (
        tables_dir
        / "experiment_049_hybrid_confusion_matrix.csv"
    )

    failure_summary_file = (
        tables_dir
        / "experiment_049_failure_summary.csv"
    )

    severity_file = (
        tables_dir
        / "experiment_049_per_fault_severity.csv"
    )

    profile_file = (
        tables_dir
        / "experiment_049_per_fault_profile.csv"
    )

    critical_file = (
        tables_dir
        / "experiment_049_critical_failure_scenarios.csv"
    )

    diagnosis_cm.to_csv(
        diagnosis_cm_file
    )

    hybrid_cm.to_csv(
        hybrid_cm_file
    )

    failure_summary.to_csv(
        failure_summary_file,
        index=False,
    )

    severity_by_fault.to_csv(
        severity_file,
        index=False,
    )

    profile_by_fault.to_csv(
        profile_file,
        index=False,
    )

    critical.to_csv(
        critical_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PHASE-7 CONFUSION MATRIX
    # ======================================================

    confusion_figure = (
        figures_dir
        / "experiment_049_phase7_confusion_matrix.png"
    )

    matrix = (
        diagnosis_cm
        .to_numpy()
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.imshow(
        matrix,
        aspect="auto",
    )

    plt.xticks(
        np.arange(
            len(
                diagnosis_cm.columns
            )
        ),
        diagnosis_cm.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(
            len(
                diagnosis_cm.index
            )
        ),
        diagnosis_cm.index,
    )

    plt.xlabel(
        "Predicted Diagnosis"
    )

    plt.ylabel(
        "True Fault"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo Phase-7 Confusion Matrix"
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            plt.text(
                column,
                row,
                str(
                    int(
                        matrix[
                            row,
                            column
                        ]
                    )
                ),
                ha="center",
                va="center",
            )

    plt.colorbar(
        label="Samples"
    )

    plt.tight_layout()

    plt.savefig(
        confusion_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — NOMINAL COLLAPSE vs CORRECT CLASSIFICATION
    # ======================================================

    classification_figure = (
        figures_dir
        / "experiment_049_failure_modes.png"
    )

    x = np.arange(
        len(
            failure_summary
        )
    )

    width = 0.35

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        failure_summary[
            "raw_accuracy"
        ],
        width=width,
        label="Correct diagnosis fraction",
    )

    plt.bar(
        x + width / 2,
        failure_summary[
            "raw_nominal_prediction_fraction"
        ],
        width=width,
        label="Predicted nominal fraction",
    )

    plt.xticks(
        x,
        failure_summary[
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
        "Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo Diagnosis Failure Modes"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        classification_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — SOLAR + WHEEL SEVERITY RESPONSE
    # ======================================================

    critical_figure = (
        figures_dir
        / "experiment_049_critical_fault_severity.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    for fault_type in [
        "solar_array_degradation",
        "reaction_wheel_degradation",
    ]:

        subset = (
            scenarios[
                scenarios[
                    "fault_type"
                ]
                ==
                fault_type
            ]
            .sort_values(
                "severity"
            )
        )

        plt.scatter(
            subset[
                "severity"
            ],
            subset[
                "hybrid_detection_recall"
            ],
            label=(
                fault_type
                +
                " detection"
            ),
        )

        plt.scatter(
            subset[
                "severity"
            ],
            subset[
                "raw_diagnosis_accuracy"
            ],
            marker="x",
            label=(
                fault_type
                +
                " diagnosis"
            ),
        )

    plt.xlabel(
        "Fault Severity"
    )

    plt.ylabel(
        "Recall / Accuracy"
    )

    plt.ylim(
        -0.02,
        1.05,
    )

    plt.title(
        "SENTINEL-XAI - "
        "Critical Generalization Failures vs Severity"
    )

    plt.grid(True)
    plt.legend(
        fontsize=8
    )
    plt.tight_layout()

    plt.savefig(
        critical_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # STRUCTURAL CHECKS
    # ======================================================

    checks = {

        "All six faults present in diagnosis analysis":
            (
                set(
                    diagnosis[
                        "fault_type"
                    ]
                )
                ==
                set(
                    FAULT_TYPES
                )
            ),

        "Diagnosis confusion matrix accounts for all samples":
            (
                int(
                    diagnosis_cm
                    .to_numpy()
                    .sum()
                )
                ==
                len(
                    diagnosis
                )
            ),

        "All 120 scenarios analyzed":
            (
                len(
                    scenarios
                )
                ==
                120
            ),

        "Both critical failure classes included":
            (
                set(
                    critical[
                        "fault_type"
                    ]
                )
                ==
                {
                    "solar_array_degradation",
                    "reaction_wheel_degradation",
                }
            ),
    }

    # ======================================================
    # TERMINAL
    # ======================================================

    print()
    print("=" * 100)

    print(
        "SENTINEL-XAI - EXPERIMENT 049"
    )

    print(
        "MONTE CARLO GENERALIZATION FAILURE ANALYSIS"
    )

    print("=" * 100)

    print()

    print(
        "This experiment performs analysis only."
    )

    print(
        "Models retrained: NO"
    )

    print(
        "Thresholds changed: NO"
    )

    print()

    print("-" * 100)

    print(
        "PER-FAULT FAILURE SUMMARY"
    )

    print("-" * 100)

    for _, row in (
        failure_summary.iterrows()
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
            f"  Dominant prediction: "
            f"{row['dominant_raw_prediction']}"
        )

        print(
            f"  Dominant prediction fraction: "
            f"{row['dominant_prediction_fraction']:.4f}"
        )

        print(
            f"  Predicted nominal fraction: "
            f"{row['raw_nominal_prediction_fraction']:.4f}"
        )

        print(
            f"  Accepted correct fraction: "
            f"{row['accepted_correct_fraction']:.4f}"
        )

        print(
            f"  Accepted wrong fraction: "
            f"{row['accepted_wrong_fraction']:.4f}"
        )

        print(
            f"  Mean confidence when correct: "
            f"{row['mean_confidence_when_correct']:.4f}"
        )

        print(
            f"  Mean confidence when wrong: "
            f"{row['mean_confidence_when_wrong']:.4f}"
        )

        print(
            f"  Hybrid alarm fraction: "
            f"{row['hybrid_alarm_fraction']:.4f}"
        )

        print(
            f"  Hybrid predicted nominal fraction: "
            f"{row['hybrid_nominal_fraction']:.4f}"
        )

    print()

    print("-" * 100)

    print(
        "SOLAR / WHEEL SCENARIO SUMMARY"
    )

    print("-" * 100)

    for fault_type in [
        "solar_array_degradation",
        "reaction_wheel_degradation",
    ]:

        subset = (
            scenarios[
                scenarios[
                    "fault_type"
                ]
                ==
                fault_type
            ]
        )

        print()

        print(
            fault_type
        )

        print(
            f"  Scenarios: "
            f"{len(subset)}"
        )

        print(
            f"  Mean severity: "
            f"{subset['severity'].mean():.4f}"
        )

        print(
            f"  Mean hybrid recall: "
            f"{subset['hybrid_detection_recall'].mean():.4f}"
        )

        print(
            f"  Mean raw diagnosis accuracy: "
            f"{subset['raw_diagnosis_accuracy'].mean():.4f}"
        )

        print(
            f"  Successful scenarios: "
            f"{int(subset['scenario_success_bool'].sum())}"
            f"/{len(subset)}"
        )

    print()

    print("-" * 100)

    print(
        "STRUCTURAL CHECKS"
    )

    print("-" * 100)

    passed = 0

    for name, result in checks.items():

        print()

        print(
            f"{name}: "
            f"{bool(result)}"
        )

        if result:
            passed += 1

    print()

    print(
        f"Passed checks: "
        f"{passed}/{len(checks)}"
    )

    print()

    print(
        "RESULT: GENERALIZATION FAILURE "
        "ANALYSIS COMPLETE"
    )

    print()

    print(
        "No performance pass/fail criterion is "
        "used here because these data are the "
        "independent Monte Carlo test set."
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        diagnosis_cm_file
    )

    print(
        hybrid_cm_file
    )

    print(
        failure_summary_file
    )

    print(
        severity_file
    )

    print(
        profile_file
    )

    print(
        critical_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        confusion_figure
    )

    print(
        classification_figure
    )

    print(
        critical_figure
    )

    print("=" * 100)


if __name__ == "__main__":
    main()