from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


# ==========================================================
# HELPERS
# ==========================================================

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


# ==========================================================
# MAIN
# ==========================================================

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
    # LOAD RANDOM-FOREST TEST PREDICTIONS
    # ======================================================

    rf = load_csv(
        tables_dir
        / "experiment_037_diagnosis_predictions.csv"
    )

    # Avoid floating-point merge problems.
    rf["time_key"] = (
        rf["time_h"]
        .round(6)
    )

    # ======================================================
    # LOAD PHASE-6 FAULT-SCENARIO PREDICTIONS
    # ======================================================

    hybrid_fault = load_csv(
        tables_dir
        / "experiment_034_hybrid_predictions.csv"
    )

    hybrid_fault["time_key"] = (
        hybrid_fault["time_h"]
        .round(6)
    )

    hybrid_fault_reference = (
        hybrid_fault[
            [
                "dataset",
                "time_key",
                "hybrid_label",
                "hybrid_confidence",
            ]
        ]
        .rename(
            columns={
                "dataset":
                    "source_dataset",

                "hybrid_label":
                    "phase6_label",

                "hybrid_confidence":
                    "phase6_confidence",
            }
        )
    )

    # ======================================================
    # LOAD PHASE-6 NOMINAL PREDICTIONS
    # ======================================================

    hybrid_nominal = load_csv(
        tables_dir
        / "experiment_033_hybrid_nominal_validation.csv"
    )

    hybrid_nominal["time_key"] = (
        hybrid_nominal["time_h"]
        .round(6)
    )

    nominal_reference = pd.DataFrame(
        {
            "source_dataset":
                "nominal",

            "time_key":
                hybrid_nominal[
                    "time_key"
                ].values,

            "phase6_label":
                hybrid_nominal[
                    "hybrid_fault"
                ].values,

            "phase6_confidence":
                hybrid_nominal[
                    "hybrid_confidence"
                ].values,
        }
    )

    # ======================================================
    # COMPLETE PHASE-6 REFERENCE
    # ======================================================

    hybrid_reference = pd.concat(
        [
            hybrid_fault_reference,
            nominal_reference,
        ],
        ignore_index=True,
    )

    # ======================================================
    # MERGE EXACT TEST SAMPLES
    # ======================================================

    comparison = rf.merge(
        hybrid_reference,
        on=[
            "source_dataset",
            "time_key",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_reference = int(
        comparison[
            "phase6_label"
        ]
        .isna()
        .sum()
    )

    if missing_reference > 0:

        raise RuntimeError(
            f"{missing_reference} Phase-6 predictions "
            f"could not be matched."
        )

    comparison[
        "phase6_correct"
    ] = (
        comparison[
            "phase6_label"
        ]
        ==
        comparison[
            "true_label"
        ]
    )

    comparison[
        "phase7_correct"
    ] = (
        comparison[
            "predicted_label"
        ]
        ==
        comparison[
            "true_label"
        ]
    )

    # ======================================================
    # GLOBAL ACCURACY
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

    # ======================================================
    # CLASSES
    # ======================================================

    classes = [

        "nominal",

        "solar_array_degradation",

        "battery_degradation",

        "thermal_anomaly",

        "reaction_wheel_degradation",

        "battery_voltage_sensor_drift",

        "telemetry_dropout",
    ]

    # ======================================================
    # PER-CLASS COMPARISON
    # ======================================================

    per_class_rows = []

    for class_name in classes:

        subset = comparison[
            comparison[
                "true_label"
            ]
            ==
            class_name
        ]

        phase6_recall = safe_divide(
            (
                subset[
                    "phase6_label"
                ]
                ==
                class_name
            ).sum(),
            len(subset),
        )

        phase7_recall = safe_divide(
            (
                subset[
                    "predicted_label"
                ]
                ==
                class_name
            ).sum(),
            len(subset),
        )

        per_class_rows.append(
            {
                "class":
                    class_name,

                "samples":
                    len(subset),

                "phase6_recall":
                    phase6_recall,

                "phase7_recall":
                    phase7_recall,

                "improvement":
                    phase7_recall
                    -
                    phase6_recall,
            }
        )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    # ======================================================
    # CONFIDENCE ANALYSIS
    # ======================================================

    correct_rows = comparison[
        comparison[
            "phase7_correct"
        ]
    ]

    incorrect_rows = comparison[
        ~comparison[
            "phase7_correct"
        ]
    ]

    mean_correct_confidence = float(
        correct_rows[
            "confidence"
        ].mean()
    )

    if len(
        incorrect_rows
    ) > 0:

        mean_incorrect_confidence = float(
            incorrect_rows[
                "confidence"
            ].mean()
        )

        max_incorrect_confidence = float(
            incorrect_rows[
                "confidence"
            ].max()
        )

    else:

        mean_incorrect_confidence = (
            float("nan")
        )

        max_incorrect_confidence = (
            float("nan")
        )

    # ======================================================
    # CONFIDENCE BINS
    # ======================================================

    bins = [
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    labels = [
        "<0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    comparison[
        "confidence_bin"
    ] = pd.cut(
        comparison[
            "confidence"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    confidence_rows = []

    for label in labels:

        subset = comparison[
            comparison[
                "confidence_bin"
            ]
            ==
            label
        ]

        if len(
            subset
        ) == 0:

            continue

        confidence_rows.append(
            {
                "confidence_bin":
                    label,

                "samples":
                    len(subset),

                "accuracy":
                    float(
                        subset[
                            "phase7_correct"
                        ].mean()
                    ),
            }
        )

    confidence_df = pd.DataFrame(
        confidence_rows
    )

    # ======================================================
    # DISAGREEMENTS
    # ======================================================

    disagreements = comparison[
        comparison[
            "phase6_label"
        ]
        !=
        comparison[
            "predicted_label"
        ]
    ].copy()

    both_correct = int(
        (
            comparison[
                "phase6_correct"
            ]
            &
            comparison[
                "phase7_correct"
            ]
        ).sum()
    )

    phase7_only_correct = int(
        (
            ~comparison[
                "phase6_correct"
            ]
            &
            comparison[
                "phase7_correct"
            ]
        ).sum()
    )

    phase6_only_correct = int(
        (
            comparison[
                "phase6_correct"
            ]
            &
            ~comparison[
                "phase7_correct"
            ]
        ).sum()
    )

    both_wrong = int(
        (
            ~comparison[
                "phase6_correct"
            ]
            &
            ~comparison[
                "phase7_correct"
            ]
        ).sum()
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    comparison_file = (
        tables_dir
        / "experiment_038_phase6_vs_phase7_predictions.csv"
    )

    per_class_file = (
        tables_dir
        / "experiment_038_per_class_comparison.csv"
    )

    confidence_file = (
        tables_dir
        / "experiment_038_confidence_analysis.csv"
    )

    disagreement_file = (
        tables_dir
        / "experiment_038_diagnosis_disagreements.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    per_class_df.to_csv(
        per_class_file,
        index=False,
    )

    confidence_df.to_csv(
        confidence_file,
        index=False,
    )

    disagreements.to_csv(
        disagreement_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PER-CLASS COMPARISON
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_038_phase6_vs_phase7_recall.png"
    )

    x = np.arange(
        len(
            per_class_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        x - width / 2,
        per_class_df[
            "phase6_recall"
        ],
        width=width,
        label="Phase 6 Hybrid",
    )

    plt.bar(
        x + width / 2,
        per_class_df[
            "phase7_recall"
        ],
        width=width,
        label="Phase 7 Random Forest",
    )

    plt.xticks(
        x,
        per_class_df[
            "class"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Classification Recall"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Phase 6 vs Phase 7 Diagnosis"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        recall_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — CONFIDENCE VS ACCURACY
    # ======================================================

    confidence_figure = (
        figures_dir
        / "experiment_038_confidence_vs_accuracy.png"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        confidence_df[
            "confidence_bin"
        ],
        confidence_df[
            "accuracy"
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Random Forest Confidence"
    )

    plt.ylabel(
        "Observed Accuracy"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Diagnosis Confidence vs Accuracy"
    )

    plt.grid(True)
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
    print("=" * 86)

    print(
        "SENTINEL-XAI - EXPERIMENT 038"
    )

    print(
        "PHASE 6 HYBRID vs PHASE 7 DIAGNOSIS"
    )

    print("=" * 86)

    print()

    print(
        f"Matched test samples: "
        f"{len(comparison)}"
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
        f"{phase7_accuracy - phase6_accuracy:+.4f}"
    )

    print()

    print("-" * 86)

    print(
        "PER-CLASS RECALL"
    )

    print("-" * 86)

    for _, row in (
        per_class_df.iterrows()
    ):

        print()

        print(
            row["class"]
        )

        print(
            f"  Phase 6: "
            f"{row['phase6_recall']:.4f}"
        )

        print(
            f"  Phase 7: "
            f"{row['phase7_recall']:.4f}"
        )

        print(
            f"  Change:  "
            f"{row['improvement']:+.4f}"
        )

    print()

    print("-" * 86)

    print(
        "DECISION COMPARISON"
    )

    print("-" * 86)

    print()

    print(
        f"Both correct: "
        f"{both_correct}"
    )

    print(
        f"Phase 7 only correct: "
        f"{phase7_only_correct}"
    )

    print(
        f"Phase 6 only correct: "
        f"{phase6_only_correct}"
    )

    print(
        f"Both wrong: "
        f"{both_wrong}"
    )

    print()

    print(
        f"Phase 6 / Phase 7 disagreements: "
        f"{len(disagreements)}"
    )

    print()

    print("-" * 86)

    print(
        "CONFIDENCE ANALYSIS"
    )

    print("-" * 86)

    print()

    print(
        f"Mean confidence when correct: "
        f"{mean_correct_confidence:.4f}"
    )

    print(
        f"Mean confidence when wrong: "
        f"{mean_incorrect_confidence:.4f}"
    )

    print(
        f"Maximum confidence among errors: "
        f"{max_incorrect_confidence:.4f}"
    )

    print()

    print(
        confidence_df.to_string(
            index=False
        )
    )

    print()

    print("=" * 86)

    print(
        "Tables saved to:"
    )

    print(
        comparison_file
    )

    print(
        per_class_file
    )

    print(
        confidence_file
    )

    print(
        disagreement_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        recall_figure
    )

    print(
        confidence_figure
    )

    print("=" * 86)


if __name__ == "__main__":
    main()