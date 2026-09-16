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


from src.ml.feature_pipeline import (
    build_ml_features,
    feature_matrix,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


# ==========================================================
# HELPERS
# ==========================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_csv(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def calculate_binary_metrics(
    df,
):

    tp = int(
        (
            (df["true_fault"] == 1)
            &
            (df["predicted_fault"] == 1)
        ).sum()
    )

    tn = int(
        (
            (df["true_fault"] == 0)
            &
            (df["predicted_fault"] == 0)
        ).sum()
    )

    fp = int(
        (
            (df["true_fault"] == 0)
            &
            (df["predicted_fault"] == 1)
        ).sum()
    )

    fn = int(
        (
            (df["true_fault"] == 1)
            &
            (df["predicted_fault"] == 0)
        ).sum()
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "false_positive_rate":
            false_positive_rate,
    }


# ==========================================================
# EVALUATE ONE DATASET
# ==========================================================

def evaluate_dataset(
    dataset_name,
    df,
    detector,
):

    features = build_ml_features(
        df
    )

    x = feature_matrix(
        features
    )

    # ======================================================
    # INVALID / MISSING TELEMETRY
    #
    # Isolation Forest should only score valid numerical
    # telemetry. Missing required inputs are directly
    # considered anomalous observations.
    # ======================================================

    missing_input = (
        x.isna()
        .any(axis=1)
        .to_numpy()
    )

    valid_input = (
        ~missing_input
    )

    scores = np.full(
        len(x),
        np.nan,
        dtype=float,
    )

    if valid_input.any():

        scores[
            valid_input
        ] = detector.anomaly_score(
            x.loc[
                valid_input
            ]
        )

    raw_anomalies = np.zeros(
        len(x),
        dtype=bool,
    )

    raw_anomalies[
        valid_input
    ] = (
        scores[
            valid_input
        ]
        >
        detector.threshold
    )

    # Missing required telemetry is anomalous.
    raw_anomalies[
        missing_input
    ] = True

    # ======================================================
    # 3-SAMPLE PERSISTENCE
    # ======================================================

    persistence_filter = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent_anomalies = []
    persistence_counts = []

    for raw_anomaly in raw_anomalies:

        result = (
            persistence_filter.update(
                bool(
                    raw_anomaly
                )
            )
        )

        persistent_anomalies.append(
            result.persistent_anomaly
        )

        persistence_counts.append(
            result.consecutive_count
        )

    persistent_anomalies = np.array(
        persistent_anomalies,
        dtype=bool,
    )

    # ======================================================
    # GROUND TRUTH
    # ======================================================

    if "fault_label" in df.columns:

        true_labels = (
            df["fault_label"]
            .fillna("nominal")
            .astype(str)
        )

    else:

        true_labels = pd.Series(
            ["nominal"] * len(df),
            index=df.index,
        )

    true_fault = (
        true_labels
        != "nominal"
    ).astype(int)

    # ======================================================
    # RESULT TABLE
    # ======================================================

    return pd.DataFrame(
        {
            "dataset":
                dataset_name,

            "time_h":
                df["time_h"].values,

            "true_label":
                true_labels.values,

            "true_fault":
                true_fault.values,

            "anomaly_score":
                scores,

            "threshold":
                detector.threshold,

            "missing_ml_input":
                missing_input,

            "raw_anomaly":
                raw_anomalies,

            "persistence_count":
                persistence_counts,

            "persistent_anomaly":
                persistent_anomalies,

            "predicted_fault":
                persistent_anomalies.astype(
                    int
                ),
        }
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD FIXED MODEL FROM EXPERIMENT 025
    # ======================================================

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_025_isolation_forest.joblib"
    )

    detector = joblib.load(
        model_file
    )

    # ======================================================
    # DATASETS
    # ======================================================

    files = {

        "nominal":
            project_root
            / "data"
            / "nominal"
            / "nominal_spacecraft_telemetry.csv",

        "solar_array_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_006_solar_degradation.csv",

        "battery_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_007_battery_degradation.csv",

        "thermal_anomaly":
            project_root
            / "data"
            / "faults"
            / "experiment_008_thermal_anomaly.csv",

        "reaction_wheel_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_009_reaction_wheel_degradation.csv",

        "battery_voltage_sensor_drift":
            project_root
            / "data"
            / "faults"
            / "experiment_010_sensor_drift.csv",

        "telemetry_dropout":
            project_root
            / "data"
            / "faults"
            / "experiment_011_telemetry_dropout.csv",
    }

    results = []

    print()
    print(
        "Running fixed Isolation Forest "
        "on all spacecraft datasets..."
    )

    for name, path in files.items():

        print(
            f"  Processing {name}..."
        )

        df = load_csv(
            path
        )

        evaluated = evaluate_dataset(
            dataset_name=name,
            df=df,
            detector=detector,
        )

        results.append(
            evaluated
        )

    results_df = pd.concat(
        results,
        ignore_index=True,
    )

    # ======================================================
    # GLOBAL METRICS
    # ======================================================

    phase5 = (
        calculate_binary_metrics(
            results_df
        )
    )

    # ======================================================
    # PER-FAULT RECALL
    # ======================================================

    fault_types = [
        "solar_array_degradation",
        "battery_degradation",
        "thermal_anomaly",
        "reaction_wheel_degradation",
        "battery_voltage_sensor_drift",
        "telemetry_dropout",
    ]

    per_fault_rows = []

    for fault_type in fault_types:

        subset = results_df[
            results_df[
                "true_label"
            ]
            ==
            fault_type
        ]

        raw_detected = int(
            subset[
                "raw_anomaly"
            ].sum()
        )

        persistent_detected = int(
            subset[
                "persistent_anomaly"
            ].sum()
        )

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "fault_samples":
                    len(subset),

                "raw_recall":
                    safe_divide(
                        raw_detected,
                        len(subset),
                    ),

                "persistent_recall":
                    safe_divide(
                        persistent_detected,
                        len(subset),
                    ),
            }
        )

    per_fault_df = pd.DataFrame(
        per_fault_rows
    )

    # ======================================================
    # DETECTION DELAYS
    # ======================================================

    fault_start_times = {

        "solar_array_degradation":
            2.0,

        "battery_degradation":
            2.0,

        "thermal_anomaly":
            2.0,

        "reaction_wheel_degradation":
            2.0,

        "battery_voltage_sensor_drift":
            2.0,

        "telemetry_dropout":
            3.0,
    }

    delay_rows = []

    for fault_type, start_h in fault_start_times.items():

        subset = results_df[
            results_df[
                "dataset"
            ]
            ==
            fault_type
        ]

        detected_rows = subset[
            (
                subset["time_h"]
                >= start_h
            )
            &
            (
                subset[
                    "persistent_anomaly"
                ]
            )
        ]

        if len(
            detected_rows
        ) > 0:

            first_detection_h = float(
                detected_rows[
                    "time_h"
                ].iloc[0]
            )

            delay_min = (
                first_detection_h
                - start_h
            ) * 60.0

            detected = True

        else:

            first_detection_h = (
                float("nan")
            )

            delay_min = (
                float("nan")
            )

            detected = False

        delay_rows.append(
            {
                "fault_type":
                    fault_type,

                "fault_start_h":
                    start_h,

                "first_detection_h":
                    first_detection_h,

                "detection_delay_min":
                    delay_min,

                "detected":
                    detected,
            }
        )

    delay_df = pd.DataFrame(
        delay_rows
    )

    detected_delays = delay_df[
        delay_df["detected"]
        == True
    ]

    if len(
        detected_delays
    ) > 0:

        phase5_mean_delay = float(
            detected_delays[
                "detection_delay_min"
            ].mean()
        )

    else:

        phase5_mean_delay = (
            float("nan")
        )

    # ======================================================
    # LOAD PHASE 3 + PHASE 4 FOR FAIR BINARY COMPARISON
    # ======================================================

    phase3_predictions = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_013_rule_based_predictions.csv"
    )

    phase4_predictions = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_023_refined_predictions.csv"
    )

    phase3 = (
        calculate_binary_metrics(
            phase3_predictions
        )
    )

    phase4 = (
        calculate_binary_metrics(
            phase4_predictions
        )
    )

    phase3_delay_file = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_014_detection_delay.csv"
    )

    phase4_delay_file = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_023_detection_delays.csv"
    )

    phase3_mean_delay = float(
        phase3_delay_file.loc[
            phase3_delay_file[
                "detected"
            ]
            == True,
            "detection_delay_min",
        ].mean()
    )

    phase4_mean_delay = float(
        phase4_delay_file.loc[
            phase4_delay_file[
                "detected"
            ]
            == True,
            "detection_delay_min",
        ].mean()
    )

    # ======================================================
    # COMPARISON TABLE
    # ======================================================

    comparison_df = pd.DataFrame(
        [
            {
                "method":
                    "Phase 3 Rule-Based",

                "precision":
                    phase3[
                        "precision"
                    ],

                "recall":
                    phase3[
                        "recall"
                    ],

                "f1":
                    phase3[
                        "f1"
                    ],

                "false_positive_rate":
                    phase3[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    phase3_mean_delay,
            },

            {
                "method":
                    "Phase 4 Model-Based",

                "precision":
                    phase4[
                        "precision"
                    ],

                "recall":
                    phase4[
                        "recall"
                    ],

                "f1":
                    phase4[
                        "f1"
                    ],

                "false_positive_rate":
                    phase4[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    phase4_mean_delay,
            },

            {
                "method":
                    "Phase 5 Isolation Forest",

                "precision":
                    phase5[
                        "precision"
                    ],

                "recall":
                    phase5[
                        "recall"
                    ],

                "f1":
                    phase5[
                        "f1"
                    ],

                "false_positive_rate":
                    phase5[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    phase5_mean_delay,
            },
        ]
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
        / "experiment_027_ml_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_027_ml_per_fault_metrics.csv"
    )

    delay_file = (
        tables_dir
        / "experiment_027_ml_detection_delays.csv"
    )

    comparison_file = (
        tables_dir
        / "experiment_027_phase3_phase4_phase5.csv"
    )

    results_df.to_csv(
        predictions_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    delay_df.to_csv(
        delay_file,
        index=False,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — ML PER-FAULT RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_027_ml_per_fault_recall.png"
    )

    x = np.arange(
        len(
            per_fault_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        x - width / 2,
        per_fault_df[
            "raw_recall"
        ],
        width=width,
        label="Raw ML anomaly",
    )

    plt.bar(
        x + width / 2,
        per_fault_df[
            "persistent_recall"
        ],
        width=width,
        label="3-sample persistent alarm",
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
        "Fault Recall"
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.title(
        "SENTINEL-XAI - Isolation Forest Fault Recall"
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
    # FIGURE 2 — DETECTION DELAY
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_027_ml_detection_delays.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        delay_df[
            "fault_type"
        ],
        delay_df[
            "detection_delay_min"
        ],
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.ylabel(
        "Detection Delay [minutes]"
    )

    plt.title(
        "SENTINEL-XAI - Isolation Forest Detection Delay"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        delay_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — PHASE 3 / 4 / 5 COMPARISON
    # ======================================================

    comparison_figure = (
        figures_dir
        / "experiment_027_detection_method_comparison.png"
    )

    metric_names = [
        "precision",
        "recall",
        "f1",
    ]

    x = np.arange(
        len(
            metric_names
        )
    )

    width = 0.25

    plt.figure(
        figsize=(10, 5)
    )

    for index, row in comparison_df.iterrows():

        values = [
            row[
                metric
            ]
            for metric
            in metric_names
        ]

        offset = (
            index - 1
        ) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=row[
                "method"
            ],
        )

    plt.xticks(
        x,
        [
            "Precision",
            "Recall",
            "F1",
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - Rules vs Physics vs ML"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        comparison_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 82)

    print(
        "SENTINEL-XAI - EXPERIMENT 027"
    )

    print(
        "ISOLATION FOREST ALL-FAULT EVALUATION"
    )

    print("=" * 82)

    print()

    print(
        "Fixed anomaly threshold: "
        f"{detector.threshold:.6f}"
    )

    print(
        "Persistence requirement: "
        "3 consecutive samples"
    )

    print()

    print("-" * 82)
    print(
        "PHASE 5 GLOBAL BINARY METRICS"
    )
    print("-" * 82)

    print(
        f"True positives:  "
        f"{phase5['tp']}"
    )

    print(
        f"True negatives:  "
        f"{phase5['tn']}"
    )

    print(
        f"False positives: "
        f"{phase5['fp']}"
    )

    print(
        f"False negatives: "
        f"{phase5['fn']}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{phase5['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{phase5['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{phase5['recall']:.4f}"
    )

    print(
        f"F1 score:  "
        f"{phase5['f1']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{phase5['false_positive_rate']:.4f}"
    )

    print()

    print("-" * 82)
    print(
        "PER-FAULT ML RECALL"
    )
    print("-" * 82)

    for _, row in per_fault_df.iterrows():

        print()
        print(
            row[
                "fault_type"
            ]
        )

        print(
            f"  Raw anomaly recall: "
            f"{row['raw_recall']:.4f}"
        )

        print(
            f"  Persistent recall:  "
            f"{row['persistent_recall']:.4f}"
        )

    print()

    print("-" * 82)
    print(
        "DETECTION DELAYS"
    )
    print("-" * 82)

    for _, row in delay_df.iterrows():

        print()
        print(
            row[
                "fault_type"
            ]
        )

        if row[
            "detected"
        ]:

            print(
                f"  Delay: "
                f"{row['detection_delay_min']:.1f} min"
            )

        else:

            print(
                "  Delay: NOT DETECTED"
            )

    print()

    print(
        f"Phase 5 mean detected delay: "
        f"{phase5_mean_delay:.1f} min"
    )

    print()

    print("-" * 82)
    print(
        "PHASE 3 vs PHASE 4 vs PHASE 5"
    )
    print("-" * 82)

    for _, row in comparison_df.iterrows():

        print()
        print(
            row["method"]
        )

        print(
            f"  Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{row['f1']:.4f}"
        )

        print(
            f"  FPR:       "
            f"{row['false_positive_rate']:.4f}"
        )

        print(
            f"  Mean delay: "
            f"{row['mean_detection_delay_min']:.1f} min"
        )

    print()

    print("=" * 82)

    print()
    print(
        "Tables saved to:"
    )

    print(
        predictions_file
    )

    print(
        per_fault_file
    )

    print(
        delay_file
    )

    print(
        comparison_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        recall_figure
    )

    print(
        delay_figure
    )

    print(
        comparison_figure
    )

    print("=" * 82)


if __name__ == "__main__":
    main()          