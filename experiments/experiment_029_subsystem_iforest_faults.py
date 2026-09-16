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
)

from src.ml.subsystem_isolation_forest import (
    SUBSYSTEM_FEATURES,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


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
# SCORE DATASET
# ==========================================================

def evaluate_dataset(
    dataset_name,
    df,
    detector,
):

    features = build_ml_features(
        df
    )

    number_of_samples = len(
        features
    )

    result = pd.DataFrame(
        {
            "dataset":
                dataset_name,

            "time_h":
                df["time_h"].values,
        }
    )

    # ======================================================
    # SCORE EACH SUBSYSTEM
    # ======================================================

    subsystem_anomaly_columns = []

    subsystem_ratio_columns = []

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = features[
            columns
        ]

        # ----------------------------------------------
        # Missing required telemetry
        # ----------------------------------------------

        missing_input = (
            x.isna()
            .any(axis=1)
            .to_numpy()
        )

        valid_input = (
            ~missing_input
        )

        scores = np.full(
            number_of_samples,
            np.nan,
            dtype=float,
        )

        threshold = (
            detector.thresholds[
                subsystem
            ]
        )

        # ----------------------------------------------
        # Score valid rows
        # ----------------------------------------------

        if valid_input.any():

            valid_features = (
                x.loc[
                    valid_input
                ]
            )

            scores[
                valid_input
            ] = (
                -detector.models[
                    subsystem
                ].score_samples(
                    valid_features
                )
            )

        score_ratio = np.full(
            number_of_samples,
            np.nan,
            dtype=float,
        )

        score_ratio[
            valid_input
        ] = (
            scores[
                valid_input
            ]
            /
            threshold
        )

        raw_anomaly = np.zeros(
            number_of_samples,
            dtype=bool,
        )

        raw_anomaly[
            valid_input
        ] = (
            scores[
                valid_input
            ]
            >
            threshold
        )

        # Missing required telemetry is considered
        # anomalous input for that subsystem.
        raw_anomaly[
            missing_input
        ] = True

        # Give missing telemetry an anomaly ratio > 1
        # only for visualization / fusion purposes.
        score_ratio[
            missing_input
        ] = 1.25

        result[
            f"{subsystem}_score"
        ] = scores

        result[
            f"{subsystem}_threshold"
        ] = threshold

        result[
            f"{subsystem}_ratio"
        ] = score_ratio

        result[
            f"{subsystem}_anomaly"
        ] = raw_anomaly

        result[
            f"{subsystem}_missing"
        ] = missing_input

        subsystem_anomaly_columns.append(
            f"{subsystem}_anomaly"
        )

        subsystem_ratio_columns.append(
            f"{subsystem}_ratio"
        )

    # ======================================================
    # GLOBAL ML ANOMALY
    # ======================================================

    result[
        "maximum_score_ratio"
    ] = result[
        subsystem_ratio_columns
    ].max(
        axis=1
    )

    result[
        "raw_anomaly"
    ] = result[
        subsystem_anomaly_columns
    ].any(
        axis=1
    )

    # ======================================================
    # TEMPORAL PERSISTENCE
    # ======================================================

    persistence_filter = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent = []
    persistence_count = []

    for raw_value in result[
        "raw_anomaly"
    ]:

        persistence_result = (
            persistence_filter.update(
                bool(raw_value)
            )
        )

        persistent.append(
            persistence_result
            .persistent_anomaly
        )

        persistence_count.append(
            persistence_result
            .consecutive_count
        )

    result[
        "persistence_count"
    ] = persistence_count

    result[
        "persistent_anomaly"
    ] = persistent

    # ======================================================
    # TRUE LABELS
    # ======================================================

    if "fault_label" in df.columns:

        labels = (
            df["fault_label"]
            .fillna("nominal")
            .astype(str)
        )

    else:

        labels = pd.Series(
            ["nominal"] * len(df),
            index=df.index,
        )

    result[
        "true_label"
    ] = labels.values

    result[
        "true_fault"
    ] = (
        labels.values
        != "nominal"
    ).astype(
        int
    )

    result[
        "predicted_fault"
    ] = (
        result[
            "persistent_anomaly"
        ]
        .astype(int)
    )

    return result


# ==========================================================
# METRICS
# ==========================================================

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
        2.0
        * precision
        * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    false_positive_rate = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positive_rate":
            false_positive_rate,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # FIXED SUBSYSTEM MODEL
    # ======================================================

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_028_subsystem_iforest.joblib"
    )

    detector = joblib.load(
        model_file
    )

    # ======================================================
    # SIX FAULT DATASETS
    # ======================================================

    files = {

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

    all_fault_results = []

    print()
    print(
        "Running subsystem Isolation Forest "
        "on all six fault datasets..."
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

        all_fault_results.append(
            evaluated
        )

    fault_results = pd.concat(
        all_fault_results,
        ignore_index=True,
    )

    # ======================================================
    # INDEPENDENT NOMINAL HOLDOUT
    #
    # Same untouched final 20% of 24h nominal telemetry
    # used in Experiment 028.
    # ======================================================

    nominal_24h_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry_24h.csv"
    )

    nominal_24h = load_csv(
        nominal_24h_file
    )

    nominal_features = build_ml_features(
        nominal_24h
    )

    n_nominal = len(
        nominal_features
    )

    nominal_test_start = int(
        0.80
        * n_nominal
    )

    nominal_test_df = (
        nominal_24h
        .iloc[
            nominal_test_start:
        ]
        .copy()
    )

    nominal_results = evaluate_dataset(
        dataset_name=(
            "nominal_holdout"
        ),
        df=nominal_test_df,
        detector=detector,
    )

    # ======================================================
    # FAULT-ACTIVE SAMPLES ONLY
    #
    # Positive class.
    # ======================================================

    active_fault_results = (
        fault_results[
            fault_results[
                "true_fault"
            ]
            == 1
        ]
        .copy()
    )

    # ======================================================
    # GLOBAL HOLDOUT EVALUATION
    #
    # Negatives:
    #   untouched 24h nominal holdout
    #
    # Positives:
    #   active fault samples
    # ======================================================

    evaluation_df = pd.concat(
        [
            nominal_results,
            active_fault_results,
        ],
        ignore_index=True,
    )

    phase5 = (
        calculate_binary_metrics(
            evaluation_df
        )
    )

    # ======================================================
    # PER-FAULT RECALL
    # ======================================================

    fault_types = list(
        files.keys()
    )

    per_fault_rows = []

    for fault_type in fault_types:

        subset = (
            active_fault_results[
                active_fault_results[
                    "true_label"
                ]
                ==
                fault_type
            ]
        )

        raw_count = int(
            subset[
                "raw_anomaly"
            ].sum()
        )

        persistent_count = int(
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
                        raw_count,
                        len(subset),
                    ),

                "persistent_recall":
                    safe_divide(
                        persistent_count,
                        len(subset),
                    ),
            }
        )

    per_fault_df = pd.DataFrame(
        per_fault_rows
    )

    # ======================================================
    # SUBSYSTEM RESPONSE MATRIX
    # ======================================================

    subsystem_rows = []

    for fault_type in fault_types:

        subset = (
            active_fault_results[
                active_fault_results[
                    "true_label"
                ]
                ==
                fault_type
            ]
        )

        row = {
            "fault_type":
                fault_type,
        }

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            row[
                f"{subsystem}_anomaly_rate"
            ] = float(
                subset[
                    f"{subsystem}_anomaly"
                ].mean()
            )

        subsystem_rows.append(
            row
        )

    subsystem_df = pd.DataFrame(
        subsystem_rows
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

    for fault_type, start_h in (
        fault_start_times.items()
    ):

        subset = (
            fault_results[
                fault_results[
                    "dataset"
                ]
                ==
                fault_type
            ]
        )

        correct_alarm = subset[
            (
                subset[
                    "time_h"
                ]
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
            correct_alarm
        ) > 0:

            first_detection_h = float(
                correct_alarm[
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

        # ----------------------------------------------
        # Persistent alarms before actual fault start
        # ----------------------------------------------

        pre_fault_false_alarms = int(
            subset[
                (
                    subset["time_h"]
                    < start_h
                )
                &
                (
                    subset[
                        "persistent_anomaly"
                    ]
                )
            ].shape[0]
        )

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

                "pre_fault_persistent_alarms":
                    pre_fault_false_alarms,
            }
        )

    delay_df = pd.DataFrame(
        delay_rows
    )

    detected_delays = (
        delay_df[
            delay_df[
                "detected"
            ]
            == True
        ]
    )

    if len(
        detected_delays
    ) > 0:

        mean_delay = float(
            detected_delays[
                "detection_delay_min"
            ].mean()
        )

    else:

        mean_delay = (
            float("nan")
        )

    # ======================================================
    # LOAD ORIGINAL GLOBAL ML BASELINE
    # ======================================================

    baseline_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_027_ml_per_fault_metrics.csv"
    )

    baseline_per_fault = load_csv(
        baseline_file
    )

    baseline_predictions = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_027_ml_predictions.csv"
    )

    baseline_metrics = (
        calculate_binary_metrics(
            baseline_predictions
        )
    )

    # ======================================================
    # OUTPUT DIRECTORIES
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
        / "experiment_029_subsystem_ml_predictions.csv"
    )

    nominal_file = (
        tables_dir
        / "experiment_029_nominal_holdout_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_029_per_fault_metrics.csv"
    )

    subsystem_file = (
        tables_dir
        / "experiment_029_subsystem_response_matrix.csv"
    )

    delay_file = (
        tables_dir
        / "experiment_029_detection_delays.csv"
    )

    active_fault_results.to_csv(
        predictions_file,
        index=False,
    )

    nominal_results.to_csv(
        nominal_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    subsystem_df.to_csv(
        subsystem_file,
        index=False,
    )

    delay_df.to_csv(
        delay_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PER-FAULT RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_029_subsystem_ml_recall.png"
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
        label="Raw subsystem anomaly",
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

    plt.title(
        "SENTINEL-XAI - "
        "Subsystem Isolation Forest Fault Recall"
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
    # FIGURE 2 — SUBSYSTEM RESPONSE MATRIX
    # ======================================================

    response_figure = (
        figures_dir
        / "experiment_029_subsystem_response_matrix.png"
    )

    subsystem_names = list(
        SUBSYSTEM_FEATURES.keys()
    )

    matrix = np.array(
        [
            [
                row[
                    f"{subsystem}_anomaly_rate"
                ]
                for subsystem
                in subsystem_names
            ]
            for _, row
            in subsystem_df.iterrows()
        ]
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    plt.xticks(
        range(
            len(
                subsystem_names
            )
        ),
        subsystem_names,
    )

    plt.yticks(
        range(
            len(
                subsystem_df
            )
        ),
        subsystem_df[
            "fault_type"
        ],
    )

    plt.xlabel(
        "Subsystem Isolation Forest"
    )

    plt.ylabel(
        "True Fault"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Subsystem ML Response Matrix"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        response_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — DETECTION DELAY
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_029_detection_delays.png"
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
        "SENTINEL-XAI - "
        "Subsystem ML Detection Delay"
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
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 82)

    print(
        "SENTINEL-XAI - EXPERIMENT 029"
    )

    print(
        "SUBSYSTEM ISOLATION FOREST "
        "ALL-FAULT EVALUATION"
    )

    print("=" * 82)

    print()
    print(
        "INDEPENDENT HOLDOUT METRICS"
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

    print(
        f"Untouched nominal samples: "
        f"{len(nominal_results)}"
    )

    print(
        "Persistent nominal alarms: "
        f"{int(nominal_results['persistent_anomaly'].sum())}"
    )

    print()

    print("-" * 82)
    print(
        "PER-FAULT RESULTS"
    )
    print("-" * 82)

    for _, row in (
        per_fault_df.iterrows()
    ):

        print()

        print(
            row["fault_type"]
        )

        print(
            f"  Raw recall: "
            f"{row['raw_recall']:.4f}"
        )

        print(
            f"  Persistent recall: "
            f"{row['persistent_recall']:.4f}"
        )

    print()

    print("-" * 82)
    print(
        "DETECTION DELAYS"
    )
    print("-" * 82)

    for _, row in (
        delay_df.iterrows()
    ):

        print()

        print(
            row["fault_type"]
        )

        if row["detected"]:

            print(
                f"  Delay: "
                f"{row['detection_delay_min']:.1f} min"
            )

        else:

            print(
                "  Delay: NOT DETECTED"
            )

        print(
            "  Pre-fault persistent alarms: "
            f"{int(row['pre_fault_persistent_alarms'])}"
        )

    print()

    print(
        f"Mean detected delay: "
        f"{mean_delay:.1f} min"
    )

    print()

    print("-" * 82)
    print(
        "GLOBAL ML BASELINE vs SUBSYSTEM ML"
    )
    print("-" * 82)

    print()

    print(
        "Experiment 027 global Isolation Forest"
    )

    print(
        f"  Recall: "
        f"{baseline_metrics['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{baseline_metrics['f1']:.4f}"
    )

    print()

    print(
        "Experiment 029 subsystem Isolation Forest"
    )

    print(
        f"  Recall: "
        f"{phase5['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{phase5['f1']:.4f}"
    )

    print()

    print("=" * 82)

    print(
        "Tables saved to:"
    )

    print(
        predictions_file
    )

    print(
        nominal_file
    )

    print(
        per_fault_file
    )

    print(
        subsystem_file
    )

    print(
        delay_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        recall_figure
    )

    print(
        response_figure
    )

    print(
        delay_figure
    )

    print("=" * 82)


if __name__ == "__main__":
    main()