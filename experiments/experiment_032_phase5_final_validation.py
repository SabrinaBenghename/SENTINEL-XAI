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
# SCORE ONE COMPLETE SCENARIO
# ==========================================================

def evaluate_dataset(
    dataset_name,
    df,
    detector,
    force_nominal=False,
):

    features = build_ml_features(
        df
    )

    n = len(
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

    anomaly_columns = []
    ratio_columns = []

    # ======================================================
    # SCORE EACH SUBSYSTEM
    # ======================================================

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = features[
            columns
        ]

        missing = (
            x.isna()
            .any(axis=1)
            .to_numpy()
        )

        valid = (
            ~missing
        )

        scores = np.full(
            n,
            np.nan,
            dtype=float,
        )

        if valid.any():

            scores[
                valid
            ] = (
                -detector.models[
                    subsystem
                ].score_samples(
                    x.loc[
                        valid
                    ]
                )
            )

        threshold = (
            detector.thresholds[
                subsystem
            ]
        )

        ratios = np.full(
            n,
            np.nan,
            dtype=float,
        )

        ratios[
            valid
        ] = (
            scores[
                valid
            ]
            /
            threshold
        )

        raw_anomaly = np.zeros(
            n,
            dtype=bool,
        )

        raw_anomaly[
            valid
        ] = (
            scores[
                valid
            ]
            >
            threshold
        )

        # Missing telemetry cannot be passed normally
        # through the Isolation Forest.
        #
        # Missing required input is therefore treated
        # as anomalous telemetry input.
        raw_anomaly[
            missing
        ] = True

        ratios[
            missing
        ] = 1.25

        result[
            f"{subsystem}_score"
        ] = scores

        result[
            f"{subsystem}_threshold"
        ] = threshold

        result[
            f"{subsystem}_ratio"
        ] = ratios

        result[
            f"{subsystem}_anomaly"
        ] = raw_anomaly

        result[
            f"{subsystem}_missing"
        ] = missing

        anomaly_columns.append(
            f"{subsystem}_anomaly"
        )

        ratio_columns.append(
            f"{subsystem}_ratio"
        )

    # ======================================================
    # GLOBAL RAW ML DECISION
    # ======================================================

    result[
        "maximum_score_ratio"
    ] = result[
        ratio_columns
    ].max(
        axis=1
    )

    result[
        "raw_anomaly"
    ] = result[
        anomaly_columns
    ].any(
        axis=1
    )

    # ======================================================
    # THREE-SAMPLE PERSISTENCE
    # ======================================================

    persistence = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent_values = []
    persistence_counts = []

    for value in result[
        "raw_anomaly"
    ]:

        state = persistence.update(
            bool(value)
        )

        persistent_values.append(
            state.persistent_anomaly
        )

        persistence_counts.append(
            state.consecutive_count
        )

    result[
        "persistence_count"
    ] = persistence_counts

    result[
        "persistent_anomaly"
    ] = persistent_values

    result[
        "predicted_fault"
    ] = (
        result[
            "persistent_anomaly"
        ]
        .astype(int)
    )

    # ======================================================
    # GROUND TRUTH
    # ======================================================

    if force_nominal:

        labels = pd.Series(
            ["nominal"] * n,
            index=df.index,
        )

    elif "fault_label" in df.columns:

        labels = (
            df[
                "fault_label"
            ]
            .fillna(
                "nominal"
            )
            .astype(str)
        )

    else:

        labels = pd.Series(
            ["nominal"] * n,
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
    ).astype(int)

    return result


# ==========================================================
# BINARY METRICS
# ==========================================================

def calculate_metrics(
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
    # LOAD FROZEN EXPERIMENT-031 MODEL
    # ======================================================

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_031_regime_balanced_subsystem_iforest.joblib"
    )

    detector = joblib.load(
        model_file
    )

    # ======================================================
    # FAULT SCENARIOS
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

    scenario_results = []

    print()
    print(
        "Running frozen Phase-5 detector "
        "on all six complete scenarios..."
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

        scenario_results.append(
            evaluated
        )

    scenario_df = pd.concat(
        scenario_results,
        ignore_index=True,
    )

    # ======================================================
    # END-TO-END SCENARIO METRICS
    #
    # IMPORTANT:
    # Includes the healthy PRE-FAULT samples.
    # ======================================================

    scenario_metrics = (
        calculate_metrics(
            scenario_df
        )
    )

    # ======================================================
    # SEPARATE 24H HELD-OUT NOMINAL EVALUATION
    # ======================================================

    nominal_24h = load_csv(
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry_24h.csv"
    )

    holdout_start = int(
        0.80
        * len(
            nominal_24h
        )
    )

    nominal_holdout_df = (
        nominal_24h
        .iloc[
            holdout_start:
        ]
        .copy()
    )

    nominal_holdout_result = (
        evaluate_dataset(
            dataset_name=(
                "24h_nominal_holdout"
            ),
            df=nominal_holdout_df,
            detector=detector,
            force_nominal=True,
        )
    )

    holdout_false_alarms = int(
        nominal_holdout_result[
            "persistent_anomaly"
        ].sum()
    )

    # ======================================================
    # 6H HEALTHY STARTUP CHECK
    # ======================================================

    nominal_6h = load_csv(
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    nominal_6h_result = (
        evaluate_dataset(
            dataset_name=(
                "6h_nominal_startup"
            ),
            df=nominal_6h,
            detector=detector,
            force_nominal=True,
        )
    )

    nominal_6h_false_alarms = int(
        nominal_6h_result[
            "persistent_anomaly"
        ].sum()
    )

    # ======================================================
    # ACTIVE FAULT RESULTS
    # ======================================================

    active_fault_df = (
        scenario_df[
            scenario_df[
                "true_fault"
            ]
            == 1
        ]
        .copy()
    )

    # ======================================================
    # PER-FAULT RECALL
    # ======================================================

    per_fault_rows = []

    for fault_type in (
        files.keys()
    ):

        subset = (
            active_fault_df[
                active_fault_df[
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
    # PRE-FAULT FALSE ALARMS
    # ======================================================

    start_times = {

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

    prefault_rows = []

    delay_rows = []

    for fault_type, start_h in (
        start_times.items()
    ):

        subset = (
            scenario_df[
                scenario_df[
                    "dataset"
                ]
                ==
                fault_type
            ]
        )

        prefault = subset[
            subset[
                "time_h"
            ]
            <
            start_h
        ]

        prefault_alarm_count = int(
            prefault[
                "persistent_anomaly"
            ].sum()
        )

        prefault_rows.append(
            {
                "fault_type":
                    fault_type,

                "prefault_samples":
                    len(prefault),

                "persistent_false_alarms":
                    prefault_alarm_count,

                "prefault_false_alarm_rate":
                    safe_divide(
                        prefault_alarm_count,
                        len(prefault),
                    ),
            }
        )

        # ==================================================
        # DETECTION DELAY
        # ==================================================

        detected = subset[
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
            detected
        ) > 0:

            first_detection_h = float(
                detected[
                    "time_h"
                ].iloc[0]
            )

            delay_min = (
                first_detection_h
                - start_h
            ) * 60.0

            detected_flag = True

        else:

            first_detection_h = (
                float("nan")
            )

            delay_min = (
                float("nan")
            )

            detected_flag = False

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
                    detected_flag,
            }
        )

    prefault_df = pd.DataFrame(
        prefault_rows
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
    # SUBSYSTEM RESPONSE MATRIX
    # ======================================================

    subsystem_rows = []

    for fault_type in (
        files.keys()
    ):

        subset = (
            active_fault_df[
                active_fault_df[
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
    # SAVE
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
        / "experiment_032_phase5_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_032_phase5_per_fault.csv"
    )

    prefault_file = (
        tables_dir
        / "experiment_032_phase5_prefault_false_alarms.csv"
    )

    delay_file = (
        tables_dir
        / "experiment_032_phase5_detection_delays.csv"
    )

    subsystem_file = (
        tables_dir
        / "experiment_032_phase5_subsystem_matrix.csv"
    )

    scenario_df.to_csv(
        predictions_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    prefault_df.to_csv(
        prefault_file,
        index=False,
    )

    delay_df.to_csv(
        delay_file,
        index=False,
    )

    subsystem_df.to_csv(
        subsystem_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PER-FAULT RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_032_phase5_fault_recall.png"
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
        label="Raw anomaly",
    )

    plt.bar(
        x + width / 2,
        per_fault_df[
            "persistent_recall"
        ],
        width=width,
        label="Persistent alarm",
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
        "Recall"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Final Phase-5 Fault Recall"
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
    # FIGURE 2 — DELAYS
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_032_phase5_detection_delays.png"
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
        "Final Phase-5 Detection Delay"
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
    # FIGURE 3 — SUBSYSTEM RESPONSE MATRIX
    # ======================================================

    response_figure = (
        figures_dir
        / "experiment_032_phase5_subsystem_response.png"
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
        "Subsystem ML Detector"
    )

    plt.ylabel(
        "True Fault"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Final Phase-5 Subsystem Response"
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
    # TERMINAL SUMMARY
    # ======================================================

    print()
    print("=" * 84)

    print(
        "SENTINEL-XAI - EXPERIMENT 032"
    )

    print(
        "PHASE 5 FINAL UNSUPERVISED ML EVALUATION"
    )

    print("=" * 84)

    print()

    print(
        "Detector configuration:"
    )

    print(
        "  4 subsystem Isolation Forest models"
    )

    print(
        "  300 trees per model"
    )

    print(
        "  Nominal-only training"
    )

    print(
        "  99th-percentile regime-balanced calibration"
    )

    print(
        "  3-sample persistence"
    )

    print()

    print("-" * 84)
    print(
        "END-TO-END SCENARIO METRICS"
    )
    print(
        "(healthy pre-fault samples INCLUDED)"
    )
    print("-" * 84)

    print(
        f"True positives:  "
        f"{scenario_metrics['tp']}"
    )

    print(
        f"True negatives:  "
        f"{scenario_metrics['tn']}"
    )

    print(
        f"False positives: "
        f"{scenario_metrics['fp']}"
    )

    print(
        f"False negatives: "
        f"{scenario_metrics['fn']}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{scenario_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{scenario_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{scenario_metrics['recall']:.4f}"
    )

    print(
        f"F1 score:  "
        f"{scenario_metrics['f1']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{scenario_metrics['false_positive_rate']:.4f}"
    )

    print()

    print("-" * 84)
    print(
        "HEALTHY GENERALIZATION CHECKS"
    )
    print("-" * 84)

    print(
        f"24h untouched holdout samples: "
        f"{len(nominal_holdout_result)}"
    )

    print(
        f"24h holdout persistent alarms: "
        f"{holdout_false_alarms}"
    )

    print()

    print(
        f"6h nominal samples: "
        f"{len(nominal_6h_result)}"
    )

    print(
        f"6h nominal persistent alarms: "
        f"{nominal_6h_false_alarms}"
    )

    print()

    print("-" * 84)
    print(
        "PER-FAULT RECALL"
    )
    print("-" * 84)

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

    print("-" * 84)
    print(
        "PRE-FAULT FALSE ALARMS"
    )
    print("-" * 84)

    for _, row in (
        prefault_df.iterrows()
    ):

        print()

        print(
            row["fault_type"]
        )

        print(
            f"  Healthy samples: "
            f"{int(row['prefault_samples'])}"
        )

        print(
            f"  Persistent false alarms: "
            f"{int(row['persistent_false_alarms'])}"
        )

        print(
            f"  False-alarm rate: "
            f"{row['prefault_false_alarm_rate']:.4f}"
        )

    print()

    print("-" * 84)
    print(
        "DETECTION DELAYS"
    )
    print("-" * 84)

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

    print()

    print(
        f"Mean detected delay: "
        f"{mean_delay:.1f} min"
    )

    print()

    print("=" * 84)

    print(
        "PHASE 5 MODEL IS NOW FROZEN."
    )

    print(
        "No further threshold or persistence "
        "tuning will be performed."
    )

    print("=" * 84)

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
        prefault_file
    )

    print(
        delay_file
    )

    print(
        subsystem_file
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
        response_figure
    )

    print("=" * 84)


if __name__ == "__main__":
    main()