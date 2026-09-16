from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.detection.model_based_detector import (
    ModelBasedDetector,
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


# ==========================================================
# RUN DETECTOR ON ONE DATASET
# ==========================================================

def evaluate_dataset(
    dataset_name,
    df,
):

    # IMPORTANT:
    # New detector for every experiment so that
    # internal EKF/model states do not leak between runs.
    detector = (
        ModelBasedDetector()
    )

    records = []

    for index, row in df.iterrows():

        if index == 0:

            dt_s = 60.0

        else:

            dt_s = (
                row["time_s"]
                - df.iloc[
                    index - 1
                ]["time_s"]
            )

        result = detector.detect(
            row=row,
            dt_s=dt_s,
        )

        true_label = row.get(
            "fault_label",
            "nominal",
        )

        if pd.isna(true_label):
            true_label = "nominal"

        predicted_label = (
            result.predicted_fault
        )

        true_fault = (
            true_label != "nominal"
        )

        predicted_fault = (
            predicted_label != "nominal"
        )

        records.append(
            {
                "dataset":
                    dataset_name,

                "time_h":
                    row["time_h"],

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "true_fault":
                    int(true_fault),

                "predicted_fault":
                    int(predicted_fault),

                "confidence":
                    result.confidence,

                "solar_relative_residual":
                    result.solar_relative_residual,

                "battery_innovation_v":
                    result.battery_innovation_v,

                "battery_temp_residual_c":
                    result.battery_temp_residual_c,

                "electronics_temp_residual_c":
                    result.electronics_temp_residual_c,

                "wheel_current_residual_a":
                    result.wheel_current_residual_a,

                "wheel_temp_residual_c":
                    result.wheel_temp_residual_c,
            }
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# GLOBAL METRICS
# ==========================================================

def calculate_metrics(
    df,
):

    tp = (
        (
            df["true_fault"] == 1
        )
        &
        (
            df["predicted_fault"] == 1
        )
    ).sum()

    tn = (
        (
            df["true_fault"] == 0
        )
        &
        (
            df["predicted_fault"] == 0
        )
    ).sum()

    fp = (
        (
            df["true_fault"] == 0
        )
        &
        (
            df["predicted_fault"] == 1
        )
    ).sum()

    fn = (
        (
            df["true_fault"] == 1
        )
        &
        (
            df["predicted_fault"] == 0
        )
    ).sum()

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

    false_positive_rate = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    fault_rows = df[
        df["true_fault"] == 1
    ]

    classification_accuracy = (
        safe_divide(
            (
                fault_rows["true_label"]
                ==
                fault_rows["predicted_label"]
            ).sum(),
            len(fault_rows),
        )
    )

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),

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

        "classification_accuracy":
            classification_accuracy,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

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
        "Running model-based detector "
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
        )

        results.append(
            evaluated
        )

    results_df = pd.concat(
        results,
        ignore_index=True,
    )

    # ======================================================
    # GLOBAL MODEL-BASED METRICS
    # ======================================================

    phase4 = calculate_metrics(
        results_df
    )

    # ======================================================
    # PER-FAULT PERFORMANCE
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
            results_df["true_label"]
            == fault_type
        ]

        detected = (
            subset["predicted_fault"]
            == 1
        ).sum()

        correctly_classified = (
            subset["predicted_label"]
            == fault_type
        ).sum()

        detection_recall = (
            safe_divide(
                detected,
                len(subset),
            )
        )

        classification_recall = (
            safe_divide(
                correctly_classified,
                len(subset),
            )
        )

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "samples":
                    len(subset),

                "detection_recall":
                    detection_recall,

                "classification_recall":
                    classification_recall,
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
            results_df["dataset"]
            == fault_type
        ]

        correct = subset[
            (
                subset["time_h"]
                >= start_h
            )
            &
            (
                subset["predicted_label"]
                == fault_type
            )
        ]

        if len(correct) > 0:

            first_detection_h = float(
                correct[
                    "time_h"
                ].iloc[0]
            )

            delay_min = (
                (
                    first_detection_h
                    - start_h
                )
                * 60.0
            )

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

    detected_delay_df = delay_df[
        delay_df["detected"]
        == True
    ]

    if len(detected_delay_df) > 0:

        phase4_mean_delay = float(
            detected_delay_df[
                "detection_delay_min"
            ].mean()
        )

    else:

        phase4_mean_delay = (
            float("nan")
        )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    labels = [
        "nominal",
        "solar_array_degradation",
        "battery_degradation",
        "thermal_anomaly",
        "reaction_wheel_degradation",
        "battery_voltage_sensor_drift",
        "telemetry_dropout",
    ]

    confusion = pd.crosstab(
        results_df["true_label"],
        results_df["predicted_label"],
    )

    confusion = confusion.reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )

    # ======================================================
    # PHASE 3 BASELINE
    # ======================================================

    phase3_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_013_rule_based_predictions.csv"
    )

    phase3_delay_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_014_detection_delay.csv"
    )

    phase3_predictions = load_csv(
        phase3_file
    )

    phase3_delay = load_csv(
        phase3_delay_file
    )

    phase3 = calculate_metrics(
        phase3_predictions
    )

    phase3_detected_delays = (
        phase3_delay[
            phase3_delay["detected"]
            == True
        ]
    )

    phase3_mean_delay = float(
        phase3_detected_delays[
            "detection_delay_min"
        ].mean()
    )

    # ======================================================
    # COMPARISON TABLE
    # ======================================================

    comparison_df = pd.DataFrame(
        [
            {
                "method":
                    "Phase 3 - Rule Based",

                "precision":
                    phase3["precision"],

                "recall":
                    phase3["recall"],

                "f1":
                    phase3["f1"],

                "classification_accuracy":
                    phase3[
                        "classification_accuracy"
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
                    "Phase 4 - Model Based",

                "precision":
                    phase4["precision"],

                "recall":
                    phase4["recall"],

                "f1":
                    phase4["f1"],

                "classification_accuracy":
                    phase4[
                        "classification_accuracy"
                    ],

                "false_positive_rate":
                    phase4[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    phase4_mean_delay,
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

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_file = (
        tables_dir
        / "experiment_021_model_based_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_021_per_fault_metrics.csv"
    )

    delay_file = (
        tables_dir
        / "experiment_021_detection_delays.csv"
    )

    confusion_file = (
        tables_dir
        / "experiment_021_confusion_matrix.csv"
    )

    comparison_file = (
        tables_dir
        / "experiment_021_phase3_vs_phase4.csv"
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

    confusion.to_csv(
        confusion_file
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    # ======================================================
    # FIGURE DIRECTORY
    # ======================================================

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
    # FIGURE 1 — PER-FAULT RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_021_model_based_recall.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        per_fault_df["fault_type"],
        per_fault_df[
            "classification_recall"
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.ylabel(
        "Classification Recall"
    )

    plt.title(
        "SENTINEL-XAI - Model-Based Fault Recall"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        recall_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — CONFUSION MATRIX
    # ======================================================

    confusion_figure = (
        figures_dir
        / "experiment_021_model_based_confusion_matrix.png"
    )

    matrix = (
        confusion.values
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        aspect="auto",
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.title(
        "SENTINEL-XAI - Model-Based Confusion Matrix"
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
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        confusion_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — PHASE 3 VS PHASE 4
    # ======================================================

    comparison_figure = (
        figures_dir
        / "experiment_021_phase3_vs_phase4.png"
    )

    metric_names = [
        "precision",
        "recall",
        "f1",
        "classification_accuracy",
    ]

    phase3_values = [
        phase3[name]
        for name in metric_names
    ]

    phase4_values = [
        phase4[name]
        for name in metric_names
    ]

    x = np.arange(
        len(metric_names)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        x - width / 2,
        phase3_values,
        width=width,
        label="Phase 3 Rule-Based",
    )

    plt.bar(
        x + width / 2,
        phase4_values,
        width=width,
        label="Phase 4 Model-Based",
    )

    plt.xticks(
        x,
        [
            "Precision",
            "Recall",
            "F1",
            "Classification Accuracy",
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
        "SENTINEL-XAI - Classical vs Model-Based Detection"
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
    # FIGURE 4 — DETECTION DELAYS
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_021_detection_delays.png"
    )

    plot_delay = delay_df.copy()

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        plot_delay["fault_type"],
        plot_delay[
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
        "SENTINEL-XAI - Model-Based Detection Delay"
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
    # TERMINAL SUMMARY
    # ======================================================

    print()
    print("=" * 80)

    print(
        "SENTINEL-XAI - EXPERIMENT 021"
    )

    print(
        "MODEL-BASED FAULT DETECTION EVALUATION"
    )

    print("=" * 80)

    print()
    print("Phase 4 global metrics:")
    print()

    print(
        f"True positives:  "
        f"{phase4['tp']}"
    )

    print(
        f"True negatives:  "
        f"{phase4['tn']}"
    )

    print(
        f"False positives: "
        f"{phase4['fp']}"
    )

    print(
        f"False negatives: "
        f"{phase4['fn']}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{phase4['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{phase4['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{phase4['recall']:.4f}"
    )

    print(
        f"F1 score:  "
        f"{phase4['f1']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{phase4['false_positive_rate']:.4f}"
    )

    print(
        f"Fault classification accuracy: "
        f"{phase4['classification_accuracy']:.4f}"
    )

    print()

    print("-" * 80)
    print("PER-FAULT RESULTS")
    print("-" * 80)

    for _, row in per_fault_df.iterrows():

        print()
        print(
            row["fault_type"]
        )

        print(
            f"  Detection recall: "
            f"{row['detection_recall']:.4f}"
        )

        print(
            f"  Classification recall: "
            f"{row['classification_recall']:.4f}"
        )

    print()

    print("-" * 80)
    print("DETECTION DELAYS")
    print("-" * 80)

    for _, row in delay_df.iterrows():

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
        f"Phase 4 mean detected delay: "
        f"{phase4_mean_delay:.1f} min"
    )

    print()

    print("-" * 80)
    print("PHASE 3 vs PHASE 4")
    print("-" * 80)

    print()

    print(
        f"Recall: "
        f"{phase3['recall']:.4f}"
        f" -> "
        f"{phase4['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{phase3['f1']:.4f}"
        f" -> "
        f"{phase4['f1']:.4f}"
    )

    print(
        f"Classification accuracy: "
        f"{phase3['classification_accuracy']:.4f}"
        f" -> "
        f"{phase4['classification_accuracy']:.4f}"
    )

    print(
        f"Mean detection delay: "
        f"{phase3_mean_delay:.1f} min"
        f" -> "
        f"{phase4_mean_delay:.1f} min"
    )

    print()

    print("=" * 80)

    print()
    print("Tables saved to:")
    print(predictions_file)
    print(per_fault_file)
    print(delay_file)
    print(confusion_file)
    print(comparison_file)

    print()

    print("Figures saved to:")
    print(recall_figure)
    print(confusion_figure)
    print(comparison_figure)
    print(delay_figure)

    print("=" * 80)


if __name__ == "__main__":
    main()