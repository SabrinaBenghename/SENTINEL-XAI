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


from src.detection.model_based_detector_refined import (
    RefinedModelBasedDetector,
)


# ==========================================================
# HELPERS
# ==========================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


def load_csv(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


def calculate_metrics(df):

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

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    fault_rows = df[
        df["true_fault"] == 1
    ]

    classification_accuracy = safe_divide(
        (
            fault_rows["true_label"]
            ==
            fault_rows["predicted_label"]
        ).sum(),
        len(fault_rows),
    )

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "false_positive_rate":
            false_positive_rate,

        "classification_accuracy":
            classification_accuracy,
    }


# ==========================================================
# RUN REFINED DETECTOR ON ONE DATASET
# ==========================================================

def evaluate_dataset(
    dataset_name,
    df,
):

    # Fresh detector for every independent scenario.
    detector = (
        RefinedModelBasedDetector()
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
            true_label
            != "nominal"
        )

        predicted_fault = (
            predicted_label
            != "nominal"
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

                "solar_relative_residual":
                    result.solar_relative_residual,
            }
        )

    return pd.DataFrame(
        records
    )


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

    all_results = []

    print()
    print(
        "Running refined model-based detector "
        "on all spacecraft datasets..."
    )

    for name, path in files.items():

        print(
            f"  Processing {name}..."
        )

        df = load_csv(path)

        evaluated = evaluate_dataset(
            dataset_name=name,
            df=df,
        )

        all_results.append(
            evaluated
        )

    refined_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    refined_metrics = (
        calculate_metrics(
            refined_df
        )
    )

    # ======================================================
    # PER-FAULT METRICS
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

        subset = refined_df[
            refined_df["true_label"]
            == fault_type
        ]

        detected_count = (
            subset["predicted_fault"]
            == 1
        ).sum()

        correct_count = (
            subset["predicted_label"]
            == fault_type
        ).sum()

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "samples":
                    len(subset),

                "detection_recall":
                    safe_divide(
                        detected_count,
                        len(subset),
                    ),

                "classification_recall":
                    safe_divide(
                        correct_count,
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

    start_times = {
        "solar_array_degradation": 2.0,
        "battery_degradation": 2.0,
        "thermal_anomaly": 2.0,
        "reaction_wheel_degradation": 2.0,
        "battery_voltage_sensor_drift": 2.0,
        "telemetry_dropout": 3.0,
    }

    delay_rows = []

    for fault_type, start_h in start_times.items():

        subset = refined_df[
            refined_df["dataset"]
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

    detected_delays = delay_df[
        delay_df["detected"] == True
    ]

    refined_mean_delay = float(
        detected_delays[
            "detection_delay_min"
        ].mean()
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
        refined_df["true_label"],
        refined_df["predicted_label"],
    )

    confusion = confusion.reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )

    # ======================================================
    # LOAD PREVIOUS METHODS
    # ======================================================

    phase3_predictions = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_013_rule_based_predictions.csv"
    )

    phase3_delay = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_014_detection_delay.csv"
    )

    original_phase4_predictions = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_021_model_based_predictions.csv"
    )

    original_phase4_delay = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_021_detection_delays.csv"
    )

    original_phase4_per_fault = load_csv(
        project_root
        / "results"
        / "tables"
        / "experiment_021_per_fault_metrics.csv"
    )

    phase3_metrics = (
        calculate_metrics(
            phase3_predictions
        )
    )

    original_phase4_metrics = (
        calculate_metrics(
            original_phase4_predictions
        )
    )

    phase3_mean_delay = float(
        phase3_delay.loc[
            phase3_delay["detected"] == True,
            "detection_delay_min",
        ].mean()
    )

    original_phase4_mean_delay = float(
        original_phase4_delay.loc[
            original_phase4_delay["detected"] == True,
            "detection_delay_min",
        ].mean()
    )

    # ======================================================
    # SENSOR-DRIFT IMPROVEMENT
    # ======================================================

    original_sensor_recall = float(
        original_phase4_per_fault.loc[
            original_phase4_per_fault["fault_type"]
            ==
            "battery_voltage_sensor_drift",
            "classification_recall",
        ].iloc[0]
    )

    refined_sensor_recall = float(
        per_fault_df.loc[
            per_fault_df["fault_type"]
            ==
            "battery_voltage_sensor_drift",
            "classification_recall",
        ].iloc[0]
    )

    # ======================================================
    # METHOD COMPARISON
    # ======================================================

    comparison_df = pd.DataFrame(
        [
            {
                "method":
                    "Phase 3 Rule-Based",

                "precision":
                    phase3_metrics["precision"],

                "recall":
                    phase3_metrics["recall"],

                "f1":
                    phase3_metrics["f1"],

                "classification_accuracy":
                    phase3_metrics[
                        "classification_accuracy"
                    ],

                "false_positive_rate":
                    phase3_metrics[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    phase3_mean_delay,
            },

            {
                "method":
                    "Phase 4 Original",

                "precision":
                    original_phase4_metrics[
                        "precision"
                    ],

                "recall":
                    original_phase4_metrics[
                        "recall"
                    ],

                "f1":
                    original_phase4_metrics[
                        "f1"
                    ],

                "classification_accuracy":
                    original_phase4_metrics[
                        "classification_accuracy"
                    ],

                "false_positive_rate":
                    original_phase4_metrics[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    original_phase4_mean_delay,
            },

            {
                "method":
                    "Phase 4 Refined",

                "precision":
                    refined_metrics["precision"],

                "recall":
                    refined_metrics["recall"],

                "f1":
                    refined_metrics["f1"],

                "classification_accuracy":
                    refined_metrics[
                        "classification_accuracy"
                    ],

                "false_positive_rate":
                    refined_metrics[
                        "false_positive_rate"
                    ],

                "mean_detection_delay_min":
                    refined_mean_delay,
            },
        ]
    )

    # ======================================================
    # FINAL PHASE-4 VALIDATION CHECKS
    # ======================================================

    all_fault_types_detected = (
        int(
            delay_df[
                "detected"
            ].sum()
        )
        == 6
    )

    checks = [
        {
            "check":
                "Nominal false alarms",

            "value":
                refined_metrics["fp"],

            "criterion":
                "== 0",

            "pass":
                refined_metrics["fp"] == 0,
        },

        {
            "check":
                "Precision",

            "value":
                refined_metrics["precision"],

            "criterion":
                ">= 0.95",

            "pass":
                refined_metrics["precision"] >= 0.95,
        },

        {
            "check":
                "Recall improves over Phase 3",

            "value":
                refined_metrics["recall"],

            "criterion":
                f"> {phase3_metrics['recall']:.4f}",

            "pass":
                refined_metrics["recall"]
                >
                phase3_metrics["recall"],
        },

        {
            "check":
                "F1 improves over Phase 3",

            "value":
                refined_metrics["f1"],

            "criterion":
                f"> {phase3_metrics['f1']:.4f}",

            "pass":
                refined_metrics["f1"]
                >
                phase3_metrics["f1"],
        },

        {
            "check":
                "Classification improves over Phase 3",

            "value":
                refined_metrics[
                    "classification_accuracy"
                ],

            "criterion":
                (
                    f"> "
                    f"{phase3_metrics['classification_accuracy']:.4f}"
                ),

            "pass":
                (
                    refined_metrics[
                        "classification_accuracy"
                    ]
                    >
                    phase3_metrics[
                        "classification_accuracy"
                    ]
                ),
        },

        {
            "check":
                "Detection delay improves over Phase 3",

            "value":
                refined_mean_delay,

            "criterion":
                f"< {phase3_mean_delay:.1f} min",

            "pass":
                refined_mean_delay
                <
                phase3_mean_delay,
        },

        {
            "check":
                "Sensor-drift diagnosis improves",

            "value":
                refined_sensor_recall,

            "criterion":
                (
                    f"> "
                    f"{original_sensor_recall:.4f}"
                ),

            "pass":
                refined_sensor_recall
                >
                original_sensor_recall,
        },

        {
            "check":
                "All six faults detected",

            "value":
                int(
                    delay_df[
                        "detected"
                    ].sum()
                ),

            "criterion":
                "== 6",

            "pass":
                all_fault_types_detected,
        },
    ]

    checks_df = pd.DataFrame(
        checks
    )

    global_validation = bool(
        checks_df["pass"].all()
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
        / "experiment_023_refined_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_023_per_fault_metrics.csv"
    )

    delay_file = (
        tables_dir
        / "experiment_023_detection_delays.csv"
    )

    confusion_file = (
        tables_dir
        / "experiment_023_confusion_matrix.csv"
    )

    comparison_file = (
        tables_dir
        / "experiment_023_method_comparison.csv"
    )

    validation_file = (
        tables_dir
        / "experiment_023_phase4_validation.csv"
    )

    refined_df.to_csv(
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
        confusion_file,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    checks_df.to_csv(
        validation_file,
        index=False,
    )

    # ======================================================
    # FIGURES
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
    # FIGURE 1 - PER-FAULT RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_023_refined_fault_recall.png"
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
        "SENTINEL-XAI - Refined Model-Based Fault Recall"
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
    # FIGURE 2 - CONFUSION MATRIX
    # ======================================================

    confusion_figure = (
        figures_dir
        / "experiment_023_refined_confusion_matrix.png"
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
        "SENTINEL-XAI - Refined Model-Based Confusion Matrix"
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
    # FIGURE 3 - THREE-METHOD COMPARISON
    # ======================================================

    comparison_figure = (
        figures_dir
        / "experiment_023_method_comparison.png"
    )

    metric_names = [
        "precision",
        "recall",
        "f1",
        "classification_accuracy",
    ]

    x = np.arange(
        len(metric_names)
    )

    width = 0.25

    plt.figure(
        figsize=(11, 5)
    )

    for index, row in comparison_df.iterrows():

        values = [
            row[name]
            for name in metric_names
        ]

        offset = (
            index - 1
        ) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=row["method"],
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
        "SENTINEL-XAI - Detection Method Comparison"
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
    # FIGURE 4 - DETECTION DELAYS
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_023_refined_detection_delays.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        delay_df["fault_type"],
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
        "SENTINEL-XAI - Refined Model-Based Detection Delay"
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
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 82)
    print("SENTINEL-XAI - EXPERIMENT 023")
    print("PHASE 4 FINAL MODEL-BASED VALIDATION")
    print("=" * 82)

    print()
    print("REFINED GLOBAL METRICS")
    print("-" * 82)

    print(
        f"True positives:  "
        f"{refined_metrics['tp']}"
    )

    print(
        f"True negatives:  "
        f"{refined_metrics['tn']}"
    )

    print(
        f"False positives: "
        f"{refined_metrics['fp']}"
    )

    print(
        f"False negatives: "
        f"{refined_metrics['fn']}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{refined_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{refined_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{refined_metrics['recall']:.4f}"
    )

    print(
        f"F1 score:  "
        f"{refined_metrics['f1']:.4f}"
    )

    print(
        f"Classification accuracy: "
        f"{refined_metrics['classification_accuracy']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{refined_metrics['false_positive_rate']:.4f}"
    )

    print(
        f"Mean detection delay: "
        f"{refined_mean_delay:.1f} min"
    )

    # ======================================================
    # PER-FAULT
    # ======================================================

    print()
    print("-" * 82)
    print("PER-FAULT RESULTS")
    print("-" * 82)

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

    # ======================================================
    # DELAYS
    # ======================================================

    print()
    print("-" * 82)
    print("DETECTION DELAYS")
    print("-" * 82)

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

    # ======================================================
    # METHOD COMPARISON
    # ======================================================

    print()
    print("-" * 82)
    print("METHOD COMPARISON")
    print("-" * 82)

    print()
    print(
        f"Phase 3 recall: "
        f"{phase3_metrics['recall']:.4f}"
    )

    print(
        f"Original Phase 4 recall: "
        f"{original_phase4_metrics['recall']:.4f}"
    )

    print(
        f"Refined Phase 4 recall: "
        f"{refined_metrics['recall']:.4f}"
    )

    print()

    print(
        f"Phase 3 F1: "
        f"{phase3_metrics['f1']:.4f}"
    )

    print(
        f"Original Phase 4 F1: "
        f"{original_phase4_metrics['f1']:.4f}"
    )

    print(
        f"Refined Phase 4 F1: "
        f"{refined_metrics['f1']:.4f}"
    )

    print()

    print(
        f"Sensor-drift classification recall: "
        f"{original_sensor_recall:.4f}"
        f" -> "
        f"{refined_sensor_recall:.4f}"
    )

    print()

    print(
        f"Mean detection delay: "
        f"{phase3_mean_delay:.1f}"
        f" -> "
        f"{original_phase4_mean_delay:.1f}"
        f" -> "
        f"{refined_mean_delay:.1f} min"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print()
    print("-" * 82)
    print("FINAL PHASE-4 VALIDATION CHECKS")
    print("-" * 82)

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
    print("-" * 82)

    print(
        f"Passed checks: "
        f"{int(checks_df['pass'].sum())}"
        f"/"
        f"{len(checks_df)}"
    )

    print(
        f"GLOBAL PHASE 4 VALIDATION: "
        f"{global_validation}"
    )

    print("-" * 82)

    print()
    print("Tables saved to:")
    print(predictions_file)
    print(per_fault_file)
    print(delay_file)
    print(confusion_file)
    print(comparison_file)
    print(validation_file)

    print()

    print("Figures saved to:")
    print(recall_figure)
    print(confusion_figure)
    print(comparison_figure)
    print(delay_figure)

    print("=" * 82)


if __name__ == "__main__":
    main()