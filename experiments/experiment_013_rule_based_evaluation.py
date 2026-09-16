from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.detection.rule_based_detector import (
    RuleBasedDetector,
)


def evaluate_dataset(
    detector,
    dataframe,
    dataset_name,
):
    records = []

    for _, row in dataframe.iterrows():

        result = detector.detect(row)

        true_label = row.get(
            "fault_label",
            "nominal",
        )

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
                "dataset": dataset_name,
                "time_h": row["time_h"],

                "true_label": true_label,
                "predicted_label": predicted_label,

                "true_fault": int(true_fault),
                "predicted_fault": int(predicted_fault),

                "confidence": result.confidence,
            }
        )

    return pd.DataFrame(records)


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():

    detector = RuleBasedDetector()

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
    print("Running classical detector on all datasets...")

    for dataset_name, file_path in files.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset:\n{file_path}"
            )

        df = pd.read_csv(
            file_path
        )

        result_df = evaluate_dataset(
            detector=detector,
            dataframe=df,
            dataset_name=dataset_name,
        )

        all_results.append(
            result_df
        )

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ======================================================
    # BINARY FAULT-DETECTION METRICS
    # ======================================================

    tp = (
        (
            results_df["true_fault"] == 1
        )
        &
        (
            results_df["predicted_fault"] == 1
        )
    ).sum()

    tn = (
        (
            results_df["true_fault"] == 0
        )
        &
        (
            results_df["predicted_fault"] == 0
        )
    ).sum()

    fp = (
        (
            results_df["true_fault"] == 0
        )
        &
        (
            results_df["predicted_fault"] == 1
        )
    ).sum()

    fn = (
        (
            results_df["true_fault"] == 1
        )
        &
        (
            results_df["predicted_fault"] == 0
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

    # ======================================================
    # MULTI-CLASS FAULT CLASSIFICATION ACCURACY
    # ======================================================

    fault_rows = results_df[
        results_df["true_fault"] == 1
    ]

    correct_fault_classification = (
        fault_rows["true_label"]
        ==
        fault_rows["predicted_label"]
    ).sum()

    fault_classification_accuracy = (
        safe_divide(
            correct_fault_classification,
            len(fault_rows),
        )
    )

    # ======================================================
    # PER-FAULT RECALL
    # ======================================================

    per_fault_rows = []

    fault_types = [
        "solar_array_degradation",
        "battery_degradation",
        "thermal_anomaly",
        "reaction_wheel_degradation",
        "battery_voltage_sensor_drift",
        "telemetry_dropout",
    ]

    for fault_type in fault_types:

        subset = results_df[
            results_df["true_label"]
            == fault_type
        ]

        detected_any_fault = (
            subset["predicted_fault"]
            == 1
        ).sum()

        correctly_classified = (
            subset["predicted_label"]
            == fault_type
        ).sum()

        detection_recall = safe_divide(
            detected_any_fault,
            len(subset),
        )

        classification_recall = safe_divide(
            correctly_classified,
            len(subset),
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
    # SAVE RESULTS
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

    results_file = (
        tables_dir
        / "experiment_013_rule_based_predictions.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_013_per_fault_metrics.csv"
    )

    confusion_file = (
        tables_dir
        / "experiment_013_confusion_matrix.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    confusion.to_csv(
        confusion_file,
    )

    # ======================================================
    # FIGURE 1 - PER-FAULT CLASSIFICATION RECALL
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

    recall_figure = (
        figures_dir
        / "experiment_013_per_fault_recall.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        per_fault_df["fault_type"],
        per_fault_df["classification_recall"],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Classification Recall"
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.title(
        "SENTINEL-XAI - Classical Rule-Based Fault Recall"
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
        / "experiment_013_confusion_matrix.png"
    )

    matrix = confusion.values

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
        "SENTINEL-XAI - Rule-Based Confusion Matrix"
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
                str(matrix[i, j]),
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
    # PRINT SUMMARY
    # ======================================================

    print()
    print("=" * 76)
    print("SENTINEL-XAI - EXPERIMENT 013")
    print("CLASSICAL RULE-BASED DETECTION EVALUATION")
    print("=" * 76)

    print()
    print("Binary fault detection:")
    print()

    print(
        f"True positives:  {tp}"
    )

    print(
        f"True negatives:  {tn}"
    )

    print(
        f"False positives: {fp}"
    )

    print(
        f"False negatives: {fn}"
    )

    print()

    print(
        f"Accuracy:  "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1 score:  "
        f"{f1:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{false_positive_rate:.4f}"
    )

    print()

    print(
        f"Fault-type classification accuracy: "
        f"{fault_classification_accuracy:.4f}"
    )

    print()

    print("-" * 76)
    print("PER-FAULT RESULTS")
    print("-" * 76)

    for _, row in per_fault_df.iterrows():

        print()

        print(
            row["fault_type"]
        )

        print(
            f"  Samples: "
            f"{int(row['samples'])}"
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
    print("=" * 76)

    print()
    print("Figures saved to:")
    print(recall_figure)
    print(confusion_figure)

    print()
    print("Tables saved to:")
    print(results_file)
    print(per_fault_file)
    print(confusion_file)

    print("=" * 76)


if __name__ == "__main__":
    main()