from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def main():

    # ======================================================
    # INPUT FILES
    # ======================================================

    predictions_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_013_rule_based_predictions.csv"
    )

    per_fault_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_013_per_fault_metrics.csv"
    )

    delay_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_014_detection_delay.csv"
    )

    require_file(predictions_file)
    require_file(per_fault_file)
    require_file(delay_file)

    predictions_df = pd.read_csv(predictions_file)
    per_fault_df = pd.read_csv(per_fault_file)
    delay_df = pd.read_csv(delay_file)

    # ======================================================
    # RECOMPUTE GLOBAL METRICS
    # ======================================================

    tp = (
        (
            predictions_df["true_fault"] == 1
        )
        &
        (
            predictions_df["predicted_fault"] == 1
        )
    ).sum()

    tn = (
        (
            predictions_df["true_fault"] == 0
        )
        &
        (
            predictions_df["predicted_fault"] == 0
        )
    ).sum()

    fp = (
        (
            predictions_df["true_fault"] == 0
        )
        &
        (
            predictions_df["predicted_fault"] == 1
        )
    ).sum()

    fn = (
        (
            predictions_df["true_fault"] == 1
        )
        &
        (
            predictions_df["predicted_fault"] == 0
        )
    ).sum()

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)

    fault_rows = predictions_df[
        predictions_df["true_fault"] == 1
    ]

    fault_classification_accuracy = safe_divide(
        (
            fault_rows["true_label"]
            == fault_rows["predicted_label"]
        ).sum(),
        len(fault_rows),
    )

    # ======================================================
    # HELPER LOOKUPS
    # ======================================================

    def get_classification_recall(fault_name):
        row = per_fault_df[
            per_fault_df["fault_type"] == fault_name
        ]

        if len(row) == 0:
            return float("nan")

        return float(
            row["classification_recall"].iloc[0]
        )

    def get_detection_recall(fault_name):
        row = per_fault_df[
            per_fault_df["fault_type"] == fault_name
        ]

        if len(row) == 0:
            return float("nan")

        return float(
            row["detection_recall"].iloc[0]
        )

    def get_delay_minutes(fault_name):
        row = delay_df[
            delay_df["fault_type"] == fault_name
        ]

        if len(row) == 0:
            return float("nan")

        return float(
            row["detection_delay_min"].iloc[0]
        )

    # ======================================================
    # VALIDATION CHECKS
    # ======================================================

    nominal_false_alarms = fp

    detected_fault_count = int(
        delay_df["detected"].sum()
    )

    nonnegative_delays = bool(
        (
            delay_df.loc[
                delay_df["detected"] == True,
                "detection_delay_min",
            ] >= 0.0
        ).all()
    )

    mean_detected_delay = float(
        delay_df.loc[
            delay_df["detected"] == True,
            "detection_delay_min",
        ].mean()
    )

    checks = [
        {
            "check": "Predictions file loaded",
            "value": 1.0,
            "criterion": "file exists",
            "pass": True,
        },
        {
            "check": "Per-fault metrics file loaded",
            "value": 1.0,
            "criterion": "file exists",
            "pass": True,
        },
        {
            "check": "Delay file loaded",
            "value": 1.0,
            "criterion": "file exists",
            "pass": True,
        },
        {
            "check": "Nominal false alarms",
            "value": nominal_false_alarms,
            "criterion": "== 0",
            "pass": nominal_false_alarms == 0,
        },
        {
            "check": "Global precision",
            "value": precision,
            "criterion": ">= 0.95",
            "pass": precision >= 0.95,
        },
        {
            "check": "Global recall",
            "value": recall,
            "criterion": ">= 0.50",
            "pass": recall >= 0.50,
        },
        {
            "check": "Global F1 score",
            "value": f1,
            "criterion": ">= 0.70",
            "pass": f1 >= 0.70,
        },
        {
            "check": "False-positive rate",
            "value": false_positive_rate,
            "criterion": "<= 0.05",
            "pass": false_positive_rate <= 0.05,
        },
        {
            "check": "Fault classification accuracy",
            "value": fault_classification_accuracy,
            "criterion": ">= 0.50",
            "pass": fault_classification_accuracy >= 0.50,
        },
        {
            "check": "Solar classification recall",
            "value": get_classification_recall(
                "solar_array_degradation"
            ),
            "criterion": ">= 0.50",
            "pass": get_classification_recall(
                "solar_array_degradation"
            ) >= 0.50,
        },
        {
            "check": "Battery classification recall",
            "value": get_classification_recall(
                "battery_degradation"
            ),
            "criterion": ">= 0.05",
            "pass": get_classification_recall(
                "battery_degradation"
            ) >= 0.05,
        },
        {
            "check": "Thermal classification recall",
            "value": get_classification_recall(
                "thermal_anomaly"
            ),
            "criterion": ">= 0.60",
            "pass": get_classification_recall(
                "thermal_anomaly"
            ) >= 0.60,
        },
        {
            "check": "Wheel classification recall",
            "value": get_classification_recall(
                "reaction_wheel_degradation"
            ),
            "criterion": ">= 0.30",
            "pass": get_classification_recall(
                "reaction_wheel_degradation"
            ) >= 0.30,
        },
        {
            "check": "Sensor-drift classification recall",
            "value": get_classification_recall(
                "battery_voltage_sensor_drift"
            ),
            "criterion": ">= 0.95",
            "pass": get_classification_recall(
                "battery_voltage_sensor_drift"
            ) >= 0.95,
        },
        {
            "check": "Telemetry-dropout classification recall",
            "value": get_classification_recall(
                "telemetry_dropout"
            ),
            "criterion": "== 1.00",
            "pass": get_classification_recall(
                "telemetry_dropout"
            ) == 1.0,
        },
        {
            "check": "All fault types detected at least once",
            "value": detected_fault_count,
            "criterion": "== 6",
            "pass": detected_fault_count == 6,
        },
        {
            "check": "Detection delays are nonnegative",
            "value": float(nonnegative_delays),
            "criterion": "True",
            "pass": nonnegative_delays,
        },
        {
            "check": "Mean detected delay [min]",
            "value": mean_detected_delay,
            "criterion": "<= 40",
            "pass": mean_detected_delay <= 40.0,
        },
    ]

    checks_df = pd.DataFrame(checks)

    # ======================================================
    # SAVE TABLE
    # ======================================================

    tables_dir = (
        project_root
        / "results"
        / "tables"
    )
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_file = (
        tables_dir
        / "experiment_015_phase3_validation_summary.csv"
    )

    checks_df.to_csv(
        summary_file,
        index=False,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_file = (
        figures_dir
        / "experiment_015_phase3_validation.png"
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        checks_df["check"],
        checks_df["pass"].astype(int),
    )

    plt.ylim(-0.1, 1.1)
    plt.ylabel("Validation Pass")
    plt.xlabel("Validation Check")
    plt.title("SENTINEL-XAI - Phase 3 Validation Summary")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # GLOBAL RESULT
    # ======================================================

    passed_checks = int(
        checks_df["pass"].sum()
    )

    total_checks = len(checks_df)

    global_validation = bool(
        checks_df["pass"].all()
    )

    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    print()
    print("=" * 80)
    print("SENTINEL-XAI - EXPERIMENT 015")
    print("PHASE 3 FINAL VALIDATION")
    print("=" * 80)
    print()

    print("Global metrics:")
    print(
        f"  Precision: {precision:.4f}"
    )
    print(
        f"  Recall: {recall:.4f}"
    )
    print(
        f"  F1 score: {f1:.4f}"
    )
    print(
        f"  False-positive rate: {false_positive_rate:.4f}"
    )
    print(
        f"  Fault classification accuracy: "
        f"{fault_classification_accuracy:.4f}"
    )
    print(
        f"  Mean detected delay: {mean_detected_delay:.1f} min"
    )

    print()
    print("-" * 80)
    print("VALIDATION CHECKS")
    print("-" * 80)

    for _, row in checks_df.iterrows():

        print(
            f"{row['check']}"
        )
        print(
            f"  Value: {row['value']}"
        )
        print(
            f"  Criterion: {row['criterion']}"
        )
        print(
            f"  PASS: {bool(row['pass'])}"
        )
        print()

    print("-" * 80)
    print(
        f"Passed checks: {passed_checks}/{total_checks}"
    )
    print(
        f"GLOBAL PHASE 3 VALIDATION: {global_validation}"
    )
    print("-" * 80)
    print()

    print("Summary CSV saved to:")
    print(summary_file)
    print()

    print("Figure saved to:")
    print(figure_file)
    print("=" * 80)


if __name__ == "__main__":
    main()