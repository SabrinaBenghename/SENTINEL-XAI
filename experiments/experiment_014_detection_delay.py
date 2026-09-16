from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


def main():

    # ======================================================
    # LOAD RULE-BASED PREDICTIONS
    # ======================================================

    predictions_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_013_rule_based_predictions.csv"
    )

    df = pd.read_csv(
        predictions_file
    )

    # ======================================================
    # SCHEDULED FAULT START TIMES
    # ======================================================

    fault_start_times = {
        "solar_array_degradation": 2.0,
        "battery_degradation": 2.0,
        "thermal_anomaly": 2.0,
        "reaction_wheel_degradation": 2.0,
        "battery_voltage_sensor_drift": 2.0,
        "telemetry_dropout": 3.0,
    }

    results = []

    # ======================================================
    # CALCULATE FIRST CORRECT DETECTION
    # ======================================================

    for fault_type, start_h in fault_start_times.items():

        subset = df[
            df["dataset"]
            == fault_type
        ].copy()

        after_start = subset[
            subset["time_h"]
            >= start_h
        ]

        correct_detections = after_start[
            after_start["predicted_label"]
            == fault_type
        ]

        if len(correct_detections) > 0:

            first_detection_h = (
                correct_detections[
                    "time_h"
                ].iloc[0]
            )

            delay_h = (
                first_detection_h
                - start_h
            )

            delay_min = (
                delay_h * 60.0
            )

            detected = True

        else:

            first_detection_h = float("nan")
            delay_h = float("nan")
            delay_min = float("nan")

            detected = False

        results.append(
            {
                "fault_type":
                    fault_type,

                "fault_start_h":
                    start_h,

                "first_detection_h":
                    first_detection_h,

                "detection_delay_h":
                    delay_h,

                "detection_delay_min":
                    delay_min,

                "detected":
                    detected,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # ======================================================
    # SAVE TABLE
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

    output_csv = (
        tables_dir
        / "experiment_014_detection_delay.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    # ======================================================
    # PLOT
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

    output_figure = (
        figures_dir
        / "experiment_014_detection_delay.png"
    )

    plot_df = results_df[
        results_df["detected"]
        == True
    ]

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        plot_df["fault_type"],
        plot_df["detection_delay_min"],
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.ylabel(
        "Detection Delay [minutes]"
    )

    plt.title(
        "SENTINEL-XAI - Classical Rule-Based Detection Delay"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 76)

    print(
        "SENTINEL-XAI - EXPERIMENT 014"
    )

    print(
        "CLASSICAL RULE-BASED DETECTION DELAY"
    )

    print("=" * 76)

    for _, row in results_df.iterrows():

        print()
        print(
            row["fault_type"]
        )

        print(
            f"  Fault start: "
            f"{row['fault_start_h']:.2f} h"
        )

        if row["detected"]:

            print(
                f"  First correct detection: "
                f"{row['first_detection_h']:.2f} h"
            )

            print(
                f"  Detection delay: "
                f"{row['detection_delay_min']:.1f} min"
            )

        else:

            print(
                "  Detection delay: NOT DETECTED"
            )

    detected_rows = results_df[
        results_df["detected"]
        == True
    ]

    if len(detected_rows) > 0:

        mean_delay = (
            detected_rows[
                "detection_delay_min"
            ].mean()
        )

        print()
        print("-" * 76)

        print(
            f"Mean delay among detected faults: "
            f"{mean_delay:.1f} min"
        )

    print()
    print("Table saved to:")
    print(output_csv)

    print()
    print("Figure saved to:")
    print(output_figure)

    print("=" * 76)


if __name__ == "__main__":
    main()