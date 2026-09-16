from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def main():

    # ======================================================
    # FILES
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    solar_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_006_solar_degradation.csv"
    )

    battery_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_007_battery_degradation.csv"
    )

    thermal_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_008_thermal_anomaly.csv"
    )

    wheel_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_009_reaction_wheel_degradation.csv"
    )

    sensor_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_010_sensor_drift.csv"
    )

    dropout_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_011_telemetry_dropout.csv"
    )

    required_files = [
        nominal_file,
        solar_file,
        battery_file,
        thermal_file,
        wheel_file,
        sensor_file,
        dropout_file,
    ]

    for path in required_files:
        require_file(path)

    # ======================================================
    # LOAD DATA
    # ======================================================

    nominal_df = pd.read_csv(nominal_file)

    solar_df = pd.read_csv(solar_file)
    battery_df = pd.read_csv(battery_file)
    thermal_df = pd.read_csv(thermal_file)
    wheel_df = pd.read_csv(wheel_file)
    sensor_df = pd.read_csv(sensor_file)
    dropout_df = pd.read_csv(dropout_file)

    # ======================================================
    # VALIDATION METRICS
    # ======================================================

    results = []

    # ------------------------------------------------------
    # 006 SOLAR ARRAY DEGRADATION
    # ------------------------------------------------------

    solar_fault_mask = solar_df["time_h"] >= 2.5

    nominal_solar_after = nominal_df.loc[
        solar_fault_mask,
        "solar_power_w"
    ].mean()

    faulty_solar_after = solar_df.loc[
        solar_fault_mask,
        "solar_power_w"
    ].mean()

    observed_solar_reduction_percent = (
        100.0
        * (
            nominal_solar_after
            - faulty_solar_after
        )
        / nominal_solar_after
    )

    solar_final_soc_drop = (
        nominal_df["battery_soc_true"].iloc[-1]
        - solar_df["battery_soc_true"].iloc[-1]
    )

    solar_pass = (
        observed_solar_reduction_percent > 25.0
        and solar_final_soc_drop > 5.0
    )

    results.append(
        {
            "experiment": "006",
            "fault": "solar_array_degradation",
            "main_metric": observed_solar_reduction_percent,
            "metric_label": "solar reduction [%]",
            "secondary_metric": solar_final_soc_drop,
            "secondary_label": "final SOC drop [points]",
            "pass": solar_pass,
        }
    )

    # ------------------------------------------------------
    # 007 BATTERY DEGRADATION
    # ------------------------------------------------------

    battery_final_soc_drop = (
        nominal_df["battery_soc_true"].iloc[-1]
        - battery_df["battery_soc_true"].iloc[-1]
    )

    battery_max_solar_difference = (
        (
            battery_df["solar_power_w"]
            - nominal_df["solar_power_w"]
        )
        .abs()
        .max()
    )

    battery_pass = (
        battery_final_soc_drop > 5.0
        and battery_max_solar_difference < 1e-9
    )

    results.append(
        {
            "experiment": "007",
            "fault": "battery_degradation",
            "main_metric": battery_final_soc_drop,
            "metric_label": "final SOC drop [points]",
            "secondary_metric": battery_max_solar_difference,
            "secondary_label": "max solar diff [W]",
            "pass": battery_pass,
        }
    )

    # ------------------------------------------------------
    # 008 THERMAL ANOMALY
    # ------------------------------------------------------

    thermal_fault_mask = thermal_df["time_h"] >= 2.5

    battery_temp_rise = (
        thermal_df.loc[
            thermal_fault_mask,
            "battery_temp_true"
        ].mean()
        -
        nominal_df.loc[
            thermal_fault_mask,
            "battery_temp_true"
        ].mean()
    )

    electronics_temp_rise = (
        thermal_df.loc[
            thermal_fault_mask,
            "electronics_temp_true"
        ].mean()
        -
        nominal_df.loc[
            thermal_fault_mask,
            "electronics_temp_true"
        ].mean()
    )

    thermal_pass = (
        battery_temp_rise > 1.0
        and electronics_temp_rise > 1.0
    )

    results.append(
        {
            "experiment": "008",
            "fault": "thermal_anomaly",
            "main_metric": battery_temp_rise,
            "metric_label": "battery temp rise [C]",
            "secondary_metric": electronics_temp_rise,
            "secondary_label": "electronics temp rise [C]",
            "pass": thermal_pass,
        }
    )

    # ------------------------------------------------------
    # 009 REACTION-WHEEL DEGRADATION
    # ------------------------------------------------------

    wheel_fault_mask = wheel_df["time_h"] >= 2.5

    wheel_current_increase = (
        wheel_df.loc[
            wheel_fault_mask,
            "wheel_current_true"
        ].mean()
        -
        nominal_df.loc[
            wheel_fault_mask,
            "wheel_current_true"
        ].mean()
    )

    wheel_temp_increase = (
        wheel_df.loc[
            wheel_fault_mask,
            "wheel_temp_true"
        ].mean()
        -
        nominal_df.loc[
            wheel_fault_mask,
            "wheel_temp_true"
        ].mean()
    )

    wheel_pass = (
        wheel_current_increase > 0.0
        and wheel_temp_increase > 0.0
    )

    results.append(
        {
            "experiment": "009",
            "fault": "reaction_wheel_degradation",
            "main_metric": wheel_current_increase,
            "metric_label": "wheel current increase [A]",
            "secondary_metric": wheel_temp_increase,
            "secondary_label": "wheel temp rise [C]",
            "pass": wheel_pass,
        }
    )

    # ------------------------------------------------------
    # 010 SENSOR DRIFT
    # ------------------------------------------------------

    sensor_fault_mask = sensor_df["time_h"] >= 2.5

    mean_drift_residual = sensor_df.loc[
        sensor_fault_mask,
        "voltage_residual"
    ].mean()

    expected_bias = 0.60 * 0.50

    sensor_pass = (
        mean_drift_residual > 0.25
        and abs(
            mean_drift_residual - expected_bias
        ) < 0.05
    )

    results.append(
        {
            "experiment": "010",
            "fault": "battery_voltage_sensor_drift",
            "main_metric": mean_drift_residual,
            "metric_label": "mean drift residual [V]",
            "secondary_metric": expected_bias,
            "secondary_label": "expected bias [V]",
            "pass": sensor_pass,
        }
    )

    # ------------------------------------------------------
    # 011 TELEMETRY DROPOUT
    # ------------------------------------------------------

    missing_before = dropout_df.loc[
        dropout_df["time_h"] < 3.0,
        "missing_measurements"
    ].sum()

    missing_after = dropout_df.loc[
        dropout_df["time_h"] >= 3.0,
        "missing_measurements"
    ].sum()

    max_missing_channels = (
        dropout_df["missing_measurements"].max()
    )

    dropout_pass = (
        missing_before == 0
        and missing_after > 0
        and max_missing_channels == 8
    )

    results.append(
        {
            "experiment": "011",
            "fault": "telemetry_dropout",
            "main_metric": missing_after,
            "metric_label": "missing after fault",
            "secondary_metric": max_missing_channels,
            "secondary_label": "max missing/sample",
            "pass": dropout_pass,
        }
    )

    # ======================================================
    # SUMMARY TABLE
    # ======================================================

    results_df = pd.DataFrame(results)

    # ======================================================
    # SAVE SUMMARY CSV
    # ======================================================

    summary_csv = (
        project_root
        / "results"
        / "tables"
        / "experiment_012_phase2_validation_summary.csv"
    )

    summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        summary_csv,
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

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "experiment_012_phase2_validation.png"
    )

    numeric_pass = results_df["pass"].astype(int)

    plt.figure(figsize=(10, 5))
    plt.bar(
        results_df["experiment"] + " - " + results_df["fault"],
        numeric_pass,
    )

    plt.ylim(-0.1, 1.1)
    plt.ylabel("Validation Pass")
    plt.xlabel("Experiment")
    plt.title("SENTINEL-XAI - Phase 2 Validation Summary")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # GLOBAL VALIDATION
    # ======================================================

    total_pass = int(results_df["pass"].sum())
    total_tests = len(results_df)
    global_pass = bool(results_df["pass"].all())

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 76)
    print("SENTINEL-XAI - EXPERIMENT 012")
    print("PHASE 2 FINAL VALIDATION")
    print("=" * 76)
    print()

    for _, row in results_df.iterrows():

        print(
            f"Experiment {row['experiment']} | "
            f"{row['fault']}"
        )

        print(
            f"  {row['metric_label']}: "
            f"{row['main_metric']:.4f}"
        )

        print(
            f"  {row['secondary_label']}: "
            f"{row['secondary_metric']:.4f}"
        )

        print(
            f"  PASS: {bool(row['pass'])}"
        )

        print()

    print("-" * 76)

    print(
        f"Passed checks: {total_pass}/{total_tests}"
    )

    print(
        f"GLOBAL PHASE 2 VALIDATION: {global_pass}"
    )

    print("-" * 76)
    print()

    print("Summary CSV saved to:")
    print(summary_csv)
    print()

    print("Figure saved to:")
    print(figure_path)

    print("=" * 76)


if __name__ == "__main__":
    main()