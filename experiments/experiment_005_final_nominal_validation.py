from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root),
)

from src.simulation.spacecraft_simulator import (
    run_simulation,
)


def check(
    condition,
    name,
):
    if condition:
        print(f"[PASS] {name}")
        return True

    print(f"[FAIL] {name}")
    return False


def main():

    # ======================================================
    # 1. RUN LONG NOMINAL SIMULATION
    # ======================================================

    duration_hours = 24.0
    dt_s = 60.0

    print()
    print("Running 24-hour unified nominal simulation...")

    telemetry = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
    )

    df = pd.DataFrame(
        telemetry
    )

    # ======================================================
    # 2. SAVE 24-HOUR DATASET
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry_24h.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ======================================================
    # 3. SENSOR RESIDUALS
    #
    # residual = measured - true
    # ======================================================

    df["battery_voltage_residual"] = (
        df["battery_voltage_measured"]
        - df["battery_voltage_true"]
    )

    df["battery_current_residual"] = (
        df["battery_current_measured"]
        - df["battery_current_true"]
    )

    df["battery_soc_residual"] = (
        df["battery_soc_measured"]
        - df["battery_soc_true"]
    )

    df["battery_temp_residual"] = (
        df["battery_temp_measured"]
        - df["battery_temp_true"]
    )

    df["electronics_temp_residual"] = (
        df["electronics_temp_measured"]
        - df["electronics_temp_true"]
    )

    df["wheel_speed_residual"] = (
        df["wheel_speed_measured"]
        - df["wheel_speed_true"]
    )

    df["wheel_current_residual"] = (
        df["wheel_current_measured"]
        - df["wheel_current_true"]
    )

    df["wheel_temp_residual"] = (
        df["wheel_temp_measured"]
        - df["wheel_temp_true"]
    )

    # ======================================================
    # 4. VALIDATION CHECKS
    # ======================================================

    print()
    print("=" * 72)
    print("SENTINEL-XAI - EXPERIMENT 005")
    print("FINAL NOMINAL SPACECRAFT VALIDATION")
    print("=" * 72)

    expected_samples = int(
        duration_hours * 3600 / dt_s
    ) + 1

    results = []

    results.append(
        check(
            len(df) == expected_samples,
            f"Expected sample count ({expected_samples})",
        )
    )

    results.append(
        check(
            not df.isna().any().any(),
            "No NaN / missing telemetry",
        )
    )

    results.append(
        check(
            set(df["fault_label"].unique())
            == {"nominal"},
            "All health labels are nominal",
        )
    )

    # ------------------------------------------------------
    # Broad simulator sanity bounds.
    #
    # These are NOT spacecraft operational limits.
    # They only detect obvious simulation failures.
    # ------------------------------------------------------

    results.append(
        check(
            df["battery_soc_true"].between(
                0.0,
                100.0,
            ).all(),
            "Battery SOC remains between 0 and 100 %",
        )
    )

    results.append(
        check(
            df["battery_temp_true"].between(
                10.0,
                45.0,
            ).all(),
            "Battery temperature remains bounded",
        )
    )

    results.append(
        check(
            df["electronics_temp_true"].between(
                10.0,
                50.0,
            ).all(),
            "Electronics temperature remains bounded",
        )
    )

    results.append(
        check(
            df["wheel_speed_true"].between(
                2000.0,
                4000.0,
            ).all(),
            "Reaction-wheel speed remains bounded",
        )
    )

    results.append(
        check(
            df["wheel_current_true"].between(
                0.0,
                1.0,
            ).all(),
            "Reaction-wheel current remains bounded",
        )
    )

    # ======================================================
    # 5. RESIDUAL STATISTICS
    # ======================================================

    print()
    print("-" * 72)
    print("SENSOR RESIDUAL STATISTICS")
    print("-" * 72)

    residual_columns = {
        "Battery voltage [V]":
            "battery_voltage_residual",

        "Battery current [A]":
            "battery_current_residual",

        "Battery SOC [%]":
            "battery_soc_residual",

        "Battery temperature [C]":
            "battery_temp_residual",

        "Electronics temperature [C]":
            "electronics_temp_residual",

        "Wheel speed [RPM]":
            "wheel_speed_residual",

        "Wheel current [A]":
            "wheel_current_residual",

        "Wheel temperature [C]":
            "wheel_temp_residual",
    }

    for name, column in residual_columns.items():

        mean_error = df[column].mean()
        std_error = df[column].std()

        print(
            f"{name:<30}"
            f" mean = {mean_error:>9.4f}"
            f"   std = {std_error:>9.4f}"
        )

    # ======================================================
    # 6. PHYSICAL SUMMARY
    # ======================================================

    print()
    print("-" * 72)
    print("24-HOUR PHYSICAL SUMMARY")
    print("-" * 72)

    print(
        f"Battery SOC: "
        f"{df['battery_soc_true'].min():.2f} "
        f"to "
        f"{df['battery_soc_true'].max():.2f} %"
    )

    print(
        f"Battery temperature: "
        f"{df['battery_temp_true'].min():.2f} "
        f"to "
        f"{df['battery_temp_true'].max():.2f} C"
    )

    print(
        f"Electronics temperature: "
        f"{df['electronics_temp_true'].min():.2f} "
        f"to "
        f"{df['electronics_temp_true'].max():.2f} C"
    )

    print(
        f"Wheel speed: "
        f"{df['wheel_speed_true'].min():.1f} "
        f"to "
        f"{df['wheel_speed_true'].max():.1f} RPM"
    )

    print(
        f"Wheel temperature: "
        f"{df['wheel_temp_true'].min():.2f} "
        f"to "
        f"{df['wheel_temp_true'].max():.2f} C"
    )

    # ======================================================
    # 7. FINAL VALIDATION FIGURE
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
        / "experiment_005_sensor_residuals.png"
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    axes[0].plot(
        df["time_h"],
        df["battery_voltage_residual"],
    )

    axes[0].set_ylabel(
        "Voltage residual [V]"
    )

    axes[0].grid(True)

    axes[1].plot(
        df["time_h"],
        df["battery_temp_residual"],
    )

    axes[1].set_ylabel(
        "Battery temp residual [C]"
    )

    axes[1].grid(True)

    axes[2].plot(
        df["time_h"],
        df["wheel_speed_residual"],
    )

    axes[2].set_ylabel(
        "Wheel speed residual [RPM]"
    )

    axes[2].set_xlabel(
        "Time [hours]"
    )

    axes[2].grid(True)

    fig.suptitle(
        "SENTINEL-XAI - Nominal Sensor Measurement Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        output_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FINAL RESULT
    # ======================================================

    print()
    print("=" * 72)

    if all(results):
        print(
            "FINAL RESULT: PHASE 1 NOMINAL VALIDATION PASSED"
        )
    else:
        print(
            "FINAL RESULT: VALIDATION FAILURE - REVIEW REQUIRED"
        )

    print("=" * 72)

    print()
    print("24-hour telemetry saved to:")
    print(output_csv)

    print()
    print("Validation figure saved to:")
    print(output_figure)


if __name__ == "__main__":
    main()