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


from src.simulation.spacecraft_simulator import (
    run_simulation,
)

from src.faults.fault_manager import (
    FaultConfig,
)


def main():

    duration_hours = 6.0
    dt_s = 60.0

    fault_start_h = 2.0
    ramp_duration_h = 0.5
    requested_severity = 0.60

    thermal_fault = FaultConfig(
        fault_type="thermal_anomaly",
        start_time_h=fault_start_h,
        severity=requested_severity,
        profile="gradual",
        ramp_duration_h=ramp_duration_h,
    )

    # ======================================================
    # RUN NOMINAL
    # ======================================================

    print()
    print("Running nominal spacecraft...")

    nominal = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=None,
    )

    # ======================================================
    # RUN FAULTY
    # ======================================================

    print(
        "Running spacecraft with thermal anomaly..."
    )

    faulty = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=[
            thermal_fault,
        ],
    )

    nominal_df = pd.DataFrame(
        nominal
    )

    faulty_df = pd.DataFrame(
        faulty
    )

    # ======================================================
    # SAVE DATA
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "faults"
        / "experiment_008_thermal_anomaly.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faulty_df.to_csv(
        output_csv,
        index=False,
    )

    # ======================================================
    # ANALYSIS
    # ======================================================

    full_fault_time_h = (
        fault_start_h
        + ramp_duration_h
    )

    fault_mask = (
        faulty_df["time_h"]
        >= full_fault_time_h
    )

    nominal_battery_temp_mean = (
        nominal_df.loc[
            fault_mask,
            "battery_temp_true",
        ].mean()
    )

    faulty_battery_temp_mean = (
        faulty_df.loc[
            fault_mask,
            "battery_temp_true",
        ].mean()
    )

    battery_temp_difference = (
        faulty_battery_temp_mean
        - nominal_battery_temp_mean
    )

    nominal_electronics_temp_mean = (
        nominal_df.loc[
            fault_mask,
            "electronics_temp_true",
        ].mean()
    )

    faulty_electronics_temp_mean = (
        faulty_df.loc[
            fault_mask,
            "electronics_temp_true",
        ].mean()
    )

    electronics_temp_difference = (
        faulty_electronics_temp_mean
        - nominal_electronics_temp_mean
    )

    max_solar_difference = (
        (
            faulty_df["solar_power_w"]
            - nominal_df["solar_power_w"]
        )
        .abs()
        .max()
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
    # FIGURE 1 - BATTERY TEMPERATURE
    # ======================================================

    battery_figure = (
        figures_dir
        / "experiment_008_battery_temperature.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        nominal_df["time_h"],
        nominal_df["battery_temp_true"],
        label="Nominal Battery Temperature",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["battery_temp_true"],
        label="Thermal-Anomaly Battery Temperature",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Fault Start",
    )

    plt.axvline(
        full_fault_time_h,
        linestyle=":",
        label="Full Severity",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Battery Temperature [C]")

    plt.title(
        "SENTINEL-XAI - Battery Thermal Anomaly"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        battery_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 - ELECTRONICS TEMPERATURE
    # ======================================================

    electronics_figure = (
        figures_dir
        / "experiment_008_electronics_temperature.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        nominal_df["time_h"],
        nominal_df["electronics_temp_true"],
        label="Nominal Electronics Temperature",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["electronics_temp_true"],
        label="Thermal-Anomaly Electronics Temperature",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Fault Start",
    )

    plt.axvline(
        full_fault_time_h,
        linestyle=":",
        label="Full Severity",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Electronics Temperature [C]")

    plt.title(
        "SENTINEL-XAI - Electronics Thermal Anomaly"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        electronics_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 - FAULT PROGRESSION
    # ======================================================

    severity_figure = (
        figures_dir
        / "experiment_008_thermal_fault_severity.png"
    )

    plt.figure(
        figsize=(11, 4)
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["thermal_anomaly"]
        * 100.0,
        label="Thermal Anomaly Severity",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Severity [%]")

    plt.title(
        "SENTINEL-XAI - Thermal Fault Progression"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        severity_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 72)
    print("SENTINEL-XAI - EXPERIMENT 008")
    print("THERMAL ANOMALY")
    print("=" * 72)

    print(
        f"Fault start: "
        f"{fault_start_h:.2f} h"
    )

    print(
        f"Ramp duration: "
        f"{ramp_duration_h:.2f} h"
    )

    print(
        f"Requested severity: "
        f"{requested_severity * 100:.1f} %"
    )

    print()

    print(
        f"Mean battery-temperature increase "
        f"after full severity: "
        f"{battery_temp_difference:.2f} C"
    )

    print(
        f"Mean electronics-temperature increase "
        f"after full severity: "
        f"{electronics_temp_difference:.2f} C"
    )

    print()

    print(
        f"Maximum solar-power difference: "
        f"{max_solar_difference:.6f} W"
    )

    print(
        f"Active fault samples: "
        f"{faulty_df['fault_active'].sum()}"
    )

    print()

    print("Fault telemetry saved to:")
    print(output_csv)

    print()

    print("Figures saved to:")
    print(battery_figure)
    print(electronics_figure)
    print(severity_figure)

    print("=" * 72)


if __name__ == "__main__":
    main()