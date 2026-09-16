from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

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

    # ======================================================
    # EXPERIMENT CONFIGURATION
    # ======================================================

    duration_hours = 6.0
    dt_s = 60.0

    fault_start_h = 2.0
    ramp_duration_h = 0.5
    requested_severity = 0.35

    battery_fault = FaultConfig(
        fault_type="battery_degradation",
        start_time_h=fault_start_h,
        severity=requested_severity,
        profile="gradual",
        ramp_duration_h=ramp_duration_h,
    )

    # ======================================================
    # RUN NOMINAL CASE
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
    # RUN FAULTY CASE
    # ======================================================

    print(
        "Running spacecraft with battery degradation..."
    )

    faulty = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=[
            battery_fault,
        ],
    )

    nominal_df = pd.DataFrame(
        nominal
    )

    faulty_df = pd.DataFrame(
        faulty
    )

    # ======================================================
    # SAVE FAULT DATASET
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "faults"
        / "experiment_007_battery_degradation.csv"
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

    nominal_final_soc = (
        nominal_df[
            "battery_soc_true"
        ].iloc[-1]
    )

    faulty_final_soc = (
        faulty_df[
            "battery_soc_true"
        ].iloc[-1]
    )

    final_soc_difference = (
        faulty_final_soc
        - nominal_final_soc
    )

    nominal_min_soc = (
        nominal_df[
            "battery_soc_true"
        ].min()
    )

    faulty_min_soc = (
        faulty_df[
            "battery_soc_true"
        ].min()
    )

    solar_difference = (
        faulty_df["solar_power_w"]
        - nominal_df["solar_power_w"]
    )

    max_solar_difference = (
        solar_difference.abs().max()
    )

    # ======================================================
    # FIGURES DIRECTORY
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
    # FIGURE 1 - BATTERY SOC
    # ======================================================

    soc_figure = (
        figures_dir
        / "experiment_007_battery_soc.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        nominal_df["time_h"],
        nominal_df["battery_soc_true"],
        label="Nominal SOC",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["battery_soc_true"],
        label="Degraded Battery SOC",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Fault Start",
    )

    plt.axvline(
        fault_start_h + ramp_duration_h,
        linestyle=":",
        label="Full Severity",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Battery SOC [%]"
    )

    plt.title(
        "SENTINEL-XAI - Battery Degradation Impact"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        soc_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 - BATTERY VOLTAGE
    # ======================================================

    voltage_figure = (
        figures_dir
        / "experiment_007_battery_voltage.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        nominal_df["time_h"],
        nominal_df["battery_voltage_true"],
        label="Nominal Voltage",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["battery_voltage_true"],
        label="Degraded Battery Voltage",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Fault Start",
    )

    plt.axvline(
        fault_start_h + ramp_duration_h,
        linestyle=":",
        label="Full Severity",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Battery Voltage [V]"
    )

    plt.title(
        "SENTINEL-XAI - Battery Voltage Under Degradation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        voltage_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 - FAULT SEVERITY
    # ======================================================

    severity_figure = (
        figures_dir
        / "experiment_007_battery_fault_severity.png"
    )

    plt.figure(
        figsize=(11, 4)
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["battery_degradation"]
        * 100.0,
        label="Battery Degradation",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Degradation [%]"
    )

    plt.title(
        "SENTINEL-XAI - Battery Fault Progression"
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
    print("SENTINEL-XAI - EXPERIMENT 007")
    print("BATTERY DEGRADATION")
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
        f"Requested degradation: "
        f"{requested_severity * 100:.1f} %"
    )

    print()

    print(
        f"Nominal final SOC: "
        f"{nominal_final_soc:.2f} %"
    )

    print(
        f"Degraded final SOC: "
        f"{faulty_final_soc:.2f} %"
    )

    print(
        f"Final SOC difference: "
        f"{final_soc_difference:.2f} percentage points"
    )

    print()

    print(
        f"Nominal minimum SOC: "
        f"{nominal_min_soc:.2f} %"
    )

    print(
        f"Degraded minimum SOC: "
        f"{faulty_min_soc:.2f} %"
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
    print(soc_figure)
    print(voltage_figure)
    print(severity_figure)

    print("=" * 72)


if __name__ == "__main__":
    main()