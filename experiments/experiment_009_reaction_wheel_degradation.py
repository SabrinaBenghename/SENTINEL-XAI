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

    wheel_fault = FaultConfig(
        fault_type="reaction_wheel_degradation",
        start_time_h=fault_start_h,
        severity=requested_severity,
        profile="gradual",
        ramp_duration_h=ramp_duration_h,
    )

    # ======================================================
    # NOMINAL RUN
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
    # FAULTY RUN
    # ======================================================

    print(
        "Running spacecraft with reaction-wheel degradation..."
    )

    faulty = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=[
            wheel_fault,
        ],
    )

    nominal_df = pd.DataFrame(nominal)
    faulty_df = pd.DataFrame(faulty)

    # ======================================================
    # SAVE DATA
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "faults"
        / "experiment_009_reaction_wheel_degradation.csv"
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

    nominal_current_mean = (
        nominal_df.loc[
            fault_mask,
            "wheel_current_true",
        ].mean()
    )

    faulty_current_mean = (
        faulty_df.loc[
            fault_mask,
            "wheel_current_true",
        ].mean()
    )

    current_increase = (
        faulty_current_mean
        - nominal_current_mean
    )

    nominal_temp_mean = (
        nominal_df.loc[
            fault_mask,
            "wheel_temp_true",
        ].mean()
    )

    faulty_temp_mean = (
        faulty_df.loc[
            fault_mask,
            "wheel_temp_true",
        ].mean()
    )

    temp_increase = (
        faulty_temp_mean
        - nominal_temp_mean
    )

    nominal_speed = (
        nominal_df[
            "wheel_speed_true"
        ]
    )

    faulty_speed = (
        faulty_df[
            "wheel_speed_true"
        ]
    )

    maximum_speed_difference = (
        (
            faulty_speed
            - nominal_speed
        )
        .abs()
        .max()
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

    # ------------------------------------------------------
    # FIGURE 1 - MOTOR CURRENT
    # ------------------------------------------------------

    current_figure = (
        figures_dir
        / "experiment_009_wheel_current.png"
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        nominal_df["time_h"],
        nominal_df["wheel_current_true"],
        label="Nominal Wheel Current",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["wheel_current_true"],
        label="Degraded Wheel Current",
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
    plt.ylabel("Motor Current [A]")

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Current Degradation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        current_figure,
        dpi=200,
    )

    plt.show()

    # ------------------------------------------------------
    # FIGURE 2 - WHEEL TEMPERATURE
    # ------------------------------------------------------

    temperature_figure = (
        figures_dir
        / "experiment_009_wheel_temperature.png"
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        nominal_df["time_h"],
        nominal_df["wheel_temp_true"],
        label="Nominal Wheel Temperature",
    )

    plt.plot(
        faulty_df["time_h"],
        faulty_df["wheel_temp_true"],
        label="Degraded Wheel Temperature",
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
    plt.ylabel("Wheel Temperature [C]")

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Thermal Impact"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        temperature_figure,
        dpi=200,
    )

    plt.show()

    # ------------------------------------------------------
    # FIGURE 3 - FAULT PROGRESSION
    # ------------------------------------------------------

    severity_figure = (
        figures_dir
        / "experiment_009_wheel_fault_severity.png"
    )

    plt.figure(figsize=(11, 4))

    plt.plot(
        faulty_df["time_h"],
        faulty_df["wheel_degradation"]
        * 100.0,
        label="Wheel Degradation",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Degradation [%]")

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Fault Progression"
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
    print("SENTINEL-XAI - EXPERIMENT 009")
    print("REACTION-WHEEL DEGRADATION")
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
        f"Nominal mean wheel current: "
        f"{nominal_current_mean:.4f} A"
    )

    print(
        f"Degraded mean wheel current: "
        f"{faulty_current_mean:.4f} A"
    )

    print(
        f"Mean current increase: "
        f"{current_increase:.4f} A"
    )

    print()

    print(
        f"Mean wheel-temperature increase: "
        f"{temp_increase:.2f} C"
    )

    print(
        f"Maximum wheel-speed difference: "
        f"{maximum_speed_difference:.3f} RPM"
    )

    print()

    print(
        f"Active fault samples: "
        f"{faulty_df['fault_active'].sum()}"
    )

    print()

    print("Fault telemetry saved to:")
    print(output_csv)

    print()

    print("Figures saved to:")
    print(current_figure)
    print(temperature_figure)
    print(severity_figure)

    print("=" * 72)


if __name__ == "__main__":
    main()