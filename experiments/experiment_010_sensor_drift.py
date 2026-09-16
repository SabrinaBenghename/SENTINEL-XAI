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
    FaultManager,
)

from src.faults.sensor_faults import (
    apply_battery_voltage_drift,
)


def main():

    duration_hours = 6.0
    dt_s = 60.0

    fault_start_h = 2.0
    ramp_duration_h = 0.5
    requested_severity = 0.60

    drift_fault = FaultConfig(
        fault_type="battery_voltage_sensor_drift",
        start_time_h=fault_start_h,
        severity=requested_severity,
        profile="gradual",
        ramp_duration_h=ramp_duration_h,
    )

    manager = FaultManager(
        faults=[
            drift_fault,
        ]
    )

    print()
    print("Running healthy spacecraft with drifting voltage sensor...")

    telemetry = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=None,
    )

    df = pd.DataFrame(
        telemetry
    )

    # Keep original healthy measurement
    df["battery_voltage_measured_nominal"] = (
        df["battery_voltage_measured"]
    )

    severities = []
    drifted_measurements = []

    for index, row in df.iterrows():

        state = manager.get_state(
            fault=drift_fault,
            time_s=row["time_s"],
        )

        severity = (
            state.effective_severity
        )

        severities.append(
            severity
        )

        drifted_value = (
            apply_battery_voltage_drift(
                measured_voltage_v=row[
                    "battery_voltage_measured"
                ],
                severity=severity,
                max_bias_v=0.50,
            )
        )

        drifted_measurements.append(
            drifted_value
        )

    df["sensor_drift"] = severities

    df["battery_voltage_measured"] = (
        drifted_measurements
    )

    df["voltage_residual"] = (
        df["battery_voltage_measured"]
        - df["battery_voltage_true"]
    )

    df["fault_active"] = (
        df["sensor_drift"] > 0.0
    ).astype(int)

    df["fault_type"] = (
        df["fault_active"]
        .map(
            {
                0: "none",
                1: "battery_voltage_sensor_drift",
            }
        )
    )

    df["fault_label"] = (
        df["fault_active"]
        .map(
            {
                0: "nominal",
                1: "battery_voltage_sensor_drift",
            }
        )
    )

    # ======================================================
    # SAVE
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "faults"
        / "experiment_010_sensor_drift.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ======================================================
    # ANALYSIS
    # ======================================================

    full_fault_h = (
        fault_start_h
        + ramp_duration_h
    )

    mask = (
        df["time_h"]
        >= full_fault_h
    )

    nominal_residual = (
        df.loc[
            df["time_h"] < fault_start_h,
            "battery_voltage_measured_nominal",
        ]
        -
        df.loc[
            df["time_h"] < fault_start_h,
            "battery_voltage_true",
        ]
    )

    fault_residual = (
        df.loc[
            mask,
            "voltage_residual",
        ]
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

    figure = (
        figures_dir
        / "experiment_010_sensor_drift.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        df["time_h"],
        df["battery_voltage_true"],
        label="True Battery Voltage",
    )

    plt.plot(
        df["time_h"],
        df["battery_voltage_measured"],
        label="Drifting Sensor Measurement",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Fault Start",
    )

    plt.axvline(
        full_fault_h,
        linestyle=":",
        label="Full Severity",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Battery Voltage [V]")

    plt.title(
        "SENTINEL-XAI - Battery Voltage Sensor Drift"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 72)
    print("SENTINEL-XAI - EXPERIMENT 010")
    print("BATTERY-VOLTAGE SENSOR DRIFT")
    print("=" * 72)

    print(
        f"Fault start: "
        f"{fault_start_h:.2f} h"
    )

    print(
        f"Requested severity: "
        f"{requested_severity * 100:.1f} %"
    )

    print()

    print(
        f"Mean nominal voltage residual: "
        f"{nominal_residual.mean():.4f} V"
    )

    print(
        f"Mean residual after full drift: "
        f"{fault_residual.mean():.4f} V"
    )

    print(
        f"Expected maximum drift bias: "
        f"{requested_severity * 0.50:.3f} V"
    )

    print()

    print(
        f"Active fault samples: "
        f"{df['fault_active'].sum()}"
    )

    print()

    print("Fault telemetry saved to:")
    print(output_csv)

    print()
    print("Figure saved to:")
    print(figure)

    print("=" * 72)


if __name__ == "__main__":
    main()