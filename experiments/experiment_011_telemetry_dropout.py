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
    apply_telemetry_dropout,
)


def main():

    duration_hours = 6.0
    dt_s = 60.0

    fault_start_h = 3.0

    dropout_fault = FaultConfig(
        fault_type="telemetry_dropout",
        start_time_h=fault_start_h,
        severity=1.0,
        profile="abrupt",
    )

    manager = FaultManager(
        faults=[
            dropout_fault,
        ]
    )

    print()
    print(
        "Running healthy spacecraft with telemetry dropout..."
    )

    telemetry = run_simulation(
        duration_hours=duration_hours,
        dt_s=dt_s,
        sensor_seed=42,
        faults=None,
    )

    df = pd.DataFrame(
        telemetry
    )

    # ======================================================
    # MEASURED CHANNELS THAT CAN DISAPPEAR
    # ======================================================

    measured_columns = [
        "battery_soc_measured",
        "battery_voltage_measured",
        "battery_current_measured",
        "battery_temp_measured",
        "electronics_temp_measured",
        "wheel_speed_measured",
        "wheel_current_measured",
        "wheel_temp_measured",
    ]

    dropout_severity = []

    # ======================================================
    # APPLY DROPOUT
    # ======================================================

    for index, row in df.iterrows():

        state = manager.get_state(
            fault=dropout_fault,
            time_s=row["time_s"],
        )

        severity = (
            state.effective_severity
        )

        dropout_severity.append(
            severity
        )

        if severity > 0.0:

            for column in measured_columns:

                df.at[
                    index,
                    column,
                ] = apply_telemetry_dropout(
                    value=row[column],
                    severity=severity,
                )

    df["telemetry_dropout"] = (
        dropout_severity
    )

    df["fault_active"] = (
        df["telemetry_dropout"]
        > 0.0
    ).astype(int)

    df["fault_type"] = (
        df["fault_active"]
        .map(
            {
                0: "none",
                1: "telemetry_dropout",
            }
        )
    )

    df["fault_label"] = (
        df["fault_active"]
        .map(
            {
                0: "nominal",
                1: "telemetry_dropout",
            }
        )
    )

    # ======================================================
    # NUMBER OF MISSING CHANNELS PER SAMPLE
    # ======================================================

    df["missing_measurements"] = (
        df[
            measured_columns
        ]
        .isna()
        .sum(axis=1)
    )

    # ======================================================
    # SAVE DATASET
    # ======================================================

    output_csv = (
        project_root
        / "data"
        / "faults"
        / "experiment_011_telemetry_dropout.csv"
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

    before_fault = (
        df["time_h"]
        < fault_start_h
    )

    after_fault = (
        df["time_h"]
        >= fault_start_h
    )

    missing_before = (
        df.loc[
            before_fault,
            "missing_measurements",
        ].sum()
    )

    missing_after = (
        df.loc[
            after_fault,
            "missing_measurements",
        ].sum()
    )

    max_missing_channels = (
        df["missing_measurements"]
        .max()
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

    figure = (
        figures_dir
        / "experiment_011_telemetry_dropout.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        df["time_h"],
        df["missing_measurements"],
        label="Missing Measured Channels",
    )

    plt.axvline(
        fault_start_h,
        linestyle="--",
        label="Dropout Start",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Missing Channels"
    )

    plt.title(
        "SENTINEL-XAI - Telemetry Dropout"
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

    print(
        "SENTINEL-XAI - EXPERIMENT 011"
    )

    print(
        "TELEMETRY DROPOUT"
    )

    print("=" * 72)

    print(
        f"Dropout start: "
        f"{fault_start_h:.2f} h"
    )

    print()

    print(
        f"Missing measurements before fault: "
        f"{missing_before}"
    )

    print(
        f"Missing measurements after fault: "
        f"{missing_after}"
    )

    print(
        f"Maximum missing channels per sample: "
        f"{max_missing_channels}"
    )

    print()

    print(
        f"Active fault samples: "
        f"{df['fault_active'].sum()}"
    )

    print()

    print(
        "Fault telemetry saved to:"
    )

    print(
        output_csv
    )

    print()

    print(
        "Figure saved to:"
    )

    print(
        figure
    )

    print("=" * 72)


if __name__ == "__main__":
    main()