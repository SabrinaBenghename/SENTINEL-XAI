from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.estimation.battery_ekf import (
    BatteryEKF,
)


# ==========================================================
# RUN EKF ON ONE DATASET
# ==========================================================

def run_ekf(
    df,
    dataset_name,
):

    ekf = BatteryEKF(
        initial_soc=0.85,
        initial_covariance=0.01,
        battery_capacity_wh=1120.0,
        process_noise=1e-6,
        voltage_noise_std_v=0.03,
    )

    records = []

    for index, row in df.iterrows():

        if index == 0:
            dt_s = 60.0

        else:
            dt_s = (
                row["time_s"]
                - df.iloc[
                    index - 1
                ]["time_s"]
            )

        voltage = row[
            "battery_voltage_measured"
        ]

        current = row[
            "battery_current_measured"
        ]

        # Skip missing measurements
        if (
            pd.isna(voltage)
            or pd.isna(current)
        ):
            continue

        state = ekf.step(
            time_s=row["time_s"],
            dt_s=dt_s,
            battery_current_a=current,
            battery_voltage_measured_v=voltage,
        )

        records.append(
            {
                "dataset":
                    dataset_name,

                "time_h":
                    row["time_h"],

                "innovation_v":
                    state.innovation_v,

                "soc_estimated":
                    state.soc_estimated,

                "voltage_predicted_v":
                    state.voltage_predicted_v,

                "voltage_measured_v":
                    state.voltage_measured_v,
            }
        )

    return pd.DataFrame(
        records
    )


def main():

    # ======================================================
    # DATASETS
    # ======================================================

    files = {

        "nominal":
            project_root
            / "data"
            / "nominal"
            / "nominal_spacecraft_telemetry.csv",

        "solar_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_006_solar_degradation.csv",

        "battery_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_007_battery_degradation.csv",

        "sensor_drift":
            project_root
            / "data"
            / "faults"
            / "experiment_010_sensor_drift.csv",
    }

    results = {}

    print()
    print(
        "Running battery EKF across fault datasets..."
    )

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset:\n{path}"
            )

        df = pd.read_csv(
            path
        )

        results[name] = run_ekf(
            df=df,
            dataset_name=name,
        )

    # ======================================================
    # NOMINAL INNOVATION THRESHOLD
    #
    # Estimate the baseline directly from healthy data.
    # ======================================================

    nominal_innovation = (
        results["nominal"][
            "innovation_v"
        ]
    )

    # Ignore first few minutes of convergence
    nominal_steady = (
        results["nominal"]
        [
            results["nominal"]["time_h"]
            >= 0.25
        ]
    )

    nominal_mean = (
        nominal_steady[
            "innovation_v"
        ].mean()
    )

    nominal_std = (
        nominal_steady[
            "innovation_v"
        ].std()
    )

    three_sigma_threshold = (
        3.0
        * nominal_std
    )

    # ======================================================
    # METRICS AFTER FAULT START
    # ======================================================

    summary = []

    for name, df in results.items():

        if name == "nominal":

            analysis_df = df[
                df["time_h"] >= 2.0
            ]

        else:

            analysis_df = df[
                df["time_h"] >= 2.0
            ]

        innovation = (
            analysis_df[
                "innovation_v"
            ]
        )

        mean_innovation = float(
            innovation.mean()
        )

        mean_absolute_innovation = float(
            innovation.abs().mean()
        )

        rms_innovation = float(
            np.sqrt(
                np.mean(
                    innovation ** 2
                )
            )
        )

        max_absolute_innovation = float(
            innovation.abs().max()
        )

        exceedances = int(
            (
                np.abs(
                    innovation
                    - nominal_mean
                )
                > three_sigma_threshold
            ).sum()
        )

        exceedance_fraction = (
            exceedances
            / len(analysis_df)
            if len(analysis_df) > 0
            else 0.0
        )

        summary.append(
            {
                "dataset":
                    name,

                "mean_innovation_v":
                    mean_innovation,

                "mean_abs_innovation_v":
                    mean_absolute_innovation,

                "rms_innovation_v":
                    rms_innovation,

                "max_abs_innovation_v":
                    max_absolute_innovation,

                "three_sigma_exceedances":
                    exceedances,

                "exceedance_fraction":
                    exceedance_fraction,
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    # ======================================================
    # SAVE TABLES
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

    summary_file = (
        tables_dir
        / "experiment_017_battery_ekf_fault_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    combined_df = pd.concat(
        results.values(),
        ignore_index=True,
    )

    residual_file = (
        tables_dir
        / "experiment_017_battery_ekf_residuals.csv"
    )

    combined_df.to_csv(
        residual_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — INNOVATIONS
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

    innovation_figure = (
        figures_dir
        / "experiment_017_battery_ekf_fault_innovations.png"
    )

    plt.figure(
        figsize=(12, 6)
    )

    for name, df in results.items():

        plt.plot(
            df["time_h"],
            df["innovation_v"],
            label=name,
        )

    plt.axhline(
        nominal_mean
        + three_sigma_threshold,
        linestyle="--",
        label="+3 sigma nominal",
    )

    plt.axhline(
        nominal_mean
        - three_sigma_threshold,
        linestyle="--",
        label="-3 sigma nominal",
    )

    plt.axvline(
        2.0,
        linestyle=":",
        label="Fault Start",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Voltage Innovation [V]"
    )

    plt.title(
        "SENTINEL-XAI - Battery EKF Innovations Under Faults"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        innovation_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — RMS INNOVATION
    # ======================================================

    rms_figure = (
        figures_dir
        / "experiment_017_battery_ekf_rms.png"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        summary_df["dataset"],
        summary_df["rms_innovation_v"],
    )

    plt.xlabel(
        "Dataset"
    )

    plt.ylabel(
        "Innovation RMS [V]"
    )

    plt.title(
        "SENTINEL-XAI - EKF Residual Energy by Scenario"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        rms_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    print()
    print("=" * 78)

    print(
        "SENTINEL-XAI - EXPERIMENT 017"
    )

    print(
        "BATTERY EKF FAULT RESIDUAL ANALYSIS"
    )

    print("=" * 78)

    print()

    print(
        f"Nominal innovation mean: "
        f"{nominal_mean:.5f} V"
    )

    print(
        f"Nominal innovation std: "
        f"{nominal_std:.5f} V"
    )

    print(
        f"3-sigma threshold: "
        f"{three_sigma_threshold:.5f} V"
    )

    print()

    print("-" * 78)

    for _, row in summary_df.iterrows():

        print()
        print(
            row["dataset"]
        )

        print(
            f"  Mean innovation: "
            f"{row['mean_innovation_v']:.5f} V"
        )

        print(
            f"  Mean absolute innovation: "
            f"{row['mean_abs_innovation_v']:.5f} V"
        )

        print(
            f"  RMS innovation: "
            f"{row['rms_innovation_v']:.5f} V"
        )

        print(
            f"  Max absolute innovation: "
            f"{row['max_abs_innovation_v']:.5f} V"
        )

        print(
            f"  3-sigma exceedances: "
            f"{int(row['three_sigma_exceedances'])}"
        )

        print(
            f"  Exceedance fraction: "
            f"{row['exceedance_fraction']:.4f}"
        )

    print()
    print("=" * 78)

    print("Tables saved to:")
    print(summary_file)
    print(residual_file)

    print()

    print("Figures saved to:")
    print(innovation_figure)
    print(rms_figure)

    print("=" * 78)


if __name__ == "__main__":
    main()