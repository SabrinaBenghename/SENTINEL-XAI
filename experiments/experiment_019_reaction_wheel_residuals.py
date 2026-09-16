from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


# ==========================================================
# HEALTHY REACTION-WHEEL PARAMETERS
# ==========================================================

WHEEL_INERTIA = 0.015

NOMINAL_FRICTION_COEFFICIENT = 2.0e-6

MOTOR_TORQUE_CONSTANT = 0.04

IDLE_CURRENT_A = 0.15

TRACKING_TIME_CONSTANT_S = 120.0


# ==========================================================
# LOAD DATA
# ==========================================================

def load_csv(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


# ==========================================================
# HEALTHY PHYSICS CURRENT PREDICTION
# ==========================================================

def predict_nominal_current(
    target_speed_rpm,
    measured_speed_rpm,
):

    # RPM -> rad/s

    target_rad_s = (
        target_speed_rpm
        * 2.0
        * np.pi
        / 60.0
    )

    measured_rad_s = (
        measured_speed_rpm
        * 2.0
        * np.pi
        / 60.0
    )

    # ======================================================
    # CONTROLLER MODEL
    # ======================================================

    speed_error = (
        target_rad_s
        - measured_rad_s
    )

    angular_acceleration = (
        speed_error
        / TRACKING_TIME_CONSTANT_S
    )

    # ======================================================
    # EXPECTED INERTIA TORQUE
    #
    # tau = J * alpha
    # ======================================================

    inertia_torque = (
        WHEEL_INERTIA
        * angular_acceleration
    )

    # ======================================================
    # EXPECTED NOMINAL FRICTION
    # ======================================================

    friction_torque = (
        NOMINAL_FRICTION_COEFFICIENT
        * measured_rad_s
    )

    expected_motor_torque = (
        inertia_torque
        + friction_torque
    )

    # ======================================================
    # EXPECTED MOTOR CURRENT
    # ======================================================

    expected_current = (
        IDLE_CURRENT_A
        + np.abs(
            expected_motor_torque
        )
        / MOTOR_TORQUE_CONSTANT
    )

    return expected_current


# ==========================================================
# BUILD RESIDUALS
# ==========================================================

def build_residuals(
    dataset_name,
    df,
    nominal_reference,
):

    target_speed = (
        df[
            "wheel_target_speed_rpm"
        ].to_numpy()
    )

    measured_speed = (
        df[
            "wheel_speed_measured"
        ].to_numpy()
    )

    measured_current = (
        df[
            "wheel_current_measured"
        ].to_numpy()
    )

    measured_temperature = (
        df[
            "wheel_temp_measured"
        ].to_numpy()
    )

    # ======================================================
    # CURRENT PREDICTION
    # ======================================================

    predicted_current = (
        predict_nominal_current(
            target_speed_rpm=target_speed,
            measured_speed_rpm=measured_speed,
        )
    )

    current_residual = (
        measured_current
        - predicted_current
    )

    # ======================================================
    # TEMPERATURE REFERENCE
    #
    # Use healthy nominal thermal trajectory.
    # ======================================================

    nominal_temperature = np.interp(
        df["time_h"].to_numpy(),
        nominal_reference[
            "time_h"
        ].to_numpy(),
        nominal_reference[
            "wheel_temp_true"
        ].to_numpy(),
    )

    temperature_residual = (
        measured_temperature
        - nominal_temperature
    )

    return pd.DataFrame(
        {
            "dataset":
                dataset_name,

            "time_h":
                df["time_h"],

            "wheel_target_speed_rpm":
                target_speed,

            "wheel_speed_measured":
                measured_speed,

            "wheel_current_measured":
                measured_current,

            "wheel_current_predicted":
                predicted_current,

            "wheel_current_residual_a":
                current_residual,

            "wheel_temp_measured":
                measured_temperature,

            "wheel_temp_nominal_ref":
                nominal_temperature,

            "wheel_temp_residual_c":
                temperature_residual,
        }
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # DATASETS
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    wheel_fault_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_009_reaction_wheel_degradation.csv"
    )

    thermal_fault_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_008_thermal_anomaly.csv"
    )

    nominal_df = load_csv(
        nominal_file
    )

    wheel_fault_df = load_csv(
        wheel_fault_file
    )

    thermal_fault_df = load_csv(
        thermal_fault_file
    )

    # ======================================================
    # BUILD RESIDUAL DATASETS
    # ======================================================

    results = {

        "nominal":
            build_residuals(
                dataset_name="nominal",
                df=nominal_df,
                nominal_reference=nominal_df,
            ),

        "wheel_degradation":
            build_residuals(
                dataset_name="wheel_degradation",
                df=wheel_fault_df,
                nominal_reference=nominal_df,
            ),

        "thermal_anomaly":
            build_residuals(
                dataset_name="thermal_anomaly",
                df=thermal_fault_df,
                nominal_reference=nominal_df,
            ),
    }

    # ======================================================
    # NOMINAL STATISTICS
    # ======================================================

    nominal_steady = (
        results["nominal"][
            results["nominal"]["time_h"]
            >= 0.25
        ]
    )

    current_mean = float(
        nominal_steady[
            "wheel_current_residual_a"
        ].mean()
    )

    current_std = float(
        nominal_steady[
            "wheel_current_residual_a"
        ].std()
    )

    temp_mean = float(
        nominal_steady[
            "wheel_temp_residual_c"
        ].mean()
    )

    temp_std = float(
        nominal_steady[
            "wheel_temp_residual_c"
        ].std()
    )

    current_threshold = (
        3.0
        * current_std
    )

    temp_threshold = (
        3.0
        * temp_std
    )

    # ======================================================
    # ANALYSIS
    # ======================================================

    summary = []

    for name, df in results.items():

        analysis = (
            df[
                df["time_h"]
                >= 2.0
            ]
        )

        current_residual = (
            analysis[
                "wheel_current_residual_a"
            ]
        )

        temp_residual = (
            analysis[
                "wheel_temp_residual_c"
            ]
        )

        current_exceedances = int(
            (
                np.abs(
                    current_residual
                    - current_mean
                )
                >
                current_threshold
            )
            .sum()
        )

        temp_exceedances = int(
            (
                np.abs(
                    temp_residual
                    - temp_mean
                )
                >
                temp_threshold
            )
            .sum()
        )

        n = len(analysis)

        summary.append(
            {
                "dataset":
                    name,

                "current_mean_residual_a":
                    float(
                        current_residual.mean()
                    ),

                "current_rms_residual_a":
                    float(
                        np.sqrt(
                            np.mean(
                                current_residual ** 2
                            )
                        )
                    ),

                "current_max_abs_residual_a":
                    float(
                        current_residual
                        .abs()
                        .max()
                    ),

                "current_3sigma_exceedances":
                    current_exceedances,

                "current_exceedance_fraction":
                    (
                        current_exceedances / n
                        if n > 0
                        else 0.0
                    ),

                "temp_mean_residual_c":
                    float(
                        temp_residual.mean()
                    ),

                "temp_rms_residual_c":
                    float(
                        np.sqrt(
                            np.mean(
                                temp_residual ** 2
                            )
                        )
                    ),

                "temp_max_abs_residual_c":
                    float(
                        temp_residual
                        .abs()
                        .max()
                    ),

                "temp_3sigma_exceedances":
                    temp_exceedances,

                "temp_exceedance_fraction":
                    (
                        temp_exceedances / n
                        if n > 0
                        else 0.0
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    combined_df = pd.concat(
        results.values(),
        ignore_index=True,
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
        / "experiment_019_wheel_residual_summary.csv"
    )

    residual_file = (
        tables_dir
        / "experiment_019_wheel_residuals.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    combined_df.to_csv(
        residual_file,
        index=False,
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
    # CURRENT RESIDUAL
    # ------------------------------------------------------

    current_figure = (
        figures_dir
        / "experiment_019_wheel_current_residuals.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    for name, df in results.items():

        plt.plot(
            df["time_h"],
            df["wheel_current_residual_a"],
            label=name,
        )

    plt.axhline(
        current_mean
        + current_threshold,
        linestyle="--",
        label="+3 sigma nominal",
    )

    plt.axhline(
        current_mean
        - current_threshold,
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
        "Wheel Current Residual [A]"
    )

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Current Residuals"
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
    # TEMPERATURE RESIDUAL
    # ------------------------------------------------------

    temperature_figure = (
        figures_dir
        / "experiment_019_wheel_temperature_residuals.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    for name, df in results.items():

        plt.plot(
            df["time_h"],
            df["wheel_temp_residual_c"],
            label=name,
        )

    plt.axhline(
        temp_mean
        + temp_threshold,
        linestyle="--",
        label="+3 sigma nominal",
    )

    plt.axhline(
        temp_mean
        - temp_threshold,
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
        "Wheel Temperature Residual [C]"
    )

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Thermal Residuals"
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
    # RMS COMPARISON
    # ------------------------------------------------------

    rms_figure = (
        figures_dir
        / "experiment_019_wheel_residual_rms.png"
    )

    x = np.arange(
        len(summary_df)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        x - width / 2,
        summary_df[
            "current_rms_residual_a"
        ],
        width=width,
        label="Current RMS [A]",
    )

    plt.bar(
        x + width / 2,
        summary_df[
            "temp_rms_residual_c"
        ],
        width=width,
        label="Temperature RMS [C]",
    )

    plt.xticks(
        x,
        summary_df["dataset"],
        rotation=20,
        ha="right",
    )

    plt.xlabel(
        "Dataset"
    )

    plt.ylabel(
        "Residual RMS"
    )

    plt.title(
        "SENTINEL-XAI - Reaction-Wheel Residual Comparison"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        rms_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL SUMMARY
    # ======================================================

    print()
    print("=" * 78)
    print("SENTINEL-XAI - EXPERIMENT 019")
    print("REACTION-WHEEL MODEL-BASED RESIDUAL ANALYSIS")
    print("=" * 78)

    print()

    print(
        f"Nominal current residual mean: "
        f"{current_mean:.6f} A"
    )

    print(
        f"Nominal current residual std: "
        f"{current_std:.6f} A"
    )

    print(
        f"Current 3-sigma threshold: "
        f"{current_threshold:.6f} A"
    )

    print()

    print(
        f"Nominal wheel-temperature residual std: "
        f"{temp_std:.5f} C"
    )

    print(
        f"Temperature 3-sigma threshold: "
        f"{temp_threshold:.5f} C"
    )

    print()

    print("-" * 78)

    for _, row in summary_df.iterrows():

        print()
        print(
            row["dataset"]
        )

        print(
            f"  Current RMS residual: "
            f"{row['current_rms_residual_a']:.6f} A"
        )

        print(
            f"  Current exceedances: "
            f"{int(row['current_3sigma_exceedances'])}"
        )

        print(
            f"  Current exceedance fraction: "
            f"{row['current_exceedance_fraction']:.4f}"
        )

        print()

        print(
            f"  Temperature RMS residual: "
            f"{row['temp_rms_residual_c']:.5f} C"
        )

        print(
            f"  Temperature exceedances: "
            f"{int(row['temp_3sigma_exceedances'])}"
        )

        print(
            f"  Temperature exceedance fraction: "
            f"{row['temp_exceedance_fraction']:.4f}"
        )

    print()
    print("=" * 78)

    print("Tables saved to:")
    print(summary_file)
    print(residual_file)

    print()

    print("Figures saved to:")
    print(current_figure)
    print(temperature_figure)
    print(rms_figure)

    print("=" * 78)


if __name__ == "__main__":
    main()