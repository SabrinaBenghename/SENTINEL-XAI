from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


# ==========================================================
# LOAD CSV
# ==========================================================

def load_required_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


# ==========================================================
# INTERPOLATE NOMINAL MODEL REFERENCE
# ==========================================================

def interpolate_nominal_reference(
    nominal_df: pd.DataFrame,
    target_time_h: np.ndarray,
    column_name: str,
) -> np.ndarray:

    return np.interp(
        target_time_h,
        nominal_df["time_h"].to_numpy(),
        nominal_df[column_name].to_numpy(),
    )


# ==========================================================
# BUILD THERMAL RESIDUAL DATASET
# ==========================================================

def build_residual_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    nominal_df: pd.DataFrame,
) -> pd.DataFrame:

    time_h = (
        df["time_h"]
        .to_numpy()
    )

    # ======================================================
    # NOMINAL PHYSICS PREDICTIONS
    #
    # Use TRUE nominal trajectory as the reference model.
    # ======================================================

    battery_nominal_reference = (
        interpolate_nominal_reference(
            nominal_df=nominal_df,
            target_time_h=time_h,
            column_name="battery_temp_true",
        )
    )

    electronics_nominal_reference = (
        interpolate_nominal_reference(
            nominal_df=nominal_df,
            target_time_h=time_h,
            column_name="electronics_temp_true",
        )
    )

    # ======================================================
    # ACTUAL SENSOR MEASUREMENTS
    # ======================================================

    battery_measured = (
        df["battery_temp_measured"]
        .to_numpy()
    )

    electronics_measured = (
        df["electronics_temp_measured"]
        .to_numpy()
    )

    # ======================================================
    # RESIDUALS
    #
    # residual = measured - nominal prediction
    # ======================================================

    battery_residual = (
        battery_measured
        - battery_nominal_reference
    )

    electronics_residual = (
        electronics_measured
        - electronics_nominal_reference
    )

    return pd.DataFrame(
        {
            "dataset":
                dataset_name,

            "time_h":
                time_h,

            "battery_temp_measured":
                battery_measured,

            "battery_temp_nominal_ref":
                battery_nominal_reference,

            "battery_residual_c":
                battery_residual,

            "electronics_temp_measured":
                electronics_measured,

            "electronics_temp_nominal_ref":
                electronics_nominal_reference,

            "electronics_residual_c":
                electronics_residual,
        }
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # INPUT FILES
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    thermal_fault_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_008_thermal_anomaly.csv"
    )

    battery_fault_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_007_battery_degradation.csv"
    )

    nominal_df = (
        load_required_csv(
            nominal_file
        )
    )

    thermal_fault_df = (
        load_required_csv(
            thermal_fault_file
        )
    )

    battery_fault_df = (
        load_required_csv(
            battery_fault_file
        )
    )

    # ======================================================
    # BUILD RESIDUAL DATASETS
    # ======================================================

    results = {

        "nominal":
            build_residual_dataset(
                dataset_name="nominal",
                df=nominal_df,
                nominal_df=nominal_df,
            ),

        "thermal_anomaly":
            build_residual_dataset(
                dataset_name="thermal_anomaly",
                df=thermal_fault_df,
                nominal_df=nominal_df,
            ),

        "battery_degradation":
            build_residual_dataset(
                dataset_name="battery_degradation",
                df=battery_fault_df,
                nominal_df=nominal_df,
            ),
    }

    combined_df = pd.concat(
        results.values(),
        ignore_index=True,
    )

    # ======================================================
    # NOMINAL RESIDUAL STATISTICS
    #
    # Ignore first 15 min.
    # ======================================================

    nominal_steady = (
        results["nominal"][
            results["nominal"]["time_h"]
            >= 0.25
        ]
    )

    battery_nominal_mean = float(
        nominal_steady[
            "battery_residual_c"
        ].mean()
    )

    battery_nominal_std = float(
        nominal_steady[
            "battery_residual_c"
        ].std()
    )

    electronics_nominal_mean = float(
        nominal_steady[
            "electronics_residual_c"
        ].mean()
    )

    electronics_nominal_std = float(
        nominal_steady[
            "electronics_residual_c"
        ].std()
    )

    battery_threshold = (
        3.0
        * battery_nominal_std
    )

    electronics_threshold = (
        3.0
        * electronics_nominal_std
    )

    # ======================================================
    # ANALYSIS AFTER FAULT START
    # ======================================================

    summary_rows = []

    for name, df in results.items():

        analysis_df = (
            df[
                df["time_h"]
                >= 2.0
            ]
        )

        battery_residual = (
            analysis_df[
                "battery_residual_c"
            ]
        )

        electronics_residual = (
            analysis_df[
                "electronics_residual_c"
            ]
        )

        # --------------------------------------------------
        # Battery threshold exceedances
        # --------------------------------------------------

        battery_exceedances = int(
            (
                np.abs(
                    battery_residual
                    - battery_nominal_mean
                )
                >
                battery_threshold
            )
            .sum()
        )

        # --------------------------------------------------
        # Electronics threshold exceedances
        # --------------------------------------------------

        electronics_exceedances = int(
            (
                np.abs(
                    electronics_residual
                    - electronics_nominal_mean
                )
                >
                electronics_threshold
            )
            .sum()
        )

        number_of_samples = (
            len(
                analysis_df
            )
        )

        summary_rows.append(
            {
                "dataset":
                    name,

                # BATTERY
                "battery_mean_residual_c":
                    float(
                        battery_residual.mean()
                    ),

                "battery_mean_abs_residual_c":
                    float(
                        battery_residual.abs().mean()
                    ),

                "battery_rms_residual_c":
                    float(
                        np.sqrt(
                            np.mean(
                                battery_residual ** 2
                            )
                        )
                    ),

                "battery_max_abs_residual_c":
                    float(
                        battery_residual.abs().max()
                    ),

                "battery_3sigma_exceedances":
                    battery_exceedances,

                "battery_exceedance_fraction":
                    (
                        battery_exceedances
                        / number_of_samples
                        if number_of_samples > 0
                        else 0.0
                    ),

                # ELECTRONICS
                "electronics_mean_residual_c":
                    float(
                        electronics_residual.mean()
                    ),

                "electronics_mean_abs_residual_c":
                    float(
                        electronics_residual.abs().mean()
                    ),

                "electronics_rms_residual_c":
                    float(
                        np.sqrt(
                            np.mean(
                                electronics_residual ** 2
                            )
                        )
                    ),

                "electronics_max_abs_residual_c":
                    float(
                        electronics_residual.abs().max()
                    ),

                "electronics_3sigma_exceedances":
                    electronics_exceedances,

                "electronics_exceedance_fraction":
                    (
                        electronics_exceedances
                        / number_of_samples
                        if number_of_samples > 0
                        else 0.0
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ======================================================
    # SAVE RESULTS
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
        / "experiment_018_thermal_residual_summary.csv"
    )

    residual_file = (
        tables_dir
        / "experiment_018_thermal_residuals.csv"
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
    # FIGURE 1
    # BATTERY TEMPERATURE RESIDUAL
    # ======================================================

    battery_figure = (
        figures_dir
        / "experiment_018_battery_temp_residuals.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    for name, df in results.items():

        plt.plot(
            df["time_h"],
            df["battery_residual_c"],
            label=name,
        )

    plt.axhline(
        battery_nominal_mean
        + battery_threshold,
        linestyle="--",
        label="+3 sigma nominal",
    )

    plt.axhline(
        battery_nominal_mean
        - battery_threshold,
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
        "Battery Temperature Residual [C]"
    )

    plt.title(
        "SENTINEL-XAI - Battery Thermal Residuals"
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
    # FIGURE 2
    # ELECTRONICS TEMPERATURE RESIDUAL
    # ======================================================

    electronics_figure = (
        figures_dir
        / "experiment_018_electronics_temp_residuals.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    for name, df in results.items():

        plt.plot(
            df["time_h"],
            df["electronics_residual_c"],
            label=name,
        )

    plt.axhline(
        electronics_nominal_mean
        + electronics_threshold,
        linestyle="--",
        label="+3 sigma nominal",
    )

    plt.axhline(
        electronics_nominal_mean
        - electronics_threshold,
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
        "Electronics Temperature Residual [C]"
    )

    plt.title(
        "SENTINEL-XAI - Electronics Thermal Residuals"
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
    # FIGURE 3
    # RMS COMPARISON
    # ======================================================

    rms_figure = (
        figures_dir
        / "experiment_018_thermal_residual_rms.png"
    )

    x = np.arange(
        len(
            summary_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        x - width / 2,
        summary_df[
            "battery_rms_residual_c"
        ],
        width=width,
        label="Battery Residual RMS",
    )

    plt.bar(
        x + width / 2,
        summary_df[
            "electronics_rms_residual_c"
        ],
        width=width,
        label="Electronics Residual RMS",
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
        "Residual RMS [C]"
    )

    plt.title(
        "SENTINEL-XAI - Thermal Residual Energy"
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

    print(
        "SENTINEL-XAI - EXPERIMENT 018"
    )

    print(
        "THERMAL RESIDUAL ANALYSIS"
    )

    print("=" * 78)

    print()

    print(
        f"Nominal battery residual mean: "
        f"{battery_nominal_mean:.5f} C"
    )

    print(
        f"Nominal battery residual std: "
        f"{battery_nominal_std:.5f} C"
    )

    print(
        f"Battery 3-sigma threshold: "
        f"{battery_threshold:.5f} C"
    )

    print()

    print(
        f"Nominal electronics residual mean: "
        f"{electronics_nominal_mean:.5f} C"
    )

    print(
        f"Nominal electronics residual std: "
        f"{electronics_nominal_std:.5f} C"
    )

    print(
        f"Electronics 3-sigma threshold: "
        f"{electronics_threshold:.5f} C"
    )

    print()

    print("-" * 78)

    for _, row in summary_df.iterrows():

        print()
        print(
            row["dataset"]
        )

        print(
            f"  Battery RMS residual: "
            f"{row['battery_rms_residual_c']:.5f} C"
        )

        print(
            f"  Battery exceedances: "
            f"{int(row['battery_3sigma_exceedances'])}"
        )

        print(
            f"  Battery exceedance fraction: "
            f"{row['battery_exceedance_fraction']:.4f}"
        )

        print()

        print(
            f"  Electronics RMS residual: "
            f"{row['electronics_rms_residual_c']:.5f} C"
        )

        print(
            f"  Electronics exceedances: "
            f"{int(row['electronics_3sigma_exceedances'])}"
        )

        print(
            f"  Electronics exceedance fraction: "
            f"{row['electronics_exceedance_fraction']:.4f}"
        )

    print()
    print("=" * 78)

    print(
        "Tables saved to:"
    )

    print(
        summary_file
    )

    print(
        residual_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        battery_figure
    )

    print(
        electronics_figure
    )

    print(
        rms_figure
    )

    print("=" * 78)


if __name__ == "__main__":
    main()