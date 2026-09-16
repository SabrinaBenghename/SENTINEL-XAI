from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "results"
        / "tables"
        / "battery_ekf_nominal.csv"
    )

    df = pd.read_csv(input_file)

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
    # FIGURE 1 - SOC ESTIMATION
    # ======================================================

    soc_figure = (
        figures_dir
        / "experiment_016_battery_ekf_soc.png"
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        df["time_h"],
        df["soc_true"] * 100.0,
        label="True SOC",
    )

    plt.plot(
        df["time_h"],
        df["soc_estimated"] * 100.0,
        label="EKF Estimated SOC",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Battery SOC [%]")

    plt.title(
        "SENTINEL-XAI - Battery EKF State Estimation"
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
    # FIGURE 2 - VOLTAGE INNOVATION
    # ======================================================

    innovation_figure = (
        figures_dir
        / "experiment_016_battery_ekf_innovation.png"
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        df["time_h"],
        df["innovation_v"],
        label="Voltage Innovation",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Innovation [V]")

    plt.title(
        "SENTINEL-XAI - Nominal Battery EKF Innovation"
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
    # FIGURE 3 - SOC ESTIMATION ERROR
    # ======================================================

    error_figure = (
        figures_dir
        / "experiment_016_battery_ekf_error.png"
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        df["time_h"],
        df["soc_error"] * 100.0,
        label="SOC Estimation Error",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("SOC Error [percentage points]")

    plt.title(
        "SENTINEL-XAI - Battery EKF Estimation Error"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        error_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 72)
    print("SENTINEL-XAI - EXPERIMENT 016")
    print("BATTERY EKF NOMINAL VALIDATION")
    print("=" * 72)

    print(
        f"Maximum absolute SOC error: "
        f"{(df['soc_error'].abs().max() * 100.0):.3f} percentage points"
    )

    print(
        f"Final absolute SOC error: "
        f"{(abs(df['soc_error'].iloc[-1]) * 100.0):.3f} percentage points"
    )

    print(
        f"Mean innovation: "
        f"{df['innovation_v'].mean():.5f} V"
    )

    print(
        f"Innovation standard deviation: "
        f"{df['innovation_v'].std():.5f} V"
    )

    print()
    print("Figures saved to:")
    print(soc_figure)
    print(innovation_figure)
    print(error_figure)

    print("=" * 72)


if __name__ == "__main__":
    main()