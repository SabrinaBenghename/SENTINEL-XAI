from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]

    telemetry_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_power_telemetry.csv"
    )

    output_file = (
        project_root
        / "results"
        / "figures"
        / "experiment_001_nominal_power.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(telemetry_file)

    time_h = df["time_s"] / 3600.0

    # --------------------------------------------------
    # Figure 1: Solar power and spacecraft load
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["solar_power_w"],
        label="Solar Power",
    )

    plt.plot(
        time_h,
        df["load_power_w"],
        label="Spacecraft Load",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Power [W]")
    plt.title("SENTINEL-XAI - Nominal Spacecraft Power")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    power_plot = (
        project_root
        / "results"
        / "figures"
        / "experiment_001_power_balance.png"
    )

    plt.savefig(
        power_plot,
        dpi=200,
    )

    plt.show()

    # --------------------------------------------------
    # Figure 2: Battery state of charge
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["battery_soc_percent"],
        label="Battery SOC",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("State of Charge [%]")
    plt.title("SENTINEL-XAI - Battery State of Charge")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
    )

    plt.show()

    print()
    print("=" * 60)
    print("SENTINEL-XAI - EXPERIMENT 001")
    print("=" * 60)

    print(
        f"Minimum SOC: "
        f"{df['battery_soc_percent'].min():.2f} %"
    )

    print(
        f"Maximum SOC: "
        f"{df['battery_soc_percent'].max():.2f} %"
    )

    print(
        f"Average solar power: "
        f"{df['solar_power_w'].mean():.2f} W"
    )

    print(
        f"Average load power: "
        f"{df['load_power_w'].mean():.2f} W"
    )

    print()
    print("Figures saved to:")
    print(power_plot)
    print(output_file)

    print("=" * 60)


if __name__ == "__main__":
    main()