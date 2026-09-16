from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_reaction_wheel_telemetry.csv"
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(input_file)

    time_h = df["time_s"] / 3600.0

    # ==================================================
    # FIGURE 1 - WHEEL SPEED TRACKING
    # ==================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["target_speed_rpm"],
        label="Target Speed",
    )

    plt.plot(
        time_h,
        df["wheel_speed_rpm"],
        label="Actual Wheel Speed",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Wheel Speed [RPM]")

    plt.title(
        "SENTINEL-XAI - Reaction Wheel Speed Tracking"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    speed_figure = (
        figures_dir
        / "experiment_004_wheel_speed.png"
    )

    plt.savefig(
        speed_figure,
        dpi=200,
    )

    plt.show()

    # ==================================================
    # FIGURE 2 - MOTOR CURRENT
    # ==================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["motor_current_a"],
        label="Motor Current",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Current [A]")

    plt.title(
        "SENTINEL-XAI - Nominal Reaction Wheel Current"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    current_figure = (
        figures_dir
        / "experiment_004_wheel_current.png"
    )

    plt.savefig(
        current_figure,
        dpi=200,
    )

    plt.show()

    # ==================================================
    # FIGURE 3 - WHEEL TEMPERATURE
    # ==================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["wheel_temperature_c"],
        label="Wheel Temperature",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Temperature [°C]")

    plt.title(
        "SENTINEL-XAI - Nominal Reaction Wheel Temperature"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    temperature_figure = (
        figures_dir
        / "experiment_004_wheel_temperature.png"
    )

    plt.savefig(
        temperature_figure,
        dpi=200,
    )

    plt.show()

    # ==================================================
    # SUMMARY
    # ==================================================

    tracking_error = (
        df["target_speed_rpm"]
        - df["wheel_speed_rpm"]
    )

    print()
    print("=" * 65)
    print("SENTINEL-XAI - EXPERIMENT 004")
    print("NOMINAL REACTION-WHEEL VALIDATION")
    print("=" * 65)

    print(
        f"Wheel speed range: "
        f"{df['wheel_speed_rpm'].min():.1f} "
        f"to "
        f"{df['wheel_speed_rpm'].max():.1f} RPM"
    )

    print(
        f"Mean absolute tracking error: "
        f"{tracking_error.abs().mean():.2f} RPM"
    )

    print(
        f"Maximum motor current: "
        f"{df['motor_current_a'].max():.3f} A"
    )

    print(
        f"Wheel temperature range: "
        f"{df['wheel_temperature_c'].min():.2f} "
        f"to "
        f"{df['wheel_temperature_c'].max():.2f} °C"
    )

    print()
    print("Figures saved to:")
    print(speed_figure)
    print(current_figure)
    print(temperature_figure)

    print("=" * 65)


if __name__ == "__main__":
    main()