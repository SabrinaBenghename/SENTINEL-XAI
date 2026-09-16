from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_thermal_telemetry.csv"
    )

    output_file = (
        project_root
        / "results"
        / "figures"
        / "experiment_002_nominal_thermal.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(input_file)

    time_h = df["time_s"] / 3600.0

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_h,
        df["battery_temp_c"],
        label="Battery Temperature",
    )

    plt.plot(
        time_h,
        df["electronics_temp_c"],
        label="Electronics Temperature",
    )

    plt.xlabel("Time [hours]")
    plt.ylabel("Temperature [°C]")

    plt.title(
        "SENTINEL-XAI - Nominal Thermal Behaviour"
    )

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
    print("SENTINEL-XAI - EXPERIMENT 002")
    print("=" * 60)

    print(
        f"Battery temperature range: "
        f"{df['battery_temp_c'].min():.2f} "
        f"to "
        f"{df['battery_temp_c'].max():.2f} °C"
    )

    print(
        f"Electronics temperature range: "
        f"{df['electronics_temp_c'].min():.2f} "
        f"to "
        f"{df['electronics_temp_c'].max():.2f} °C"
    )

    print()
    print("Figure saved to:")
    print(output_file)

    print("=" * 60)


if __name__ == "__main__":
    main()