from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Allow imports from the project src folder
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root)
)


from src.simulation.power_model import SpacecraftPowerModel
from src.simulation.thermal_model import ThermalModel


def main():

    duration_hours = 24.0
    dt_s = 60.0

    power_model = SpacecraftPowerModel()
    thermal_model = ThermalModel()

    records = []

    number_of_steps = int(
        duration_hours * 3600.0 / dt_s
    )

    for step_index in range(number_of_steps + 1):

        time_s = step_index * dt_s

        # ------------------------------------------
        # Power simulation
        # ------------------------------------------

        power_state = power_model.step(
            time_s=time_s,
            dt_s=dt_s,
        )

        # ------------------------------------------
        # Thermal simulation
        # ------------------------------------------

        (
            battery_temp_c,
            electronics_temp_c,
            battery_heat_w,
            electronics_heat_w,
        ) = thermal_model.step(
            dt_s=dt_s,
            battery_current_a=power_state.battery_current_a,
            load_power_w=power_state.load_power_w,
            in_sunlight=power_state.in_sunlight,
        )

        records.append(
            {
                "time_s": time_s,
                "time_h": time_s / 3600.0,
                "battery_temp_c": battery_temp_c,
                "electronics_temp_c": electronics_temp_c,
                "battery_soc_percent":
                    power_state.battery_soc * 100.0,
                "in_sunlight":
                    int(power_state.in_sunlight),
            }
        )

    df = pd.DataFrame(records)

    # --------------------------------------------------
    # Save telemetry
    # --------------------------------------------------

    output_csv = (
        project_root
        / "data"
        / "nominal"
        / "experiment_003_24h_thermal.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------
    # Last 6 hours
    # --------------------------------------------------

    last_6h = df[
        df["time_h"] >= 18.0
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    output_figure = (
        project_root
        / "results"
        / "figures"
        / "experiment_003_24h_thermal_stability.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        df["time_h"],
        df["battery_temp_c"],
        label="Battery Temperature",
    )

    plt.plot(
        df["time_h"],
        df["electronics_temp_c"],
        label="Electronics Temperature",
    )

    plt.xlabel(
        "Time [hours]"
    )

    plt.ylabel(
        "Temperature [°C]"
    )

    plt.title(
        "SENTINEL-XAI - 24 h Nominal Thermal Stability"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_figure,
        dpi=200,
    )

    plt.show()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 65)
    print("SENTINEL-XAI - EXPERIMENT 003")
    print("24-HOUR NOMINAL THERMAL STABILITY")
    print("=" * 65)

    print(
        f"Battery temperature overall: "
        f"{df['battery_temp_c'].min():.2f} "
        f"to "
        f"{df['battery_temp_c'].max():.2f} °C"
    )

    print(
        f"Electronics temperature overall: "
        f"{df['electronics_temp_c'].min():.2f} "
        f"to "
        f"{df['electronics_temp_c'].max():.2f} °C"
    )

    print()

    print("Last 6 hours:")

    print(
        f"Battery: "
        f"{last_6h['battery_temp_c'].min():.2f} "
        f"to "
        f"{last_6h['battery_temp_c'].max():.2f} °C"
    )

    print(
        f"Electronics: "
        f"{last_6h['electronics_temp_c'].min():.2f} "
        f"to "
        f"{last_6h['electronics_temp_c'].max():.2f} °C"
    )

    print()

    print(
        f"Final battery temperature: "
        f"{df['battery_temp_c'].iloc[-1]:.2f} °C"
    )

    print(
        f"Final electronics temperature: "
        f"{df['electronics_temp_c'].iloc[-1]:.2f} °C"
    )

    print()

    print("Telemetry saved to:")
    print(output_csv)

    print()

    print("Figure saved to:")
    print(output_figure)

    print("=" * 65)


if __name__ == "__main__":
    main()