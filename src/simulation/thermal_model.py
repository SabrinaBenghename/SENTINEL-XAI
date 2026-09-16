from pathlib import Path

import pandas as pd


class ThermalModel:
    """
    Simplified spacecraft thermal model.

    Two thermal nodes:
        1. Battery
        2. Electronics

    thermal_anomaly:
        0.0 = healthy
        1.0 = maximum modeled anomaly

    The anomaly represents:
        - increased internal heat generation
        - reduced thermal dissipation
    """

    def __init__(self):

        # ==================================================
        # INITIAL TEMPERATURES
        # ==================================================

        self.battery_temp_c = 22.0
        self.electronics_temp_c = 24.0

        # ==================================================
        # BATTERY THERMAL PARAMETERS
        # ==================================================

        self.battery_thermal_capacity = 18000.0

        self.nominal_battery_thermal_conductance = 1.8

        self.battery_internal_resistance = 0.08

        self.nominal_battery_background_heat = 1.0

        # ==================================================
        # ELECTRONICS THERMAL PARAMETERS
        # ==================================================

        self.electronics_thermal_capacity = 30000.0

        self.nominal_electronics_thermal_conductance = 4.0

        self.nominal_electronics_heat_fraction = 0.05

    def step(
        self,
        dt_s,
        battery_current_a,
        load_power_w,
        in_sunlight,
        thermal_anomaly=0.0,
    ):

        # ==================================================
        # FAULT SEVERITY
        # ==================================================

        thermal_anomaly = max(
            0.0,
            min(
                1.0,
                thermal_anomaly,
            ),
        )

        # Increasing anomaly:
        #
        # 1. adds internal heat
        # 2. reduces heat dissipation

        battery_background_heat = (
            self.nominal_battery_background_heat
            + 15.0 * thermal_anomaly
        )

        battery_thermal_conductance = (
            self.nominal_battery_thermal_conductance
            * (
                1.0
                - 0.60 * thermal_anomaly
            )
        )

        electronics_heat_fraction = (
            self.nominal_electronics_heat_fraction
            * (
                1.0
                + 1.50 * thermal_anomaly
            )
        )

        electronics_thermal_conductance = (
            self.nominal_electronics_thermal_conductance
            * (
                1.0
                - 0.60 * thermal_anomaly
            )
        )

        # ==================================================
        # BATTERY ENVIRONMENT
        # ==================================================

        if in_sunlight:
            battery_environment_temp_c = 22.0
        else:
            battery_environment_temp_c = 19.0

        # ==================================================
        # BATTERY HEATING
        #
        # Q = I^2 R
        # ==================================================

        battery_heat_w = (
            battery_current_a ** 2
            * self.battery_internal_resistance
            + battery_background_heat
        )

        battery_heat_loss_w = (
            battery_thermal_conductance
            * (
                self.battery_temp_c
                - battery_environment_temp_c
            )
        )

        battery_temp_rate = (
            battery_heat_w
            - battery_heat_loss_w
        ) / self.battery_thermal_capacity

        self.battery_temp_c += (
            battery_temp_rate
            * dt_s
        )

        # ==================================================
        # ELECTRONICS ENVIRONMENT
        # ==================================================

        if in_sunlight:
            electronics_environment_temp_c = 22.5
        else:
            electronics_environment_temp_c = 20.5

        # ==================================================
        # ELECTRONICS HEATING
        # ==================================================

        electronics_heat_w = (
            electronics_heat_fraction
            * load_power_w
        )

        electronics_heat_loss_w = (
            electronics_thermal_conductance
            * (
                self.electronics_temp_c
                - electronics_environment_temp_c
            )
        )

        electronics_temp_rate = (
            electronics_heat_w
            - electronics_heat_loss_w
        ) / self.electronics_thermal_capacity

        self.electronics_temp_c += (
            electronics_temp_rate
            * dt_s
        )

        return (
            self.battery_temp_c,
            self.electronics_temp_c,
            battery_heat_w,
            electronics_heat_w,
        )


# ==========================================================
# NOMINAL STANDALONE SIMULATION
# ==========================================================

def run_thermal_simulation():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_power_telemetry.csv"
    )

    output_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_thermal_telemetry.csv"
    )

    df = pd.read_csv(
        input_file
    )

    model = ThermalModel()

    battery_temperatures = []
    electronics_temperatures = []

    battery_heat_values = []
    electronics_heat_values = []

    for index, row in df.iterrows():

        if index == 0:
            dt_s = 60.0

        else:
            dt_s = (
                row["time_s"]
                - df.iloc[index - 1]["time_s"]
            )

        (
            battery_temp,
            electronics_temp,
            battery_heat,
            electronics_heat,
        ) = model.step(
            dt_s=dt_s,

            battery_current_a=(
                row["battery_current_a"]
            ),

            load_power_w=(
                row["load_power_w"]
            ),

            in_sunlight=bool(
                row["in_sunlight"]
            ),

            thermal_anomaly=0.0,
        )

        battery_temperatures.append(
            battery_temp
        )

        electronics_temperatures.append(
            electronics_temp
        )

        battery_heat_values.append(
            battery_heat
        )

        electronics_heat_values.append(
            electronics_heat
        )

    thermal_df = pd.DataFrame(
        {
            "time_s":
                df["time_s"],

            "time_min":
                df["time_min"],

            "in_sunlight":
                df["in_sunlight"],

            "battery_temp_c":
                battery_temperatures,

            "electronics_temp_c":
                electronics_temperatures,

            "battery_heat_w":
                battery_heat_values,

            "electronics_heat_w":
                electronics_heat_values,
        }
    )

    thermal_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 60)

    print(
        "SENTINEL-XAI - NOMINAL THERMAL SIMULATION"
    )

    print("=" * 60)

    print(
        f"Battery temperature range: "
        f"{thermal_df['battery_temp_c'].min():.2f} "
        f"to "
        f"{thermal_df['battery_temp_c'].max():.2f} C"
    )

    print(
        f"Electronics temperature range: "
        f"{thermal_df['electronics_temp_c'].min():.2f} "
        f"to "
        f"{thermal_df['electronics_temp_c'].max():.2f} C"
    )

    print(
        f"Final battery temperature: "
        f"{thermal_df['battery_temp_c'].iloc[-1]:.2f} C"
    )

    print(
        f"Final electronics temperature: "
        f"{thermal_df['electronics_temp_c'].iloc[-1]:.2f} C"
    )

    print()
    print(
        "Telemetry saved to:"
    )

    print(
        output_file
    )

    print("=" * 60)


if __name__ == "__main__":
    run_thermal_simulation()