from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PowerState:
    time_s: float
    in_sunlight: bool

    solar_power_w: float
    load_power_w: float
    battery_power_w: float

    battery_voltage_v: float
    battery_current_a: float
    battery_soc: float


class SpacecraftPowerModel:
    """
    Simplified spacecraft electrical-power model.

    Fault inputs
    ------------

    solar_degradation:
        0.0 -> healthy solar array
        0.3 -> 30% solar power loss

    battery_degradation:
        0.0 -> healthy battery
        0.3 -> 30% usable-capacity loss
    """

    def __init__(
        self,
        battery_capacity_ah: float = 40.0,
        nominal_battery_voltage_v: float = 28.0,
        initial_soc: float = 0.90,
        solar_array_power_w: float = 600.0,
        base_load_power_w: float = 350.0,
        orbit_period_min: float = 90.0,
        sunlight_fraction: float = 0.62,
    ):
        self.nominal_capacity_wh = (
            battery_capacity_ah
            * nominal_battery_voltage_v
        )

        self.battery_soc = initial_soc

        self.solar_array_power_w = (
            solar_array_power_w
        )

        self.base_load_power_w = (
            base_load_power_w
        )

        self.orbit_period_s = (
            orbit_period_min
            * 60.0
        )

        self.sunlight_fraction = (
            sunlight_fraction
        )

    def _is_in_sunlight(
        self,
        time_s: float,
    ) -> bool:

        orbit_phase_s = (
            time_s
            % self.orbit_period_s
        )

        sunlight_duration_s = (
            self.sunlight_fraction
            * self.orbit_period_s
        )

        return (
            orbit_phase_s
            < sunlight_duration_s
        )

    def _solar_power(
        self,
        time_s: float,
        in_sunlight: bool,
        solar_degradation: float = 0.0,
    ) -> float:

        if not in_sunlight:
            return 0.0

        variation = (
            1.0
            + 0.03
            * math.sin(
                2.0
                * math.pi
                * time_s
                / self.orbit_period_s
            )
        )

        solar_degradation = max(
            0.0,
            min(
                1.0,
                solar_degradation,
            ),
        )

        nominal_power_w = (
            self.solar_array_power_w
            * variation
        )

        return (
            nominal_power_w
            * (
                1.0
                - solar_degradation
            )
        )

    def _load_power(
        self,
        time_s: float,
    ) -> float:

        variation = (
            20.0
            * math.sin(
                2.0
                * math.pi
                * time_s
                / (30.0 * 60.0)
            )
        )

        return (
            self.base_load_power_w
            + variation
        )

    def _battery_voltage(
        self,
        soc: float,
    ) -> float:

        return (
            26.0
            + 4.0 * soc
        )

    def step(
        self,
        time_s: float,
        dt_s: float,
        solar_degradation: float = 0.0,
        battery_degradation: float = 0.0,
    ) -> PowerState:

        # ==================================================
        # 1. SUNLIGHT / ECLIPSE
        # ==================================================

        in_sunlight = (
            self._is_in_sunlight(
                time_s
            )
        )

        # ==================================================
        # 2. SOLAR GENERATION
        # ==================================================

        solar_power_w = (
            self._solar_power(
                time_s=time_s,
                in_sunlight=in_sunlight,
                solar_degradation=(
                    solar_degradation
                ),
            )
        )

        # ==================================================
        # 3. SPACECRAFT LOAD
        # ==================================================

        load_power_w = (
            self._load_power(
                time_s
            )
        )

        # ==================================================
        # 4. BATTERY POWER
        #
        # Positive -> discharge
        # Negative -> charge
        # ==================================================

        battery_power_w = (
            load_power_w
            - solar_power_w
        )

        # ==================================================
        # 5. BATTERY DEGRADATION
        # ==================================================

        battery_degradation = max(
            0.0,
            min(
                0.95,
                battery_degradation,
            ),
        )

        effective_capacity_wh = (
            self.nominal_capacity_wh
            * (
                1.0
                - battery_degradation
            )
        )

        # ==================================================
        # 6. SOC UPDATE
        #
        # Smaller effective capacity means the same
        # energy flow changes SOC faster.
        # ==================================================

        soc_change = (
            -battery_power_w
            * dt_s
            / 3600.0
            / effective_capacity_wh
        )

        self.battery_soc += (
            soc_change
        )

        self.battery_soc = max(
            0.0,
            min(
                1.0,
                self.battery_soc,
            ),
        )

        # ==================================================
        # 7. BATTERY VOLTAGE
        # ==================================================

        battery_voltage_v = (
            self._battery_voltage(
                self.battery_soc
            )
        )

        # ==================================================
        # 8. BATTERY CURRENT
        #
        # P = V I
        # ==================================================

        battery_current_a = (
            battery_power_w
            / battery_voltage_v
        )

        return PowerState(
            time_s=time_s,
            in_sunlight=in_sunlight,
            solar_power_w=solar_power_w,
            load_power_w=load_power_w,
            battery_power_w=battery_power_w,
            battery_voltage_v=battery_voltage_v,
            battery_current_a=battery_current_a,
            battery_soc=self.battery_soc,
        )


def run_nominal_simulation(
    duration_hours: float = 6.0,
    dt_s: float = 60.0,
):

    model = (
        SpacecraftPowerModel()
    )

    telemetry = []

    number_of_steps = int(
        duration_hours
        * 3600.0
        / dt_s
    )

    for step_index in range(
        number_of_steps + 1
    ):

        time_s = (
            step_index
            * dt_s
        )

        state = model.step(
            time_s=time_s,
            dt_s=dt_s,
            solar_degradation=0.0,
            battery_degradation=0.0,
        )

        telemetry.append(
            state
        )

    return telemetry


def save_telemetry(
    telemetry: list[PowerState],
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "time_s",
                "time_min",
                "in_sunlight",
                "solar_power_w",
                "load_power_w",
                "battery_power_w",
                "battery_voltage_v",
                "battery_current_a",
                "battery_soc",
                "battery_soc_percent",
            ]
        )

        for state in telemetry:

            writer.writerow(
                [
                    state.time_s,
                    state.time_s / 60.0,
                    int(state.in_sunlight),
                    state.solar_power_w,
                    state.load_power_w,
                    state.battery_power_w,
                    state.battery_voltage_v,
                    state.battery_current_a,
                    state.battery_soc,
                    state.battery_soc * 100.0,
                ]
            )


def print_summary(
    telemetry: list[PowerState],
):

    soc_values = [
        state.battery_soc
        for state in telemetry
    ]

    solar_values = [
        state.solar_power_w
        for state in telemetry
    ]

    load_values = [
        state.load_power_w
        for state in telemetry
    ]

    final_state = (
        telemetry[-1]
    )

    print()
    print("=" * 60)
    print(
        "SENTINEL-XAI - NOMINAL POWER SIMULATION"
    )
    print("=" * 60)

    print(
        f"Simulation duration: "
        f"{final_state.time_s / 3600.0:.2f} h"
    )

    print(
        f"Final battery SOC: "
        f"{final_state.battery_soc * 100.0:.2f} %"
    )

    print(
        f"Minimum battery SOC: "
        f"{min(soc_values) * 100.0:.2f} %"
    )

    print(
        f"Maximum battery SOC: "
        f"{max(soc_values) * 100.0:.2f} %"
    )

    print(
        f"Average solar power: "
        f"{sum(solar_values) / len(solar_values):.2f} W"
    )

    print(
        f"Average spacecraft load: "
        f"{sum(load_values) / len(load_values):.2f} W"
    )

    print("=" * 60)


if __name__ == "__main__":

    telemetry = (
        run_nominal_simulation()
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    output_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_power_telemetry.csv"
    )

    save_telemetry(
        telemetry,
        output_file,
    )

    print_summary(
        telemetry
    )

    print()
    print("Telemetry saved to:")
    print(output_file)