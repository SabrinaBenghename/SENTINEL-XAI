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

    Sign convention:
        battery_power_w > 0  -> battery is discharging
        battery_power_w < 0  -> battery is charging

    Supported faults:
        solar_degradation:
            0.0 = healthy
            0.30 = 30% solar-array power loss

        battery_degradation:
            0.0 = healthy
            0.30 = 30% effective battery-capacity loss
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

        # ==================================================
        # BATTERY
        # ==================================================

        self.battery_capacity_ah = (
            battery_capacity_ah
        )

        self.nominal_battery_voltage_v = (
            nominal_battery_voltage_v
        )

        self.capacity_wh = (
            battery_capacity_ah
            * nominal_battery_voltage_v
        )

        self.energy_wh = (
            initial_soc
            * self.capacity_wh
        )

        # ==================================================
        # POWER SYSTEM
        # ==================================================

        self.solar_array_power_w = (
            solar_array_power_w
        )

        self.base_load_power_w = (
            base_load_power_w
        )

        # ==================================================
        # ORBIT / SUNLIGHT MODEL
        # ==================================================

        self.orbit_period_s = (
            orbit_period_min
            * 60.0
        )

        self.sunlight_fraction = (
            sunlight_fraction
        )

    # ======================================================
    # SUNLIGHT / ECLIPSE
    # ======================================================

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

    # ======================================================
    # SOLAR GENERATION
    # ======================================================

    def _solar_power(
        self,
        time_s: float,
        in_sunlight: bool,
        solar_degradation: float = 0.0,
    ) -> float:

        if not in_sunlight:
            return 0.0

        # Small nominal orbital variation
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

        nominal_power_w = (
            self.solar_array_power_w
            * variation
        )

        # Keep severity physically valid
        solar_degradation = max(
            0.0,
            min(
                1.0,
                solar_degradation,
            ),
        )

        degraded_power_w = (
            nominal_power_w
            * (
                1.0
                - solar_degradation
            )
        )

        return degraded_power_w

    # ======================================================
    # SPACECRAFT LOAD
    # ======================================================

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

    # ======================================================
    # BATTERY VOLTAGE MODEL
    # ======================================================

    def _battery_voltage(
        self,
        soc: float,
    ) -> float:

        """
        Simplified SOC-voltage relationship.

        SOC = 0.0 -> approximately 26 V
        SOC = 1.0 -> approximately 30 V
        """

        return (
            26.0
            + 4.0 * soc
        )

    # ======================================================
    # ONE POWER-SYSTEM STEP
    # ======================================================

    def step(
        self,
        time_s: float,
        dt_s: float,
        solar_degradation: float = 0.0,
        battery_degradation: float = 0.0,
    ) -> PowerState:

        # --------------------------------------------------
        # 1. SUNLIGHT STATE
        # --------------------------------------------------

        in_sunlight = (
            self._is_in_sunlight(
                time_s
            )
        )

        # --------------------------------------------------
        # 2. SOLAR GENERATION
        # --------------------------------------------------

        solar_power_w = (
            self._solar_power(
                time_s=time_s,
                in_sunlight=in_sunlight,
                solar_degradation=(
                    solar_degradation
                ),
            )
        )

        # --------------------------------------------------
        # 3. SPACECRAFT LOAD
        # --------------------------------------------------

        load_power_w = (
            self._load_power(
                time_s
            )
        )

        # --------------------------------------------------
        # 4. BATTERY POWER BALANCE
        #
        # Positive -> discharge
        # Negative -> charge
        # --------------------------------------------------

        battery_power_w = (
            load_power_w
            - solar_power_w
        )

        # --------------------------------------------------
        # 5. BATTERY DEGRADATION
        #
        # Example:
        #
        # 0.30 degradation
        # -> 70% effective usable capacity
        #
        # Therefore the same energy transfer creates
        # a larger SOC change.
        # --------------------------------------------------

        battery_degradation = max(
            0.0,
            min(
                0.95,
                battery_degradation,
            ),
        )

        capacity_factor = (
            1.0
            - battery_degradation
        )

        # --------------------------------------------------
        # 6. ENERGY UPDATE
        #
        # Energy [Wh] = Power [W] * time [h]
        #
        # Dividing by capacity_factor makes SOC change
        # faster as battery degradation increases.
        # --------------------------------------------------

        energy_change_wh = (
            -battery_power_w
            * dt_s
            / 3600.0
            / capacity_factor
        )

        self.energy_wh += (
            energy_change_wh
        )

        # Keep stored-energy state inside physical limits
        self.energy_wh = max(
            0.0,
            min(
                self.energy_wh,
                self.capacity_wh,
            ),
        )

        # --------------------------------------------------
        # 7. STATE OF CHARGE
        # --------------------------------------------------

        battery_soc = (
            self.energy_wh
            / self.capacity_wh
        )

        # --------------------------------------------------
        # 8. BATTERY VOLTAGE
        # --------------------------------------------------

        battery_voltage_v = (
            self._battery_voltage(
                battery_soc
            )
        )

        # --------------------------------------------------
        # 9. BATTERY CURRENT
        #
        # P = V * I
        #
        # I = P / V
        # --------------------------------------------------

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
            battery_soc=battery_soc,
        )


# ==========================================================
# NOMINAL SIMULATION
# ==========================================================

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


# ==========================================================
# SAVE TELEMETRY
# ==========================================================

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


# ==========================================================
# SUMMARY
# ==========================================================

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


# ==========================================================
# DEFAULT RUN = HEALTHY
# ==========================================================

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
    print(
        "Telemetry saved to:"
    )

    print(
        output_file
    )