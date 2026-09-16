from __future__ import annotations

import csv
import sys
from pathlib import Path


# ==========================================================
# PROJECT PATH
# ==========================================================

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ==========================================================
# IMPORTS
# ==========================================================

from src.simulation.power_model import SpacecraftPowerModel
from src.simulation.thermal_model import ThermalModel
from src.simulation.reaction_wheel_model import ReactionWheelModel
from src.simulation.sensor_model import SensorModel

from src.faults.fault_manager import (
    FaultConfig,
    FaultManager,
)


class SpacecraftSimulator:

    def __init__(
        self,
        sensor_seed: int = 42,
        faults: list[FaultConfig] | None = None,
    ):

        self.power_model = SpacecraftPowerModel()

        self.thermal_model = ThermalModel()

        self.reaction_wheel_model = ReactionWheelModel()

        self.sensor_model = SensorModel(
            seed=sensor_seed
        )

        self.fault_manager = FaultManager(
            faults=faults
        )

    # ======================================================
    # GET FAULT SEVERITY
    # ======================================================

    def _get_fault_severity(
        self,
        time_s: float,
        fault_type: str,
    ) -> float:

        severities = []

        for fault in self.fault_manager.faults:

            if fault.fault_type != fault_type:
                continue

            state = self.fault_manager.get_state(
                fault=fault,
                time_s=time_s,
            )

            severities.append(
                state.effective_severity
            )

        if not severities:
            return 0.0

        return max(severities)

    # ======================================================
    # FAULT METADATA
    # ======================================================

    def _get_fault_metadata(
        self,
        time_s: float,
    ) -> dict:

        states = self.fault_manager.get_all_states(
            time_s=time_s
        )

        active_states = [
            state
            for state in states
            if state.active
        ]

        if not active_states:

            return {
                "fault_active": 0,
                "fault_type": "none",
                "fault_progress": 0.0,
                "fault_severity": 0.0,
                "fault_label": "nominal",
            }

        primary = max(
            active_states,
            key=lambda state:
                state.effective_severity,
        )

        return {
            "fault_active": 1,
            "fault_type":
                primary.fault_type,

            "fault_progress":
                primary.progress,

            "fault_severity":
                primary.effective_severity,

            "fault_label":
                primary.label,
        }

    # ======================================================
    # SIMULATION STEP
    # ======================================================

    def step(
        self,
        time_s: float,
        dt_s: float,
    ) -> dict:

        # ==================================================
        # 1. FAULT STATES
        # ==================================================

        solar_degradation = self._get_fault_severity(
            time_s,
            "solar_array_degradation",
        )

        battery_degradation = self._get_fault_severity(
            time_s,
            "battery_degradation",
        )

        thermal_anomaly = self._get_fault_severity(
            time_s,
            "thermal_anomaly",
        )

        wheel_degradation = self._get_fault_severity(
            time_s,
            "reaction_wheel_degradation",
        )

        fault_metadata = self._get_fault_metadata(
            time_s
        )

        # ==================================================
        # 2. POWER SYSTEM
        # ==================================================

        power = self.power_model.step(
            time_s=time_s,
            dt_s=dt_s,

            solar_degradation=(
                solar_degradation
            ),

            battery_degradation=(
                battery_degradation
            ),
        )

        # ==================================================
        # 3. THERMAL SYSTEM
        # ==================================================

        (
            battery_temp_c,
            electronics_temp_c,
            battery_heat_w,
            electronics_heat_w,
        ) = self.thermal_model.step(

            dt_s=dt_s,

            battery_current_a=(
                power.battery_current_a
            ),

            load_power_w=(
                power.load_power_w
            ),

            in_sunlight=(
                power.in_sunlight
            ),

            thermal_anomaly=(
                thermal_anomaly
            ),
        )

        # ==================================================
        # 4. REACTION WHEEL
        # ==================================================

        wheel = self.reaction_wheel_model.step(
            time_s=time_s,
            dt_s=dt_s,

            wheel_degradation=(
                wheel_degradation
            ),
        )

        # ==================================================
        # 5. TRUE VALUES
        # ==================================================

        battery_soc_true = (
            power.battery_soc
            * 100.0
        )

        battery_voltage_true = (
            power.battery_voltage_v
        )

        battery_current_true = (
            power.battery_current_a
        )

        battery_temp_true = (
            battery_temp_c
        )

        electronics_temp_true = (
            electronics_temp_c
        )

        wheel_speed_true = (
            wheel.wheel_speed_rpm
        )

        wheel_current_true = (
            wheel.motor_current_a
        )

        wheel_temp_true = (
            wheel.wheel_temperature_c
        )

        # ==================================================
        # 6. SENSOR MEASUREMENTS
        # ==================================================

        battery_soc_measured = (
            self.sensor_model
            .battery_soc_percent(
                battery_soc_true
            )
        )

        battery_voltage_measured = (
            self.sensor_model
            .battery_voltage(
                battery_voltage_true
            )
        )

        battery_current_measured = (
            self.sensor_model
            .battery_current(
                battery_current_true
            )
        )

        battery_temp_measured = (
            self.sensor_model
            .battery_temperature(
                battery_temp_true
            )
        )

        electronics_temp_measured = (
            self.sensor_model
            .electronics_temperature(
                electronics_temp_true
            )
        )

        wheel_speed_measured = (
            self.sensor_model
            .wheel_speed(
                wheel_speed_true
            )
        )

        wheel_current_measured = (
            self.sensor_model
            .wheel_current(
                wheel_current_true
            )
        )

        wheel_temp_measured = (
            self.sensor_model
            .wheel_temperature(
                wheel_temp_true
            )
        )

        # ==================================================
        # 7. TELEMETRY RECORD
        # ==================================================

        return {

            # ---------------- TIME ----------------

            "time_s":
                time_s,

            "time_min":
                time_s / 60.0,

            "time_h":
                time_s / 3600.0,

            "in_sunlight":
                int(power.in_sunlight),

            # ---------------- POWER ----------------

            "solar_power_w":
                power.solar_power_w,

            "load_power_w":
                power.load_power_w,

            "battery_power_w":
                power.battery_power_w,

            # ---------------- BATTERY ----------------

            "battery_soc_true":
                battery_soc_true,

            "battery_soc_measured":
                battery_soc_measured,

            "battery_voltage_true":
                battery_voltage_true,

            "battery_voltage_measured":
                battery_voltage_measured,

            "battery_current_true":
                battery_current_true,

            "battery_current_measured":
                battery_current_measured,

            # ---------------- THERMAL ----------------

            "battery_temp_true":
                battery_temp_true,

            "battery_temp_measured":
                battery_temp_measured,

            "electronics_temp_true":
                electronics_temp_true,

            "electronics_temp_measured":
                electronics_temp_measured,

            "battery_heat_w":
                battery_heat_w,

            "electronics_heat_w":
                electronics_heat_w,

            # ---------------- REACTION WHEEL ----------------

            "wheel_target_speed_rpm":
                wheel.target_speed_rpm,

            "wheel_speed_true":
                wheel_speed_true,

            "wheel_speed_measured":
                wheel_speed_measured,

            "wheel_torque_nm":
                wheel.motor_torque_nm,

            "wheel_current_true":
                wheel_current_true,

            "wheel_current_measured":
                wheel_current_measured,

            "wheel_temp_true":
                wheel_temp_true,

            "wheel_temp_measured":
                wheel_temp_measured,

            # ---------------- PHYSICAL FAULT VALUES ----------------

            "solar_degradation":
                solar_degradation,

            "battery_degradation":
                battery_degradation,

            "thermal_anomaly":
                thermal_anomaly,

            "wheel_degradation":
                wheel_degradation,

            # ---------------- GROUND TRUTH ----------------

            "fault_active":
                fault_metadata[
                    "fault_active"
                ],

            "fault_type":
                fault_metadata[
                    "fault_type"
                ],

            "fault_progress":
                fault_metadata[
                    "fault_progress"
                ],

            "fault_severity":
                fault_metadata[
                    "fault_severity"
                ],

            "fault_label":
                fault_metadata[
                    "fault_label"
                ],
        }


# ==========================================================
# RUN SIMULATION
# ==========================================================

def run_simulation(
    duration_hours: float = 6.0,
    dt_s: float = 60.0,
    sensor_seed: int = 42,
    faults: list[FaultConfig] | None = None,
):

    simulator = SpacecraftSimulator(
        sensor_seed=sensor_seed,
        faults=faults,
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

        record = simulator.step(
            time_s=time_s,
            dt_s=dt_s,
        )

        telemetry.append(
            record
        )

    return telemetry


# ==========================================================
# SAVE TELEMETRY
# ==========================================================

def save_telemetry(
    telemetry: list[dict],
    output_file: Path,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not telemetry:
        raise ValueError(
            "Telemetry is empty."
        )

    fieldnames = list(
        telemetry[0].keys()
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            telemetry
        )


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(
    telemetry: list[dict],
):

    last = telemetry[-1]

    battery_soc = [
        row["battery_soc_true"]
        for row in telemetry
    ]

    battery_temperature = [
        row["battery_temp_true"]
        for row in telemetry
    ]

    wheel_speed = [
        row["wheel_speed_true"]
        for row in telemetry
    ]

    active_fault_samples = sum(
        row["fault_active"]
        for row in telemetry
    )

    print()
    print("=" * 72)

    print(
        "SENTINEL-XAI - UNIFIED SPACECRAFT SIMULATION"
    )

    print("=" * 72)

    print(
        f"Simulation duration: "
        f"{last['time_h']:.2f} h"
    )

    print(
        f"Telemetry samples: "
        f"{len(telemetry)}"
    )

    print()

    print(
        f"Battery SOC range: "
        f"{min(battery_soc):.2f} "
        f"to "
        f"{max(battery_soc):.2f} %"
    )

    print(
        f"Battery temperature range: "
        f"{min(battery_temperature):.2f} "
        f"to "
        f"{max(battery_temperature):.2f} C"
    )

    print(
        f"Reaction-wheel speed range: "
        f"{min(wheel_speed):.1f} "
        f"to "
        f"{max(wheel_speed):.1f} RPM"
    )

    print()

    print(
        f"Active fault samples: "
        f"{active_fault_samples}"
    )

    if active_fault_samples == 0:

        print(
            "Health state: NOMINAL"
        )

    else:

        labels = sorted(
            {
                row["fault_label"]
                for row in telemetry
                if row["fault_active"]
            }
        )

        print(
            "Faults present: "
            + ", ".join(labels)
        )

    print("=" * 72)


# ==========================================================
# DEFAULT RUN = NOMINAL
# ==========================================================

if __name__ == "__main__":

    telemetry = run_simulation(
        duration_hours=6.0,
        dt_s=60.0,
        sensor_seed=42,
        faults=None,
    )

    output_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
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
        "Unified telemetry saved to:"
    )

    print(
        output_file
    )