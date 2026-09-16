from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReactionWheelState:
    time_s: float

    target_speed_rpm: float
    wheel_speed_rpm: float

    angular_acceleration_rad_s2: float

    inertia_torque_nm: float
    friction_torque_nm: float
    motor_torque_nm: float

    motor_current_a: float
    wheel_temperature_c: float


class ReactionWheelModel:
    """
    Simplified spacecraft reaction-wheel model.

    wheel_degradation:
        0.0 = healthy wheel
        1.0 = maximum modeled degradation

    Degradation increases:
        - mechanical friction
        - motor electrical resistance
    """

    def __init__(
        self,
        initial_speed_rpm: float = 3000.0,
        wheel_inertia_kg_m2: float = 0.015,
        friction_coefficient: float = 2.0e-6,
        motor_torque_constant_nm_per_a: float = 0.04,
        idle_current_a: float = 0.15,
        tracking_time_constant_s: float = 120.0,
    ):

        self.wheel_inertia = (
            wheel_inertia_kg_m2
        )

        self.nominal_friction_coefficient = (
            friction_coefficient
        )

        self.motor_torque_constant = (
            motor_torque_constant_nm_per_a
        )

        self.idle_current_a = (
            idle_current_a
        )

        self.tracking_time_constant_s = (
            tracking_time_constant_s
        )

        # RPM -> rad/s
        self.wheel_speed_rad_s = (
            initial_speed_rpm
            * 2.0
            * math.pi
            / 60.0
        )

        # ==================================================
        # THERMAL PARAMETERS
        # ==================================================

        self.wheel_temperature_c = 25.0

        self.environment_temperature_c = 22.0

        self.thermal_capacity_j_per_k = 8000.0

        self.thermal_conductance_w_per_k = 1.0

        self.nominal_motor_resistance_ohm = 1.2

    # ======================================================
    # TARGET SPEED
    # ======================================================

    def _target_speed_rpm(
        self,
        time_s: float,
    ) -> float:

        return (
            3000.0
            + 500.0
            * math.sin(
                2.0
                * math.pi
                * time_s
                / (30.0 * 60.0)
            )
        )

    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        time_s: float,
        dt_s: float,
        wheel_degradation: float = 0.0,
    ) -> ReactionWheelState:

        wheel_degradation = max(
            0.0,
            min(
                1.0,
                wheel_degradation,
            ),
        )

        # ==================================================
        # DEGRADATION PARAMETERS
        # ==================================================

        # At full degradation:
        # friction becomes 6x nominal
        friction_coefficient = (
            self.nominal_friction_coefficient
            * (
                1.0
                + 5.0 * wheel_degradation
            )
        )

        # At full degradation:
        # resistance becomes 3x nominal
        motor_resistance_ohm = (
            self.nominal_motor_resistance_ohm
            * (
                1.0
                + 2.0 * wheel_degradation
            )
        )

        # ==================================================
        # 1. TARGET SPEED
        # ==================================================

        target_speed_rpm = (
            self._target_speed_rpm(
                time_s
            )
        )

        target_speed_rad_s = (
            target_speed_rpm
            * 2.0
            * math.pi
            / 60.0
        )

        # ==================================================
        # 2. SPEED CONTROL
        # ==================================================

        speed_error = (
            target_speed_rad_s
            - self.wheel_speed_rad_s
        )

        angular_acceleration = (
            speed_error
            / self.tracking_time_constant_s
        )

        # ==================================================
        # 3. INERTIA TORQUE
        #
        # tau = J * alpha
        # ==================================================

        inertia_torque = (
            self.wheel_inertia
            * angular_acceleration
        )

        # ==================================================
        # 4. FRICTION TORQUE
        # ==================================================

        friction_torque = (
            friction_coefficient
            * self.wheel_speed_rad_s
        )

        # ==================================================
        # 5. TOTAL MOTOR TORQUE
        # ==================================================

        motor_torque = (
            inertia_torque
            + friction_torque
        )

        # ==================================================
        # 6. MOTOR CURRENT
        #
        # tau = Kt * I
        # ==================================================

        motor_current = (
            self.idle_current_a
            + abs(
                motor_torque
            )
            / self.motor_torque_constant
        )

        # ==================================================
        # 7. UPDATE SPEED
        # ==================================================

        self.wheel_speed_rad_s += (
            angular_acceleration
            * dt_s
        )

        wheel_speed_rpm = (
            self.wheel_speed_rad_s
            * 60.0
            / (
                2.0
                * math.pi
            )
        )

        # ==================================================
        # 8. ELECTRICAL HEATING
        #
        # Q = I^2 R
        # ==================================================

        electrical_heat_w = (
            motor_current ** 2
            * motor_resistance_ohm
        )

        # ==================================================
        # 9. FRICTION HEATING
        #
        # P = tau * omega
        # ==================================================

        friction_heat_w = abs(
            friction_torque
            * self.wheel_speed_rad_s
        )

        total_heat_w = (
            electrical_heat_w
            + friction_heat_w
        )

        # ==================================================
        # 10. HEAT LOSS
        # ==================================================

        heat_loss_w = (
            self.thermal_conductance_w_per_k
            * (
                self.wheel_temperature_c
                - self.environment_temperature_c
            )
        )

        temperature_rate = (
            total_heat_w
            - heat_loss_w
        ) / self.thermal_capacity_j_per_k

        self.wheel_temperature_c += (
            temperature_rate
            * dt_s
        )

        return ReactionWheelState(
            time_s=time_s,

            target_speed_rpm=(
                target_speed_rpm
            ),

            wheel_speed_rpm=(
                wheel_speed_rpm
            ),

            angular_acceleration_rad_s2=(
                angular_acceleration
            ),

            inertia_torque_nm=(
                inertia_torque
            ),

            friction_torque_nm=(
                friction_torque
            ),

            motor_torque_nm=(
                motor_torque
            ),

            motor_current_a=(
                motor_current
            ),

            wheel_temperature_c=(
                self.wheel_temperature_c
            ),
        )


# ==========================================================
# NOMINAL SIMULATION
# ==========================================================

def run_nominal_simulation(
    duration_hours: float = 6.0,
    dt_s: float = 60.0,
):

    model = ReactionWheelModel()

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
            wheel_degradation=0.0,
        )

        telemetry.append(
            state
        )

    return telemetry


# ==========================================================
# SAVE
# ==========================================================

def save_telemetry(
    telemetry: list[ReactionWheelState],
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
                "target_speed_rpm",
                "wheel_speed_rpm",
                "angular_acceleration_rad_s2",
                "inertia_torque_nm",
                "friction_torque_nm",
                "motor_torque_nm",
                "motor_current_a",
                "wheel_temperature_c",
            ]
        )

        for state in telemetry:

            writer.writerow(
                [
                    state.time_s,
                    state.time_s / 60.0,

                    state.target_speed_rpm,
                    state.wheel_speed_rpm,

                    state.angular_acceleration_rad_s2,

                    state.inertia_torque_nm,
                    state.friction_torque_nm,
                    state.motor_torque_nm,

                    state.motor_current_a,

                    state.wheel_temperature_c,
                ]
            )


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary(
    telemetry: list[ReactionWheelState],
):

    speeds = [
        state.wheel_speed_rpm
        for state in telemetry
    ]

    currents = [
        state.motor_current_a
        for state in telemetry
    ]

    torques = [
        state.motor_torque_nm
        for state in telemetry
    ]

    temperatures = [
        state.wheel_temperature_c
        for state in telemetry
    ]

    print()
    print("=" * 65)

    print(
        "SENTINEL-XAI - NOMINAL REACTION WHEEL"
    )

    print("=" * 65)

    print(
        f"Wheel speed range: "
        f"{min(speeds):.1f} "
        f"to "
        f"{max(speeds):.1f} RPM"
    )

    print(
        f"Motor current range: "
        f"{min(currents):.3f} "
        f"to "
        f"{max(currents):.3f} A"
    )

    print(
        f"Motor torque range: "
        f"{min(torques):.6f} "
        f"to "
        f"{max(torques):.6f} N m"
    )

    print(
        f"Wheel temperature range: "
        f"{min(temperatures):.2f} "
        f"to "
        f"{max(temperatures):.2f} C"
    )

    print("=" * 65)


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
        / "nominal_reaction_wheel_telemetry.csv"
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