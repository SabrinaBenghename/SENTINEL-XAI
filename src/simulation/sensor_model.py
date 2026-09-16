from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorNoiseConfig:
    battery_voltage_std_v: float = 0.03
    battery_current_std_a: float = 0.03
    battery_soc_std_percent: float = 0.20

    battery_temperature_std_c: float = 0.10
    electronics_temperature_std_c: float = 0.10

    wheel_speed_std_rpm: float = 5.0
    wheel_current_std_a: float = 0.005
    wheel_temperature_std_c: float = 0.08


class SensorModel:
    """
    Adds realistic measurement noise to nominal spacecraft telemetry.

    The physical simulator provides the true value.

    The sensor model provides:

        measured_value = true_value + noise
    """

    def __init__(
        self,
        seed: int = 42,
        config: SensorNoiseConfig | None = None,
    ):
        self.rng = np.random.default_rng(seed)

        if config is None:
            config = SensorNoiseConfig()

        self.config = config

    def battery_voltage(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.battery_voltage_std_v,
            )
        )

    def battery_current(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.battery_current_std_a,
            )
        )

    def battery_soc_percent(
        self,
        true_value: float,
    ) -> float:

        measured = (
            true_value
            + self.rng.normal(
                0.0,
                self.config.battery_soc_std_percent,
            )
        )

        return max(
            0.0,
            min(100.0, measured),
        )

    def battery_temperature(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.battery_temperature_std_c,
            )
        )

    def electronics_temperature(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.electronics_temperature_std_c,
            )
        )

    def wheel_speed(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.wheel_speed_std_rpm,
            )
        )

    def wheel_current(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.wheel_current_std_a,
            )
        )

    def wheel_temperature(
        self,
        true_value: float,
    ) -> float:

        return (
            true_value
            + self.rng.normal(
                0.0,
                self.config.wheel_temperature_std_c,
            )
        )


if __name__ == "__main__":

    sensor = SensorModel(
        seed=42,
    )

    print()
    print("=" * 60)
    print("SENTINEL-XAI - SENSOR MODEL TEST")
    print("=" * 60)

    true_voltage = 29.10
    true_temperature = 25.00
    true_wheel_speed = 3000.0

    print()
    print(f"True battery voltage: {true_voltage:.3f} V")

    for _ in range(5):
        measured = sensor.battery_voltage(
            true_voltage
        )

        print(
            f"Measured voltage:    "
            f"{measured:.3f} V"
        )

    print()
    print(
        f"True battery temperature: "
        f"{true_temperature:.2f} °C"
    )

    for _ in range(5):
        measured = sensor.battery_temperature(
            true_temperature
        )

        print(
            f"Measured temperature:     "
            f"{measured:.2f} °C"
        )

    print()
    print(
        f"True reaction-wheel speed: "
        f"{true_wheel_speed:.1f} RPM"
    )

    for _ in range(5):
        measured = sensor.wheel_speed(
            true_wheel_speed
        )

        print(
            f"Measured wheel speed:      "
            f"{measured:.1f} RPM"
        )

    print()
    print("=" * 60)