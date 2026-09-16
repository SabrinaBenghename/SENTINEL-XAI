from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.estimation.battery_ekf import BatteryEKF
from src.simulation.thermal_model import ThermalModel
from src.simulation.reaction_wheel_model import ReactionWheelModel


@dataclass
class ModelBasedDetectionResult:
    detected: bool
    predicted_fault: str
    confidence: float

    evidence: list[str]

    solar_residual_w: float
    solar_relative_residual: float

    battery_innovation_v: float

    battery_temp_residual_c: float
    electronics_temp_residual_c: float

    wheel_current_residual_a: float
    wheel_temp_residual_c: float


class ModelBasedDetector:
    """
    SENTINEL-XAI model-based fault detector.

    Uses parallel healthy models:

        Solar physics model
        Battery EKF
        Thermal predictor
        Reaction-wheel predictor

    The detector does NOT use the ground-truth fault label.

    It compares observed telemetry with expected healthy
    behaviour and looks for persistent residuals.
    """

    def __init__(
        self,
        warmup_minutes: float = 30.0,
        rolling_window: int = 10,
        persistence_samples: int = 3,
    ):

        # ==================================================
        # HEALTHY MODELS
        # ==================================================

        self.battery_ekf = BatteryEKF(
            initial_soc=0.85,
            initial_covariance=0.01,
            battery_capacity_wh=1120.0,
            process_noise=1e-6,
            voltage_noise_std_v=0.03,
        )

        self.thermal_predictor = ThermalModel()

        self.wheel_predictor = ReactionWheelModel()

        # ==================================================
        # DETECTOR CONFIGURATION
        # ==================================================

        self.warmup_h = (
            warmup_minutes / 60.0
        )

        self.rolling_window = (
            rolling_window
        )

        self.persistence_samples = (
            persistence_samples
        )

        # Solar degradation is detected when observed
        # sunlight generation is more than 10% below
        # the healthy physical prediction.
        self.solar_relative_threshold = 0.10

        # ==================================================
        # NOMINAL CALIBRATION
        # ==================================================

        self.calibrated = False

        self.calibration = {
            "battery_innovation": [],
            "battery_temp": [],
            "electronics_temp": [],
            "wheel_current": [],
            "wheel_temp": [],
        }

        self.stats = {}

        # ==================================================
        # ROLLING BATTERY RESIDUAL
        # ==================================================

        self.battery_history = deque(
            maxlen=rolling_window
        )

        # ==================================================
        # PERSISTENCE COUNTERS
        # ==================================================

        self.counters = {
            "solar": 0,
            "thermal": 0,
            "wheel": 0,
            "sensor_drift": 0,
            "battery": 0,
        }

    # ======================================================
    # HELPERS
    # ======================================================

    @staticmethod
    def _is_missing(value) -> bool:

        if value is None:
            return True

        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_std(
        values,
        minimum=1e-6,
    ):

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        return max(
            std,
            minimum,
        )

    def _update_counter(
        self,
        name,
        condition,
    ):

        if condition:

            self.counters[name] += 1

        else:

            self.counters[name] = 0

        return (
            self.counters[name]
            >= self.persistence_samples
        )

    # ======================================================
    # SOLAR MODEL
    # ======================================================

    @staticmethod
    def expected_solar_power(
        time_s,
        in_sunlight,
    ):

        if not in_sunlight:
            return 0.0

        orbit_period_s = (
            90.0 * 60.0
        )

        nominal_power_w = 600.0

        variation = (
            1.0
            + 0.03
            * math.sin(
                2.0
                * math.pi
                * time_s
                / orbit_period_s
            )
        )

        return (
            nominal_power_w
            * variation
        )

    # ======================================================
    # NOMINAL CALIBRATION
    # ======================================================

    def _finish_calibration(self):

        self.stats = {}

        for name, values in self.calibration.items():

            self.stats[name] = {
                "mean": float(
                    np.mean(values)
                ),

                "std": self._safe_std(
                    values
                ),
            }

        self.calibrated = True

    # ======================================================
    # DETECTION STEP
    # ======================================================

    def detect(
        self,
        row,
        dt_s: float,
    ) -> ModelBasedDetectionResult:

        time_s = float(
            row["time_s"]
        )

        time_h = float(
            row["time_h"]
        )

        # ==================================================
        # 1. TELEMETRY DROPOUT
        # ==================================================

        measured_channels = [
            "battery_soc_measured",
            "battery_voltage_measured",
            "battery_current_measured",
            "battery_temp_measured",
            "electronics_temp_measured",
            "wheel_speed_measured",
            "wheel_current_measured",
            "wheel_temp_measured",
        ]

        missing_channels = [
            name
            for name in measured_channels
            if self._is_missing(
                row[name]
            )
        ]

        if missing_channels:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "telemetry_dropout"
                ),

                confidence=1.0,

                evidence=[
                    f"{len(missing_channels)} "
                    f"measured channels missing"
                ],

                solar_residual_w=float("nan"),
                solar_relative_residual=float("nan"),

                battery_innovation_v=float("nan"),

                battery_temp_residual_c=float("nan"),
                electronics_temp_residual_c=float("nan"),

                wheel_current_residual_a=float("nan"),
                wheel_temp_residual_c=float("nan"),
            )

        # ==================================================
        # TELEMETRY VALUES
        # ==================================================

        in_sunlight = bool(
            row["in_sunlight"]
        )

        solar_observed = float(
            row["solar_power_w"]
        )

        battery_voltage = float(
            row["battery_voltage_measured"]
        )

        battery_current = float(
            row["battery_current_measured"]
        )

        battery_temp = float(
            row["battery_temp_measured"]
        )

        electronics_temp = float(
            row["electronics_temp_measured"]
        )

        wheel_current = float(
            row["wheel_current_measured"]
        )

        wheel_temp = float(
            row["wheel_temp_measured"]
        )

        # ==================================================
        # 2. SOLAR RESIDUAL
        # ==================================================

        solar_expected = (
            self.expected_solar_power(
                time_s=time_s,
                in_sunlight=in_sunlight,
            )
        )

        solar_residual = (
            solar_observed
            - solar_expected
        )

        if (
            in_sunlight
            and solar_expected > 1.0
        ):

            solar_relative_residual = (
                solar_residual
                / solar_expected
            )

        else:

            solar_relative_residual = 0.0

        # ==================================================
        # 3. BATTERY EKF
        # ==================================================

        battery_state = (
            self.battery_ekf.step(
                time_s=time_s,
                dt_s=dt_s,

                battery_current_a=(
                    battery_current
                ),

                battery_voltage_measured_v=(
                    battery_voltage
                ),
            )
        )

        battery_innovation = (
            battery_state.innovation_v
        )

        # ==================================================
        # 4. HEALTHY THERMAL PREDICTOR
        # ==================================================

        (
            battery_temp_predicted,
            electronics_temp_predicted,
            _,
            _,
        ) = self.thermal_predictor.step(

            dt_s=dt_s,

            battery_current_a=(
                battery_current
            ),

            load_power_w=float(
                row["load_power_w"]
            ),

            in_sunlight=in_sunlight,

            thermal_anomaly=0.0,
        )

        battery_temp_residual = (
            battery_temp
            - battery_temp_predicted
        )

        electronics_temp_residual = (
            electronics_temp
            - electronics_temp_predicted
        )

        # ==================================================
        # 5. HEALTHY REACTION-WHEEL PREDICTOR
        # ==================================================

        healthy_wheel = (
            self.wheel_predictor.step(
                time_s=time_s,
                dt_s=dt_s,
                wheel_degradation=0.0,
            )
        )

        wheel_current_residual = (
            wheel_current
            - healthy_wheel.motor_current_a
        )

        wheel_temp_residual = (
            wheel_temp
            - healthy_wheel.wheel_temperature_c
        )

        # ==================================================
        # 6. CALIBRATION PERIOD
        # ==================================================

        if not self.calibrated:

            self.calibration[
                "battery_innovation"
            ].append(
                battery_innovation
            )

            self.calibration[
                "battery_temp"
            ].append(
                battery_temp_residual
            )

            self.calibration[
                "electronics_temp"
            ].append(
                electronics_temp_residual
            )

            self.calibration[
                "wheel_current"
            ].append(
                wheel_current_residual
            )

            self.calibration[
                "wheel_temp"
            ].append(
                wheel_temp_residual
            )

            if time_h >= self.warmup_h:

                self._finish_calibration()

            return ModelBasedDetectionResult(
                detected=False,
                predicted_fault="nominal",
                confidence=1.0,

                evidence=[
                    "Nominal residual calibration"
                ],

                solar_residual_w=(
                    solar_residual
                ),

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # ==================================================
        # 7. NORMALIZED RESIDUAL SCORES
        # ==================================================

        battery_mean = (
            self.stats[
                "battery_innovation"
            ]["mean"]
        )

        battery_std = (
            self.stats[
                "battery_innovation"
            ]["std"]
        )

        battery_temp_z = (
            (
                battery_temp_residual
                - self.stats[
                    "battery_temp"
                ]["mean"]
            )
            /
            self.stats[
                "battery_temp"
            ]["std"]
        )

        electronics_temp_z = (
            (
                electronics_temp_residual
                - self.stats[
                    "electronics_temp"
                ]["mean"]
            )
            /
            self.stats[
                "electronics_temp"
            ]["std"]
        )

        wheel_current_z = (
            (
                wheel_current_residual
                - self.stats[
                    "wheel_current"
                ]["mean"]
            )
            /
            self.stats[
                "wheel_current"
            ]["std"]
        )

        wheel_temp_z = (
            (
                wheel_temp_residual
                - self.stats[
                    "wheel_temp"
                ]["mean"]
            )
            /
            self.stats[
                "wheel_temp"
            ]["std"]
        )

        # ==================================================
        # 8. BATTERY ROLLING STATISTICS
        # ==================================================

        centered_battery_innovation = (
            battery_innovation
            - battery_mean
        )

        self.battery_history.append(
            centered_battery_innovation
        )

        battery_window = np.array(
            self.battery_history,
            dtype=float,
        )

        battery_rolling_mean = float(
            np.mean(
                battery_window
            )
        )

        battery_rolling_rms = float(
            np.sqrt(
                np.mean(
                    battery_window ** 2
                )
            )
        )

        # ==================================================
        # 9. FAULT CONDITIONS
        # ==================================================

        solar_condition = (
            in_sunlight
            and
            solar_relative_residual
            < -self.solar_relative_threshold
        )

        thermal_score = max(
            abs(
                battery_temp_z
            ),
            abs(
                electronics_temp_z
            ),
        )

        thermal_condition = (
            thermal_score > 3.0
        )

        wheel_score = max(
            abs(
                wheel_current_z
            ),
            abs(
                wheel_temp_z
            ),
        )

        wheel_condition = (
            wheel_score > 3.0
        )

        # Persistent positive EKF innovation is highly
        # characteristic of the voltage sensor drift.
        sensor_drift_condition = (
            len(
                battery_window
            )
            >= self.rolling_window
            and
            battery_rolling_mean
            >
            2.5
            * battery_std
        )

        # Battery degradation raises residual energy,
        # but does not create the same strong positive
        # persistent bias as voltage sensor drift.
        battery_condition = (
            len(
                battery_window
            )
            >= self.rolling_window
            and
            battery_rolling_rms
            >
            1.7
            * battery_std
            and
            abs(
                battery_rolling_mean
            )
            <=
            2.5
            * battery_std
        )

        solar_alarm = (
            self._update_counter(
                "solar",
                solar_condition,
            )
        )

        thermal_alarm = (
            self._update_counter(
                "thermal",
                thermal_condition,
            )
        )

        wheel_alarm = (
            self._update_counter(
                "wheel",
                wheel_condition,
            )
        )

        sensor_alarm = (
            self._update_counter(
                "sensor_drift",
                sensor_drift_condition,
            )
        )

        battery_alarm = (
            self._update_counter(
                "battery",
                battery_condition,
            )
        )

        # ==================================================
        # 10. DIAGNOSIS
        # ==================================================

        # Solar fault
        if solar_alarm:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "solar_array_degradation"
                ),

                confidence=0.95,

                evidence=[
                    (
                        "Solar generation is "
                        f"{abs(solar_relative_residual) * 100:.1f}% "
                        "below healthy model prediction"
                    )
                ],

                solar_residual_w=solar_residual,

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # Thermal fault
        if thermal_alarm:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "thermal_anomaly"
                ),

                confidence=0.95,

                evidence=[
                    (
                        "Thermal residual exceeds "
                        f"nominal model by {thermal_score:.1f} sigma"
                    )
                ],

                solar_residual_w=solar_residual,

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # Reaction-wheel fault
        if wheel_alarm:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "reaction_wheel_degradation"
                ),

                confidence=0.90,

                evidence=[
                    (
                        "Reaction-wheel residual exceeds "
                        f"healthy model by {wheel_score:.1f} sigma"
                    )
                ],

                solar_residual_w=solar_residual,

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # Voltage sensor drift
        if sensor_alarm:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "battery_voltage_sensor_drift"
                ),

                confidence=0.95,

                evidence=[
                    (
                        "Persistent positive EKF innovation: "
                        f"{battery_rolling_mean:.4f} V"
                    )
                ],

                solar_residual_w=solar_residual,

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # Battery degradation
        if battery_alarm:

            return ModelBasedDetectionResult(
                detected=True,

                predicted_fault=(
                    "battery_degradation"
                ),

                confidence=0.85,

                evidence=[
                    (
                        "Battery EKF residual energy elevated: "
                        f"rolling RMS = "
                        f"{battery_rolling_rms:.4f} V"
                    )
                ],

                solar_residual_w=solar_residual,

                solar_relative_residual=(
                    solar_relative_residual
                ),

                battery_innovation_v=(
                    battery_innovation
                ),

                battery_temp_residual_c=(
                    battery_temp_residual
                ),

                electronics_temp_residual_c=(
                    electronics_temp_residual
                ),

                wheel_current_residual_a=(
                    wheel_current_residual
                ),

                wheel_temp_residual_c=(
                    wheel_temp_residual
                ),
            )

        # ==================================================
        # NOMINAL
        # ==================================================

        return ModelBasedDetectionResult(
            detected=False,
            predicted_fault="nominal",
            confidence=1.0,

            evidence=[
                "Model residuals remain within nominal bounds"
            ],

            solar_residual_w=solar_residual,

            solar_relative_residual=(
                solar_relative_residual
            ),

            battery_innovation_v=(
                battery_innovation
            ),

            battery_temp_residual_c=(
                battery_temp_residual
            ),

            electronics_temp_residual_c=(
                electronics_temp_residual
            ),

            wheel_current_residual_a=(
                wheel_current_residual
            ),

            wheel_temp_residual_c=(
                wheel_temp_residual
            ),
        )