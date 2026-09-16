from __future__ import annotations

from collections import deque

import numpy as np

from src.detection.model_based_detector import (
    ModelBasedDetector,
    ModelBasedDetectionResult,
)


class RefinedModelBasedDetector:
    """
    Refined Phase-4 detector.

    Uses the original model-based detector for:

        solar degradation
        battery degradation
        thermal anomaly
        reaction-wheel degradation
        telemetry dropout

    Adds a dedicated persistent-positive-innovation
    diagnostic for battery-voltage sensor drift.

    Motivation:

        Battery degradation:
            elevated EKF residual energy,
            but weak / nonpersistent positive bias.

        Voltage sensor drift:
            persistent positive EKF innovation.

    This refinement is based on the residual signatures
    observed during the dedicated Phase-4 residual analysis.
    """

    def __init__(
        self,
        rolling_window: int = 10,
        persistence_samples: int = 3,
        sensor_bias_sigma_threshold: float = 1.5,
    ):

        self.base_detector = (
            ModelBasedDetector()
        )

        self.rolling_window = (
            rolling_window
        )

        self.persistence_samples = (
            persistence_samples
        )

        self.sensor_bias_sigma_threshold = (
            sensor_bias_sigma_threshold
        )

        self.innovation_history = deque(
            maxlen=rolling_window
        )

        self.sensor_counter = 0

    def detect(
        self,
        row,
        dt_s: float,
    ) -> ModelBasedDetectionResult:

        # ==================================================
        # RUN ORIGINAL MODEL-BASED DETECTOR
        # ==================================================

        result = (
            self.base_detector.detect(
                row=row,
                dt_s=dt_s,
            )
        )

        # ==================================================
        # DO NOT INTERFERE WITH CALIBRATION
        # ==================================================

        if not self.base_detector.calibrated:

            return result

        # ==================================================
        # MISSING BATTERY INNOVATION
        #
        # Typically telemetry dropout.
        # ==================================================

        innovation = (
            result.battery_innovation_v
        )

        if np.isnan(
            innovation
        ):

            return result

        # ==================================================
        # NOMINAL EKF STATISTICS
        # ==================================================

        nominal_mean = (
            self.base_detector.stats[
                "battery_innovation"
            ]["mean"]
        )

        nominal_std = (
            self.base_detector.stats[
                "battery_innovation"
            ]["std"]
        )

        centered_innovation = (
            innovation
            - nominal_mean
        )

        self.innovation_history.append(
            centered_innovation
        )

        # ==================================================
        # WAIT FOR A FULL WINDOW
        # ==================================================

        if (
            len(
                self.innovation_history
            )
            <
            self.rolling_window
        ):

            return result

        innovation_window = np.array(
            self.innovation_history,
            dtype=float,
        )

        rolling_mean = float(
            np.mean(
                innovation_window
            )
        )

        rolling_rms = float(
            np.sqrt(
                np.mean(
                    innovation_window ** 2
                )
            )
        )

        # ==================================================
        # SENSOR-DRIFT SIGNATURE
        #
        # Persistent POSITIVE bias.
        # ==================================================

        sensor_condition = (
            rolling_mean
            >
            self.sensor_bias_sigma_threshold
            * nominal_std
        )

        if sensor_condition:

            self.sensor_counter += 1

        else:

            self.sensor_counter = 0

        sensor_alarm = (
            self.sensor_counter
            >= self.persistence_samples
        )

        # ==================================================
        # IMPORTANT:
        #
        # Only override NOMINAL or BATTERY-DEGRADATION
        # decisions.
        #
        # We do not overwrite:
        #   solar faults
        #   thermal faults
        #   wheel faults
        #   telemetry dropout
        # ==================================================

        allowed_to_override = (
            result.predicted_fault
            in
            [
                "nominal",
                "battery_degradation",
                "battery_voltage_sensor_drift",
            ]
        )

        if (
            sensor_alarm
            and allowed_to_override
        ):

            return ModelBasedDetectionResult(

                detected=True,

                predicted_fault=(
                    "battery_voltage_sensor_drift"
                ),

                confidence=0.95,

                evidence=[
                    (
                        "Persistent positive battery EKF "
                        "innovation detected"
                    ),

                    (
                        f"Rolling innovation mean = "
                        f"{rolling_mean:.4f} V"
                    ),

                    (
                        f"Nominal innovation sigma = "
                        f"{nominal_std:.4f} V"
                    ),

                    (
                        f"Rolling innovation RMS = "
                        f"{rolling_rms:.4f} V"
                    ),
                ],

                solar_residual_w=(
                    result.solar_residual_w
                ),

                solar_relative_residual=(
                    result.solar_relative_residual
                ),

                battery_innovation_v=(
                    result.battery_innovation_v
                ),

                battery_temp_residual_c=(
                    result.battery_temp_residual_c
                ),

                electronics_temp_residual_c=(
                    result.electronics_temp_residual_c
                ),

                wheel_current_residual_a=(
                    result.wheel_current_residual_a
                ),

                wheel_temp_residual_c=(
                    result.wheel_temp_residual_c
                ),
            )

        return result