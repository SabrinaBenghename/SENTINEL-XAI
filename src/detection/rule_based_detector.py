from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DetectionResult:

    detected: bool

    predicted_fault: str

    confidence: float

    evidence: list[str]


class RuleBasedDetector:
    """
    Classical SENTINEL-XAI fault detector.

    Uses only spacecraft telemetry.

    No machine learning.
    No EKF.
    No ground-truth fault labels.

    The rules are intentionally simple because this
    detector will become the classical baseline that
    later ML and hybrid approaches must outperform.
    """

    def __init__(self):

        # ==================================================
        # RULE THRESHOLDS
        # ==================================================

        # Healthy solar power is approximately 580-620 W
        # when in sunlight.
        self.solar_low_threshold_w = 500.0

        # Nominal battery temperature is about 22-25.4 C
        self.battery_temp_high_c = 28.0

        # Nominal electronics temperature is about 24-26.2 C
        self.electronics_temp_high_c = 28.5

        # Nominal wheel current peaks near 0.231 A
        self.wheel_current_high_a = 0.245

        # Battery degradation eventually creates deeper SOC
        # excursions while solar generation remains healthy.
        self.battery_soc_low_percent = 78.0

        # Voltage sensor drift threshold.
        #
        # Nominal residual noise is only a few hundredths V.
        self.voltage_residual_threshold_v = 0.12

        self.measured_channels = [
            "battery_soc_measured",
            "battery_voltage_measured",
            "battery_current_measured",
            "battery_temp_measured",
            "electronics_temp_measured",
            "wheel_speed_measured",
            "wheel_current_measured",
            "wheel_temp_measured",
        ]

    # ======================================================
    # HELPER
    # ======================================================

    @staticmethod
    def _is_missing(value) -> bool:

        if value is None:
            return True

        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    # ======================================================
    # DETECTION
    # ======================================================

    def detect(
        self,
        row,
    ) -> DetectionResult:

        evidence = []

        # ==================================================
        # RULE 1
        # TELEMETRY DROPOUT
        # ==================================================

        missing_channels = []

        for channel in self.measured_channels:

            value = row[channel]

            if self._is_missing(value):

                missing_channels.append(
                    channel
                )

        if missing_channels:

            evidence.append(
                f"{len(missing_channels)} measured "
                f"telemetry channels are missing"
            )

            return DetectionResult(
                detected=True,
                predicted_fault="telemetry_dropout",
                confidence=1.0,
                evidence=evidence,
            )

        # ==================================================
        # READ TELEMETRY
        # ==================================================

        in_sunlight = bool(
            row["in_sunlight"]
        )

        solar_power = float(
            row["solar_power_w"]
        )

        battery_soc = float(
            row["battery_soc_measured"]
        )

        battery_voltage = float(
            row["battery_voltage_measured"]
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

        # ==================================================
        # RULE 2
        # SOLAR-ARRAY DEGRADATION
        # ==================================================

        if (
            in_sunlight
            and solar_power
            < self.solar_low_threshold_w
        ):

            evidence.append(
                f"Solar power is only "
                f"{solar_power:.1f} W during sunlight"
            )

            evidence.append(
                f"Expected healthy sunlight power "
                f"is above "
                f"{self.solar_low_threshold_w:.1f} W"
            )

            return DetectionResult(
                detected=True,
                predicted_fault=(
                    "solar_array_degradation"
                ),
                confidence=0.95,
                evidence=evidence,
            )

        # ==================================================
        # RULE 3
        # THERMAL ANOMALY
        # ==================================================

        if (
            battery_temp
            > self.battery_temp_high_c
            or
            electronics_temp
            > self.electronics_temp_high_c
        ):

            if (
                battery_temp
                > self.battery_temp_high_c
            ):

                evidence.append(
                    f"Battery temperature high: "
                    f"{battery_temp:.2f} C"
                )

            if (
                electronics_temp
                > self.electronics_temp_high_c
            ):

                evidence.append(
                    f"Electronics temperature high: "
                    f"{electronics_temp:.2f} C"
                )

            return DetectionResult(
                detected=True,
                predicted_fault="thermal_anomaly",
                confidence=0.90,
                evidence=evidence,
            )

        # ==================================================
        # RULE 4
        # REACTION-WHEEL DEGRADATION
        # ==================================================

        if (
            wheel_current
            > self.wheel_current_high_a
        ):

            evidence.append(
                f"Reaction-wheel current high: "
                f"{wheel_current:.3f} A"
            )

            evidence.append(
                f"Nominal upper region is below "
                f"{self.wheel_current_high_a:.3f} A"
            )

            return DetectionResult(
                detected=True,
                predicted_fault=(
                    "reaction_wheel_degradation"
                ),
                confidence=0.85,
                evidence=evidence,
            )

        # ==================================================
        # RULE 5
        # BATTERY-VOLTAGE SENSOR DRIFT
        #
        # Our simple battery physics says:
        #
        # V_expected = 26 + 4 * SOC
        #
        # SOC must be fraction 0 -> 1.
        # ==================================================

        expected_voltage = (
            26.0
            + 4.0
            * (
                battery_soc
                / 100.0
            )
        )

        voltage_residual = (
            battery_voltage
            - expected_voltage
        )

        if (
            abs(voltage_residual)
            > self.voltage_residual_threshold_v
        ):

            evidence.append(
                f"Voltage residual is "
                f"{voltage_residual:.3f} V"
            )

            evidence.append(
                f"Allowed residual is approximately "
                f"+/- "
                f"{self.voltage_residual_threshold_v:.3f} V"
            )

            return DetectionResult(
                detected=True,
                predicted_fault=(
                    "battery_voltage_sensor_drift"
                ),
                confidence=0.90,
                evidence=evidence,
            )

        # ==================================================
        # RULE 6
        # BATTERY DEGRADATION
        #
        # Only evaluate while sunlight is available and
        # solar generation itself appears healthy.
        #
        # This reduces confusion with solar degradation.
        # ==================================================

        if (
            in_sunlight
            and solar_power
            >= self.solar_low_threshold_w
            and battery_soc
            < self.battery_soc_low_percent
        ):

            evidence.append(
                f"Battery SOC unusually low: "
                f"{battery_soc:.2f} %"
            )

            evidence.append(
                "Solar generation is currently healthy"
            )

            return DetectionResult(
                detected=True,
                predicted_fault="battery_degradation",
                confidence=0.75,
                evidence=evidence,
            )

        # ==================================================
        # NO FAULT DETECTED
        # ==================================================

        return DetectionResult(
            detected=False,
            predicted_fault="nominal",
            confidence=1.0,
            evidence=[
                "No classical rule threshold exceeded"
            ],
        )


# ==========================================================
# SMALL MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    detector = RuleBasedDetector()

    example = {
        "in_sunlight": 1,

        "solar_power_w": 600.0,

        "battery_soc_measured": 90.0,
        "battery_voltage_measured": 29.60,
        "battery_current_measured": 1.0,

        "battery_temp_measured": 25.0,
        "electronics_temp_measured": 25.0,

        "wheel_speed_measured": 3000.0,
        "wheel_current_measured": 0.19,
        "wheel_temp_measured": 23.0,
    }

    result = detector.detect(
        example
    )

    print()
    print("=" * 65)
    print("SENTINEL-XAI - RULE-BASED DETECTOR TEST")
    print("=" * 65)

    print(
        f"Detected: "
        f"{result.detected}"
    )

    print(
        f"Predicted fault: "
        f"{result.predicted_fault}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.2f}"
    )

    print()

    print("Evidence:")

    for item in result.evidence:
        print(
            f"- {item}"
        )

    print("=" * 65)