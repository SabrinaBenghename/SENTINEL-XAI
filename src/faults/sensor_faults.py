from __future__ import annotations

import math


def apply_battery_voltage_drift(
    measured_voltage_v: float,
    severity: float,
    max_bias_v: float = 0.50,
) -> float:
    """
    Apply a positive battery-voltage sensor bias.

    severity:
        0.0 -> no drift
        1.0 -> maximum bias

    Default maximum bias:
        +0.50 V
    """

    severity = max(
        0.0,
        min(1.0, severity),
    )

    bias_v = (
        max_bias_v
        * severity
    )

    return (
        measured_voltage_v
        + bias_v
    )


def apply_telemetry_dropout(
    value: float,
    severity: float,
):
    """
    Drop a telemetry measurement.

    For Phase 2 we treat any active dropout
    as a missing measurement.
    """

    if severity > 0.0:
        return math.nan

    return value