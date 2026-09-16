from __future__ import annotations


# ==========================================================
# HUMAN-READABLE LABELS
# ==========================================================

FAULT_NAMES = {

    "nominal":
        "Nominal spacecraft operation",

    "solar_array_degradation":
        "Solar-array degradation",

    "battery_degradation":
        "Battery degradation",

    "thermal_anomaly":
        "Thermal anomaly",

    "reaction_wheel_degradation":
        "Reaction-wheel degradation",

    "battery_voltage_sensor_drift":
        "Battery-voltage sensor drift",

    "telemetry_dropout":
        "Telemetry dropout",

    "uncertain":
        "Uncertain diagnosis",
}


FEATURE_NAMES = {

    "solar_relative_residual":
        "solar-power model residual",

    "battery_innovation_v":
        "battery EKF voltage innovation",

    "battery_temp_residual_c":
        "battery thermal residual",

    "electronics_temp_residual_c":
        "electronics thermal residual",

    "wheel_current_residual_a":
        "reaction-wheel current residual",

    "wheel_temp_residual_c":
        "reaction-wheel temperature residual",

    "battery_temp_measured":
        "measured battery temperature",

    "electronics_temp_measured":
        "measured electronics temperature",

    "battery_voltage_measured":
        "measured battery voltage",

    "battery_current_measured":
        "measured battery current",

    "battery_soc_measured":
        "measured battery state of charge",

    "wheel_current_measured":
        "measured reaction-wheel current",

    "wheel_temp_measured":
        "measured reaction-wheel temperature",

    "wheel_speed_measured":
        "measured reaction-wheel speed",
}


UNITS = {

    "battery_temp_measured": "°C",

    "electronics_temp_measured": "°C",

    "wheel_temp_measured": "°C",

    "battery_voltage_measured": "V",

    "battery_current_measured": "A",

    "wheel_current_measured": "A",

    "wheel_speed_measured": "RPM",

    "battery_soc_measured": "%",

    "solar_relative_residual": "",

    "battery_innovation_v": "V",

    "battery_temp_residual_c": "°C",

    "electronics_temp_residual_c": "°C",

    "wheel_current_residual_a": "A",

    "wheel_temp_residual_c": "°C",
}


# ==========================================================
# HELPERS
# ==========================================================

def humanize_fault(
    fault_name,
):

    return FAULT_NAMES.get(
        fault_name,
        fault_name.replace(
            "_",
            " ",
        ).title(),
    )


def humanize_feature(
    feature,
):

    return FEATURE_NAMES.get(
        feature,
        feature.replace(
            "_",
            " ",
        ),
    )


# ==========================================================
# EVIDENCE CATEGORY
# ==========================================================

def evidence_category(
    feature,
):

    if (
        feature.startswith(
            "rule_"
        )
    ):

        return "rules"

    if (
        feature.startswith(
            "physics_"
        )
        or
        "residual" in feature
        or
        "innovation" in feature
    ):

        return "physics"

    if feature.startswith(
        "ml_"
    ):

        return "ml"

    if (
        feature.startswith(
            "missing_"
        )
        or
        feature
        ==
        "missing_count"
    ):

        return "missing_data"

    return "telemetry"


# ==========================================================
# RULE / PHYSICS PREDICTION FEATURE
# ==========================================================

def explain_prediction_flag(
    feature,
    value,
    shap_value,
    diagnosis,
):

    if feature.startswith(
        "rule_pred_"
    ):

        source = (
            "rule-based detector"
        )

        competing_fault = (
            feature[
                len(
                    "rule_pred_"
                ):
            ]
        )

    else:

        source = (
            "physics/model detector"
        )

        competing_fault = (
            feature[
                len(
                    "physics_pred_"
                ):
            ]
        )

    competing_name = (
        humanize_fault(
            competing_fault
        )
    )

    diagnosis_name = (
        humanize_fault(
            diagnosis
        )
    )

    if value >= 0.5:

        if (
            competing_fault
            ==
            diagnosis
        ):

            return (
                f"The {source} independently identified "
                f"{competing_name}. This agreement pushed "
                f"the learned classifier toward the same "
                f"diagnosis (SHAP {shap_value:+.4f})."
            )

        return (
            f"The {source} identified "
            f"{competing_name}. The classifier used this "
            f"cross-detector pattern while selecting "
            f"{diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    return (
        f"The {source} did not identify "
        f"{competing_name}. The absence of this competing "
        f"diagnosis supported {diagnosis_name} "
        f"(SHAP {shap_value:+.4f})."
    )


# ==========================================================
# GENERIC FEATURE EXPLANATION
# ==========================================================

def explain_feature(
    feature,
    value,
    shap_value,
    diagnosis,
):

    diagnosis_name = (
        humanize_fault(
            diagnosis
        )
    )

    # ======================================================
    # ONE-HOT DIAGNOSIS FLAGS
    # ======================================================

    if (
        feature.startswith(
            "rule_pred_"
        )
        or
        feature.startswith(
            "physics_pred_"
        )
    ):

        return explain_prediction_flag(
            feature=feature,
            value=float(value),
            shap_value=float(
                shap_value
            ),
            diagnosis=diagnosis,
        )

    # ======================================================
    # HARD DETECTION FLAGS
    # ======================================================

    if feature == "rule_detected":

        if value >= 0.5:

            state = (
                "reported an active fault"
            )

        else:

            state = (
                "remained nominal"
            )

        return (
            f"The rule-based detector {state}; this "
            f"contributed toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    if feature == "physics_detected":

        if value >= 0.5:

            state = (
                "reported an active model-based fault"
            )

        else:

            state = (
                "remained nominal"
            )

        return (
            f"The physics/model detector {state}; this "
            f"contributed toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # CONFIDENCE FEATURES
    # ======================================================

    if feature == "rule_confidence":

        return (
            f"Rule-based confidence was "
            f"{float(value):.3f}; this evidence pushed "
            f"the classifier toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    if feature == "physics_confidence":

        return (
            f"Physics/model confidence was "
            f"{float(value):.3f}; this evidence pushed "
            f"the classifier toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # ML RATIOS
    # ======================================================

    if (
        feature.startswith(
            "ml_"
        )
        and
        feature.endswith(
            "_ratio"
        )
    ):

        subsystem = (
            feature
            .replace(
                "ml_",
                "",
                1,
            )
            .replace(
                "_ratio",
                "",
            )
        )

        ratio = float(
            value
        )

        if ratio > 1.0:

            state = (
                "above the learned anomaly boundary"
            )

        else:

            state = (
                "below the learned anomaly boundary"
            )

        return (
            f"The {subsystem} ML anomaly ratio was "
            f"{ratio:.3f}, {state}. Its pattern "
            f"contributed toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    if feature == "ml_persistent_anomaly":

        if value >= 0.5:

            state = (
                "a persistent ML anomaly was active"
            )

        else:

            state = (
                "no persistent ML anomaly was active"
            )

        return (
            f"At this sample, {state}; this contributed "
            f"toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # MISSING TELEMETRY
    # ======================================================

    if feature == "missing_count":

        return (
            f"{int(round(float(value)))} required telemetry "
            f"channels were missing. This contributed "
            f"toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    if feature.startswith(
        "missing_"
    ):

        channel = feature[
            len(
                "missing_"
            ):
        ]

        readable_channel = (
            humanize_feature(
                channel
            )
        )

        if value >= 0.5:

            state = "was missing"

        else:

            state = "was available"

        return (
            f"{readable_channel.capitalize()} {state}; "
            f"this contributed toward {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # RESIDUAL / INNOVATION FEATURES
    # ======================================================

    if (
        "residual" in feature
        or
        "innovation" in feature
    ):

        readable = (
            humanize_feature(
                feature
            )
        )

        unit = UNITS.get(
            feature,
            "",
        )

        return (
            f"The {readable} was "
            f"{float(value):+.4f} {unit}".rstrip()
            +
            f". This model mismatch contributed toward "
            f"{diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # MEASURED TELEMETRY
    # ======================================================

    if feature.endswith(
        "_measured"
    ):

        readable = (
            humanize_feature(
                feature
            )
        )

        unit = UNITS.get(
            feature,
            "",
        )

        return (
            f"The {readable} was "
            f"{float(value):.4f} {unit}".rstrip()
            +
            f". The classifier used this operating value "
            f"as evidence for {diagnosis_name} "
            f"(SHAP {shap_value:+.4f})."
        )

    # ======================================================
    # FALLBACK
    # ======================================================

    readable = (
        humanize_feature(
            feature
        )
    )

    return (
        f"{readable.capitalize()} had value "
        f"{float(value):.4f} and contributed toward "
        f"{diagnosis_name} "
        f"(SHAP {shap_value:+.4f})."
    )