from __future__ import annotations

import numpy as np
import pandas as pd


from src.detection.rule_based_detector import (
    RuleBasedDetector,
)

from src.detection.model_based_detector_refined import (
    RefinedModelBasedDetector,
)

from src.ml.feature_pipeline import (
    ALL_FEATURES,
    build_ml_features,
)

from src.ml.subsystem_isolation_forest import (
    SUBSYSTEM_FEATURES,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


# ==========================================================
# FAULT CLASSES
# ==========================================================

FAULT_CLASSES = [

    "solar_array_degradation",

    "battery_degradation",

    "thermal_anomaly",

    "reaction_wheel_degradation",

    "battery_voltage_sensor_drift",

    "telemetry_dropout",
]


MEASURED_CHANNELS = [

    "battery_soc_measured",

    "battery_voltage_measured",

    "battery_current_measured",

    "battery_temp_measured",

    "electronics_temp_measured",

    "wheel_speed_measured",

    "wheel_current_measured",

    "wheel_temp_measured",
]


# ==========================================================
# MODEL-BASED RESIDUAL FEATURES
# ==========================================================

MODEL_RESIDUAL_FEATURES = [

    "solar_relative_residual",

    "battery_innovation_v",

    "battery_temp_residual_c",

    "electronics_temp_residual_c",

    "wheel_current_residual_a",

    "wheel_temp_residual_c",
]


# ==========================================================
# RULE-BASED DIAGNOSTIC FEATURES
# ==========================================================

RULE_FEATURES = [

    "rule_detected",

    "rule_confidence",

] + [

    f"rule_pred_{fault_type}"

    for fault_type
    in FAULT_CLASSES
]


# ==========================================================
# PHYSICS DIAGNOSTIC FEATURES
# ==========================================================

PHYSICS_FEATURES = [

    "physics_detected",

    "physics_confidence",

] + [

    f"physics_pred_{fault_type}"

    for fault_type
    in FAULT_CLASSES
]


# ==========================================================
# ML EVIDENCE FEATURES
# ==========================================================

ML_FEATURES = [

    "ml_power_ratio",

    "ml_battery_ratio",

    "ml_thermal_ratio",

    "ml_wheel_ratio",

    "ml_max_ratio",

    "ml_persistent_anomaly",
]


# ==========================================================
# MISSING-DATA FEATURES
# ==========================================================

MISSING_FEATURES = [

    f"missing_{channel}"

    for channel
    in MEASURED_CHANNELS

] + [

    "missing_count",
]


# ==========================================================
# FINAL DIAGNOSTIC FEATURE LIST
# ==========================================================

DIAGNOSTIC_FEATURES = (

    ALL_FEATURES

    + MODEL_RESIDUAL_FEATURES

    + RULE_FEATURES

    + PHYSICS_FEATURES

    + ML_FEATURES

    + MISSING_FEATURES
)


# ==========================================================
# ML EVIDENCE
# ==========================================================

def calculate_ml_evidence(
    df,
    ml_detector,
):

    telemetry_features = (
        build_ml_features(
            df
        )
    )

    number_of_samples = len(
        telemetry_features
    )

    subsystem_ratios = {}

    raw_components = []

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = telemetry_features[
            columns
        ]

        missing = (
            x.isna()
            .any(axis=1)
            .to_numpy()
        )

        valid = (
            ~missing
        )

        scores = np.full(
            number_of_samples,
            np.nan,
            dtype=float,
        )

        threshold = (
            ml_detector.thresholds[
                subsystem
            ]
        )

        if valid.any():

            scores[
                valid
            ] = (

                -ml_detector.models[
                    subsystem
                ].score_samples(

                    x.loc[
                        valid
                    ]
                )
            )

        ratios = np.full(
            number_of_samples,
            np.nan,
            dtype=float,
        )

        ratios[
            valid
        ] = (

            scores[
                valid
            ]

            /

            threshold
        )

        raw_anomaly = np.zeros(
            number_of_samples,
            dtype=bool,
        )

        raw_anomaly[
            valid
        ] = (

            scores[
                valid
            ]

            >

            threshold
        )

        # Missing required telemetry is anomalous input.
        raw_anomaly[
            missing
        ] = True

        # Numeric indicator for missing telemetry.
        ratios[
            missing
        ] = 1.25

        subsystem_ratios[
            subsystem
        ] = ratios

        raw_components.append(
            raw_anomaly
        )

    raw_matrix = np.vstack(
        raw_components
    )

    global_raw = np.any(
        raw_matrix,
        axis=0,
    )

    # ======================================================
    # ML PERSISTENCE
    # ======================================================

    persistence = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent = []

    for value in global_raw:

        state = persistence.update(
            bool(value)
        )

        persistent.append(
            state.persistent_anomaly
        )

    return (
        telemetry_features,
        subsystem_ratios,
        np.array(
            persistent,
            dtype=bool,
        ),
    )


# ==========================================================
# ONE-HOT DIAGNOSIS
# ==========================================================

def prediction_flags(
    predicted_label,
    prefix,
):

    flags = {}

    for fault_type in FAULT_CLASSES:

        flags[
            f"{prefix}_pred_{fault_type}"
        ] = int(
            predicted_label
            ==
            fault_type
        )

    return flags


# ==========================================================
# BUILD DATASET
# ==========================================================

def build_diagnostic_dataset(
    df,
    dataset_name,
    ml_detector,
    force_nominal=False,
):

    # ======================================================
    # ML EVIDENCE FIRST
    # ======================================================

    (
        telemetry_features,
        ml_ratios,
        ml_persistent,
    ) = calculate_ml_evidence(
        df=df,
        ml_detector=ml_detector,
    )

    # ======================================================
    # FRESH STATEFUL DETECTORS
    # ======================================================

    rule_detector = (
        RuleBasedDetector()
    )

    physics_detector = (
        RefinedModelBasedDetector()
    )

    records = []

    for position, (_, row) in enumerate(
        df.iterrows()
    ):

        # ==================================================
        # TIME STEP
        # ==================================================

        if position == 0:

            dt_s = 60.0

        else:

            dt_s = (

                float(
                    df.iloc[
                        position
                    ]["time_s"]
                )

                -

                float(
                    df.iloc[
                        position - 1
                    ]["time_s"]
                )
            )

        # ==================================================
        # RULE DETECTOR
        # ==================================================

        rule_result = (
            rule_detector.detect(
                row
            )
        )

        # ==================================================
        # PHYSICS DETECTOR
        # ==================================================

        physics_result = (
            physics_detector.detect(
                row=row,
                dt_s=dt_s,
            )
        )

        # ==================================================
        # TARGET LABEL
        # ==================================================

        if force_nominal:

            target_label = (
                "nominal"
            )

        else:

            target_label = row.get(
                "fault_label",
                "nominal",
            )

            if pd.isna(
                target_label
            ):

                target_label = (
                    "nominal"
                )

            target_label = str(
                target_label
            )

        # ==================================================
        # METADATA
        #
        # These columns are saved for analysis but are NOT
        # part of DIAGNOSTIC_FEATURES.
        # ==================================================

        record = {

            "source_dataset":
                dataset_name,

            "time_h":
                float(
                    row["time_h"]
                ),

            "target_label":
                target_label,
        }

        # ==================================================
        # OBSERVABLE TELEMETRY FEATURES
        # ==================================================

        for feature in ALL_FEATURES:

            record[
                feature
            ] = (
                telemetry_features
                .iloc[
                    position
                ][
                    feature
                ]
            )

        # ==================================================
        # PHYSICS RESIDUALS
        # ==================================================

        record[
            "solar_relative_residual"
        ] = (
            physics_result
            .solar_relative_residual
        )

        record[
            "battery_innovation_v"
        ] = (
            physics_result
            .battery_innovation_v
        )

        record[
            "battery_temp_residual_c"
        ] = (
            physics_result
            .battery_temp_residual_c
        )

        record[
            "electronics_temp_residual_c"
        ] = (
            physics_result
            .electronics_temp_residual_c
        )

        record[
            "wheel_current_residual_a"
        ] = (
            physics_result
            .wheel_current_residual_a
        )

        record[
            "wheel_temp_residual_c"
        ] = (
            physics_result
            .wheel_temp_residual_c
        )

        # ==================================================
        # RULE EVIDENCE
        # ==================================================

        record[
            "rule_detected"
        ] = int(
            rule_result.detected
        )

        record[
            "rule_confidence"
        ] = float(
            rule_result.confidence
        )

        record.update(
            prediction_flags(
                predicted_label=(
                    rule_result
                    .predicted_fault
                ),
                prefix="rule",
            )
        )

        # ==================================================
        # PHYSICS EVIDENCE
        # ==================================================

        record[
            "physics_detected"
        ] = int(
            physics_result.detected
        )

        record[
            "physics_confidence"
        ] = float(
            physics_result.confidence
        )

        record.update(
            prediction_flags(
                predicted_label=(
                    physics_result
                    .predicted_fault
                ),
                prefix="physics",
            )
        )

        # ==================================================
        # ML SUBSYSTEM EVIDENCE
        # ==================================================

        ml_row_ratios = {

            subsystem:
                float(
                    ml_ratios[
                        subsystem
                    ][
                        position
                    ]
                )

            for subsystem
            in SUBSYSTEM_FEATURES
        }

        record[
            "ml_power_ratio"
        ] = (
            ml_row_ratios[
                "power"
            ]
        )

        record[
            "ml_battery_ratio"
        ] = (
            ml_row_ratios[
                "battery"
            ]
        )

        record[
            "ml_thermal_ratio"
        ] = (
            ml_row_ratios[
                "thermal"
            ]
        )

        record[
            "ml_wheel_ratio"
        ] = (
            ml_row_ratios[
                "wheel"
            ]
        )

        record[
            "ml_max_ratio"
        ] = max(
            ml_row_ratios.values()
        )

        record[
            "ml_persistent_anomaly"
        ] = int(
            ml_persistent[
                position
            ]
        )

        # ==================================================
        # MISSING TELEMETRY EVIDENCE
        # ==================================================

        missing_count = 0

        for channel in (
            MEASURED_CHANNELS
        ):

            missing = int(
                pd.isna(
                    row[
                        channel
                    ]
                )
            )

            record[
                f"missing_{channel}"
            ] = missing

            missing_count += (
                missing
            )

        record[
            "missing_count"
        ] = missing_count

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )