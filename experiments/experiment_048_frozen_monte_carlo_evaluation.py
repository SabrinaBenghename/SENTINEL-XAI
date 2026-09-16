from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# ==========================================================
# PROJECT ROOT
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


# ==========================================================
# SENTINEL IMPORTS
# ==========================================================

from src.detection.rule_based_detector import (
    RuleBasedDetector,
)

from src.detection.model_based_detector_refined import (
    RefinedModelBasedDetector,
)

from src.detection.hybrid_fusion import (
    HybridFusionEngine,
)

from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    calculate_ml_evidence,
    build_diagnostic_dataset,
)


# ==========================================================
# CONSTANTS
# ==========================================================

CONFIDENCE_THRESHOLD = 0.70

DT_S = 60.0


FAULT_TYPES = [
    "solar_array_degradation",
    "battery_degradation",
    "thermal_anomaly",
    "reaction_wheel_degradation",
    "battery_voltage_sensor_drift",
    "telemetry_dropout",
]


# ==========================================================
# BASIC HELPERS
# ==========================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return float(
        numerator
        /
        denominator
    )


def load_csv(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def bool_series(
    values,
):

    """
    Robust conversion for bool / int / string columns.
    """

    series = pd.Series(
        values
    )

    if pd.api.types.is_bool_dtype(
        series
    ):

        return (
            series
            .astype(bool)
            .to_numpy()
        )

    if pd.api.types.is_numeric_dtype(
        series
    ):

        return (
            series
            .fillna(0)
            .astype(float)
            .ne(0)
            .to_numpy()
        )

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "t",
            ]
        )
        .to_numpy()
    )


# ==========================================================
# DETECTOR RESULT ADAPTERS
# ==========================================================

def get_result_value(
    result,
    names,
    default=None,
):

    if isinstance(
        result,
        dict,
    ):

        for name in names:

            if name in result:
                return result[
                    name
                ]

    for name in names:

        if hasattr(
            result,
            name,
        ):

            return getattr(
                result,
                name,
            )

    return default


def normalize_fault_label(
    label,
):

    if label is None:
        return "nominal"

    label = str(
        label
    ).strip()

    if label.lower() in {
        "",
        "none",
        "nominal",
        "healthy",
        "no_fault",
        "no fault",
    }:

        return "nominal"

    aliases = {

        "solar_degradation":
            "solar_array_degradation",

        "wheel_degradation":
            "reaction_wheel_degradation",

        "sensor_drift":
            "battery_voltage_sensor_drift",

        "battery_voltage_drift":
            "battery_voltage_sensor_drift",

        "dropout":
            "telemetry_dropout",
    }

    return aliases.get(
        label,
        label,
    )


def extract_detection(
    result,
):

    detected = get_result_value(
        result,
        [
            "detected",
            "fault_detected",
            "alarm",
            "is_fault",
        ],
        default=None,
    )

    label = get_result_value(
        result,
        [
            "predicted_fault",
            "predicted_label",
            "diagnosis",
            "fault_type",
            "label",
        ],
        default=None,
    )

    label = normalize_fault_label(
        label
    )

    if detected is None:

        detected = (
            label
            !=
            "nominal"
        )

    confidence = get_result_value(
        result,
        [
            "confidence",
            "score",
            "diagnostic_confidence",
        ],
        default=np.nan,
    )

    try:

        confidence = float(
            confidence
        )

    except Exception:

        confidence = float(
            "nan"
        )

    return (
        bool(
            detected
        ),
        label,
        confidence,
    )


# ==========================================================
# PHASE-5 ML OUTPUT ADAPTER
# ==========================================================

def ml_evidence_to_frame(
    raw_evidence,
    expected_rows,
):

    """
    Normalize calculate_ml_evidence() output.

    Actual SENTINEL Phase-5 output:

        power
        battery
        thermal
        wheel
        ml_persistent_anomaly

    Canonical names required by this experiment:

        ml_power_ratio
        ml_battery_ratio
        ml_thermal_ratio
        ml_wheel_ratio
        ml_persistent_anomaly
    """

    # ======================================================
    # NORMALIZE RETURN TYPE
    # ======================================================

    if isinstance(
        raw_evidence,
        pd.DataFrame,
    ):

        result = (
            raw_evidence
            .copy()
            .reset_index(
                drop=True
            )
        )

    elif isinstance(
        raw_evidence,
        dict,
    ):

        result = pd.DataFrame(
            raw_evidence
        )

    elif isinstance(
        raw_evidence,
        list,
    ):

        result = pd.DataFrame(
            raw_evidence
        )

    elif isinstance(
        raw_evidence,
        tuple,
    ):

        result = None

        frame_candidate = None
        persistent_candidate = None

        for item in raw_evidence:

            if isinstance(
                item,
                pd.DataFrame,
            ):

                if len(
                    item
                ) == expected_rows:

                    frame_candidate = (
                        item.copy()
                    )

            elif isinstance(
                item,
                dict,
            ):

                try:

                    possible = pd.DataFrame(
                        item
                    )

                    if len(
                        possible
                    ) == expected_rows:

                        frame_candidate = (
                            possible
                        )

                except Exception:

                    pass

            else:

                try:

                    array = np.asarray(
                        item
                    )

                    if (
                        array.ndim
                        ==
                        1
                        and
                        len(
                            array
                        )
                        ==
                        expected_rows
                    ):

                        persistent_candidate = (
                            array
                        )

                except Exception:

                    pass

        if frame_candidate is None:

            raise TypeError(
                "Could not interpret tuple returned "
                "by calculate_ml_evidence()."
            )

        result = (
            frame_candidate
            .reset_index(
                drop=True
            )
        )

        if (
            persistent_candidate
            is not None
            and
            "ml_persistent_anomaly"
            not in
            result.columns
        ):

            result[
                "ml_persistent_anomaly"
            ] = (
                persistent_candidate
            )

    else:

        raise TypeError(
            "Unsupported calculate_ml_evidence() "
            f"return type: "
            f"{type(raw_evidence).__name__}"
        )

    # ======================================================
    # ACTUAL -> CANONICAL COLUMN NAMES
    # ======================================================

    aliases = {

        "ml_power_ratio": [
            "ml_power_ratio",
            "power_ratio",
            "power",
        ],

        "ml_battery_ratio": [
            "ml_battery_ratio",
            "battery_ratio",
            "battery",
        ],

        "ml_thermal_ratio": [
            "ml_thermal_ratio",
            "thermal_ratio",
            "thermal",
        ],

        "ml_wheel_ratio": [
            "ml_wheel_ratio",
            "wheel_ratio",
            "wheel",
        ],

        "ml_persistent_anomaly": [
            "ml_persistent_anomaly",
            "persistent_anomaly",
            "persistent_alarm",
            "ml_persistent_alarm",
        ],
    }

    for canonical, candidates in (
        aliases.items()
    ):

        if canonical in result.columns:
            continue

        source = None

        for candidate in candidates:

            if candidate in result.columns:

                source = candidate
                break

        if source is not None:

            result[
                canonical
            ] = result[
                source
            ]

    # ======================================================
    # VALIDATION
    # ======================================================

    required = [
        "ml_power_ratio",
        "ml_battery_ratio",
        "ml_thermal_ratio",
        "ml_wheel_ratio",
        "ml_persistent_anomaly",
    ]

    missing = [
        column
        for column in required
        if column
        not in
        result.columns
    ]

    if missing:

        raise ValueError(
            "ML evidence is missing required columns: "
            f"{missing}\n"
            f"Available columns: "
            f"{list(result.columns)}"
        )

    if len(
        result
    ) != expected_rows:

        raise ValueError(
            "ML evidence length mismatch. "
            f"Expected {expected_rows}, "
            f"got {len(result)}."
        )

    # ======================================================
    # NUMERIC RATIOS
    # ======================================================

    ratio_columns = [
        "ml_power_ratio",
        "ml_battery_ratio",
        "ml_thermal_ratio",
        "ml_wheel_ratio",
    ]

    for column in ratio_columns:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    result[
        "ml_persistent_anomaly"
    ] = bool_series(
        result[
            "ml_persistent_anomaly"
        ]
    )

    result[
        "ml_max_ratio"
    ] = (
        result[
            ratio_columns
        ]
        .max(
            axis=1
        )
    )

    return (
        result
        .reset_index(
            drop=True
        )
    )


# ==========================================================
# LOCATE FROZEN PHASE-5 MODEL
# ==========================================================

def find_phase5_model():

    candidates = [

        (
            project_root
            / "results"
            / "models"
            / "experiment_031_regime_balanced_subsystem_iforest.joblib"
        ),

        (
            project_root
            / "results"
            / "models"
            / "experiment_028_subsystem_iforest.joblib"
        ),
    ]

    for path in candidates:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Frozen Phase-5 Isolation Forest "
        "model was not found."
    )


# ==========================================================
# RUN PHASE 3 / 4 / 5 / 6
# ==========================================================

def evaluate_detectors(
    df,
    ml_detector,
):

    rule_detector = (
        RuleBasedDetector()
    )

    model_detector = (
        RefinedModelBasedDetector()
    )

    hybrid_engine = (
        HybridFusionEngine()
    )

    # ======================================================
    # PHASE 5 EVIDENCE
    # ======================================================

    raw_ml = (
        calculate_ml_evidence(
            df,
            ml_detector,
        )
    )

    ml_df = (
        ml_evidence_to_frame(
            raw_ml,
            expected_rows=len(
                df
            ),
        )
    )

    rows = []

    # ======================================================
    # SAMPLE LOOP
    # ======================================================

    for index in range(
        len(
            df
        )
    ):

        telemetry_row = (
            df.iloc[
                index
            ]
        )

        # ==================================================
        # PHASE 3 — RULES
        # ==================================================

        rule_result = (
            rule_detector.detect(
                telemetry_row
            )
        )

        (
            rule_detected,
            rule_label,
            rule_confidence,
        ) = extract_detection(
            rule_result
        )

        # ==================================================
        # PHASE 4 — PHYSICS / MODEL
        # ==================================================

        model_result = (
            model_detector.detect(
                telemetry_row,
                dt_s=DT_S,
            )
        )

        (
            model_detected,
            model_label,
            model_confidence,
        ) = extract_detection(
            model_result
        )

        # ==================================================
        # PHASE 5 — ML SUPPORT
        # ==================================================

        ml_row = (
            ml_df.iloc[
                index
            ]
        )

        ml_persistent = bool(
            ml_row[
                "ml_persistent_anomaly"
            ]
        )

        subsystem_ratios = {

            "power":
                float(
                    ml_row[
                        "ml_power_ratio"
                    ]
                ),

            "battery":
                float(
                    ml_row[
                        "ml_battery_ratio"
                    ]
                ),

            "thermal":
                float(
                    ml_row[
                        "ml_thermal_ratio"
                    ]
                ),

            "wheel":
                float(
                    ml_row[
                        "ml_wheel_ratio"
                    ]
                ),
        }

        # ==================================================
        # PHASE 6 — HYBRID
        # ==================================================

        hybrid_result = (
            hybrid_engine.fuse(

                rule_result=(
                    rule_result
                ),

                model_result=(
                    model_result
                ),

                ml_persistent_anomaly=(
                    ml_persistent
                ),

                ml_subsystem_ratios=(
                    subsystem_ratios
                ),
            )
        )

        (
            hybrid_detected,
            hybrid_label,
            hybrid_confidence,
        ) = extract_detection(
            hybrid_result
        )

        rows.append(
            {
                "time_h":
                    float(
                        telemetry_row[
                            "time_h"
                        ]
                    ),

                "fault_active":
                    bool_series(
                        [
                            telemetry_row[
                                "fault_active"
                            ]
                        ]
                    )[0],

                "true_label":
                    str(
                        telemetry_row[
                            "fault_label"
                        ]
                    ),

                "rule_detected":
                    rule_detected,

                "rule_label":
                    rule_label,

                "rule_confidence":
                    rule_confidence,

                "model_detected":
                    model_detected,

                "model_label":
                    model_label,

                "model_confidence":
                    model_confidence,

                "ml_persistent":
                    ml_persistent,

                "ml_power_ratio":
                    float(
                        ml_row[
                            "ml_power_ratio"
                        ]
                    ),

                "ml_battery_ratio":
                    float(
                        ml_row[
                            "ml_battery_ratio"
                        ]
                    ),

                "ml_thermal_ratio":
                    float(
                        ml_row[
                            "ml_thermal_ratio"
                        ]
                    ),

                "ml_wheel_ratio":
                    float(
                        ml_row[
                            "ml_wheel_ratio"
                        ]
                    ),

                "ml_max_ratio":
                    float(
                        ml_row[
                            "ml_max_ratio"
                        ]
                    ),

                "hybrid_detected":
                    hybrid_detected,

                "hybrid_label":
                    hybrid_label,

                "hybrid_confidence":
                    hybrid_confidence,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# BINARY METRICS
# ==========================================================

def calculate_binary_metrics(
    truth,
    prediction,
):

    truth = np.asarray(
        truth,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    tp = int(
        (
            truth
            &
            prediction
        ).sum()
    )

    tn = int(
        (
            ~truth
            &
            ~prediction
        ).sum()
    )

    fp = int(
        (
            ~truth
            &
            prediction
        ).sum()
    )

    fn = int(
        (
            truth
            &
            ~prediction
        ).sum()
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2
        *
        precision
        *
        recall,
        precision
        +
        recall,
    )

    fpr = safe_divide(
        fp,
        fp + tn,
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD MONTE CARLO MANIFEST
    # ======================================================

    manifest_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_047_monte_carlo_manifest.csv"
    )

    manifest = load_csv(
        manifest_file
    )

    # ======================================================
    # LOAD FROZEN MODELS
    # ======================================================

    phase5_model_file = (
        find_phase5_model()
    )

    phase7_model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_037_random_forest_diagnoser.joblib"
    )

    if not phase7_model_file.exists():

        raise FileNotFoundError(
            f"Frozen Phase-7 model not found:\n"
            f"{phase7_model_file}"
        )

    ml_detector = joblib.load(
        phase5_model_file
    )

    phase7_diagnoser = joblib.load(
        phase7_model_file
    )

    # ======================================================
    # TARGET-LEAKAGE CHECK
    # ======================================================

    forbidden_tokens = [
        "_true",
        "fault_label",
        "fault_active",
        "effective_severity",
        "requested_severity",
        "scenario_",
        "target_label",
    ]

    forbidden_features = [
        feature
        for feature
        in DIAGNOSTIC_FEATURES
        if any(
            token
            in feature.lower()
            for token
            in forbidden_tokens
        )
    ]

    if forbidden_features:

        raise RuntimeError(
            "Evaluation truth leaked into "
            "Phase-7 diagnostic features:\n"
            f"{forbidden_features}"
        )

    # ======================================================
    # STORAGE
    # ======================================================

    detector_frames = []

    diagnosis_frames = []

    scenario_rows = []

    failures = []

    print()
    print(
        "Running frozen SENTINEL-XAI "
        "on 120 unseen Monte Carlo scenarios..."
    )
    print()

    # ======================================================
    # PROCESS SCENARIOS
    # ======================================================

    for run_number, (_, scenario) in enumerate(
        manifest.iterrows(),
        start=1,
    ):

        scenario_id = int(
            scenario[
                "scenario_id"
            ]
        )

        fault_type = str(
            scenario[
                "fault_type"
            ]
        )

        telemetry_file = Path(
            scenario[
                "output_file"
            ]
        )

        try:

            # ==================================================
            # TELEMETRY
            # ==================================================

            df = load_csv(
                telemetry_file
            )

            # ==================================================
            # PHASES 3–6
            # ==================================================

            detector_df = (
                evaluate_detectors(
                    df=df,
                    ml_detector=(
                        ml_detector
                    ),
                )
            )

            detector_df[
                "scenario_id"
            ] = scenario_id

            detector_df[
                "fault_type"
            ] = fault_type

            detector_frames.append(
                detector_df
            )

            # ==================================================
            # PHASE-7 FEATURE PIPELINE
            # ==================================================

            diagnostic_df = (
                build_diagnostic_dataset(
                    df=df,
                    dataset_name=(
                        fault_type
                    ),
                    ml_detector=(
                        ml_detector
                    ),
                    force_nominal=False,
                )
            )

            if not isinstance(
                diagnostic_df,
                pd.DataFrame,
            ):

                diagnostic_df = (
                    pd.DataFrame(
                        diagnostic_df
                    )
                )

            diagnostic_df = (
                diagnostic_df
                .reset_index(
                    drop=True
                )
            )

            missing_features = [
                feature
                for feature in
                DIAGNOSTIC_FEATURES
                if feature
                not in
                diagnostic_df.columns
            ]

            if missing_features:

                raise ValueError(
                    "Phase-7 diagnostic pipeline "
                    "did not produce these features:\n"
                    f"{missing_features}"
                )

            if (
                "time_h"
                not in
                diagnostic_df.columns
            ):

                raise ValueError(
                    "Diagnostic feature dataset "
                    "does not contain time_h."
                )

            # ==================================================
            # ACTIVE FAULT TIMES
            #
            # Ground truth used ONLY to evaluate.
            # ==================================================

            active_mask_original = bool_series(
                df[
                    "fault_active"
                ]
            )

            active_times = set(
                np.round(
                    df.loc[
                        active_mask_original,
                        "time_h",
                    ]
                    .to_numpy(
                        dtype=float
                    ),
                    8,
                )
            )

            diagnostic_df[
                "_time_key"
            ] = np.round(
                diagnostic_df[
                    "time_h"
                ]
                .to_numpy(
                    dtype=float
                ),
                8,
            )

            active_diagnosis = (
                diagnostic_df[
                    diagnostic_df[
                        "_time_key"
                    ]
                    .isin(
                        active_times
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if len(
                active_diagnosis
            ) == 0:

                raise RuntimeError(
                    "No active-fault diagnostic "
                    "samples were produced."
                )

            # ==================================================
            # FROZEN PHASE-7 RANDOM FOREST
            # ==================================================

            x_active = (
                active_diagnosis[
                    DIAGNOSTIC_FEATURES
                ]
            )

            phase7_prediction = (
                phase7_diagnoser.predict(
                    x_active
                )
            )

            raw_labels = np.asarray(
                phase7_prediction.predicted_labels,
                dtype=object,
            )

            diagnosis_confidence = np.asarray(
                phase7_prediction.confidence,
                dtype=float,
            )

            # ==================================================
            # FROZEN 0.70 CONFIDENCE GATE
            # ==================================================

            accepted = (
                diagnosis_confidence
                >=
                CONFIDENCE_THRESHOLD
            )

            final_labels = np.where(
                accepted,
                raw_labels,
                "uncertain",
            )

            true_labels = np.full(
                len(
                    active_diagnosis
                ),
                fault_type,
                dtype=object,
            )

            raw_correct = (
                raw_labels
                ==
                true_labels
            )

            accepted_correct = (
                accepted
                &
                raw_correct
            )

            # ==================================================
            # ALIGN PHASE-6 HYBRID DETECTION
            # ==================================================

            hybrid_lookup = (
                detector_df[
                    [
                        "time_h",
                        "hybrid_detected",
                        "hybrid_label",
                    ]
                ]
                .copy()
            )

            hybrid_lookup[
                "_time_key"
            ] = np.round(
                hybrid_lookup[
                    "time_h"
                ]
                .to_numpy(
                    dtype=float
                ),
                8,
            )

            active_diagnosis = (
                active_diagnosis.merge(
                    hybrid_lookup[
                        [
                            "_time_key",
                            "hybrid_detected",
                            "hybrid_label",
                        ]
                    ],
                    on="_time_key",
                    how="left",
                    validate="many_to_one",
                )
            )

            hybrid_alarm = bool_series(
                active_diagnosis[
                    "hybrid_detected"
                ]
                .fillna(False)
            )

            operational_accept = (
                hybrid_alarm
                &
                accepted
            )

            operational_correct = (
                operational_accept
                &
                raw_correct
            )

            diagnosis_output = pd.DataFrame(
                {
                    "scenario_id":
                        scenario_id,

                    "fault_type":
                        fault_type,

                    "time_h":
                        active_diagnosis[
                            "time_h"
                        ].to_numpy(),

                    "raw_prediction":
                        raw_labels,

                    "confidence":
                        diagnosis_confidence,

                    "accepted":
                        accepted,

                    "final_diagnosis":
                        final_labels,

                    "raw_correct":
                        raw_correct,

                    "hybrid_alarm":
                        hybrid_alarm,

                    "operational_accept":
                        operational_accept,

                    "operational_correct":
                        operational_correct,
                }
            )

            diagnosis_frames.append(
                diagnosis_output
            )

            # ==================================================
            # DETECTION RECALLS
            # ==================================================

            truth_active = bool_series(
                detector_df[
                    "fault_active"
                ]
            )

            prefault_mask = (
                ~truth_active
            )

            rule_active = bool_series(
                detector_df[
                    "rule_detected"
                ]
            )

            model_active = bool_series(
                detector_df[
                    "model_detected"
                ]
            )

            hybrid_active = bool_series(
                detector_df[
                    "hybrid_detected"
                ]
            )

            rule_recall = safe_divide(
                int(
                    (
                        truth_active
                        &
                        rule_active
                    ).sum()
                ),
                int(
                    truth_active.sum()
                ),
            )

            model_recall = safe_divide(
                int(
                    (
                        truth_active
                        &
                        model_active
                    ).sum()
                ),
                int(
                    truth_active.sum()
                ),
            )

            hybrid_recall = safe_divide(
                int(
                    (
                        truth_active
                        &
                        hybrid_active
                    ).sum()
                ),
                int(
                    truth_active.sum()
                ),
            )

            hybrid_prefault_alarms = int(
                (
                    prefault_mask
                    &
                    hybrid_active
                ).sum()
            )

            hybrid_prefault_fpr = safe_divide(
                hybrid_prefault_alarms,
                int(
                    prefault_mask.sum()
                ),
            )

            # ==================================================
            # DIAGNOSIS METRICS
            # ==================================================

            raw_diagnosis_accuracy = float(
                raw_correct.mean()
            )

            selective_coverage = float(
                accepted.mean()
            )

            accepted_accuracy = safe_divide(
                int(
                    accepted_correct.sum()
                ),
                int(
                    accepted.sum()
                ),
            )

            operational_coverage = float(
                operational_accept.mean()
            )

            operational_accuracy = safe_divide(
                int(
                    operational_correct.sum()
                ),
                int(
                    operational_accept.sum()
                ),
            )

            end_to_end_correct_fraction = float(
                operational_correct.mean()
            )

            scenario_success = bool(
                operational_correct.any()
            )

            # ==================================================
            # FIRST CORRECT ACCEPTED DIAGNOSIS
            # ==================================================

            if scenario_success:

                first_position = int(
                    np.where(
                        operational_correct
                    )[0][0]
                )

                first_time_h = float(
                    diagnosis_output.iloc[
                        first_position
                    ][
                        "time_h"
                    ]
                )

                diagnosis_delay_min = max(
                    0.0,
                    (
                        first_time_h
                        -
                        float(
                            scenario[
                                "fault_start_time_h"
                            ]
                        )
                    )
                    *
                    60.0,
                )

            else:

                first_time_h = float(
                    "nan"
                )

                diagnosis_delay_min = float(
                    "nan"
                )

            # ==================================================
            # SCENARIO SUMMARY
            # ==================================================

            scenario_rows.append(
                {
                    "scenario_id":
                        scenario_id,

                    "fault_type":
                        fault_type,

                    "severity":
                        float(
                            scenario[
                                "requested_severity"
                            ]
                        ),

                    "profile":
                        str(
                            scenario[
                                "profile"
                            ]
                        ),

                    "noise_scale":
                        float(
                            scenario[
                                "requested_noise_scale"
                            ]
                        ),

                    "fault_start_time_h":
                        float(
                            scenario[
                                "fault_start_time_h"
                            ]
                        ),

                    "active_samples":
                        int(
                            truth_active.sum()
                        ),

                    "rule_detection_recall":
                        rule_recall,

                    "model_detection_recall":
                        model_recall,

                    "hybrid_detection_recall":
                        hybrid_recall,

                    "hybrid_prefault_alarms":
                        hybrid_prefault_alarms,

                    "hybrid_prefault_fpr":
                        hybrid_prefault_fpr,

                    "raw_diagnosis_accuracy":
                        raw_diagnosis_accuracy,

                    "selective_coverage":
                        selective_coverage,

                    "accepted_diagnosis_accuracy":
                        accepted_accuracy,

                    "operational_coverage":
                        operational_coverage,

                    "operational_accuracy":
                        operational_accuracy,

                    "end_to_end_correct_fraction":
                        end_to_end_correct_fraction,

                    "scenario_correctly_diagnosed":
                        scenario_success,

                    "first_correct_diagnosis_time_h":
                        first_time_h,

                    "diagnosis_delay_min":
                        diagnosis_delay_min,
                }
            )

            print(
                f"[{run_number:03d}/"
                f"{len(manifest):03d}] "
                f"Scenario {scenario_id:04d} "
                f"{fault_type:<34} "
                f"OK"
            )

        except Exception as exc:

            failures.append(
                {
                    "scenario_id":
                        scenario_id,

                    "fault_type":
                        fault_type,

                    "error":
                        repr(
                            exc
                        ),
                }
            )

            print(
                f"[{run_number:03d}/"
                f"{len(manifest):03d}] "
                f"Scenario {scenario_id:04d} "
                f"{fault_type:<34} "
                f"FAILED"
            )

            print(
                f"    {exc}"
            )

    # ======================================================
    # ENSURE SOMETHING WORKED
    # ======================================================

    if len(
        scenario_rows
    ) == 0:

        raise RuntimeError(
            "Experiment 048 could not evaluate "
            "any Monte Carlo scenarios."
        )

    # ======================================================
    # COMBINE RESULTS
    # ======================================================

    detector_all = pd.concat(
        detector_frames,
        ignore_index=True,
    )

    diagnosis_all = pd.concat(
        diagnosis_frames,
        ignore_index=True,
    )

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    failure_df = pd.DataFrame(
        failures,
        columns=[
            "scenario_id",
            "fault_type",
            "error",
        ],
    )

    # ======================================================
    # GLOBAL DETECTION METRICS
    # ======================================================

    truth_active = bool_series(
        detector_all[
            "fault_active"
        ]
    )

    method_definitions = {

        "Phase 3 Rules":
            (
                "rule_detected",
                "rule_label",
            ),

        "Phase 4 Physics":
            (
                "model_detected",
                "model_label",
            ),

        "Phase 6 Hybrid":
            (
                "hybrid_detected",
                "hybrid_label",
            ),
    }

    method_rows = []

    for method_name, (
        detected_column,
        label_column,
    ) in method_definitions.items():

        predicted_active = bool_series(
            detector_all[
                detected_column
            ]
        )

        metrics = (
            calculate_binary_metrics(
                truth_active,
                predicted_active,
            )
        )

        active_rows = (
            detector_all.loc[
                truth_active
            ]
        )

        classification_accuracy = float(
            (
                active_rows[
                    label_column
                ]
                ==
                active_rows[
                    "fault_type"
                ]
            ).mean()
        )

        method_rows.append(
            {
                "method":
                    method_name,

                **metrics,

                "classification_accuracy":
                    classification_accuracy,
            }
        )

    method_df = pd.DataFrame(
        method_rows
    )

    # ======================================================
    # GLOBAL PHASE-7 DIAGNOSIS
    # ======================================================

    raw_true = (
        diagnosis_all[
            "fault_type"
        ]
        .to_numpy()
    )

    raw_pred = (
        diagnosis_all[
            "raw_prediction"
        ]
        .to_numpy()
    )

    raw_accuracy = float(
        accuracy_score(
            raw_true,
            raw_pred,
        )
    )

    raw_balanced_accuracy = float(
        balanced_accuracy_score(
            raw_true,
            raw_pred,
        )
    )

    raw_macro_f1 = float(
        f1_score(
            raw_true,
            raw_pred,
            labels=FAULT_TYPES,
            average="macro",
            zero_division=0,
        )
    )

    accepted = bool_series(
        diagnosis_all[
            "accepted"
        ]
    )

    raw_correct = bool_series(
        diagnosis_all[
            "raw_correct"
        ]
    )

    selective_coverage = float(
        accepted.mean()
    )

    accepted_accuracy = safe_divide(
        int(
            (
                accepted
                &
                raw_correct
            ).sum()
        ),
        int(
            accepted.sum()
        ),
    )

    operational_accept = bool_series(
        diagnosis_all[
            "operational_accept"
        ]
    )

    operational_correct = bool_series(
        diagnosis_all[
            "operational_correct"
        ]
    )

    operational_coverage = float(
        operational_accept.mean()
    )

    operational_accuracy = safe_divide(
        int(
            operational_correct.sum()
        ),
        int(
            operational_accept.sum()
        ),
    )

    end_to_end_correct_fraction = float(
        operational_correct.mean()
    )

    scenario_success_rate = float(
        bool_series(
            scenario_df[
                "scenario_correctly_diagnosed"
            ]
        ).mean()
    )

    delays = (
        scenario_df[
            "diagnosis_delay_min"
        ]
        .dropna()
    )

    mean_diagnosis_delay = float(
        delays.mean()
    )

    median_diagnosis_delay = float(
        delays.median()
    )

    # ======================================================
    # PER-FAULT RESULTS
    # ======================================================

    per_fault_rows = []

    for fault_type in FAULT_TYPES:

        sample_subset = (
            diagnosis_all[
                diagnosis_all[
                    "fault_type"
                ]
                ==
                fault_type
            ]
        )

        scenario_subset = (
            scenario_df[
                scenario_df[
                    "fault_type"
                ]
                ==
                fault_type
            ]
        )

        subset_accepted = bool_series(
            sample_subset[
                "accepted"
            ]
        )

        subset_correct = bool_series(
            sample_subset[
                "raw_correct"
            ]
        )

        subset_operational_accept = (
            bool_series(
                sample_subset[
                    "operational_accept"
                ]
            )
        )

        subset_operational_correct = (
            bool_series(
                sample_subset[
                    "operational_correct"
                ]
            )
        )

        success_values = bool_series(
            scenario_subset[
                "scenario_correctly_diagnosed"
            ]
        )

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "scenarios":
                    int(
                        len(
                            scenario_subset
                        )
                    ),

                "active_samples":
                    int(
                        len(
                            sample_subset
                        )
                    ),

                "hybrid_detection_recall":
                    float(
                        scenario_subset[
                            "hybrid_detection_recall"
                        ].mean()
                    ),

                "raw_diagnosis_accuracy":
                    float(
                        subset_correct.mean()
                    ),

                "selective_coverage":
                    float(
                        subset_accepted.mean()
                    ),

                "accepted_accuracy":
                    safe_divide(
                        int(
                            (
                                subset_accepted
                                &
                                subset_correct
                            ).sum()
                        ),
                        int(
                            subset_accepted.sum()
                        ),
                    ),

                "operational_coverage":
                    float(
                        subset_operational_accept.mean()
                    ),

                "operational_accuracy":
                    safe_divide(
                        int(
                            subset_operational_correct.sum()
                        ),
                        int(
                            subset_operational_accept.sum()
                        ),
                    ),

                "end_to_end_correct_fraction":
                    float(
                        subset_operational_correct.mean()
                    ),

                "scenario_success_rate":
                    float(
                        success_values.mean()
                    ),

                "mean_diagnosis_delay_min":
                    float(
                        scenario_subset[
                            "diagnosis_delay_min"
                        ]
                        .dropna()
                        .mean()
                    ),
            }
        )

    per_fault_df = pd.DataFrame(
        per_fault_rows
    )

    # ======================================================
    # SEVERITY BINS
    # ======================================================

    scenario_df[
        "severity_bin"
    ] = pd.cut(
        scenario_df[
            "severity"
        ],
        bins=[
            0.20,
            0.40,
            0.60,
            0.800001,
        ],
        labels=[
            "0.20-0.40",
            "0.40-0.60",
            "0.60-0.80",
        ],
        include_lowest=True,
    )

    severity_df = (
        scenario_df
        .groupby(
            "severity_bin",
            observed=False,
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            hybrid_detection_recall=(
                "hybrid_detection_recall",
                "mean",
            ),

            raw_diagnosis_accuracy=(
                "raw_diagnosis_accuracy",
                "mean",
            ),

            selective_coverage=(
                "selective_coverage",
                "mean",
            ),

            end_to_end_correct_fraction=(
                "end_to_end_correct_fraction",
                "mean",
            ),

            scenario_success_rate=(
                "scenario_correctly_diagnosed",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # NOISE BINS
    # ======================================================

    scenario_df[
        "noise_bin"
    ] = pd.cut(
        scenario_df[
            "noise_scale"
        ],
        bins=[
            0.75,
            1.00,
            1.25,
            1.500001,
        ],
        labels=[
            "0.75-1.00",
            "1.00-1.25",
            "1.25-1.50",
        ],
        include_lowest=True,
    )

    noise_df = (
        scenario_df
        .groupby(
            "noise_bin",
            observed=False,
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            hybrid_detection_recall=(
                "hybrid_detection_recall",
                "mean",
            ),

            raw_diagnosis_accuracy=(
                "raw_diagnosis_accuracy",
                "mean",
            ),

            selective_coverage=(
                "selective_coverage",
                "mean",
            ),

            end_to_end_correct_fraction=(
                "end_to_end_correct_fraction",
                "mean",
            ),

            scenario_success_rate=(
                "scenario_correctly_diagnosed",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # ABRUPT vs GRADUAL
    # ======================================================

    profile_df = (
        scenario_df
        .groupby(
            "profile"
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            hybrid_detection_recall=(
                "hybrid_detection_recall",
                "mean",
            ),

            raw_diagnosis_accuracy=(
                "raw_diagnosis_accuracy",
                "mean",
            ),

            selective_coverage=(
                "selective_coverage",
                "mean",
            ),

            end_to_end_correct_fraction=(
                "end_to_end_correct_fraction",
                "mean",
            ),

            scenario_success_rate=(
                "scenario_correctly_diagnosed",
                "mean",
            ),

            mean_diagnosis_delay_min=(
                "diagnosis_delay_min",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # STRUCTURAL VALIDATION
    #
    # No performance threshold yet.
    # ======================================================

    checks = {

        "All 120 scenarios evaluated":
            (
                len(
                    scenario_df
                )
                ==
                120
            ),

        "No evaluation failures":
            (
                len(
                    failures
                )
                ==
                0
            ),

        "All six fault classes evaluated":
            (
                set(
                    scenario_df[
                        "fault_type"
                    ]
                )
                ==
                set(
                    FAULT_TYPES
                )
            ),

        "Phase-7 feature set has no target leakage":
            (
                len(
                    forbidden_features
                )
                ==
                0
            ),

        "Confidence threshold remained frozen at 0.70":
            (
                CONFIDENCE_THRESHOLD
                ==
                0.70
            ),

        "Every scenario produced active diagnosis samples":
            bool(
                (
                    scenario_df[
                        "active_samples"
                    ]
                    >
                    0
                ).all()
            ),
    }

    # ======================================================
    # OUTPUT DIRECTORIES
    # ======================================================

    tables_dir = (
        project_root
        / "results"
        / "tables"
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # FILE PATHS
    # ======================================================

    scenario_file = (
        tables_dir
        / "experiment_048_scenario_metrics.csv"
    )

    detector_file = (
        tables_dir
        / "experiment_048_detector_predictions.csv"
    )

    diagnosis_file = (
        tables_dir
        / "experiment_048_diagnosis_predictions.csv"
    )

    method_file = (
        tables_dir
        / "experiment_048_detection_method_metrics.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_048_per_fault_robustness.csv"
    )

    severity_file = (
        tables_dir
        / "experiment_048_severity_robustness.csv"
    )

    noise_file = (
        tables_dir
        / "experiment_048_noise_robustness.csv"
    )

    profile_file = (
        tables_dir
        / "experiment_048_profile_robustness.csv"
    )

    failure_file = (
        tables_dir
        / "experiment_048_failures.csv"
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    scenario_df.to_csv(
        scenario_file,
        index=False,
    )

    detector_all.to_csv(
        detector_file,
        index=False,
    )

    diagnosis_all.to_csv(
        diagnosis_file,
        index=False,
    )

    method_df.to_csv(
        method_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    severity_df.to_csv(
        severity_file,
        index=False,
    )

    noise_df.to_csv(
        noise_file,
        index=False,
    )

    profile_df.to_csv(
        profile_file,
        index=False,
    )

    failure_df.to_csv(
        failure_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — DETECTOR COMPARISON
    # ======================================================

    method_figure = (
        figures_dir
        / "experiment_048_monte_carlo_detector_comparison.png"
    )

    x = np.arange(
        len(
            method_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        x - width,
        method_df[
            "precision"
        ],
        width=width,
        label="Precision",
    )

    plt.bar(
        x,
        method_df[
            "recall"
        ],
        width=width,
        label="Recall",
    )

    plt.bar(
        x + width,
        method_df[
            "f1"
        ],
        width=width,
        label="F1",
    )

    plt.xticks(
        x,
        method_df[
            "method"
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Frozen Detection on Monte Carlo Scenarios"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        method_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — PER-FAULT GENERALIZATION
    # ======================================================

    fault_figure = (
        figures_dir
        / "experiment_048_per_fault_generalization.png"
    )

    x = np.arange(
        len(
            per_fault_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width,
        per_fault_df[
            "raw_diagnosis_accuracy"
        ],
        width=width,
        label="Raw Phase-7 accuracy",
    )

    plt.bar(
        x,
        per_fault_df[
            "selective_coverage"
        ],
        width=width,
        label="0.70 confidence coverage",
    )

    plt.bar(
        x + width,
        per_fault_df[
            "end_to_end_correct_fraction"
        ],
        width=width,
        label="End-to-end correct fraction",
    )

    plt.xticks(
        x,
        per_fault_df[
            "fault_type"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Per-Fault Monte Carlo Generalization"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fault_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — SEVERITY ROBUSTNESS
    # ======================================================

    severity_figure = (
        figures_dir
        / "experiment_048_severity_robustness.png"
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        severity_df[
            "severity_bin"
        ].astype(str),
        severity_df[
            "hybrid_detection_recall"
        ],
        marker="o",
        label="Hybrid detection recall",
    )

    plt.plot(
        severity_df[
            "severity_bin"
        ].astype(str),
        severity_df[
            "raw_diagnosis_accuracy"
        ],
        marker="o",
        label="Raw diagnosis accuracy",
    )

    plt.plot(
        severity_df[
            "severity_bin"
        ].astype(str),
        severity_df[
            "end_to_end_correct_fraction"
        ],
        marker="o",
        label="End-to-end correct fraction",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Fault Severity"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Robustness vs Fault Severity"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        severity_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 4 — NOISE ROBUSTNESS
    # ======================================================

    noise_figure = (
        figures_dir
        / "experiment_048_noise_robustness.png"
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        noise_df[
            "noise_bin"
        ].astype(str),
        noise_df[
            "hybrid_detection_recall"
        ],
        marker="o",
        label="Hybrid detection recall",
    )

    plt.plot(
        noise_df[
            "noise_bin"
        ].astype(str),
        noise_df[
            "raw_diagnosis_accuracy"
        ],
        marker="o",
        label="Raw diagnosis accuracy",
    )

    plt.plot(
        noise_df[
            "noise_bin"
        ].astype(str),
        noise_df[
            "end_to_end_correct_fraction"
        ],
        marker="o",
        label="End-to-end correct fraction",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Sensor-Noise Scale"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Robustness vs Sensor Noise"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        noise_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 100)

    print(
        "SENTINEL-XAI - EXPERIMENT 048"
    )

    print(
        "FROZEN MONTE CARLO GENERALIZATION EVALUATION"
    )

    print("=" * 100)

    print()

    print(
        f"Fresh scenarios evaluated: "
        f"{len(scenario_df)}"
    )

    print(
        f"Evaluation failures: "
        f"{len(failures)}"
    )

    print(
        f"Frozen confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        "Retraining performed: NO"
    )

    print(
        "Threshold retuning performed: NO"
    )

    print()

    print("-" * 100)

    print(
        "PHASE 3 / 4 / 6 DETECTION GENERALIZATION"
    )

    print("-" * 100)

    for _, row in (
        method_df.iterrows()
    ):

        print()

        print(
            row[
                "method"
            ]
        )

        print(
            f"  Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall: "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1: "
            f"{row['f1']:.4f}"
        )

        print(
            f"  False-positive rate: "
            f"{row['false_positive_rate']:.4f}"
        )

        print(
            f"  Classification accuracy: "
            f"{row['classification_accuracy']:.4f}"
        )

    print()

    print("-" * 100)

    print(
        "PHASE 7 FROZEN DIAGNOSIS GENERALIZATION"
    )

    print("-" * 100)

    print()

    print(
        f"Raw classification accuracy: "
        f"{raw_accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{raw_balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{raw_macro_f1:.4f}"
    )

    print()

    print(
        f"Confidence-aware coverage: "
        f"{selective_coverage:.4f}"
    )

    print(
        f"Accuracy among accepted diagnoses: "
        f"{accepted_accuracy:.4f}"
    )

    print()

    print("-" * 100)

    print(
        "END-TO-END OPERATIONAL PIPELINE"
    )

    print("-" * 100)

    print()

    print(
        f"Hybrid + confidence-gate coverage: "
        f"{operational_coverage:.4f}"
    )

    print(
        f"Accuracy when operational diagnosis accepted: "
        f"{operational_accuracy:.4f}"
    )

    print(
        f"End-to-end correct active-sample fraction: "
        f"{end_to_end_correct_fraction:.4f}"
    )

    print(
        f"Scenarios with at least one correct "
        f"accepted diagnosis: "
        f"{scenario_success_rate:.4f}"
    )

    print(
        f"Mean correct-diagnosis delay: "
        f"{mean_diagnosis_delay:.2f} min"
    )

    print(
        f"Median correct-diagnosis delay: "
        f"{median_diagnosis_delay:.2f} min"
    )

    print()

    print("-" * 100)

    print(
        "PER-FAULT GENERALIZATION"
    )

    print("-" * 100)

    for _, row in (
        per_fault_df.iterrows()
    ):

        print()

        print(
            row[
                "fault_type"
            ]
        )

        print(
            f"  Raw diagnosis accuracy: "
            f"{row['raw_diagnosis_accuracy']:.4f}"
        )

        print(
            f"  Confidence coverage: "
            f"{row['selective_coverage']:.4f}"
        )

        print(
            f"  Accepted accuracy: "
            f"{row['accepted_accuracy']:.4f}"
        )

        print(
            f"  Hybrid detection recall: "
            f"{row['hybrid_detection_recall']:.4f}"
        )

        print(
            f"  End-to-end correct fraction: "
            f"{row['end_to_end_correct_fraction']:.4f}"
        )

        print(
            f"  Scenario success rate: "
            f"{row['scenario_success_rate']:.4f}"
        )

        print(
            f"  Mean diagnosis delay: "
            f"{row['mean_diagnosis_delay_min']:.2f} min"
        )

    print()

    print("-" * 100)

    print(
        "ABRUPT vs GRADUAL"
    )

    print("-" * 100)

    print()

    print(
        profile_df.to_string(
            index=False
        )
    )

    print()

    print("-" * 100)

    print(
        "STRUCTURAL VALIDATION"
    )

    print("-" * 100)

    passed = 0

    for name, result in (
        checks.items()
    ):

        print()

        print(
            f"{name}: "
            f"{bool(result)}"
        )

        if result:
            passed += 1

    print()

    print(
        f"Passed structural checks: "
        f"{passed}/"
        f"{len(checks)}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Experiment 048 imposes no new "
        "performance threshold."
    )

    print(
        "These Monte Carlo results are independent "
        "generalization measurements."
    )

    print(
        "The frozen models and 0.70 confidence "
        "threshold were not retuned."
    )

    print()

    print(
        "Frozen Phase-5 model:"
    )

    print(
        phase5_model_file
    )

    print()

    print(
        "Frozen Phase-7 model:"
    )

    print(
        phase7_model_file
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        scenario_file
    )

    print(
        method_file
    )

    print(
        per_fault_file
    )

    print(
        severity_file
    )

    print(
        noise_file
    )

    print(
        profile_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        method_figure
    )

    print(
        fault_figure
    )

    print(
        severity_figure
    )

    print(
        noise_figure
    )

    print("=" * 100)


if __name__ == "__main__":
    main()