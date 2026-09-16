from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.detection.rule_based_detector import (
    RuleBasedDetector,
)

from src.detection.model_based_detector_refined import (
    RefinedModelBasedDetector,
)

from src.detection.hybrid_fusion import (
    HybridFusionEngine,
)

from src.ml.feature_pipeline import (
    build_ml_features,
)

from src.ml.subsystem_isolation_forest import (
    SUBSYSTEM_FEATURES,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


# ==========================================================
# HELPERS
# ==========================================================

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


# ==========================================================
# ML EVIDENCE
# ==========================================================

def calculate_ml_evidence(
    df,
    detector,
):

    features = build_ml_features(
        df
    )

    number_of_samples = len(
        features
    )

    subsystem_ratios = {}
    subsystem_anomalies = {}

    global_raw_components = []

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = features[
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
            detector.thresholds[
                subsystem
            ]
        )

        if valid.any():

            scores[
                valid
            ] = (
                -detector.models[
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

        # Ratio used only for fusion/evidence visualization.
        ratios[
            missing
        ] = 1.25

        subsystem_ratios[
            subsystem
        ] = ratios

        subsystem_anomalies[
            subsystem
        ] = raw_anomaly

        global_raw_components.append(
            raw_anomaly
        )

    raw_matrix = np.vstack(
        global_raw_components
    )

    global_raw = np.any(
        raw_matrix,
        axis=0,
    )

    # ======================================================
    # TEMPORAL PERSISTENCE
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

    persistent = np.array(
        persistent,
        dtype=bool,
    )

    return (
        subsystem_ratios,
        subsystem_anomalies,
        global_raw,
        persistent,
    )


# ==========================================================
# EVALUATE ONE COMPLETE SCENARIO
# ==========================================================

def evaluate_scenario(
    dataset_name,
    df,
    ml_detector,
):

    # Fresh stateful detectors for every scenario.
    rule_detector = (
        RuleBasedDetector()
    )

    physics_detector = (
        RefinedModelBasedDetector()
    )

    fusion = (
        HybridFusionEngine()
    )

    (
        ml_ratios,
        ml_subsystem_anomalies,
        ml_raw,
        ml_persistent,
    ) = calculate_ml_evidence(
        df=df,
        detector=ml_detector,
    )

    records = []

    for index, row in df.iterrows():

        if index == 0:

            dt_s = 60.0

        else:

            dt_s = (
                row["time_s"]
                -
                df.iloc[
                    index - 1
                ]["time_s"]
            )

        # ==================================================
        # PHASE 3
        # ==================================================

        rule_result = (
            rule_detector.detect(
                row
            )
        )

        # ==================================================
        # PHASE 4
        # ==================================================

        physics_result = (
            physics_detector.detect(
                row=row,
                dt_s=dt_s,
            )
        )

        # ==================================================
        # PHASE 5
        # ==================================================

        row_ml_ratios = {

            subsystem:
                float(
                    ml_ratios[
                        subsystem
                    ][index]
                )

            for subsystem
            in SUBSYSTEM_FEATURES
        }

        # ==================================================
        # PHASE 6
        # ==================================================

        hybrid = fusion.fuse(

            rule_result=(
                rule_result
            ),

            model_result=(
                physics_result
            ),

            ml_persistent_anomaly=(
                bool(
                    ml_persistent[
                        index
                    ]
                )
            ),

            ml_subsystem_ratios=(
                row_ml_ratios
            ),
        )

        # ==================================================
        # GROUND TRUTH
        # ==================================================

        true_label = row.get(
            "fault_label",
            "nominal",
        )

        if pd.isna(
            true_label
        ):
            true_label = "nominal"

        true_label = str(
            true_label
        )

        true_fault = (
            true_label
            != "nominal"
        )

        # ==================================================
        # RECORD
        # ==================================================

        record = {

            "dataset":
                dataset_name,

            "time_h":
                row["time_h"],

            "true_label":
                true_label,

            "true_fault":
                int(
                    true_fault
                ),

            # ------------------------------
            # RULES
            # ------------------------------

            "rule_detected":
                int(
                    rule_result.detected
                ),

            "rule_label":
                rule_result.predicted_fault,

            # ------------------------------
            # PHYSICS
            # ------------------------------

            "physics_detected":
                int(
                    physics_result.detected
                ),

            "physics_label":
                physics_result.predicted_fault,

            # ------------------------------
            # ML
            # ------------------------------

            "ml_raw_anomaly":
                int(
                    ml_raw[
                        index
                    ]
                ),

            "ml_persistent_anomaly":
                int(
                    ml_persistent[
                        index
                    ]
                ),

            # ------------------------------
            # HYBRID
            # ------------------------------

            "hybrid_detected":
                int(
                    hybrid.detected
                ),

            "hybrid_label":
                hybrid.predicted_fault,

            "hybrid_confidence":
                hybrid.confidence,

            "hybrid_sources":
                "|".join(
                    hybrid.active_sources
                ),

            "hybrid_ml_support":
                int(
                    hybrid.ml_support
                ),

            "ml_max_score_ratio":
                hybrid.ml_max_score_ratio,

            "ml_dominant_subsystem":
                hybrid.ml_dominant_subsystem,
        }

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            record[
                f"ml_{subsystem}_ratio"
            ] = row_ml_ratios[
                subsystem
            ]

            record[
                f"ml_{subsystem}_anomaly"
            ] = int(
                ml_subsystem_anomalies[
                    subsystem
                ][index]
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# BINARY METRICS
# ==========================================================

def calculate_binary_metrics(
    df,
    prediction_column,
):

    tp = int(
        (
            (df["true_fault"] == 1)
            &
            (df[prediction_column] == 1)
        ).sum()
    )

    tn = int(
        (
            (df["true_fault"] == 0)
            &
            (df[prediction_column] == 0)
        ).sum()
    )

    fp = int(
        (
            (df["true_fault"] == 0)
            &
            (df[prediction_column] == 1)
        ).sum()
    )

    fn = int(
        (
            (df["true_fault"] == 1)
            &
            (df[prediction_column] == 0)
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
        2.0
        * precision
        * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    false_positive_rate = (
        safe_divide(
            fp,
            fp + tn,
        )
    )

    return {

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positive_rate":
            false_positive_rate,
    }


# ==========================================================
# CLASSIFICATION ACCURACY
# ==========================================================

def classification_accuracy(
    df,
    label_column,
):

    fault_rows = df[
        df["true_fault"] == 1
    ]

    correct = int(
        (
            fault_rows[
                label_column
            ]
            ==
            fault_rows[
                "true_label"
            ]
        ).sum()
    )

    return safe_divide(
        correct,
        len(
            fault_rows
        ),
    )


# ==========================================================
# DELAY
# ==========================================================

def calculate_delays(
    df,
    start_times,
    detection_column,
    label_column=None,
):

    rows = []

    for fault_type, start_h in (
        start_times.items()
    ):

        subset = df[
            df[
                "dataset"
            ]
            ==
            fault_type
        ]

        if label_column is None:

            detected_rows = subset[
                (
                    subset[
                        "time_h"
                    ]
                    >= start_h
                )
                &
                (
                    subset[
                        detection_column
                    ]
                    == 1
                )
            ]

        else:

            detected_rows = subset[
                (
                    subset[
                        "time_h"
                    ]
                    >= start_h
                )
                &
                (
                    subset[
                        label_column
                    ]
                    ==
                    fault_type
                )
            ]

        if len(
            detected_rows
        ) > 0:

            first_h = float(
                detected_rows[
                    "time_h"
                ].iloc[0]
            )

            delay_min = (
                first_h
                - start_h
            ) * 60.0

            detected = True

        else:

            first_h = float(
                "nan"
            )

            delay_min = float(
                "nan"
            )

            detected = False

        rows.append(
            {
                "fault_type":
                    fault_type,

                "fault_start_h":
                    start_h,

                "first_detection_h":
                    first_h,

                "delay_min":
                    delay_min,

                "detected":
                    detected,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD FROZEN PHASE-5 MODEL
    # ======================================================

    ml_model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_031_regime_balanced_subsystem_iforest.joblib"
    )

    ml_detector = joblib.load(
        ml_model_file
    )

    # ======================================================
    # SIX COMPLETE FAULT SCENARIOS
    # ======================================================

    files = {

        "solar_array_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_006_solar_degradation.csv",

        "battery_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_007_battery_degradation.csv",

        "thermal_anomaly":
            project_root
            / "data"
            / "faults"
            / "experiment_008_thermal_anomaly.csv",

        "reaction_wheel_degradation":
            project_root
            / "data"
            / "faults"
            / "experiment_009_reaction_wheel_degradation.csv",

        "battery_voltage_sensor_drift":
            project_root
            / "data"
            / "faults"
            / "experiment_010_sensor_drift.csv",

        "telemetry_dropout":
            project_root
            / "data"
            / "faults"
            / "experiment_011_telemetry_dropout.csv",
    }

    all_results = []

    print()

    print(
        "Running hybrid SENTINEL-XAI "
        "on all six complete fault scenarios..."
    )

    for name, path in files.items():

        print(
            f"  Processing {name}..."
        )

        df = load_csv(
            path
        )

        evaluated = evaluate_scenario(
            dataset_name=name,
            df=df,
            ml_detector=ml_detector,
        )

        all_results.append(
            evaluated
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ======================================================
    # GLOBAL METRICS
    # ======================================================

    rule_metrics = (
        calculate_binary_metrics(
            results,
            "rule_detected",
        )
    )

    physics_metrics = (
        calculate_binary_metrics(
            results,
            "physics_detected",
        )
    )

    ml_metrics = (
        calculate_binary_metrics(
            results,
            "ml_persistent_anomaly",
        )
    )

    hybrid_metrics = (
        calculate_binary_metrics(
            results,
            "hybrid_detected",
        )
    )

    # ======================================================
    # CLASSIFICATION
    # ======================================================

    rule_classification = (
        classification_accuracy(
            results,
            "rule_label",
        )
    )

    physics_classification = (
        classification_accuracy(
            results,
            "physics_label",
        )
    )

    hybrid_classification = (
        classification_accuracy(
            results,
            "hybrid_label",
        )
    )

    # ======================================================
    # PER-FAULT HYBRID RECALL
    # ======================================================

    per_fault_rows = []

    for fault_type in (
        files.keys()
    ):

        subset = results[
            results[
                "true_label"
            ]
            ==
            fault_type
        ]

        hybrid_detection_recall = (
            safe_divide(
                (
                    subset[
                        "hybrid_detected"
                    ]
                    == 1
                ).sum(),
                len(subset),
            )
        )

        hybrid_classification_recall = (
            safe_divide(
                (
                    subset[
                        "hybrid_label"
                    ]
                    ==
                    fault_type
                ).sum(),
                len(subset),
            )
        )

        physics_classification_recall = (
            safe_divide(
                (
                    subset[
                        "physics_label"
                    ]
                    ==
                    fault_type
                ).sum(),
                len(subset),
            )
        )

        rule_classification_recall = (
            safe_divide(
                (
                    subset[
                        "rule_label"
                    ]
                    ==
                    fault_type
                ).sum(),
                len(subset),
            )
        )

        ml_support_rate = (
            safe_divide(
                subset[
                    "hybrid_ml_support"
                ].sum(),
                len(subset),
            )
        )

        per_fault_rows.append(
            {
                "fault_type":
                    fault_type,

                "samples":
                    len(subset),

                "rule_classification_recall":
                    rule_classification_recall,

                "physics_classification_recall":
                    physics_classification_recall,

                "hybrid_detection_recall":
                    hybrid_detection_recall,

                "hybrid_classification_recall":
                    hybrid_classification_recall,

                "ml_support_rate":
                    ml_support_rate,
            }
        )

    per_fault_df = pd.DataFrame(
        per_fault_rows
    )

    # ======================================================
    # START TIMES
    # ======================================================

    start_times = {

        "solar_array_degradation":
            2.0,

        "battery_degradation":
            2.0,

        "thermal_anomaly":
            2.0,

        "reaction_wheel_degradation":
            2.0,

        "battery_voltage_sensor_drift":
            2.0,

        "telemetry_dropout":
            3.0,
    }

    # ======================================================
    # BINARY DETECTION DELAYS
    # ======================================================

    rule_binary_delay = (
        calculate_delays(
            results,
            start_times,
            "rule_detected",
        )
    )

    physics_binary_delay = (
        calculate_delays(
            results,
            start_times,
            "physics_detected",
        )
    )

    ml_binary_delay = (
        calculate_delays(
            results,
            start_times,
            "ml_persistent_anomaly",
        )
    )

    hybrid_binary_delay = (
        calculate_delays(
            results,
            start_times,
            "hybrid_detected",
        )
    )

    # ======================================================
    # CORRECT DIAGNOSIS DELAYS
    # ======================================================

    rule_diagnosis_delay = (
        calculate_delays(
            results,
            start_times,
            "rule_detected",
            label_column="rule_label",
        )
    )

    physics_diagnosis_delay = (
        calculate_delays(
            results,
            start_times,
            "physics_detected",
            label_column="physics_label",
        )
    )

    hybrid_diagnosis_delay = (
        calculate_delays(
            results,
            start_times,
            "hybrid_detected",
            label_column="hybrid_label",
        )
    )

    # ======================================================
    # MEAN DELAY HELPER
    # ======================================================

    def mean_detected_delay(
        delay_df,
    ):

        detected = delay_df[
            delay_df[
                "detected"
            ]
            == True
        ]

        if len(
            detected
        ) == 0:

            return float(
                "nan"
            )

        return float(
            detected[
                "delay_min"
            ].mean()
        )

    rule_mean_binary_delay = (
        mean_detected_delay(
            rule_binary_delay
        )
    )

    physics_mean_binary_delay = (
        mean_detected_delay(
            physics_binary_delay
        )
    )

    ml_mean_binary_delay = (
        mean_detected_delay(
            ml_binary_delay
        )
    )

    hybrid_mean_binary_delay = (
        mean_detected_delay(
            hybrid_binary_delay
        )
    )

    rule_mean_diagnosis_delay = (
        mean_detected_delay(
            rule_diagnosis_delay
        )
    )

    physics_mean_diagnosis_delay = (
        mean_detected_delay(
            physics_diagnosis_delay
        )
    )

    hybrid_mean_diagnosis_delay = (
        mean_detected_delay(
            hybrid_diagnosis_delay
        )
    )

    # ======================================================
    # METHOD COMPARISON
    # ======================================================

    comparison_df = pd.DataFrame(
        [
            {
                "method":
                    "Phase 3 Rules",

                **rule_metrics,

                "classification_accuracy":
                    rule_classification,

                "mean_binary_delay_min":
                    rule_mean_binary_delay,

                "mean_diagnosis_delay_min":
                    rule_mean_diagnosis_delay,
            },

            {
                "method":
                    "Phase 4 Physics",

                **physics_metrics,

                "classification_accuracy":
                    physics_classification,

                "mean_binary_delay_min":
                    physics_mean_binary_delay,

                "mean_diagnosis_delay_min":
                    physics_mean_diagnosis_delay,
            },

            {
                "method":
                    "Phase 5 ML",

                **ml_metrics,

                "classification_accuracy":
                    np.nan,

                "mean_binary_delay_min":
                    ml_mean_binary_delay,

                "mean_diagnosis_delay_min":
                    np.nan,
            },

            {
                "method":
                    "Phase 6 Hybrid",

                **hybrid_metrics,

                "classification_accuracy":
                    hybrid_classification,

                "mean_binary_delay_min":
                    hybrid_mean_binary_delay,

                "mean_diagnosis_delay_min":
                    hybrid_mean_diagnosis_delay,
            },
        ]
    )

    # ======================================================
    # HYBRID SOURCE CONTRIBUTION
    # ======================================================

    hybrid_alarm_rows = results[
        results[
            "hybrid_detected"
        ]
        == 1
    ]

    source_counts = (
        hybrid_alarm_rows[
            "hybrid_sources"
        ]
        .value_counts()
        .reset_index()
    )

    source_counts.columns = [
        "source_combination",
        "samples",
    ]

    # ======================================================
    # SAVE TABLES
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

    predictions_file = (
        tables_dir
        / "experiment_034_hybrid_predictions.csv"
    )

    comparison_file = (
        tables_dir
        / "experiment_034_method_comparison.csv"
    )

    per_fault_file = (
        tables_dir
        / "experiment_034_hybrid_per_fault.csv"
    )

    hybrid_delay_file = (
        tables_dir
        / "experiment_034_hybrid_diagnosis_delays.csv"
    )

    source_file = (
        tables_dir
        / "experiment_034_hybrid_source_contributions.csv"
    )

    results.to_csv(
        predictions_file,
        index=False,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    per_fault_df.to_csv(
        per_fault_file,
        index=False,
    )

    hybrid_diagnosis_delay.to_csv(
        hybrid_delay_file,
        index=False,
    )

    source_counts.to_csv(
        source_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — METHOD COMPARISON
    # ======================================================

    comparison_figure = (
        figures_dir
        / "experiment_034_detection_method_comparison.png"
    )

    metrics_to_plot = [
        "precision",
        "recall",
        "f1",
    ]

    x = np.arange(
        len(
            metrics_to_plot
        )
    )

    width = 0.20

    plt.figure(
        figsize=(11, 5)
    )

    for index, row in (
        comparison_df.iterrows()
    ):

        values = [
            row[
                metric
            ]
            for metric
            in metrics_to_plot
        ]

        offset = (
            index - 1.5
        ) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=row[
                "method"
            ],
        )

    plt.xticks(
        x,
        [
            "Precision",
            "Recall",
            "F1",
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
        "Rules vs Physics vs ML vs Hybrid"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        comparison_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — PER-FAULT CLASSIFICATION RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_034_hybrid_fault_recall.png"
    )

    x = np.arange(
        len(
            per_fault_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        x - width,
        per_fault_df[
            "rule_classification_recall"
        ],
        width=width,
        label="Rules",
    )

    plt.bar(
        x,
        per_fault_df[
            "physics_classification_recall"
        ],
        width=width,
        label="Physics",
    )

    plt.bar(
        x + width,
        per_fault_df[
            "hybrid_classification_recall"
        ],
        width=width,
        label="Hybrid",
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
        "Classification Recall"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Hybrid Per-Fault Diagnosis"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        recall_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — DIAGNOSIS DELAY
    # ======================================================

    delay_figure = (
        figures_dir
        / "experiment_034_hybrid_diagnosis_delay.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        hybrid_diagnosis_delay[
            "fault_type"
        ],
        hybrid_diagnosis_delay[
            "delay_min"
        ],
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.ylabel(
        "Correct Diagnosis Delay [minutes]"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Hybrid Fault Diagnosis Delay"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        delay_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 86)

    print(
        "SENTINEL-XAI - EXPERIMENT 034"
    )

    print(
        "HYBRID ALL-FAULT EVALUATION"
    )

    print("=" * 86)

    print()

    print(
        "All metrics below use the SAME "
        "six complete scenarios."
    )

    print(
        "Healthy pre-fault samples are included."
    )

    print()

    print("-" * 86)
    print(
        "METHOD COMPARISON"
    )
    print("-" * 86)

    for _, row in (
        comparison_df.iterrows()
    ):

        print()

        print(
            row["method"]
        )

        print(
            f"  Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{row['f1']:.4f}"
        )

        print(
            f"  FPR:       "
            f"{row['false_positive_rate']:.4f}"
        )

        if not pd.isna(
            row[
                "classification_accuracy"
            ]
        ):

            print(
                f"  Classification accuracy: "
                f"{row['classification_accuracy']:.4f}"
            )

        print(
            f"  Mean binary delay: "
            f"{row['mean_binary_delay_min']:.1f} min"
        )

        if not pd.isna(
            row[
                "mean_diagnosis_delay_min"
            ]
        ):

            print(
                f"  Mean diagnosis delay: "
                f"{row['mean_diagnosis_delay_min']:.1f} min"
            )

    # ======================================================
    # PER FAULT
    # ======================================================

    print()

    print("-" * 86)
    print(
        "HYBRID PER-FAULT RESULTS"
    )
    print("-" * 86)

    for _, row in (
        per_fault_df.iterrows()
    ):

        print()

        print(
            row["fault_type"]
        )

        print(
            f"  Detection recall: "
            f"{row['hybrid_detection_recall']:.4f}"
        )

        print(
            f"  Classification recall: "
            f"{row['hybrid_classification_recall']:.4f}"
        )

        print(
            f"  ML support rate: "
            f"{row['ml_support_rate']:.4f}"
        )

    # ======================================================
    # DELAYS
    # ======================================================

    print()

    print("-" * 86)
    print(
        "HYBRID CORRECT-DIAGNOSIS DELAYS"
    )
    print("-" * 86)

    for _, row in (
        hybrid_diagnosis_delay.iterrows()
    ):

        print()

        print(
            row["fault_type"]
        )

        if row[
            "detected"
        ]:

            print(
                f"  Delay: "
                f"{row['delay_min']:.1f} min"
            )

        else:

            print(
                "  NOT CORRECTLY DIAGNOSED"
            )

    # ======================================================
    # SOURCE CONTRIBUTIONS
    # ======================================================

    print()

    print("-" * 86)
    print(
        "HYBRID EVIDENCE SOURCE COMBINATIONS"
    )
    print("-" * 86)

    print()

    if len(
        source_counts
    ) > 0:

        print(
            source_counts.to_string(
                index=False
            )
        )

    else:

        print(
            "No hybrid alarms."
        )

    print()

    print("=" * 86)

    print(
        "Tables saved to:"
    )

    print(
        predictions_file
    )

    print(
        comparison_file
    )

    print(
        per_fault_file
    )

    print(
        hybrid_delay_file
    )

    print(
        source_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        comparison_figure
    )

    print(
        recall_figure
    )

    print(
        delay_figure
    )

    print("=" * 86)


if __name__ == "__main__":
    main()