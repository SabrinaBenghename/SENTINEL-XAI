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
# BUILD ML EVIDENCE
# ==========================================================

def calculate_ml_evidence(
    df,
    detector,
):

    features = build_ml_features(
        df
    )

    subsystem_ratios = {}

    raw_anomaly_columns = []

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = features[
            columns
        ]

        scores = (
            -detector.models[
                subsystem
            ].score_samples(
                x
            )
        )

        threshold = (
            detector.thresholds[
                subsystem
            ]
        )

        ratios = (
            scores
            /
            threshold
        )

        subsystem_ratios[
            subsystem
        ] = ratios

        raw_anomaly_columns.append(
            ratios > 1.0
        )

    raw_matrix = np.vstack(
        raw_anomaly_columns
    )

    raw_global = np.any(
        raw_matrix,
        axis=0,
    )

    persistence = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent_global = []

    for raw_value in raw_global:

        state = persistence.update(
            bool(raw_value)
        )

        persistent_global.append(
            state.persistent_anomaly
        )

    persistent_global = np.array(
        persistent_global,
        dtype=bool,
    )

    return (
        subsystem_ratios,
        raw_global,
        persistent_global,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # HEALTHY 6H DATASET
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    df = pd.read_csv(
        nominal_file
    )

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
    # ML EVIDENCE
    # ======================================================

    (
        ml_ratios,
        ml_raw,
        ml_persistent,
    ) = calculate_ml_evidence(
        df=df,
        detector=ml_detector,
    )

    # ======================================================
    # DETECTORS
    # ======================================================

    rule_detector = (
        RuleBasedDetector()
    )

    model_detector = (
        RefinedModelBasedDetector()
    )

    fusion_engine = (
        HybridFusionEngine()
    )

    # ======================================================
    # RUN HYBRID SYSTEM
    # ======================================================

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

        # ----------------------------------------------
        # Phase 3
        # ----------------------------------------------

        rule_result = (
            rule_detector.detect(
                row
            )
        )

        # ----------------------------------------------
        # Phase 4
        # ----------------------------------------------

        model_result = (
            model_detector.detect(
                row=row,
                dt_s=dt_s,
            )
        )

        # ----------------------------------------------
        # Phase 5
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Phase 6 fusion
        # ----------------------------------------------

        hybrid = fusion_engine.fuse(

            rule_result=(
                rule_result
            ),

            model_result=(
                model_result
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

        records.append(
            {
                "time_h":
                    row["time_h"],

                "rule_detected":
                    rule_result.detected,

                "rule_fault":
                    rule_result.predicted_fault,

                "physics_detected":
                    model_result.detected,

                "physics_fault":
                    model_result.predicted_fault,

                "ml_raw_anomaly":
                    bool(
                        ml_raw[
                            index
                        ]
                    ),

                "ml_persistent_anomaly":
                    bool(
                        ml_persistent[
                            index
                        ]
                    ),

                "ml_max_score_ratio":
                    hybrid.ml_max_score_ratio,

                "ml_dominant_subsystem":
                    hybrid.ml_dominant_subsystem,

                "hybrid_detected":
                    hybrid.detected,

                "hybrid_fault":
                    hybrid.predicted_fault,

                "hybrid_confidence":
                    hybrid.confidence,

                "hybrid_sources":
                    "|".join(
                        hybrid.active_sources
                    ),
            }
        )

    results = pd.DataFrame(
        records
    )

    # ======================================================
    # COUNTS
    # ======================================================

    rule_alarms = int(
        results[
            "rule_detected"
        ].sum()
    )

    physics_alarms = int(
        results[
            "physics_detected"
        ].sum()
    )

    ml_raw_alarms = int(
        results[
            "ml_raw_anomaly"
        ].sum()
    )

    ml_persistent_support = int(
        results[
            "ml_persistent_anomaly"
        ].sum()
    )

    hybrid_alarms = int(
        results[
            "hybrid_detected"
        ].sum()
    )

    # ======================================================
    # SAVE TABLE
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

    table_file = (
        tables_dir
        / "experiment_033_hybrid_nominal_validation.csv"
    )

    figure_file = (
        figures_dir
        / "experiment_033_hybrid_nominal_validation.png"
    )

    results.to_csv(
        table_file,
        index=False,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        results[
            "time_h"
        ],
        results[
            "ml_max_score_ratio"
        ],
        label=(
            "Maximum ML subsystem score ratio"
        ),
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="ML anomaly boundary",
    )

    ml_support_rows = (
        results[
            results[
                "ml_persistent_anomaly"
            ]
        ]
    )

    hybrid_alarm_rows = (
        results[
            results[
                "hybrid_detected"
            ]
        ]
    )

    if len(
        ml_support_rows
    ) > 0:

        plt.scatter(
            ml_support_rows[
                "time_h"
            ],
            ml_support_rows[
                "ml_max_score_ratio"
            ],
            label="Persistent ML support",
        )

    if len(
        hybrid_alarm_rows
    ) > 0:

        plt.scatter(
            hybrid_alarm_rows[
                "time_h"
            ],
            hybrid_alarm_rows[
                "ml_max_score_ratio"
            ],
            marker="x",
            s=100,
            label="Hybrid fault alarm",
        )

    plt.xlabel(
        "Time [h]"
    )

    plt.ylabel(
        "Normalized ML Anomaly Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Hybrid Nominal Validation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 82)

    print(
        "SENTINEL-XAI - EXPERIMENT 033"
    )

    print(
        "HYBRID FUSION NOMINAL VALIDATION"
    )

    print("=" * 82)

    print()

    print(
        f"Healthy samples: "
        f"{len(results)}"
    )

    print()

    print(
        "Individual detector activity:"
    )

    print(
        f"  Rule-based alarms: "
        f"{rule_alarms}"
    )

    print(
        f"  Physics/model alarms: "
        f"{physics_alarms}"
    )

    print(
        f"  ML raw anomalies: "
        f"{ml_raw_alarms}"
    )

    print(
        f"  ML persistent support samples: "
        f"{ml_persistent_support}"
    )

    print()

    print(
        f"HYBRID FAULT ALARMS: "
        f"{hybrid_alarms}"
    )

    print()

    if hybrid_alarms == 0:

        print(
            "RESULT: HYBRID NOMINAL "
            "VALIDATION PASSED"
        )

    else:

        print(
            "RESULT: HYBRID FALSE "
            "ALARMS OBSERVED"
        )

    print()

    if (
        ml_persistent_support > 0
        and
        hybrid_alarms == 0
    ):

        print(
            "ML-only anomalies were successfully "
            "prevented from creating hard fault alarms."
        )

    print()

    print(
        "Table saved to:"
    )

    print(
        table_file
    )

    print()

    print(
        "Figure saved to:"
    )

    print(
        figure_file
    )

    print("=" * 82)


if __name__ == "__main__":
    main()