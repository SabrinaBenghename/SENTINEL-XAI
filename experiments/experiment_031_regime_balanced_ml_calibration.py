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


from src.ml.feature_pipeline import (
    build_ml_features,
)

from src.ml.subsystem_isolation_forest import (
    SUBSYSTEM_FEATURES,
    SubsystemIsolationForest,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


# ==========================================================
# APPLY PERSISTENCE
# ==========================================================

def apply_persistence(
    raw_anomalies,
    required_samples=3,
):

    filter_ = PersistentAnomalyFilter(
        required_samples=required_samples,
    )

    persistent = []
    counters = []

    for value in raw_anomalies:

        result = filter_.update(
            bool(value)
        )

        persistent.append(
            result.persistent_anomaly
        )

        counters.append(
            result.consecutive_count
        )

    return (
        np.array(
            persistent,
            dtype=bool,
        ),
        counters,
    )


# ==========================================================
# SCORE DATASET
# ==========================================================

def score_dataset(
    feature_df,
    detector,
):

    raw_columns = []

    result = pd.DataFrame(
        {
            "time_h":
                feature_df[
                    "time_h"
                ].values,
        }
    )

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        scores = (
            detector.subsystem_score(
                feature_df,
                subsystem,
            )
        )

        threshold = (
            detector.thresholds[
                subsystem
            ]
        )

        anomalies = (
            scores > threshold
        )

        result[
            f"{subsystem}_score"
        ] = scores

        result[
            f"{subsystem}_threshold"
        ] = threshold

        result[
            f"{subsystem}_anomaly"
        ] = anomalies

        raw_columns.append(
            f"{subsystem}_anomaly"
        )

    result[
        "raw_anomaly"
    ] = result[
        raw_columns
    ].any(
        axis=1
    )

    (
        persistent,
        counters,
    ) = apply_persistence(
        result[
            "raw_anomaly"
        ].values,
        required_samples=3,
    )

    result[
        "persistence_count"
    ] = counters

    result[
        "persistent_anomaly"
    ] = persistent

    return result


# ==========================================================
# MAIN
# ==========================================================

def main():

    nominal_24h_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry_24h.csv"
    )

    nominal_6h_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    df_24h = pd.read_csv(
        nominal_24h_file
    )

    features_24h = build_ml_features(
        df_24h
    )

    # ======================================================
    # DEVELOPMENT / HOLDOUT SPLIT
    #
    # First 80%:
    #     model development
    #
    # Final 20%:
    #     held-out nominal evaluation
    # ======================================================

    n = len(
        features_24h
    )

    development_end = int(
        0.80 * n
    )

    development = (
        features_24h
        .iloc[
            :development_end
        ]
        .copy()
    )

    holdout = (
        features_24h
        .iloc[
            development_end:
        ]
        .copy()
    )

    # ======================================================
    # REGIME-BALANCED TRAIN / CALIBRATION SPLIT
    #
    # Every fifth healthy development sample is used
    # for threshold calibration.
    #
    # Therefore calibration spans:
    #
    #   startup transient
    #   eclipse / sunlight cycles
    #   thermal evolution
    #   wheel operating cycles
    #   steady-state operation
    #
    # No fault data is used.
    # ======================================================

    development_index = np.arange(
        len(development)
    )

    calibration_mask = (
        development_index % 5
        == 0
    )

    train = (
        development
        .iloc[
            ~calibration_mask
        ]
        .copy()
    )

    calibration = (
        development
        .iloc[
            calibration_mask
        ]
        .copy()
    )

    # ======================================================
    # TRAIN
    # ======================================================

    detector = (
        SubsystemIsolationForest(
            n_estimators=300,
            random_state=42,
        )
    )

    detector.fit(
        train
    )

    detector.calibrate(
        calibration,
        quantile=0.99,
    )

    # ======================================================
    # HELD-OUT 24H NOMINAL TEST
    # ======================================================

    holdout_result = score_dataset(
        holdout,
        detector,
    )

    # ======================================================
    # ORIGINAL 6H NOMINAL TEST
    #
    # This specifically verifies that the startup transient
    # no longer creates persistent alarms.
    # ======================================================

    df_6h = pd.read_csv(
        nominal_6h_file
    )

    features_6h = build_ml_features(
        df_6h
    )

    nominal_6h_result = score_dataset(
        features_6h,
        detector,
    )

    # ======================================================
    # COUNTS
    # ======================================================

    holdout_raw = int(
        holdout_result[
            "raw_anomaly"
        ].sum()
    )

    holdout_persistent = int(
        holdout_result[
            "persistent_anomaly"
        ].sum()
    )

    nominal_6h_raw = int(
        nominal_6h_result[
            "raw_anomaly"
        ].sum()
    )

    nominal_6h_persistent = int(
        nominal_6h_result[
            "persistent_anomaly"
        ].sum()
    )

    # ======================================================
    # SUBSYSTEM COUNTS ON 6H NOMINAL DATA
    # ======================================================

    subsystem_counts = {}

    for subsystem in (
        SUBSYSTEM_FEATURES
    ):

        subsystem_counts[
            subsystem
        ] = int(
            nominal_6h_result[
                f"{subsystem}_anomaly"
            ].sum()
        )

    # ======================================================
    # OUTPUT PATHS
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

    models_dir = (
        project_root
        / "results"
        / "models"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    holdout_file = (
        tables_dir
        / "experiment_031_holdout_nominal.csv"
    )

    nominal_6h_output = (
        tables_dir
        / "experiment_031_6h_nominal.csv"
    )

    model_file = (
        models_dir
        / "experiment_031_regime_balanced_subsystem_iforest.joblib"
    )

    figure_file = (
        figures_dir
        / "experiment_031_regime_balanced_nominal.png"
    )

    holdout_result.to_csv(
        holdout_file,
        index=False,
    )

    nominal_6h_result.to_csv(
        nominal_6h_output,
        index=False,
    )

    joblib.dump(
        detector,
        model_file,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    plt.figure(
        figsize=(12, 5)
    )

    max_ratio = []

    for index in range(
        len(nominal_6h_result)
    ):

        ratios = []

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            score = (
                nominal_6h_result[
                    f"{subsystem}_score"
                ].iloc[index]
            )

            threshold = (
                detector.thresholds[
                    subsystem
                ]
            )

            ratios.append(
                score / threshold
            )

        max_ratio.append(
            max(ratios)
        )

    nominal_6h_result[
        "maximum_score_ratio"
    ] = max_ratio

    plt.plot(
        nominal_6h_result[
            "time_h"
        ],
        nominal_6h_result[
            "maximum_score_ratio"
        ],
        label=(
            "Maximum subsystem anomaly ratio"
        ),
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Anomaly boundary",
    )

    raw_rows = nominal_6h_result[
        nominal_6h_result[
            "raw_anomaly"
        ]
    ]

    persistent_rows = nominal_6h_result[
        nominal_6h_result[
            "persistent_anomaly"
        ]
    ]

    if len(raw_rows) > 0:

        plt.scatter(
            raw_rows[
                "time_h"
            ],
            raw_rows[
                "maximum_score_ratio"
            ],
            label="Raw anomaly",
        )

    if len(persistent_rows) > 0:

        plt.scatter(
            persistent_rows[
                "time_h"
            ],
            persistent_rows[
                "maximum_score_ratio"
            ],
            marker="x",
            s=90,
            label="Persistent alarm",
        )

    plt.xlabel(
        "Time [h]"
    )

    plt.ylabel(
        "Normalized Anomaly Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Regime-Balanced ML Calibration"
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
        "SENTINEL-XAI - EXPERIMENT 031"
    )

    print(
        "REGIME-BALANCED SUBSYSTEM ML CALIBRATION"
    )

    print("=" * 82)

    print()

    print(
        f"Total 24h samples: "
        f"{n}"
    )

    print(
        f"Development samples: "
        f"{len(development)}"
    )

    print(
        f"Training samples: "
        f"{len(train)}"
    )

    print(
        f"Calibration samples: "
        f"{len(calibration)}"
    )

    print(
        f"Held-out nominal samples: "
        f"{len(holdout)}"
    )

    print()

    print(
        "Subsystem thresholds:"
    )

    for subsystem, threshold in (
        detector.thresholds.items()
    ):

        print(
            f"  {subsystem:<10}: "
            f"{threshold:.6f}"
        )

    print()

    print("-" * 82)

    print(
        "24H HELD-OUT NOMINAL"
    )

    print("-" * 82)

    print(
        f"Raw alarms: "
        f"{holdout_raw}"
    )

    print(
        f"Raw alarm rate: "
        f"{holdout_raw / len(holdout):.4f}"
    )

    print(
        f"Persistent alarms: "
        f"{holdout_persistent}"
    )

    print(
        f"Persistent alarm rate: "
        f"{holdout_persistent / len(holdout):.4f}"
    )

    print()

    print("-" * 82)

    print(
        "6H NOMINAL STARTUP VALIDATION"
    )

    print("-" * 82)

    print(
        f"Raw alarms: "
        f"{nominal_6h_raw}"
    )

    print(
        f"Persistent alarms: "
        f"{nominal_6h_persistent}"
    )

    print()

    print(
        "Subsystem raw alarms:"
    )

    for subsystem, count in (
        subsystem_counts.items()
    ):

        print(
            f"  {subsystem:<10}: "
            f"{count}"
        )

    print()

    # ======================================================
    # VALIDATION
    # ======================================================

    passed = (
        holdout_persistent == 0
        and
        nominal_6h_persistent == 0
    )

    if passed:

        print(
            "RESULT: REGIME-BALANCED "
            "ML CALIBRATION PASSED"
        )

    else:

        print(
            "RESULT: PERSISTENT HEALTHY "
            "ALARMS REMAIN"
        )

    print()

    print(
        "Model saved to:"
    )

    print(
        model_file
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