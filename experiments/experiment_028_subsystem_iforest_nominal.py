from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = (
    Path(__file__).resolve().parents[1]
)

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


def main():

    # ======================================================
    # USE 24-HOUR VALIDATED HEALTHY DATA
    # ======================================================

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry_24h.csv"
    )

    df = pd.read_csv(
        input_file
    )

    features = build_ml_features(
        df
    )

    # ======================================================
    # SEQUENTIAL SPLIT
    #
    # First 60%  -> training
    # Next 20%   -> calibration
    # Final 20%  -> untouched nominal test
    # ======================================================

    n = len(
        features
    )

    train_end = int(
        0.60 * n
    )

    calibration_end = int(
        0.80 * n
    )

    train = (
        features.iloc[
            :train_end
        ].copy()
    )

    calibration = (
        features.iloc[
            train_end:
            calibration_end
        ].copy()
    )

    test = (
        features.iloc[
            calibration_end:
        ].copy()
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
    # NOMINAL TEST
    # ======================================================

    prediction = (
        detector.predict(
            test
        )
    )

    raw_anomalies = (
        prediction.raw_anomaly
    )

    persistence = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent = []
    counts = []

    for value in raw_anomalies:

        result = (
            persistence.update(
                bool(value)
            )
        )

        persistent.append(
            result.persistent_anomaly
        )

        counts.append(
            result.consecutive_count
        )

    persistent = np.array(
        persistent,
        dtype=bool,
    )

    raw_count = int(
        raw_anomalies.sum()
    )

    persistent_count = int(
        persistent.sum()
    )

    # ======================================================
    # TABLE
    # ======================================================

    result_table = pd.DataFrame(
        {
            "time_h":
                test[
                    "time_h"
                ].values,

            "maximum_score_ratio":
                prediction.maximum_score_ratio,

            "raw_anomaly":
                raw_anomalies,

            "persistence_count":
                counts,

            "persistent_anomaly":
                persistent,
        }
    )

    for subsystem in (
        SUBSYSTEM_FEATURES
    ):

        result_table[
            f"{subsystem}_score"
        ] = (
            prediction
            .subsystem_scores[
                subsystem
            ]
        )

        result_table[
            f"{subsystem}_threshold"
        ] = (
            detector
            .thresholds[
                subsystem
            ]
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

    table_file = (
        tables_dir
        / "experiment_028_subsystem_iforest_nominal.csv"
    )

    model_file = (
        models_dir
        / "experiment_028_subsystem_iforest.joblib"
    )

    figure_file = (
        figures_dir
        / "experiment_028_subsystem_iforest_nominal.png"
    )

    result_table.to_csv(
        table_file,
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

    plt.plot(
        result_table[
            "time_h"
        ],
        result_table[
            "maximum_score_ratio"
        ],
        label=(
            "Maximum subsystem "
            "anomaly ratio"
        ),
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Anomaly boundary",
    )

    anomaly_rows = (
        result_table[
            result_table[
                "raw_anomaly"
            ]
        ]
    )

    persistent_rows = (
        result_table[
            result_table[
                "persistent_anomaly"
            ]
        ]
    )

    if len(
        anomaly_rows
    ) > 0:

        plt.scatter(
            anomaly_rows[
                "time_h"
            ],
            anomaly_rows[
                "maximum_score_ratio"
            ],
            label="Raw anomaly",
        )

    if len(
        persistent_rows
    ) > 0:

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
        "Subsystem Isolation Forest Nominal Validation"
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
    # OUTPUT
    # ======================================================

    print()
    print("=" * 80)

    print(
        "SENTINEL-XAI - EXPERIMENT 028"
    )

    print(
        "SUBSYSTEM ISOLATION FOREST "
        "NOMINAL VALIDATION"
    )

    print("=" * 80)

    print()

    print(
        f"Total 24h samples: "
        f"{n}"
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
        f"Untouched nominal test samples: "
        f"{len(test)}"
    )

    print()

    print(
        "Subsystem thresholds:"
    )

    for subsystem, threshold in (
        detector.thresholds.items()
    ):

        print(
            f"  {subsystem:<10} "
            f"{threshold:.6f}"
        )

    print()

    print(
        f"Raw nominal alarms: "
        f"{raw_count}"
    )

    print(
        f"Raw nominal alarm rate: "
        f"{raw_count / len(test):.4f}"
    )

    print()

    print(
        f"Persistent nominal alarms: "
        f"{persistent_count}"
    )

    print(
        f"Persistent nominal alarm rate: "
        f"{persistent_count / len(test):.4f}"
    )

    print()

    if persistent_count == 0:

        print(
            "RESULT: SUBSYSTEM ML "
            "NOMINAL VALIDATION PASSED"
        )

    else:

        print(
            "RESULT: PERSISTENT "
            "NOMINAL ALARMS OBSERVED"
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

    print("=" * 80)


if __name__ == "__main__":
    main()