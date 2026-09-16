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
    split_nominal_dataset,
    feature_matrix,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


def main():

    # ======================================================
    # PATHS
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_025_isolation_forest.joblib"
    )

    output_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_026_ml_persistence_validation.csv"
    )

    figure_file = (
        project_root
        / "results"
        / "figures"
        / "experiment_026_ml_persistence_validation.png"
    )

    # ======================================================
    # LOAD MODEL AND DATA
    # ======================================================

    detector = joblib.load(
        model_file
    )

    df = pd.read_csv(
        nominal_file
    )

    features = build_ml_features(
        df
    )

    split = split_nominal_dataset(
        features
    )

    # IMPORTANT:
    # Experiment 026 evaluates only the untouched nominal
    # test block preserved by Experiment 025.
    test_features = (
        split.test.copy()
    )

    test_x = feature_matrix(
        test_features
    )

    scores = detector.anomaly_score(
        test_x
    )

    threshold = (
        detector.threshold
    )

    raw_anomalies = (
        scores > threshold
    )

    # ======================================================
    # TEMPORAL PERSISTENCE
    # ======================================================

    persistence_filter = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent_anomalies = []
    counters = []

    for raw_anomaly in raw_anomalies:

        result = (
            persistence_filter.update(
                bool(raw_anomaly)
            )
        )

        persistent_anomalies.append(
            result.persistent_anomaly
        )

        counters.append(
            result.consecutive_count
        )

    persistent_anomalies = np.array(
        persistent_anomalies,
        dtype=bool,
    )

    # ======================================================
    # METRICS
    # ======================================================

    raw_false_alarms = int(
        raw_anomalies.sum()
    )

    persistent_false_alarms = int(
        persistent_anomalies.sum()
    )

    raw_rate = (
        raw_false_alarms
        / len(test_x)
    )

    persistent_rate = (
        persistent_false_alarms
        / len(test_x)
    )

    # ======================================================
    # SAVE TABLE
    # ======================================================

    result_table = pd.DataFrame(
        {
            "time_h":
                test_features[
                    "time_h"
                ].values,

            "anomaly_score":
                scores,

            "threshold":
                threshold,

            "raw_anomaly":
                raw_anomalies,

            "consecutive_count":
                counters,

            "persistent_anomaly":
                persistent_anomalies,
        }
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_table.to_csv(
        output_file,
        index=False,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    figure_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        result_table["time_h"],
        result_table["anomaly_score"],
        label="Anomaly score",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Threshold",
    )

    raw_rows = result_table[
        result_table["raw_anomaly"]
    ]

    persistent_rows = result_table[
        result_table[
            "persistent_anomaly"
        ]
    ]

    if len(raw_rows) > 0:

        plt.scatter(
            raw_rows["time_h"],
            raw_rows["anomaly_score"],
            label="Raw anomaly",
        )

    if len(persistent_rows) > 0:

        plt.scatter(
            persistent_rows["time_h"],
            persistent_rows[
                "anomaly_score"
            ],
            marker="x",
            s=90,
            label="Persistent alarm",
        )

    plt.xlabel(
        "Time [h]"
    )

    plt.ylabel(
        "Anomaly Score"
    )

    plt.title(
        "SENTINEL-XAI - Temporal Persistence on Nominal ML Test"
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
    # TERMINAL SUMMARY
    # ======================================================

    print()
    print("=" * 78)

    print(
        "SENTINEL-XAI - EXPERIMENT 026"
    )

    print(
        "ML TEMPORAL-PERSISTENCE VALIDATION"
    )

    print("=" * 78)

    print()

    print(
        f"Untouched nominal samples: "
        f"{len(test_x)}"
    )

    print(
        f"Anomaly threshold: "
        f"{threshold:.6f}"
    )

    print(
        "Persistence requirement: "
        "3 consecutive samples"
    )

    print()

    print(
        f"Raw false alarms: "
        f"{raw_false_alarms}"
    )

    print(
        f"Raw false-alarm rate: "
        f"{raw_rate:.4f}"
    )

    print()

    print(
        f"Persistent false alarms: "
        f"{persistent_false_alarms}"
    )

    print(
        f"Persistent false-alarm rate: "
        f"{persistent_rate:.4f}"
    )

    print()

    if persistent_false_alarms == 0:

        print(
            "RESULT: ML PERSISTENCE VALIDATION PASSED"
        )

    else:

        print(
            "RESULT: PERSISTENT NOMINAL "
            "FALSE ALARMS REMAIN"
        )

    print()

    print(
        "Table saved to:"
    )

    print(
        output_file
    )

    print()

    print(
        "Figure saved to:"
    )

    print(
        figure_file
    )

    print("=" * 78)


if __name__ == "__main__":
    main()