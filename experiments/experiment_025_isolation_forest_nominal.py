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

from src.ml.isolation_forest_detector import (
    IsolationForestAnomalyDetector,
)


# ==========================================================
# SCORE SUMMARY
# ==========================================================

def summarize_scores(
    name,
    scores,
    threshold,
):

    anomaly_count = int(
        np.sum(
            scores > threshold
        )
    )

    fraction = (
        anomaly_count
        / len(scores)
    )

    print()
    print(name)
    print(
        f"  Samples:       {len(scores)}"
    )
    print(
        f"  Mean score:    {np.mean(scores):.6f}"
    )
    print(
        f"  Std score:     {np.std(scores):.6f}"
    )
    print(
        f"  Min score:     {np.min(scores):.6f}"
    )
    print(
        f"  Max score:     {np.max(scores):.6f}"
    )
    print(
        f"  Anomalies:     {anomaly_count}"
    )
    print(
        f"  Anomaly rate:  {fraction:.4f}"
    )

    return anomaly_count


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # PATHS
    # ======================================================

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

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

    # ======================================================
    # LOAD HEALTHY TELEMETRY
    # ======================================================

    df = pd.read_csv(
        input_file
    )

    features = build_ml_features(
        df
    )

    split = split_nominal_dataset(
        features
    )

    train_x = feature_matrix(
        split.train
    )

    calibration_x = feature_matrix(
        split.calibration
    )

    test_x = feature_matrix(
        split.test
    )

    # ======================================================
    # TRAIN ISOLATION FOREST
    # ======================================================

    detector = (
        IsolationForestAnomalyDetector(
            n_estimators=300,
            random_state=42,
        )
    )

    detector.fit(
        train_x
    )

    # ======================================================
    # CALIBRATE THRESHOLD
    #
    # Calibration data is healthy but was NOT used
    # to train the Isolation Forest.
    # ======================================================

    threshold = (
        detector.calibrate_threshold(
            calibration_x,
            quantile=0.99,
        )
    )

    # ======================================================
    # SCORE ALL THREE HEALTHY SPLITS
    # ======================================================

    train_scores = (
        detector.anomaly_score(
            train_x
        )
    )

    calibration_scores = (
        detector.anomaly_score(
            calibration_x
        )
    )

    test_scores = (
        detector.anomaly_score(
            test_x
        )
    )

    # ======================================================
    # TERMINAL SUMMARY
    # ======================================================

    print()
    print("=" * 78)
    print(
        "SENTINEL-XAI - EXPERIMENT 025"
    )
    print(
        "ISOLATION FOREST TRAINING AND NOMINAL VALIDATION"
    )
    print("=" * 78)

    print()
    print(
        "Training mode: UNSUPERVISED / NOMINAL ONLY"
    )

    print(
        f"Training samples: "
        f"{len(train_x)}"
    )

    print(
        f"Features: "
        f"{train_x.shape[1]}"
    )

    print(
        f"Trees: "
        f"{detector.model.n_estimators}"
    )

    print()
    print(
        f"Calibration quantile: 0.99"
    )

    print(
        f"Anomaly threshold: "
        f"{threshold:.6f}"
    )

    print()
    print("-" * 78)

    train_anomalies = (
        summarize_scores(
            "TRAINING SET",
            train_scores,
            threshold,
        )
    )

    calibration_anomalies = (
        summarize_scores(
            "CALIBRATION SET",
            calibration_scores,
            threshold,
        )
    )

    test_anomalies = (
        summarize_scores(
            "UNTOUCHED NOMINAL TEST SET",
            test_scores,
            threshold,
        )
    )

    # ======================================================
    # SAVE SCORE TABLE
    # ======================================================

    train_table = pd.DataFrame(
        {
            "split": "train",
            "time_h":
                split.train[
                    "time_h"
                ].values,
            "anomaly_score":
                train_scores,
            "threshold":
                threshold,
            "is_anomaly":
                train_scores
                >
                threshold,
        }
    )

    calibration_table = pd.DataFrame(
        {
            "split":
                "calibration",

            "time_h":
                split.calibration[
                    "time_h"
                ].values,

            "anomaly_score":
                calibration_scores,

            "threshold":
                threshold,

            "is_anomaly":
                calibration_scores
                >
                threshold,
        }
    )

    test_table = pd.DataFrame(
        {
            "split":
                "nominal_test",

            "time_h":
                split.test[
                    "time_h"
                ].values,

            "anomaly_score":
                test_scores,

            "threshold":
                threshold,

            "is_anomaly":
                test_scores
                >
                threshold,
        }
    )

    score_table = pd.concat(
        [
            train_table,
            calibration_table,
            test_table,
        ],
        ignore_index=True,
    )

    score_file = (
        tables_dir
        / "experiment_025_isolation_forest_nominal_scores.csv"
    )

    score_table.to_csv(
        score_file,
        index=False,
    )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_file = (
        models_dir
        / "experiment_025_isolation_forest.joblib"
    )

    joblib.dump(
        detector,
        model_file,
    )

    # ======================================================
    # FIGURE 1
    # ANOMALY SCORE THROUGH TIME
    # ======================================================

    timeline_figure = (
        figures_dir
        / "experiment_025_nominal_anomaly_score.png"
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        score_table["time_h"],
        score_table["anomaly_score"],
        label="Isolation Forest anomaly score",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Calibrated anomaly threshold",
    )

    train_boundary = (
        split.train[
            "time_h"
        ].iloc[-1]
    )

    calibration_boundary = (
        split.calibration[
            "time_h"
        ].iloc[-1]
    )

    plt.axvline(
        train_boundary,
        linestyle=":",
        label="Train / calibration boundary",
    )

    plt.axvline(
        calibration_boundary,
        linestyle=":",
        label="Calibration / test boundary",
    )

    plt.xlabel(
        "Time [h]"
    )

    plt.ylabel(
        "Anomaly Score"
    )

    plt.title(
        "SENTINEL-XAI - Isolation Forest on Healthy Telemetry"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        timeline_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2
    # SCORE DISTRIBUTIONS
    # ======================================================

    distribution_figure = (
        figures_dir
        / "experiment_025_nominal_score_distribution.png"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        train_scores,
        bins=20,
        alpha=0.5,
        label="Training",
    )

    plt.hist(
        calibration_scores,
        bins=20,
        alpha=0.5,
        label="Calibration",
    )

    plt.hist(
        test_scores,
        bins=20,
        alpha=0.5,
        label="Nominal test",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label="Threshold",
    )

    plt.xlabel(
        "Anomaly Score"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "SENTINEL-XAI - Nominal Anomaly-Score Distribution"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        distribution_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # VALIDATION
    # ======================================================

    print()
    print("-" * 78)

    print(
        f"Untouched nominal false alarms: "
        f"{test_anomalies}"
    )

    if test_anomalies == 0:

        print(
            "RESULT: ISOLATION FOREST NOMINAL TEST PASSED"
        )

    else:

        print(
            "RESULT: NOMINAL FALSE ALARMS OBSERVED"
        )

        print(
            "Do not tune yet — inspect the score "
            "distribution first."
        )

    print("-" * 78)

    print()
    print(
        "Scores saved to:"
    )
    print(
        score_file
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
        "Figures saved to:"
    )
    print(
        timeline_figure
    )
    print(
        distribution_figure
    )

    print("=" * 78)


if __name__ == "__main__":
    main()