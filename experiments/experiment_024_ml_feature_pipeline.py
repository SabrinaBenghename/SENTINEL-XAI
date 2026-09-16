from pathlib import Path
import sys

import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.ml.feature_pipeline import (
    ALL_FEATURES,
    build_ml_features,
    split_nominal_dataset,
    feature_matrix,
)


def main():

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    output_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_024_nominal_ml_features.csv"
    )

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

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_file,
        index=False,
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    total_missing = int(
        features[
            ALL_FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )

    hidden_columns = [
        column
        for column in features.columns
        if (
            "true" in column.lower()
            or
            "fault" in column.lower()
            or
            "severity" in column.lower()
        )
    ]

    print()
    print("=" * 76)

    print(
        "SENTINEL-XAI - EXPERIMENT 024"
    )

    print(
        "PHASE 5 ML FEATURE PIPELINE"
    )

    print("=" * 76)

    print(
        f"Total nominal samples: "
        f"{len(features)}"
    )

    print(
        f"ML features: "
        f"{len(ALL_FEATURES)}"
    )

    print()

    print(
        f"Training samples: "
        f"{len(train_x)}"
    )

    print(
        f"Calibration samples: "
        f"{len(calibration_x)}"
    )

    print(
        f"Nominal test samples: "
        f"{len(test_x)}"
    )

    print()

    print(
        f"Missing ML values: "
        f"{total_missing}"
    )

    print(
        f"Hidden truth/fault columns present: "
        f"{len(hidden_columns)}"
    )

    print()

    print(
        "Feature list:"
    )

    for feature in ALL_FEATURES:

        print(
            f"  - {feature}"
        )

    print()

    if (
        total_missing == 0
        and len(hidden_columns) == 0
    ):

        print(
            "RESULT: ML FEATURE PIPELINE PASSED"
        )

    else:

        print(
            "RESULT: ML FEATURE PIPELINE REVIEW REQUIRED"
        )

    print()

    print(
        "Features saved to:"
    )

    print(
        output_file
    )

    print("=" * 76)


if __name__ == "__main__":
    main()