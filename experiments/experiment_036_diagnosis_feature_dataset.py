from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    FAULT_CLASSES,
    build_diagnostic_dataset,
)


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


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
    # NOMINAL DATASET
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    nominal_df = load_csv(
        nominal_file
    )

    print()

    print(
        "Building nominal diagnostic features..."
    )

    nominal_features = (
        build_diagnostic_dataset(
            df=nominal_df,
            dataset_name="nominal",
            ml_detector=ml_detector,
            force_nominal=True,
        )
    )

    # ======================================================
    # FAULT DATASETS
    # ======================================================

    fault_files = {

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

    datasets = [
        nominal_features
    ]

    for fault_type, path in (
        fault_files.items()
    ):

        print(
            f"Building {fault_type} "
            f"diagnostic features..."
        )

        df = load_csv(
            path
        )

        complete_features = (
            build_diagnostic_dataset(
                df=df,
                dataset_name=fault_type,
                ml_detector=ml_detector,
                force_nominal=False,
            )
        )

        # ==================================================
        # KEEP ONLY ACTIVE FAULT SAMPLES
        #
        # Prevents six repeated copies of the same nominal
        # pre-fault operating period.
        # ==================================================

        active_fault = (
            complete_features[
                complete_features[
                    "target_label"
                ]
                ==
                fault_type
            ]
            .copy()
        )

        datasets.append(
            active_fault
        )

    # ======================================================
    # FINAL DATASET
    # ======================================================

    diagnosis_df = pd.concat(
        datasets,
        ignore_index=True,
    )

    x = diagnosis_df[
        DIAGNOSTIC_FEATURES
    ]

    y = diagnosis_df[
        "target_label"
    ]

    # ======================================================
    # TARGET DISTRIBUTION
    # ======================================================

    class_counts = (
        y.value_counts()
        .reindex(
            [
                "nominal",
                *FAULT_CLASSES,
            ],
            fill_value=0,
        )
    )

    # ======================================================
    # LEAKAGE VALIDATION
    # ======================================================

    forbidden_tokens = [

        "_true",

        "fault_label",

        "fault_active",

        "fault_severity",

        "effective_severity",

        "requested_severity",

        "fault_progress",
    ]

    forbidden_features = []

    for feature in (
        DIAGNOSTIC_FEATURES
    ):

        lower_name = (
            feature.lower()
        )

        if any(
            token in lower_name
            for token in forbidden_tokens
        ):

            forbidden_features.append(
                feature
            )

    time_used_as_feature = (
        "time_h"
        in
        DIAGNOSTIC_FEATURES
    )

    dataset_id_used = (
        "source_dataset"
        in
        DIAGNOSTIC_FEATURES
    )

    target_used = (
        "target_label"
        in
        DIAGNOSTIC_FEATURES
    )

    # ======================================================
    # MISSING VALUES
    #
    # Missing values are expected for telemetry-dropout
    # samples. They should NOT occur in other classes.
    # ======================================================

    total_missing = int(
        x.isna()
        .sum()
        .sum()
    )

    non_dropout_mask = (
        diagnosis_df[
            "target_label"
        ]
        !=
        "telemetry_dropout"
    )

    unexpected_missing = int(
        diagnosis_df.loc[
            non_dropout_mask,
            DIAGNOSTIC_FEATURES,
        ]
        .isna()
        .sum()
        .sum()
    )

    dropout_missing = int(
        diagnosis_df.loc[
            diagnosis_df[
                "target_label"
            ]
            ==
            "telemetry_dropout",
            DIAGNOSTIC_FEATURES,
        ]
        .isna()
        .sum()
        .sum()
    )

    # ======================================================
    # CLASS CHECK
    # ======================================================

    expected_classes = set(
        [
            "nominal",
            *FAULT_CLASSES,
        ]
    )

    observed_classes = set(
        y.unique()
    )

    all_classes_present = (
        expected_classes
        ==
        observed_classes
    )

    # ======================================================
    # SAVE DATASET
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

    output_file = (
        tables_dir
        / "experiment_036_diagnosis_feature_dataset.csv"
    )

    figure_file = (
        figures_dir
        / "experiment_036_diagnosis_class_distribution.png"
    )

    diagnosis_df.to_csv(
        output_file,
        index=False,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        class_counts.index,
        class_counts.values,
    )

    plt.xlabel(
        "Diagnostic Class"
    )

    plt.ylabel(
        "Samples"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Phase-7 Diagnosis Dataset Distribution"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # VALIDATION
    # ======================================================

    validation_passed = (

        len(
            forbidden_features
        )
        == 0

        and

        not time_used_as_feature

        and

        not dataset_id_used

        and

        not target_used

        and

        unexpected_missing
        == 0

        and

        all_classes_present
    )

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 84)

    print(
        "SENTINEL-XAI - EXPERIMENT 036"
    )

    print(
        "PHASE 7 DIAGNOSIS FEATURE DATASET"
    )

    print("=" * 84)

    print()

    print(
        f"Total diagnosis samples: "
        f"{len(diagnosis_df)}"
    )

    print(
        f"Diagnostic input features: "
        f"{len(DIAGNOSTIC_FEATURES)}"
    )

    print(
        f"Target classes: "
        f"{len(observed_classes)}"
    )

    print()

    print(
        "Class distribution:"
    )

    for class_name, count in (
        class_counts.items()
    ):

        print(
            f"  {class_name:<34} "
            f"{count}"
        )

    print()

    print("-" * 84)

    print(
        "TARGET-LEAKAGE CHECK"
    )

    print("-" * 84)

    print(
        f"Forbidden truth features: "
        f"{len(forbidden_features)}"
    )

    print(
        f"Time used as model feature: "
        f"{time_used_as_feature}"
    )

    print(
        f"Dataset identity used as feature: "
        f"{dataset_id_used}"
    )

    print(
        f"Target label used as feature: "
        f"{target_used}"
    )

    print()

    print(
        f"Total missing feature cells: "
        f"{total_missing}"
    )

    print(
        f"Missing cells in dropout class: "
        f"{dropout_missing}"
    )

    print(
        f"Unexpected missing cells "
        f"outside dropout: "
        f"{unexpected_missing}"
    )

    print()

    print(
        f"All seven classes present: "
        f"{all_classes_present}"
    )

    print()

    if validation_passed:

        print(
            "RESULT: DIAGNOSIS FEATURE "
            "DATASET PASSED"
        )

    else:

        print(
            "RESULT: DIAGNOSIS FEATURE "
            "DATASET REVIEW REQUIRED"
        )

    print()

    print(
        "Dataset saved to:"
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

    print("=" * 84)


if __name__ == "__main__":
    main()