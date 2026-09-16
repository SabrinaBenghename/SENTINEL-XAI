from pathlib import Path
import sys
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ==========================================================
# PROJECT
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    build_diagnostic_dataset,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

CONFIDENCE_THRESHOLD = 0.70

TOP_K = 5

RANDOM_ABLATION_TRIALS = 10

RANDOM_SEED = 42


FAULT_TYPES = [
    "solar_array_degradation",
    "battery_degradation",
    "thermal_anomaly",
    "reaction_wheel_degradation",
    "battery_voltage_sensor_drift",
    "telemetry_dropout",
]


# ==========================================================
# PATHS
# ==========================================================

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

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# HELPERS
# ==========================================================

def load_csv(filename):

    path = (
        tables_dir
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


def as_bool(values):

    series = pd.Series(values)

    if pd.api.types.is_bool_dtype(series):

        return (
            series
            .astype(bool)
            .to_numpy()
        )

    if pd.api.types.is_numeric_dtype(series):

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


def clean_feature_name(name):

    name = str(name)

    if "__" in name:
        name = name.split("__")[-1]

    return name


def jaccard(a, b):

    a = set(a)
    b = set(b)

    union = a | b

    if len(union) == 0:
        return 1.0

    return len(a & b) / len(union)


def get_scenario_value(row, candidates):

    for candidate in candidates:

        if candidate in row.index:

            value = row[candidate]

            if not pd.isna(value):
                return value

    return np.nan


# ==========================================================
# SHAP NORMALIZATION
# ==========================================================

def normalize_shap_values(
    shap_values,
    n_samples,
    n_features,
    n_classes,
):

    # ------------------------------------------------------
    # Older SHAP:
    # list[class] -> (samples, features)
    # ------------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) != n_classes:

            raise ValueError(
                "Unexpected SHAP class count."
            )

        return np.stack(
            shap_values,
            axis=2,
        )

    values = np.asarray(
        shap_values
    )

    # ------------------------------------------------------
    # New SHAP:
    # (samples, features, classes)
    # ------------------------------------------------------

    if (
        values.ndim == 3
        and
        values.shape
        ==
        (
            n_samples,
            n_features,
            n_classes,
        )
    ):

        return values

    # ------------------------------------------------------
    # Possible:
    # (classes, samples, features)
    # ------------------------------------------------------

    if (
        values.ndim == 3
        and
        values.shape
        ==
        (
            n_classes,
            n_samples,
            n_features,
        )
    ):

        return np.transpose(
            values,
            (1, 2, 0),
        )

    # ------------------------------------------------------
    # Possible:
    # (samples, classes, features)
    # ------------------------------------------------------

    if (
        values.ndim == 3
        and
        values.shape
        ==
        (
            n_samples,
            n_classes,
            n_features,
        )
    ):

        return np.transpose(
            values,
            (0, 2, 1),
        )

    raise ValueError(
        "Unsupported SHAP array shape: "
        f"{values.shape}"
    )


# ==========================================================
# DEVELOPMENT CONSENSUS
# ==========================================================

def load_phase8_consensus():

    path = (
        tables_dir
        / "experiment_043_class_consensus_features.csv"
    )

    if not path.exists():

        print(
            "WARNING: Phase-8 consensus table not found."
        )

        return {}

    df = pd.read_csv(path)

    # ------------------------------------------------------
    # Find likely columns
    # ------------------------------------------------------

    class_column = None

    for candidate in [
        "diagnostic_class",
        "fault_type",
        "class",
        "true_class",
        "label",
    ]:

        if candidate in df.columns:

            class_column = candidate
            break

    feature_column = None

    for candidate in [
        "feature",
        "feature_name",
        "consensus_feature",
    ]:

        if candidate in df.columns:

            feature_column = candidate
            break

    rank_column = None

    for candidate in [
        "rank",
        "feature_rank",
        "consensus_rank",
    ]:

        if candidate in df.columns:

            rank_column = candidate
            break

    frequency_column = None

    for candidate in [
        "frequency",
        "fraction",
        "consensus_fraction",
        "occurrence_fraction",
        "support",
    ]:

        if candidate in df.columns:

            frequency_column = candidate
            break

    if (
        class_column is None
        or
        feature_column is None
    ):

        print(
            "WARNING: Could not interpret "
            "Phase-8 consensus table columns."
        )

        print(
            "Available columns:",
            list(df.columns),
        )

        return {}

    consensus = {}

    for fault_type in FAULT_TYPES:

        subset = (
            df[
                df[
                    class_column
                ].astype(str)
                ==
                fault_type
            ]
            .copy()
        )

        if len(subset) == 0:
            continue

        if rank_column is not None:

            subset = subset.sort_values(
                rank_column
            )

        elif frequency_column is not None:

            subset = subset.sort_values(
                frequency_column,
                ascending=False,
            )

        features = [
            clean_feature_name(feature)

            for feature in
            subset[
                feature_column
            ]
            .astype(str)
            .tolist()
        ]

        consensus[
            fault_type
        ] = features[:TOP_K]

    return consensus


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "Preparing independent Monte Carlo "
        "XAI robustness experiment..."
    )

    # ======================================================
    # LOAD FROZEN MODELS
    # ======================================================

    phase5_model_file = (
        models_dir
        / "experiment_031_regime_balanced_subsystem_iforest.joblib"
    )

    phase7_model_file = (
        models_dir
        / "experiment_037_random_forest_diagnoser.joblib"
    )

    if not phase5_model_file.exists():

        raise FileNotFoundError(
            f"Phase-5 model not found:\n"
            f"{phase5_model_file}"
        )

    if not phase7_model_file.exists():

        raise FileNotFoundError(
            f"Phase-7 model not found:\n"
            f"{phase7_model_file}"
        )

    phase5_detector = joblib.load(
        phase5_model_file
    )

    diagnoser = joblib.load(
        phase7_model_file
    )

    if not diagnoser.is_fitted:

        raise RuntimeError(
            "Frozen Phase-7 diagnoser is not fitted."
        )

    pipeline = diagnoser.pipeline

    if len(pipeline.steps) == 0:

        raise RuntimeError(
            "Phase-7 sklearn pipeline is empty."
        )

    final_estimator = (
        pipeline.steps[-1][1]
    )

    if not hasattr(
        final_estimator,
        "predict_proba",
    ):

        raise TypeError(
            "Final pipeline estimator does not "
            "support predict_proba()."
        )

    estimator_classes = np.asarray(
        final_estimator.classes_,
        dtype=object,
    )

    # ======================================================
    # PREPROCESSING PART OF PIPELINE
    # ======================================================

    if len(pipeline.steps) > 1:

        preprocessor = pipeline[:-1]

    else:

        preprocessor = None

    # ======================================================
    # LOAD INDEPENDENT RESULTS
    # ======================================================

    diagnosis = load_csv(
        "experiment_048_diagnosis_predictions.csv"
    )

    manifest = load_csv(
        "experiment_047_monte_carlo_manifest.csv"
    )

    scenario_metrics = load_csv(
        "experiment_048_scenario_metrics.csv"
    )

    diagnosis[
        "operational_correct_bool"
    ] = as_bool(
        diagnosis[
            "operational_correct"
        ]
    )

    diagnosis[
        "accepted_bool"
    ] = as_bool(
        diagnosis[
            "accepted"
        ]
    )

    # ======================================================
    # SELECT ONE CORRECT ACCEPTED OPERATIONAL DIAGNOSIS
    # FROM EACH OF THE 120 SCENARIOS
    # ======================================================

    selected_rows = []

    for scenario_id in sorted(
        diagnosis[
            "scenario_id"
        ].unique()
    ):

        subset = (
            diagnosis[
                (
                    diagnosis[
                        "scenario_id"
                    ]
                    ==
                    scenario_id
                )
                &
                (
                    diagnosis[
                        "operational_correct_bool"
                    ]
                )
            ]
            .sort_values(
                "time_h"
            )
        )

        if len(subset) == 0:
            continue

        # Use first correct accepted operational diagnosis.
        selected_rows.append(
            subset.iloc[0]
        )

    representatives = pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )

    print(
        f"Correct accepted scenario representatives: "
        f"{len(representatives)}"
    )

    # ======================================================
    # RECONSTRUCT 57 FEATURES FOR EACH REPRESENTATIVE
    # ======================================================

    feature_rows = []

    representative_metadata = []

    for number, rep in representatives.iterrows():

        scenario_id = int(
            rep[
                "scenario_id"
            ]
        )

        fault_type = str(
            rep[
                "fault_type"
            ]
        )

        target_time_h = float(
            rep[
                "time_h"
            ]
        )

        manifest_row = (
            manifest[
                manifest[
                    "scenario_id"
                ]
                ==
                scenario_id
            ]
        )

        if len(manifest_row) != 1:

            raise RuntimeError(
                f"Manifest lookup failed for "
                f"scenario {scenario_id}."
            )

        manifest_row = (
            manifest_row.iloc[0]
        )

        telemetry_file = Path(
            manifest_row[
                "output_file"
            ]
        )

        telemetry = pd.read_csv(
            telemetry_file
        )

        diagnostic_df = (
            build_diagnostic_dataset(
                df=telemetry,
                dataset_name=fault_type,
                ml_detector=phase5_detector,
                force_nominal=False,
            )
        )

        if not isinstance(
            diagnostic_df,
            pd.DataFrame,
        ):

            diagnostic_df = pd.DataFrame(
                diagnostic_df
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
                "Diagnostic features missing:\n"
                f"{missing_features}"
            )

        time_difference = np.abs(
            diagnostic_df[
                "time_h"
            ].to_numpy(
                dtype=float
            )
            -
            target_time_h
        )

        position = int(
            np.argmin(
                time_difference
            )
        )

        selected = (
            diagnostic_df.iloc[
                position
            ]
        )

        feature_rows.append(
            selected[
                DIAGNOSTIC_FEATURES
            ]
            .copy()
        )

        scenario_row = (
            scenario_metrics[
                scenario_metrics[
                    "scenario_id"
                ]
                ==
                scenario_id
            ]
        )

        if len(scenario_row) == 1:

            scenario_row = (
                scenario_row.iloc[0]
            )

            severity = float(
                scenario_row[
                    "severity"
                ]
            )

            noise_scale = float(
                scenario_row[
                    "noise_scale"
                ]
            )

            profile = str(
                scenario_row[
                    "profile"
                ]
            )

        else:

            severity = float(
                get_scenario_value(
                    manifest_row,
                    [
                        "requested_severity",
                        "severity",
                    ],
                )
            )

            noise_scale = float(
                get_scenario_value(
                    manifest_row,
                    [
                        "requested_noise_scale",
                        "sensor_noise_scale",
                    ],
                )
            )

            profile = str(
                get_scenario_value(
                    manifest_row,
                    [
                        "profile",
                    ],
                )
            )

        representative_metadata.append(
            {
                "scenario_id":
                    scenario_id,

                "fault_type":
                    fault_type,

                "time_h":
                    target_time_h,

                "prediction":
                    str(
                        rep[
                            "raw_prediction"
                        ]
                    ),

                "confidence":
                    float(
                        rep[
                            "confidence"
                        ]
                    ),

                "severity":
                    severity,

                "noise_scale":
                    noise_scale,

                "profile":
                    profile,
            }
        )

        print(
            f"[{number + 1:03d}/"
            f"{len(representatives):03d}] "
            f"Scenario {scenario_id:04d} "
            f"{fault_type}"
        )

    X_raw = pd.DataFrame(
        feature_rows
    ).reset_index(
        drop=True
    )

    metadata = pd.DataFrame(
        representative_metadata
    )

    # ======================================================
    # TRANSFORM THROUGH FROZEN DEVELOPMENT PIPELINE
    # ======================================================

    if preprocessor is not None:

        X_transformed = (
            preprocessor.transform(
                X_raw
            )
        )

        if hasattr(
            X_transformed,
            "toarray",
        ):

            X_transformed = (
                X_transformed.toarray()
            )

        X_transformed = np.asarray(
            X_transformed,
            dtype=float,
        )

        # --------------------------------------------------
        # Feature names
        # --------------------------------------------------

        transformed_names = None

        try:

            transformed_names = (
                preprocessor
                .get_feature_names_out(
                    DIAGNOSTIC_FEATURES
                )
            )

        except Exception:

            try:

                transformed_names = (
                    preprocessor
                    .get_feature_names_out()
                )

            except Exception:

                transformed_names = None

    else:

        X_transformed = (
            X_raw
            .to_numpy(
                dtype=float
            )
        )

        transformed_names = np.asarray(
            DIAGNOSTIC_FEATURES,
            dtype=object,
        )

    # ======================================================
    # FINAL FEATURE NAMES
    # ======================================================

    if (
        transformed_names is not None
        and
        len(transformed_names)
        ==
        X_transformed.shape[1]
    ):

        feature_names = [
            clean_feature_name(name)

            for name in transformed_names
        ]

    elif (
        X_transformed.shape[1]
        ==
        len(
            DIAGNOSTIC_FEATURES
        )
    ):

        feature_names = list(
            DIAGNOSTIC_FEATURES
        )

    else:

        feature_names = [
            f"feature_{index}"

            for index in range(
                X_transformed.shape[1]
            )
        ]

    # ======================================================
    # DEVELOPMENT BASELINE FOR ABLATION
    #
    # Use Phase-7 development data, NOT Monte Carlo data,
    # to avoid choosing neutral feature values from test data.
    # ======================================================

    development_file = (
        tables_dir
        / "experiment_036_diagnosis_feature_dataset.csv"
    )

    development = pd.read_csv(
        development_file
    )

    X_development_raw = (
        development[
            DIAGNOSTIC_FEATURES
        ]
    )

    if preprocessor is not None:

        X_development = (
            preprocessor.transform(
                X_development_raw
            )
        )

        if hasattr(
            X_development,
            "toarray",
        ):

            X_development = (
                X_development.toarray()
            )

        X_development = np.asarray(
            X_development,
            dtype=float,
        )

    else:

        X_development = (
            X_development_raw
            .to_numpy(
                dtype=float
            )
        )

    baseline = np.nanmedian(
        X_development,
        axis=0,
    )

    # ======================================================
    # SHAP
    # ======================================================

    print()
    print(
        "Computing SHAP values for "
        "120 unseen scenario representatives..."
    )

    explainer = shap.TreeExplainer(
        final_estimator
    )

    raw_shap = (
        explainer.shap_values(
            X_transformed
        )
    )

    shap_cube = (
        normalize_shap_values(
            raw_shap,
            n_samples=(
                X_transformed.shape[0]
            ),
            n_features=(
                X_transformed.shape[1]
            ),
            n_classes=(
                len(
                    estimator_classes
                )
            ),
        )
    )

    # ======================================================
    # ORIGINAL PROBABILITIES
    # ======================================================

    probabilities = (
        final_estimator.predict_proba(
            X_transformed
        )
    )

    estimator_predictions = (
        final_estimator.predict(
            X_transformed
        )
    )

    # ======================================================
    # EXPECTED VALUES FOR ADDITIVITY
    # ======================================================

    expected_value = np.asarray(
        explainer.expected_value
    )

    if expected_value.ndim == 0:

        expected_value = np.repeat(
            expected_value,
            len(
                estimator_classes
            ),
        )

    # ======================================================
    # SAMPLE EXPLANATIONS + FAITHFULNESS
    # ======================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_rows = []

    top_feature_rows = []

    all_feature_indices = np.arange(
        X_transformed.shape[1]
    )

    reconstruction_errors = []

    for index in range(
        len(
            metadata
        )
    ):

        predicted_label = str(
            estimator_predictions[
                index
            ]
        )

        if predicted_label not in estimator_classes:

            raise RuntimeError(
                "Predicted class not found "
                "in estimator classes."
            )

        class_index = int(
            np.where(
                estimator_classes
                ==
                predicted_label
            )[0][0]
        )

        contributions = (
            shap_cube[
                index,
                :,
                class_index,
            ]
        )

        # --------------------------------------------------
        # Top supporting SHAP evidence
        # --------------------------------------------------

        ordered = np.argsort(
            contributions
        )[::-1]

        top_indices = (
            ordered[
                :TOP_K
            ]
        )

        top_names = [
            feature_names[
                feature_index
            ]

            for feature_index in
            top_indices
        ]

        top_values = [
            float(
                contributions[
                    feature_index
                ]
            )

            for feature_index in
            top_indices
        ]

        # --------------------------------------------------
        # Additivity
        # --------------------------------------------------

        reconstructed_probability = (
            float(
                expected_value[
                    class_index
                ]
            )
            +
            float(
                np.sum(
                    contributions
                )
            )
        )

        original_probability = float(
            probabilities[
                index,
                class_index
            ]
        )

        reconstruction_error = abs(
            reconstructed_probability
            -
            original_probability
        )

        reconstruction_errors.append(
            reconstruction_error
        )

        # --------------------------------------------------
        # Top-5 ablation
        # --------------------------------------------------

        top_ablation = (
            X_transformed[
                index
            ]
            .copy()
        )

        top_ablation[
            top_indices
        ] = baseline[
            top_indices
        ]

        top_probs = (
            final_estimator.predict_proba(
                top_ablation.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        top_probability_after = float(
            top_probs[
                class_index
            ]
        )

        top_drop = (
            original_probability
            -
            top_probability_after
        )

        top_prediction_after = str(
            final_estimator.predict(
                top_ablation.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        top_flip = (
            top_prediction_after
            !=
            predicted_label
        )

        # --------------------------------------------------
        # Random-5 baseline
        # --------------------------------------------------

        candidate_random_features = np.array(
            [
                feature_index

                for feature_index in
                all_feature_indices

                if feature_index
                not in
                set(
                    top_indices
                )
            ]
        )

        random_drops = []
        random_flips = []

        for _ in range(
            RANDOM_ABLATION_TRIALS
        ):

            random_indices = (
                rng.choice(
                    candidate_random_features,
                    size=TOP_K,
                    replace=False,
                )
            )

            random_ablation = (
                X_transformed[
                    index
                ]
                .copy()
            )

            random_ablation[
                random_indices
            ] = baseline[
                random_indices
            ]

            random_probs = (
                final_estimator.predict_proba(
                    random_ablation.reshape(
                        1,
                        -1,
                    )
                )[0]
            )

            random_probability_after = float(
                random_probs[
                    class_index
                ]
            )

            random_drops.append(
                original_probability
                -
                random_probability_after
            )

            random_prediction_after = str(
                final_estimator.predict(
                    random_ablation.reshape(
                        1,
                        -1,
                    )
                )[0]
            )

            random_flips.append(
                random_prediction_after
                !=
                predicted_label
            )

        mean_random_drop = float(
            np.mean(
                random_drops
            )
        )

        random_flip_rate = float(
            np.mean(
                random_flips
            )
        )

        sample_rows.append(
            {
                **metadata.iloc[
                    index
                ].to_dict(),

                "estimator_prediction":
                    predicted_label,

                "original_probability":
                    original_probability,

                "top5_probability_drop":
                    top_drop,

                "random5_probability_drop":
                    mean_random_drop,

                "top5_beats_random":
                    bool(
                        top_drop
                        >
                        mean_random_drop
                    ),

                "top5_prediction_flip":
                    bool(
                        top_flip
                    ),

                "random5_flip_rate":
                    random_flip_rate,

                "reconstruction_error":
                    reconstruction_error,

                "top1_feature":
                    top_names[0],

                "top5_features":
                    "|".join(
                        top_names
                    ),

                "top5_positive_strength":
                    float(
                        np.sum(
                            np.maximum(
                                top_values,
                                0.0,
                            )
                        )
                    ),
            }
        )

        for rank, (
            feature_name,
            shap_value,
        ) in enumerate(
            zip(
                top_names,
                top_values,
            ),
            start=1,
        ):

            top_feature_rows.append(
                {
                    "scenario_id":
                        int(
                            metadata.iloc[
                                index
                            ][
                                "scenario_id"
                            ]
                        ),

                    "fault_type":
                        str(
                            metadata.iloc[
                                index
                            ][
                                "fault_type"
                            ]
                        ),

                    "rank":
                        rank,

                    "feature":
                        feature_name,

                    "shap_value":
                        shap_value,
                }
            )

    sample_df = pd.DataFrame(
        sample_rows
    )

    top_features_df = pd.DataFrame(
        top_feature_rows
    )

    # ======================================================
    # WITHIN-CLASS STABILITY
    # ======================================================

    phase8_consensus = (
        load_phase8_consensus()
    )

    stability_rows = []

    consensus_rows = []

    for fault_type in FAULT_TYPES:

        subset = (
            sample_df[
                sample_df[
                    "fault_type"
                ]
                ==
                fault_type
            ]
        )

        top_sets = [
            row.split("|")

            for row in
            subset[
                "top5_features"
            ]
        ]

        pairwise_values = []

        for set_a, set_b in combinations(
            top_sets,
            2,
        ):

            pairwise_values.append(
                jaccard(
                    set_a,
                    set_b,
                )
            )

        if len(pairwise_values) > 0:

            mean_pairwise = float(
                np.mean(
                    pairwise_values
                )
            )

        else:

            mean_pairwise = np.nan

        # --------------------------------------------------
        # Monte Carlo consensus
        # --------------------------------------------------

        feature_counts = {}

        for feature_set in top_sets:

            for feature in set(
                feature_set
            ):

                feature_counts[
                    feature
                ] = (
                    feature_counts.get(
                        feature,
                        0,
                    )
                    +
                    1
                )

        ordered_consensus = sorted(
            feature_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        mc_consensus = [
            feature

            for feature, _
            in ordered_consensus[
                :TOP_K
            ]
        ]

        # --------------------------------------------------
        # Sample overlap with class consensus
        # --------------------------------------------------

        sample_overlaps = [
            len(
                set(
                    feature_set
                )
                &
                set(
                    mc_consensus
                )
            )
            /
            TOP_K

            for feature_set in top_sets
        ]

        mean_consensus_overlap = float(
            np.mean(
                sample_overlaps
            )
        )

        # --------------------------------------------------
        # Development vs Monte Carlo consensus
        # --------------------------------------------------

        dev_consensus = (
            phase8_consensus.get(
                fault_type,
                [],
            )
        )

        if len(dev_consensus) > 0:

            dev_mc_jaccard = jaccard(
                dev_consensus,
                mc_consensus,
            )

            development_overlap = (
                len(
                    set(
                        dev_consensus
                    )
                    &
                    set(
                        mc_consensus
                    )
                )
                /
                TOP_K
            )

        else:

            dev_mc_jaccard = np.nan
            development_overlap = np.nan

        stability_rows.append(
            {
                "fault_type":
                    fault_type,

                "scenarios":
                    len(
                        subset
                    ),

                "mean_pairwise_top5_jaccard":
                    mean_pairwise,

                "mean_sample_consensus_overlap":
                    mean_consensus_overlap,

                "development_mc_jaccard":
                    dev_mc_jaccard,

                "development_mc_top5_overlap":
                    development_overlap,

                "mc_consensus_features":
                    "|".join(
                        mc_consensus
                    ),

                "development_consensus_features":
                    "|".join(
                        dev_consensus
                    ),
            }
        )

        for rank, (
            feature,
            count,
        ) in enumerate(
            ordered_consensus[
                :TOP_K
            ],
            start=1,
        ):

            consensus_rows.append(
                {
                    "fault_type":
                        fault_type,

                    "rank":
                        rank,

                    "feature":
                        feature,

                    "scenario_fraction":
                        count
                        /
                        len(
                            subset
                        ),
                }
            )

    stability_df = pd.DataFrame(
        stability_rows
    )

    consensus_df = pd.DataFrame(
        consensus_rows
    )

    # ======================================================
    # ADD SAMPLE CONSENSUS OVERLAP
    # ======================================================

    overlap_values = []

    for _, row in (
        sample_df.iterrows()
    ):

        fault_type = str(
            row[
                "fault_type"
            ]
        )

        class_consensus = (
            consensus_df[
                consensus_df[
                    "fault_type"
                ]
                ==
                fault_type
            ]
            .sort_values(
                "rank"
            )[
                "feature"
            ]
            .tolist()
        )

        sample_top5 = (
            str(
                row[
                    "top5_features"
                ]
            )
            .split("|")
        )

        overlap_values.append(
            len(
                set(
                    sample_top5
                )
                &
                set(
                    class_consensus
                )
            )
            /
            TOP_K
        )

    sample_df[
        "class_consensus_overlap"
    ] = overlap_values

    # ======================================================
    # FAITHFULNESS PER FAULT
    # ======================================================

    faithfulness_df = (
        sample_df
        .groupby(
            "fault_type"
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            mean_top5_probability_drop=(
                "top5_probability_drop",
                "mean",
            ),

            mean_random5_probability_drop=(
                "random5_probability_drop",
                "mean",
            ),

            shap_beats_random_fraction=(
                "top5_beats_random",
                "mean",
            ),

            top5_flip_rate=(
                "top5_prediction_flip",
                "mean",
            ),

            random5_flip_rate=(
                "random5_flip_rate",
                "mean",
            ),

            mean_consensus_overlap=(
                "class_consensus_overlap",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # RANDOMIZED-CONDITION ROBUSTNESS
    # ======================================================

    sample_df[
        "severity_bin"
    ] = pd.cut(
        sample_df[
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

    sample_df[
        "noise_bin"
    ] = pd.cut(
        sample_df[
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

    severity_robustness = (
        sample_df
        .groupby(
            "severity_bin",
            observed=False,
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            mean_consensus_overlap=(
                "class_consensus_overlap",
                "mean",
            ),

            mean_top5_probability_drop=(
                "top5_probability_drop",
                "mean",
            ),

            shap_beats_random_fraction=(
                "top5_beats_random",
                "mean",
            ),
        )
        .reset_index()
    )

    noise_robustness = (
        sample_df
        .groupby(
            "noise_bin",
            observed=False,
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            mean_consensus_overlap=(
                "class_consensus_overlap",
                "mean",
            ),

            mean_top5_probability_drop=(
                "top5_probability_drop",
                "mean",
            ),

            shap_beats_random_fraction=(
                "top5_beats_random",
                "mean",
            ),
        )
        .reset_index()
    )

    profile_robustness = (
        sample_df
        .groupby(
            "profile"
        )
        .agg(
            scenarios=(
                "scenario_id",
                "count",
            ),

            mean_consensus_overlap=(
                "class_consensus_overlap",
                "mean",
            ),

            mean_top5_probability_drop=(
                "top5_probability_drop",
                "mean",
            ),

            shap_beats_random_fraction=(
                "top5_beats_random",
                "mean",
            ),
        )
        .reset_index()
    )

    # ======================================================
    # CORRELATION WITHOUT SCIPY
    # ======================================================

    severity_rank = (
        sample_df[
            "severity"
        ]
        .rank()
    )

    noise_rank = (
        sample_df[
            "noise_scale"
        ]
        .rank()
    )

    stability_rank = (
        sample_df[
            "class_consensus_overlap"
        ]
        .rank()
    )

    faithfulness_rank = (
        sample_df[
            "top5_probability_drop"
        ]
        .rank()
    )

    severity_stability_corr = float(
        severity_rank.corr(
            stability_rank
        )
    )

    noise_stability_corr = float(
        noise_rank.corr(
            stability_rank
        )
    )

    severity_faithfulness_corr = float(
        severity_rank.corr(
            faithfulness_rank
        )
    )

    noise_faithfulness_corr = float(
        noise_rank.corr(
            faithfulness_rank
        )
    )

    # ======================================================
    # GLOBAL RESULTS
    # ======================================================

    mean_reconstruction_error = float(
        np.mean(
            reconstruction_errors
        )
    )

    max_reconstruction_error = float(
        np.max(
            reconstruction_errors
        )
    )

    mean_top_drop = float(
        sample_df[
            "top5_probability_drop"
        ].mean()
    )

    mean_random_drop = float(
        sample_df[
            "random5_probability_drop"
        ].mean()
    )

    shap_beats_random = float(
        sample_df[
            "top5_beats_random"
        ].mean()
    )

    top5_flip_rate = float(
        sample_df[
            "top5_prediction_flip"
        ].mean()
    )

    random_flip_rate = float(
        sample_df[
            "random5_flip_rate"
        ].mean()
    )

    mean_pairwise_stability = float(
        stability_df[
            "mean_pairwise_top5_jaccard"
        ].mean()
    )

    mean_consensus_stability = float(
        stability_df[
            "mean_sample_consensus_overlap"
        ].mean()
    )

    mean_development_overlap = float(
        stability_df[
            "development_mc_top5_overlap"
        ].mean()
    )

    # ======================================================
    # STRUCTURAL / SCIENTIFIC CHECKS
    # ======================================================

    validation_checks = {

        "All 120 independent scenarios explained":
            (
                len(
                    sample_df
                )
                ==
                120
            ),

        "All six fault classes represented":
            (
                set(
                    sample_df[
                        "fault_type"
                    ]
                )
                ==
                set(
                    FAULT_TYPES
                )
            ),

        "Exactly 20 explanations per fault":
            bool(
                (
                    sample_df[
                        "fault_type"
                    ]
                    .value_counts()
                    ==
                    20
                ).all()
            ),

        "SHAP reconstruction fidelity":
            (
                max_reconstruction_error
                <
                1e-5
            ),

        "SHAP top-5 damages diagnosis more than random":
            (
                mean_top_drop
                >
                mean_random_drop
            ),

        "SHAP beats random for majority of scenarios":
            (
                shap_beats_random
                >
                0.50
            ),

        "Frozen 0.70 accepted operational decisions used":
            bool(
                (
                    sample_df[
                        "confidence"
                    ]
                    >=
                    CONFIDENCE_THRESHOLD
                ).all()
            ),
    }

    # ======================================================
    # SAVE TABLES
    # ======================================================

    sample_file = (
        tables_dir
        / "experiment_051_monte_carlo_xai_samples.csv"
    )

    top_features_file = (
        tables_dir
        / "experiment_051_top_shap_features.csv"
    )

    stability_file = (
        tables_dir
        / "experiment_051_xai_stability.csv"
    )

    consensus_file = (
        tables_dir
        / "experiment_051_monte_carlo_consensus_features.csv"
    )

    faithfulness_file = (
        tables_dir
        / "experiment_051_xai_faithfulness.csv"
    )

    severity_file = (
        tables_dir
        / "experiment_051_xai_severity_robustness.csv"
    )

    noise_file = (
        tables_dir
        / "experiment_051_xai_noise_robustness.csv"
    )

    profile_file = (
        tables_dir
        / "experiment_051_xai_profile_robustness.csv"
    )

    validation_file = (
        tables_dir
        / "experiment_051_validation_checks.csv"
    )

    sample_df.to_csv(
        sample_file,
        index=False,
    )

    top_features_df.to_csv(
        top_features_file,
        index=False,
    )

    stability_df.to_csv(
        stability_file,
        index=False,
    )

    consensus_df.to_csv(
        consensus_file,
        index=False,
    )

    faithfulness_df.to_csv(
        faithfulness_file,
        index=False,
    )

    severity_robustness.to_csv(
        severity_file,
        index=False,
    )

    noise_robustness.to_csv(
        noise_file,
        index=False,
    )

    profile_robustness.to_csv(
        profile_file,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "check":
                    name,

                "pass":
                    result,
            }

            for name, result
            in validation_checks.items()
        ]
    ).to_csv(
        validation_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — STABILITY
    # ======================================================

    stability_figure = (
        figures_dir
        / "experiment_051_monte_carlo_xai_stability.png"
    )

    x = np.arange(
        len(
            stability_df
        )
    )

    width = 0.27

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width,
        stability_df[
            "mean_pairwise_top5_jaccard"
        ],
        width=width,
        label="Within-class pairwise Jaccard",
    )

    plt.bar(
        x,
        stability_df[
            "mean_sample_consensus_overlap"
        ],
        width=width,
        label="Monte Carlo consensus overlap",
    )

    plt.bar(
        x + width,
        stability_df[
            "development_mc_top5_overlap"
        ],
        width=width,
        label="Phase-8 vs Monte Carlo top-5 overlap",
    )

    plt.xticks(
        x,
        stability_df[
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
        "Similarity / Overlap"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo Explanation Stability"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        stability_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — FAITHFULNESS
    # ======================================================

    faithfulness_figure = (
        figures_dir
        / "experiment_051_monte_carlo_xai_faithfulness.png"
    )

    x = np.arange(
        len(
            faithfulness_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        faithfulness_df[
            "mean_top5_probability_drop"
        ],
        width=width,
        label="Remove top-5 SHAP features",
    )

    plt.bar(
        x + width / 2,
        faithfulness_df[
            "mean_random5_probability_drop"
        ],
        width=width,
        label="Remove 5 random features",
    )

    plt.xticks(
        x,
        faithfulness_df[
            "fault_type"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "Predicted-Class Probability Drop"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo SHAP Faithfulness"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        faithfulness_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — SEVERITY ROBUSTNESS
    # ======================================================

    severity_figure = (
        figures_dir
        / "experiment_051_xai_vs_severity.png"
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        severity_robustness[
            "severity_bin"
        ].astype(str),
        severity_robustness[
            "mean_consensus_overlap"
        ],
        marker="o",
        label="Explanation consensus overlap",
    )

    plt.plot(
        severity_robustness[
            "severity_bin"
        ].astype(str),
        severity_robustness[
            "shap_beats_random_fraction"
        ],
        marker="o",
        label="SHAP beats random fraction",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Fault Severity"
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "XAI Robustness vs Fault Severity"
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
        / "experiment_051_xai_vs_noise.png"
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        noise_robustness[
            "noise_bin"
        ].astype(str),
        noise_robustness[
            "mean_consensus_overlap"
        ],
        marker="o",
        label="Explanation consensus overlap",
    )

    plt.plot(
        noise_robustness[
            "noise_bin"
        ].astype(str),
        noise_robustness[
            "shap_beats_random_fraction"
        ],
        marker="o",
        label="SHAP beats random fraction",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Sensor-Noise Scale"
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "XAI Robustness vs Sensor Noise"
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
    print("=" * 106)

    print(
        "SENTINEL-XAI - EXPERIMENT 051"
    )

    print(
        "INDEPENDENT MONTE CARLO XAI ROBUSTNESS"
    )

    print("=" * 106)

    print()

    print(
        f"Independent scenarios explained: "
        f"{len(sample_df)}"
    )

    print(
        f"Features after frozen preprocessing: "
        f"{X_transformed.shape[1]}"
    )

    print(
        f"Top supporting features per explanation: "
        f"{TOP_K}"
    )

    print()

    print(
        "Model retraining: NO"
    )

    print(
        "Confidence threshold retuning: NO"
    )

    print(
        "SHAP parameter tuning on Monte Carlo set: NO"
    )

    print()

    print("-" * 106)

    print(
        "SHAP ADDITIVITY / FIDELITY"
    )

    print("-" * 106)

    print()

    print(
        f"Mean reconstruction error: "
        f"{mean_reconstruction_error:.10f}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{max_reconstruction_error:.10f}"
    )

    print()

    print("-" * 106)

    print(
        "MONTE CARLO FAITHFULNESS"
    )

    print("-" * 106)

    print()

    print(
        f"Mean probability drop after "
        f"top-5 SHAP ablation: "
        f"{mean_top_drop:.4f}"
    )

    print(
        f"Mean probability drop after "
        f"random-5 ablation: "
        f"{mean_random_drop:.4f}"
    )

    print(
        f"SHAP top-5 beats random fraction: "
        f"{shap_beats_random:.4f}"
    )

    print(
        f"Top-5 prediction flip rate: "
        f"{top5_flip_rate:.4f}"
    )

    print(
        f"Random-5 prediction flip rate: "
        f"{random_flip_rate:.4f}"
    )

    print()

    print("-" * 106)

    print(
        "EXPLANATION STABILITY"
    )

    print("-" * 106)

    print()

    print(
        f"Mean within-class pairwise "
        f"top-5 Jaccard: "
        f"{mean_pairwise_stability:.4f}"
    )

    print(
        f"Mean Monte Carlo consensus overlap: "
        f"{mean_consensus_stability:.4f}"
    )

    print(
        f"Mean Phase-8 / Monte-Carlo "
        f"top-5 overlap: "
        f"{mean_development_overlap:.4f}"
    )

    print()

    for _, row in (
        stability_df.iterrows()
    ):

        print(
            row[
                "fault_type"
            ]
        )

        print(
            f"  Pairwise Jaccard: "
            f"{row['mean_pairwise_top5_jaccard']:.4f}"
        )

        print(
            f"  Monte Carlo consensus overlap: "
            f"{row['mean_sample_consensus_overlap']:.4f}"
        )

        print(
            f"  Phase-8 / MC top-5 overlap: "
            f"{row['development_mc_top5_overlap']:.4f}"
        )

        print(
            f"  MC consensus: "
            f"{row['mc_consensus_features']}"
        )

        print()

    print("-" * 106)

    print(
        "RANDOMIZED-CONDITION SENSITIVITY"
    )

    print("-" * 106)

    print()

    print(
        f"Severity vs explanation stability "
        f"rank correlation: "
        f"{severity_stability_corr:.4f}"
    )

    print(
        f"Noise vs explanation stability "
        f"rank correlation: "
        f"{noise_stability_corr:.4f}"
    )

    print(
        f"Severity vs faithfulness "
        f"rank correlation: "
        f"{severity_faithfulness_corr:.4f}"
    )

    print(
        f"Noise vs faithfulness "
        f"rank correlation: "
        f"{noise_faithfulness_corr:.4f}"
    )

    print()

    print(
        "Abrupt vs gradual:"
    )

    print()

    print(
        profile_robustness.to_string(
            index=False
        )
    )

    print()

    print("-" * 106)

    print(
        "VALIDATION CHECKS"
    )

    print("-" * 106)

    passed = 0

    for name, result in (
        validation_checks.items()
    ):

        print()

        print(
            name
        )

        print(
            f"  PASS: "
            f"{bool(result)}"
        )

        if result:
            passed += 1

    print()

    print(
        f"Passed checks: "
        f"{passed}/"
        f"{len(validation_checks)}"
    )

    print()

    print(
        "Important scientific note:"
    )

    print(
        "Explanation-stability values are "
        "reported descriptively."
    )

    print(
        "No new post-hoc stability threshold "
        "is created from the Monte Carlo set."
    )

    print(
        "SHAP continues to explain model behavior, "
        "not physical causality."
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        sample_file
    )

    print(
        top_features_file
    )

    print(
        stability_file
    )

    print(
        consensus_file
    )

    print(
        faithfulness_file
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
        stability_figure
    )

    print(
        faithfulness_figure
    )

    print(
        severity_figure
    )

    print(
        noise_figure
    )

    print("=" * 106)


if __name__ == "__main__":
    main()