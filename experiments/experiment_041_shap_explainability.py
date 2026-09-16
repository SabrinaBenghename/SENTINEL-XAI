from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    FAULT_CLASSES,
)

from src.explainability.shap_explainer import (
    SentinelShapExplainer,
)


# ==========================================================
# BLOCKED SPLIT
# ==========================================================

def blocked_split(
    df,
    train_fraction=0.70,
):

    train_parts = []
    test_parts = []

    classes = [
        "nominal",
        *FAULT_CLASSES,
    ]

    for class_name in classes:

        subset = (
            df[
                df["target_label"]
                ==
                class_name
            ]
            .sort_values(
                "time_h"
            )
            .reset_index(
                drop=True
            )
        )

        split_index = int(
            len(subset)
            *
            train_fraction
        )

        train_parts.append(
            subset.iloc[
                :split_index
            ].copy()
        )

        test_parts.append(
            subset.iloc[
                split_index:
            ].copy()
        )

    return (
        pd.concat(
            train_parts,
            ignore_index=True,
        ),
        pd.concat(
            test_parts,
            ignore_index=True,
        ),
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD PHASE-7 DATASET
    # ======================================================

    dataset_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_036_diagnosis_feature_dataset.csv"
    )

    diagnosis_df = pd.read_csv(
        dataset_file
    )

    (
        train_df,
        test_df,
    ) = blocked_split(
        diagnosis_df,
        train_fraction=0.70,
    )

    test_x = test_df[
        DIAGNOSTIC_FEATURES
    ]

    test_y = (
        test_df[
            "target_label"
        ]
        .to_numpy()
    )

    # ======================================================
    # LOAD FROZEN PHASE-7 RANDOM FOREST
    # ======================================================

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_037_random_forest_diagnoser.joblib"
    )

    diagnoser = joblib.load(
        model_file
    )

    prediction = diagnoser.predict(
        test_x
    )

    predicted_labels = (
        prediction.predicted_labels
    )

    confidence = (
        prediction.confidence
    )

    correct = (
        predicted_labels
        ==
        test_y
    )

    # ======================================================
    # BUILD SHAP EXPLAINER
    # ======================================================

    print()
    print(
        "Computing SHAP explanations "
        "for Phase-7 test samples..."
    )

    sentinel_shap = (
        SentinelShapExplainer(
            diagnoser=diagnoser,
            feature_names=(
                DIAGNOSTIC_FEATURES
            ),
        )
    )

    (
        transformed_test,
        explanation,
    ) = sentinel_shap.explain(
        test_x
    )

    shap_values = (
        explanation.values
    )

    print(
        f"SHAP value shape: "
        f"{shap_values.shape}"
    )

    # ======================================================
    # GLOBAL SHAP IMPORTANCE
    #
    # Mean absolute SHAP value across:
    #   all samples
    #   all classes
    # ======================================================

    if shap_values.ndim == 3:

        global_importance_values = (
            np.abs(
                shap_values
            )
            .mean(
                axis=(0, 2)
            )
        )

    else:

        global_importance_values = (
            np.abs(
                shap_values
            )
            .mean(
                axis=0
            )
        )

    global_importance = pd.DataFrame(
        {
            "feature":
                DIAGNOSTIC_FEATURES,

            "mean_abs_shap":
                global_importance_values,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    global_importance[
        "rank"
    ] = np.arange(
        1,
        len(
            global_importance
        ) + 1,
    )

    # ======================================================
    # CLASS-SPECIFIC GLOBAL IMPORTANCE
    # ======================================================

    class_importance_rows = []

    classes = list(
        diagnoser.classes_
    )

    for class_name in classes:

        class_values = (
            sentinel_shap.class_values(
                explanation,
                class_name,
            )
        )

        mean_absolute = (
            np.abs(
                class_values
            )
            .mean(
                axis=0
            )
        )

        for feature, value in zip(
            DIAGNOSTIC_FEATURES,
            mean_absolute,
        ):

            class_importance_rows.append(
                {
                    "class":
                        class_name,

                    "feature":
                        feature,

                    "mean_abs_shap":
                        float(
                            value
                        ),
                }
            )

    class_importance = pd.DataFrame(
        class_importance_rows
    )

    class_importance[
        "rank"
    ] = (
        class_importance
        .groupby(
            "class"
        )[
            "mean_abs_shap"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # ======================================================
    # REPRESENTATIVE LOCAL EXPLANATIONS
    #
    # Highest-confidence correct accepted sample
    # for every class.
    # ======================================================

    representative_rows = []
    local_rows = []

    representatives = {}

    for class_name in classes:

        candidate_positions = np.where(
            (
                test_y
                ==
                class_name
            )
            &
            correct
            &
            (
                confidence
                >=
                0.70
            )
        )[0]

        if len(
            candidate_positions
        ) == 0:

            print(
                f"WARNING: No accepted correct sample "
                f"for {class_name}"
            )

            continue

        best_position = (
            candidate_positions[
                np.argmax(
                    confidence[
                        candidate_positions
                    ]
                )
            ]
        )

        representatives[
            class_name
        ] = best_position

        predicted_class = (
            predicted_labels[
                best_position
            ]
        )

        class_values = (
            sentinel_shap.class_values(
                explanation,
                predicted_class,
            )
        )

        local_shap = (
            class_values[
                best_position
            ]
        )

        sample_values = (
            transformed_test
            .iloc[
                best_position
            ]
            .to_numpy()
        )

        local_table = pd.DataFrame(
            {
                "feature":
                    DIAGNOSTIC_FEATURES,

                "feature_value":
                    sample_values,

                "shap_value":
                    local_shap,
            }
        )

        local_table[
            "abs_shap"
        ] = np.abs(
            local_table[
                "shap_value"
            ]
        )

        local_table = (
            local_table
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        local_table.insert(
            0,
            "true_label",
            class_name,
        )

        local_table.insert(
            1,
            "predicted_label",
            predicted_class,
        )

        local_table.insert(
            2,
            "confidence",
            float(
                confidence[
                    best_position
                ]
            ),
        )

        local_table.insert(
            3,
            "time_h",
            float(
                test_df
                .iloc[
                    best_position
                ][
                    "time_h"
                ]
            ),
        )

        local_rows.append(
            local_table
        )

        top_positive = (
            local_table[
                local_table[
                    "shap_value"
                ]
                >
                0
            ]
            .head(5)
        )

        top_features = list(
            top_positive[
                "feature"
            ]
        )

        representative_row = {

            "class":
                class_name,

            "prediction":
                predicted_class,

            "confidence":
                float(
                    confidence[
                        best_position
                    ]
                ),

            "time_h":
                float(
                    test_df
                    .iloc[
                        best_position
                    ][
                        "time_h"
                    ]
                ),
        }

        for number in range(
            1,
            6,
        ):

            if (
                len(
                    top_features
                )
                >=
                number
            ):

                representative_row[
                    f"top_supporting_feature_{number}"
                ] = (
                    top_features[
                        number - 1
                    ]
                )

            else:

                representative_row[
                    f"top_supporting_feature_{number}"
                ] = ""

        representative_rows.append(
            representative_row
        )

    local_explanations = pd.concat(
        local_rows,
        ignore_index=True,
    )

    representative_df = pd.DataFrame(
        representative_rows
    )

    # ======================================================
    # ADDITIVITY / FIDELITY CHECK
    #
    # For representative samples:
    # base value + sum(SHAP) should reconstruct
    # the model output for the predicted class.
    # ======================================================

    fidelity_rows = []

    for class_name, position in (
        representatives.items()
    ):

        predicted_class = (
            predicted_labels[
                position
            ]
        )

        class_index = list(
            diagnoser.classes_
        ).index(
            predicted_class
        )

        class_values = (
            sentinel_shap.class_values(
                explanation,
                predicted_class,
            )
        )

        shap_sum = float(
            np.sum(
                class_values[
                    position
                ]
            )
        )

        base_value = (
            sentinel_shap.base_value(
                explanation,
                position,
                predicted_class,
            )
        )

        reconstructed_output = (
            base_value
            +
            shap_sum
        )

        model_probability = float(
            prediction.probabilities[
                position,
                class_index,
            ]
        )

        error = abs(
            reconstructed_output
            -
            model_probability
        )

        fidelity_rows.append(
            {
                "class":
                    class_name,

                "model_probability":
                    model_probability,

                "base_value":
                    base_value,

                "shap_sum":
                    shap_sum,

                "reconstructed_output":
                    reconstructed_output,

                "absolute_error":
                    error,
            }
        )

    fidelity_df = pd.DataFrame(
        fidelity_rows
    )

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
    # SAVE TABLES
    # ======================================================

    global_file = (
        tables_dir
        / "experiment_041_shap_global_importance.csv"
    )

    class_file = (
        tables_dir
        / "experiment_041_shap_class_importance.csv"
    )

    local_file = (
        tables_dir
        / "experiment_041_shap_local_explanations.csv"
    )

    representative_file = (
        tables_dir
        / "experiment_041_shap_representatives.csv"
    )

    fidelity_file = (
        tables_dir
        / "experiment_041_shap_fidelity.csv"
    )

    global_importance.to_csv(
        global_file,
        index=False,
    )

    class_importance.to_csv(
        class_file,
        index=False,
    )

    local_explanations.to_csv(
        local_file,
        index=False,
    )

    representative_df.to_csv(
        representative_file,
        index=False,
    )

    fidelity_df.to_csv(
        fidelity_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — GLOBAL SHAP IMPORTANCE
    # ======================================================

    global_figure = (
        figures_dir
        / "experiment_041_shap_global_importance.png"
    )

    top_global = (
        global_importance
        .head(15)
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_global[
            "feature"
        ],
        top_global[
            "mean_abs_shap"
        ],
    )

    plt.xlabel(
        "Mean |SHAP Value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Global SHAP Feature Importance"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        global_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — BATTERY DEGRADATION LOCAL SHAP
    # ======================================================

    battery_figure = (
        figures_dir
        / "experiment_041_shap_battery_local.png"
    )

    battery_rows = (
        local_explanations[
            local_explanations[
                "true_label"
            ]
            ==
            "battery_degradation"
        ]
        .head(15)
        .sort_values(
            "shap_value",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        battery_rows[
            "feature"
        ],
        battery_rows[
            "shap_value"
        ],
    )

    plt.xlabel(
        "SHAP Contribution to "
        "Battery-Degradation Output"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Local SHAP: Battery Degradation"
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        battery_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — CLASS-SPECIFIC TOP FEATURES
    # ======================================================

    class_figure = (
        figures_dir
        / "experiment_041_shap_class_specific.png"
    )

    class_top_scores = []

    class_labels = []

    for class_name in classes:

        top_feature = (
            class_importance[
                class_importance[
                    "class"
                ]
                ==
                class_name
            ]
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .iloc[0]
        )

        class_labels.append(
            class_name
        )

        class_top_scores.append(
            top_feature[
                "mean_abs_shap"
            ]
        )

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        class_labels,
        class_top_scores,
    )

    plt.ylabel(
        "Top Feature Mean |SHAP|"
    )

    plt.xlabel(
        "Diagnostic Class"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Class-Specific SHAP Strength"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        class_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 92)

    print(
        "SENTINEL-XAI - EXPERIMENT 041"
    )

    print(
        "SHAP GLOBAL + LOCAL EXPLAINABILITY"
    )

    print("=" * 92)

    print()

    print(
        f"Explained test samples: "
        f"{len(test_x)}"
    )

    print(
        f"Features: "
        f"{len(DIAGNOSTIC_FEATURES)}"
    )

    print(
        f"Classes: "
        f"{len(classes)}"
    )

    print()

    print("-" * 92)

    print(
        "TOP 15 GLOBAL SHAP FEATURES"
    )

    print("-" * 92)

    print()

    for _, row in (
        global_importance
        .head(15)
        .iterrows()
    ):

        print(
            f"{int(row['rank']):>2}. "
            f"{row['feature']:<44} "
            f"{row['mean_abs_shap']:.6f}"
        )

    # ======================================================
    # CLASS-SPECIFIC
    # ======================================================

    print()

    print("-" * 92)

    print(
        "TOP 5 FEATURES PER CLASS"
    )

    print("-" * 92)

    for class_name in classes:

        print()
        print(
            class_name
        )

        class_top = (
            class_importance[
                class_importance[
                    "class"
                ]
                ==
                class_name
            ]
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(5)
        )

        for _, row in (
            class_top.iterrows()
        ):

            print(
                f"  {int(row['rank'])}. "
                f"{row['feature']:<42} "
                f"{row['mean_abs_shap']:.6f}"
            )

    # ======================================================
    # LOCAL
    # ======================================================

    print()

    print("-" * 92)

    print(
        "REPRESENTATIVE LOCAL EXPLANATIONS"
    )

    print("-" * 92)

    for _, row in (
        representative_df.iterrows()
    ):

        print()

        print(
            row["class"]
        )

        print(
            f"  Prediction: "
            f"{row['prediction']}"
        )

        print(
            f"  Confidence: "
            f"{row['confidence']:.4f}"
        )

        print(
            "  Top positive SHAP evidence:"
        )

        for number in range(
            1,
            6,
        ):

            feature = (
                row[
                    f"top_supporting_feature_{number}"
                ]
            )

            if (
                isinstance(
                    feature,
                    str,
                )
                and
                feature
            ):

                print(
                    f"    {number}. "
                    f"{feature}"
                )

    # ======================================================
    # FIDELITY
    # ======================================================

    max_fidelity_error = float(
        fidelity_df[
            "absolute_error"
        ].max()
    )

    mean_fidelity_error = float(
        fidelity_df[
            "absolute_error"
        ].mean()
    )

    print()

    print("-" * 92)

    print(
        "SHAP ADDITIVITY / FIDELITY CHECK"
    )

    print("-" * 92)

    print()

    print(
        f"Mean reconstruction error: "
        f"{mean_fidelity_error:.10f}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{max_fidelity_error:.10f}"
    )

    print()

    if max_fidelity_error < 1e-5:

        print(
            "RESULT: SHAP EXPLANATION "
            "FIDELITY PASSED"
        )

    else:

        print(
            "RESULT: SHAP ADDITIVITY "
            "REQUIRES REVIEW"
        )

    print()

    print("=" * 92)

    print(
        "Interpretation:"
    )

    print()

    print(
        "Positive SHAP value -> pushes the model "
        "toward the selected diagnosis."
    )

    print(
        "Negative SHAP value -> pushes the model "
        "away from that diagnosis."
    )

    print()

    print(
        "SHAP explains the model's decision; "
        "it does not prove physical causality."
    )

    print("=" * 92)

    print()

    print(
        "Tables saved to:"
    )

    print(
        global_file
    )

    print(
        class_file
    )

    print(
        local_file
    )

    print(
        representative_file
    )

    print(
        fidelity_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        global_figure
    )

    print(
        battery_figure
    )

    print(
        class_figure
    )

    print("=" * 92)


if __name__ == "__main__":
    main()