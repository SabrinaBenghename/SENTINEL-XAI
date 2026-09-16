from pathlib import Path
import sys
from collections import Counter
from itertools import combinations

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
# JACCARD SIMILARITY
# ==========================================================

def jaccard(
    a,
    b,
):

    union = (
        a
        |
        b
    )

    if len(
        union
    ) == 0:

        return 1.0

    return (
        len(
            a
            &
            b
        )
        /
        len(
            union
        )
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD DATASET
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

    train_x = train_df[
        DIAGNOSTIC_FEATURES
    ]

    test_x = test_df[
        DIAGNOSTIC_FEATURES
    ].copy()

    test_y = (
        test_df[
            "target_label"
        ]
        .to_numpy()
    )

    # ======================================================
    # LOAD FROZEN PHASE-7 DIAGNOSER
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

    probabilities = (
        prediction.probabilities
    )

    correct = (
        predicted_labels
        ==
        test_y
    )

    # ======================================================
    # ACCEPTED + CORRECT EXPLANATION POPULATION
    # ======================================================

    accepted = (
        confidence
        >=
        0.70
    )

    evaluation_mask = (
        correct
        &
        accepted
    )

    evaluation_positions = np.where(
        evaluation_mask
    )[0]

    # ======================================================
    # BACKGROUND / TYPICAL FEATURE VALUES
    #
    # Used for feature-removal experiments.
    # ======================================================

    background_values = (
        train_x
        .median(
            numeric_only=True
        )
        .reindex(
            DIAGNOSTIC_FEATURES
        )
        .fillna(0.0)
    )

    # ======================================================
    # SHAP
    # ======================================================

    print()
    print(
        "Computing SHAP values for "
        "explanation consistency testing..."
    )

    shap_explainer = (
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
    ) = shap_explainer.explain(
        test_x
    )

    # ======================================================
    # CACHE CLASS-SPECIFIC SHAP MATRICES
    # ======================================================

    class_shap = {}

    for class_name in (
        diagnoser.classes_
    ):

        class_shap[
            class_name
        ] = (
            shap_explainer
            .class_values(
                explanation,
                class_name,
            )
        )

    # ======================================================
    # TOP-K SUPPORTING FEATURES
    # ======================================================

    top_k_target = 5

    sample_rows = []

    top_feature_sets = {}

    top_feature_lists = {}

    rng = np.random.default_rng(
        42
    )

    # Copies for batched faithfulness evaluation.
    top_ablated_x = (
        test_x
        .iloc[
            evaluation_positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    random_ablated_x = (
        test_x
        .iloc[
            evaluation_positions
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    original_classes = []
    original_probabilities = []

    for output_position, test_position in enumerate(
        evaluation_positions
    ):

        predicted_class = (
            predicted_labels[
                test_position
            ]
        )

        shap_vector = (
            class_shap[
                predicted_class
            ][
                test_position
            ]
        )

        # --------------------------------------------------
        # Positive SHAP features specifically support
        # the predicted class.
        # --------------------------------------------------

        positive_indices = np.where(
            shap_vector > 0
        )[0]

        positive_indices = (
            positive_indices[
                np.argsort(
                    shap_vector[
                        positive_indices
                    ]
                )[
                    ::-1
                ]
            ]
        )

        k = min(
            top_k_target,
            len(
                positive_indices
            ),
        )

        top_indices = (
            positive_indices[
                :k
            ]
        )

        top_features = [
            DIAGNOSTIC_FEATURES[
                index
            ]
            for index
            in top_indices
        ]

        top_feature_sets[
            test_position
        ] = set(
            top_features
        )

        top_feature_lists[
            test_position
        ] = top_features

        # --------------------------------------------------
        # Random-feature comparison.
        #
        # Exclude selected SHAP-supporting features.
        # --------------------------------------------------

        available_indices = np.array(
            [
                index
                for index
                in range(
                    len(
                        DIAGNOSTIC_FEATURES
                    )
                )
                if index
                not in
                set(
                    top_indices
                )
            ]
        )

        if k > 0:

            random_indices = (
                rng.choice(
                    available_indices,
                    size=k,
                    replace=False,
                )
            )

        else:

            random_indices = (
                np.array(
                    [],
                    dtype=int,
                )
            )

        random_features = [
            DIAGNOSTIC_FEATURES[
                index
            ]
            for index
            in random_indices
        ]

        # --------------------------------------------------
        # Replace top SHAP-supporting features with
        # typical training values.
        # --------------------------------------------------

        for feature in top_features:

            top_ablated_x.loc[
                output_position,
                feature,
            ] = (
                background_values[
                    feature
                ]
            )

        # --------------------------------------------------
        # Random-feature ablation baseline.
        # --------------------------------------------------

        for feature in random_features:

            random_ablated_x.loc[
                output_position,
                feature,
            ] = (
                background_values[
                    feature
                ]
            )

        class_index = list(
            diagnoser.classes_
        ).index(
            predicted_class
        )

        baseline_probability = float(
            probabilities[
                test_position,
                class_index,
            ]
        )

        original_classes.append(
            predicted_class
        )

        original_probabilities.append(
            baseline_probability
        )

        row = {

            "test_position":
                int(
                    test_position
                ),

            "true_label":
                test_y[
                    test_position
                ],

            "predicted_label":
                predicted_class,

            "confidence":
                float(
                    confidence[
                        test_position
                    ]
                ),

            "baseline_probability":
                baseline_probability,

            "top_k_used":
                k,
        }

        for rank in range(
            1,
            top_k_target + 1,
        ):

            if len(
                top_features
            ) >= rank:

                feature = (
                    top_features[
                        rank - 1
                    ]
                )

                feature_index = (
                    DIAGNOSTIC_FEATURES.index(
                        feature
                    )
                )

                row[
                    f"top_feature_{rank}"
                ] = feature

                row[
                    f"top_shap_{rank}"
                ] = float(
                    shap_vector[
                        feature_index
                    ]
                )

            else:

                row[
                    f"top_feature_{rank}"
                ] = ""

                row[
                    f"top_shap_{rank}"
                ] = 0.0

        sample_rows.append(
            row
        )

    sample_df = pd.DataFrame(
        sample_rows
    )

    # ======================================================
    # BATCHED ABLATION PREDICTIONS
    # ======================================================

    top_prediction = (
        diagnoser.predict(
            top_ablated_x
        )
    )

    random_prediction = (
        diagnoser.predict(
            random_ablated_x
        )
    )

    faithfulness_rows = []

    for position in range(
        len(
            evaluation_positions
        )
    ):

        original_class = (
            original_classes[
                position
            ]
        )

        class_index = list(
            diagnoser.classes_
        ).index(
            original_class
        )

        baseline_probability = (
            original_probabilities[
                position
            ]
        )

        top_probability = float(
            top_prediction
            .probabilities[
                position,
                class_index,
            ]
        )

        random_probability = float(
            random_prediction
            .probabilities[
                position,
                class_index,
            ]
        )

        top_drop = (
            baseline_probability
            -
            top_probability
        )

        random_drop = (
            baseline_probability
            -
            random_probability
        )

        top_flip = (
            top_prediction
            .predicted_labels[
                position
            ]
            !=
            original_class
        )

        random_flip = (
            random_prediction
            .predicted_labels[
                position
            ]
            !=
            original_class
        )

        faithfulness_rows.append(
            {
                "true_label":
                    sample_df
                    .iloc[
                        position
                    ][
                        "true_label"
                    ],

                "predicted_label":
                    original_class,

                "baseline_probability":
                    baseline_probability,

                "top5_ablated_probability":
                    top_probability,

                "random5_ablated_probability":
                    random_probability,

                "top5_probability_drop":
                    top_drop,

                "random5_probability_drop":
                    random_drop,

                "top5_larger_drop":
                    top_drop
                    >
                    random_drop,

                "top5_prediction_flip":
                    top_flip,

                "random5_prediction_flip":
                    random_flip,
            }
        )

    faithfulness_df = pd.DataFrame(
        faithfulness_rows
    )

    # ======================================================
    # GLOBAL FAITHFULNESS METRICS
    # ======================================================

    mean_top_drop = float(
        faithfulness_df[
            "top5_probability_drop"
        ].mean()
    )

    mean_random_drop = float(
        faithfulness_df[
            "random5_probability_drop"
        ].mean()
    )

    top_wins_fraction = float(
        faithfulness_df[
            "top5_larger_drop"
        ].mean()
    )

    top_flip_rate = float(
        faithfulness_df[
            "top5_prediction_flip"
        ].mean()
    )

    random_flip_rate = float(
        faithfulness_df[
            "random5_prediction_flip"
        ].mean()
    )

    # ======================================================
    # PER-CLASS FAITHFULNESS
    # ======================================================

    classes = [
        "nominal",
        *FAULT_CLASSES,
    ]

    per_class_faithfulness_rows = []

    for class_name in classes:

        subset = faithfulness_df[
            faithfulness_df[
                "true_label"
            ]
            ==
            class_name
        ]

        per_class_faithfulness_rows.append(
            {
                "class":
                    class_name,

                "samples":
                    len(
                        subset
                    ),

                "mean_top5_probability_drop":
                    float(
                        subset[
                            "top5_probability_drop"
                        ].mean()
                    ),

                "mean_random5_probability_drop":
                    float(
                        subset[
                            "random5_probability_drop"
                        ].mean()
                    ),

                "top5_win_fraction":
                    float(
                        subset[
                            "top5_larger_drop"
                        ].mean()
                    ),

                "top5_flip_rate":
                    float(
                        subset[
                            "top5_prediction_flip"
                        ].mean()
                    ),

                "random5_flip_rate":
                    float(
                        subset[
                            "random5_prediction_flip"
                        ].mean()
                    ),
            }
        )

    per_class_faithfulness = pd.DataFrame(
        per_class_faithfulness_rows
    )

    # ======================================================
    # WITHIN-CLASS EXPLANATION STABILITY
    # ======================================================

    stability_rows = []
    consensus_rows = []

    for class_name in classes:

        class_positions = [
            position
            for position
            in evaluation_positions
            if test_y[
                position
            ]
            ==
            class_name
        ]

        class_sets = [
            top_feature_sets[
                position
            ]
            for position
            in class_positions
        ]

        class_lists = [
            top_feature_lists[
                position
            ]
            for position
            in class_positions
        ]

        # --------------------------------------------------
        # Pairwise Jaccard similarity
        # --------------------------------------------------

        pair_scores = []

        for first, second in combinations(
            class_sets,
            2,
        ):

            pair_scores.append(
                jaccard(
                    first,
                    second,
                )
            )

        if len(
            pair_scores
        ) > 0:

            mean_jaccard = float(
                np.mean(
                    pair_scores
                )
            )

        else:

            mean_jaccard = 1.0

        # --------------------------------------------------
        # Consensus / modal features
        # --------------------------------------------------

        feature_counter = Counter()

        top1_counter = Counter()

        for feature_list in class_lists:

            feature_counter.update(
                feature_list
            )

            if len(
                feature_list
            ) > 0:

                top1_counter[
                    feature_list[0]
                ] += 1

        consensus_features = [
            feature
            for feature, _
            in feature_counter.most_common(
                5
            )
        ]

        consensus_set = set(
            consensus_features
        )

        consensus_jaccards = [
            jaccard(
                feature_set,
                consensus_set,
            )
            for feature_set
            in class_sets
        ]

        mean_consensus_overlap = float(
            np.mean(
                consensus_jaccards
            )
        )

        if len(
            top1_counter
        ) > 0:

            modal_top1_feature, modal_count = (
                top1_counter
                .most_common(
                    1
                )[0]
            )

            modal_top1_fraction = (
                modal_count
                /
                len(
                    class_lists
                )
            )

        else:

            modal_top1_feature = ""
            modal_top1_fraction = 0.0

        stability_rows.append(
            {
                "class":
                    class_name,

                "samples":
                    len(
                        class_positions
                    ),

                "mean_pairwise_top5_jaccard":
                    mean_jaccard,

                "mean_consensus_top5_overlap":
                    mean_consensus_overlap,

                "modal_top1_feature":
                    modal_top1_feature,

                "modal_top1_fraction":
                    modal_top1_fraction,
            }
        )

        for rank in range(
            1,
            6,
        ):

            if len(
                consensus_features
            ) >= rank:

                feature = (
                    consensus_features[
                        rank - 1
                    ]
                )

                count = (
                    feature_counter[
                        feature
                    ]
                )

                fraction = (
                    count
                    /
                    len(
                        class_positions
                    )
                )

            else:

                feature = ""
                count = 0
                fraction = 0.0

            consensus_rows.append(
                {
                    "class":
                        class_name,

                    "rank":
                        rank,

                    "feature":
                        feature,

                    "sample_count":
                        count,

                    "sample_fraction":
                        fraction,
                }
            )

    stability_df = pd.DataFrame(
        stability_rows
    )

    consensus_df = pd.DataFrame(
        consensus_rows
    )

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

    sample_file = (
        tables_dir
        / "experiment_043_sample_top_features.csv"
    )

    faithfulness_file = (
        tables_dir
        / "experiment_043_faithfulness.csv"
    )

    per_class_file = (
        tables_dir
        / "experiment_043_per_class_faithfulness.csv"
    )

    stability_file = (
        tables_dir
        / "experiment_043_explanation_stability.csv"
    )

    consensus_file = (
        tables_dir
        / "experiment_043_class_consensus_features.csv"
    )

    sample_df.to_csv(
        sample_file,
        index=False,
    )

    faithfulness_df.to_csv(
        faithfulness_file,
        index=False,
    )

    per_class_faithfulness.to_csv(
        per_class_file,
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

    # ======================================================
    # FIGURE 1 — FAITHFULNESS
    # ======================================================

    faithfulness_figure = (
        figures_dir
        / "experiment_043_xai_faithfulness.png"
    )

    x = np.arange(
        len(
            per_class_faithfulness
        )
    )

    width = 0.35

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        x - width / 2,
        per_class_faithfulness[
            "mean_top5_probability_drop"
        ],
        width=width,
        label="Remove top-5 SHAP features",
    )

    plt.bar(
        x + width / 2,
        per_class_faithfulness[
            "mean_random5_probability_drop"
        ],
        width=width,
        label="Remove 5 random features",
    )

    plt.xticks(
        x,
        per_class_faithfulness[
            "class"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "Mean Drop in Predicted-Class Probability"
    )

    plt.title(
        "SENTINEL-XAI - "
        "SHAP Explanation Faithfulness"
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
    # FIGURE 2 — EXPLANATION STABILITY
    # ======================================================

    stability_figure = (
        figures_dir
        / "experiment_043_xai_stability.png"
    )

    x = np.arange(
        len(
            stability_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        x - width / 2,
        stability_df[
            "mean_pairwise_top5_jaccard"
        ],
        width=width,
        label="Pairwise top-5 similarity",
    )

    plt.bar(
        x + width / 2,
        stability_df[
            "mean_consensus_top5_overlap"
        ],
        width=width,
        label="Consensus top-5 overlap",
    )

    plt.xticks(
        x,
        stability_df[
            "class"
        ],
        rotation=30,
        ha="right",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Jaccard Similarity"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Within-Class Explanation Stability"
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
    # FIGURE 3 — ABLATION FLIP RATE
    # ======================================================

    flip_figure = (
        figures_dir
        / "experiment_043_xai_ablation_flip_rate.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        [
            "Top-5 SHAP\nfeatures",
            "Random 5\nfeatures",
        ],
        [
            top_flip_rate,
            random_flip_rate,
        ],
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.ylabel(
        "Prediction Flip Rate"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Explanation Ablation Impact"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        flip_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 94)

    print(
        "SENTINEL-XAI - EXPERIMENT 043"
    )

    print(
        "XAI STABILITY + FAITHFULNESS ANALYSIS"
    )

    print("=" * 94)

    print()

    print(
        f"Accepted and correct samples explained: "
        f"{len(evaluation_positions)}"
    )

    print(
        f"Top supporting features per explanation: "
        f"{top_k_target}"
    )

    print()

    print("-" * 94)

    print(
        "GLOBAL FAITHFULNESS"
    )

    print("-" * 94)

    print()

    print(
        f"Mean probability drop after removing "
        f"top-5 SHAP evidence: "
        f"{mean_top_drop:.4f}"
    )

    print(
        f"Mean probability drop after removing "
        f"5 random features: "
        f"{mean_random_drop:.4f}"
    )

    print(
        f"Top-5 SHAP ablation causes larger drop: "
        f"{top_wins_fraction:.4f}"
    )

    print()

    print(
        f"Prediction flip rate after top-5 ablation: "
        f"{top_flip_rate:.4f}"
    )

    print(
        f"Prediction flip rate after random ablation: "
        f"{random_flip_rate:.4f}"
    )

    print()

    print("-" * 94)

    print(
        "PER-CLASS FAITHFULNESS"
    )

    print("-" * 94)

    for _, row in (
        per_class_faithfulness.iterrows()
    ):

        print()

        print(
            row[
                "class"
            ]
        )

        print(
            f"  Samples: "
            f"{int(row['samples'])}"
        )

        print(
            f"  Top-5 probability drop: "
            f"{row['mean_top5_probability_drop']:.4f}"
        )

        print(
            f"  Random-5 probability drop: "
            f"{row['mean_random5_probability_drop']:.4f}"
        )

        print(
            f"  Top-5 wins: "
            f"{row['top5_win_fraction']:.4f}"
        )

        print(
            f"  Top-5 flip rate: "
            f"{row['top5_flip_rate']:.4f}"
        )

    print()

    print("-" * 94)

    print(
        "WITHIN-CLASS EXPLANATION STABILITY"
    )

    print("-" * 94)

    for _, row in (
        stability_df.iterrows()
    ):

        print()

        print(
            row[
                "class"
            ]
        )

        print(
            f"  Samples: "
            f"{int(row['samples'])}"
        )

        print(
            f"  Mean pairwise top-5 Jaccard: "
            f"{row['mean_pairwise_top5_jaccard']:.4f}"
        )

        print(
            f"  Mean consensus overlap: "
            f"{row['mean_consensus_top5_overlap']:.4f}"
        )

        print(
            f"  Most common #1 feature: "
            f"{row['modal_top1_feature']}"
        )

        print(
            f"  #1 feature frequency: "
            f"{row['modal_top1_fraction']:.4f}"
        )

    print()

    print("-" * 94)

    print(
        "CLASS CONSENSUS FEATURES"
    )

    print("-" * 94)

    for class_name in classes:

        print()
        print(
            class_name
        )

        subset = (
            consensus_df[
                consensus_df[
                    "class"
                ]
                ==
                class_name
            ]
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  {int(row['rank'])}. "
                f"{row['feature']:<42} "
                f"{row['sample_fraction']:.1%}"
            )

    print()

    print("=" * 94)

    print(
        "Interpretation:"
    )

    print()

    print(
        "Faithfulness is supported when removing "
        "SHAP-selected evidence damages the original "
        "diagnosis more than removing random features."
    )

    print()

    print(
        "Stability measures similarity of the strongest "
        "supporting features within the same fault class."
    )

    print()

    print(
        "Perfect stability is not expected because correlated "
        "telemetry, residual, rule and physics features can "
        "share explanatory credit."
    )

    print("=" * 94)

    print()

    print(
        "Tables saved to:"
    )

    print(
        sample_file
    )

    print(
        faithfulness_file
    )

    print(
        per_class_file
    )

    print(
        stability_file
    )

    print(
        consensus_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        faithfulness_figure
    )

    print(
        stability_figure
    )

    print(
        flip_figure
    )

    print("=" * 94)


if __name__ == "__main__":
    main()