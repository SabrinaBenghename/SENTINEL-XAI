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


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    FAULT_CLASSES,
)

from src.diagnosis.random_forest_diagnoser import (
    RandomForestDiagnoser,
)


# ==========================================================
# BLOCKED PER-CLASS SPLIT
# ==========================================================

def blocked_split(
    df,
    train_fraction=0.70,
):

    train_parts = []
    test_parts = []

    class_order = [
        "nominal",
        *FAULT_CLASSES,
    ]

    for class_name in class_order:

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
            * train_fraction
        )

        train_part = (
            subset.iloc[
                :split_index
            ]
            .copy()
        )

        test_part = (
            subset.iloc[
                split_index:
            ]
            .copy()
        )

        train_parts.append(
            train_part
        )

        test_parts.append(
            test_part
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return (
        train_df,
        test_df,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # LOAD DIAGNOSIS DATASET
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

    # ======================================================
    # BLOCKED TRAIN / TEST SPLIT
    # ======================================================

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

    train_y = train_df[
        "target_label"
    ]

    test_x = test_df[
        DIAGNOSTIC_FEATURES
    ]

    test_y = test_df[
        "target_label"
    ]

    # ======================================================
    # TRAIN RANDOM FOREST
    # ======================================================

    diagnoser = (
        RandomForestDiagnoser(
            n_estimators=500,
            random_state=42,
        )
    )

    diagnoser.fit(
        train_x,
        train_y,
    )

    # ======================================================
    # TEST
    # ======================================================

    prediction = diagnoser.predict(
        test_x
    )

    predicted_y = (
        prediction.predicted_labels
    )

    confidence = (
        prediction.confidence
    )

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(
        test_y,
        predicted_y,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            test_y,
            predicted_y,
        )
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        test_y,
        predicted_y,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        test_y,
        predicted_y,
        average="weighted",
        zero_division=0,
    )

    labels = [
        "nominal",
        *FAULT_CLASSES,
    ]

    matrix = confusion_matrix(
        test_y,
        predicted_y,
        labels=labels,
    )

    # ======================================================
    # PER-CLASS METRICS
    # ======================================================

    report = classification_report(
        test_y,
        predicted_y,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []

    for class_name in labels:

        values = report[
            class_name
        ]

        per_class_rows.append(
            {
                "class":
                    class_name,

                "precision":
                    values[
                        "precision"
                    ],

                "recall":
                    values[
                        "recall"
                    ],

                "f1":
                    values[
                        "f1-score"
                    ],

                "support":
                    int(
                        values[
                            "support"
                        ]
                    ),
            }
        )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    # ======================================================
    # PREDICTION TABLE
    # ======================================================

    prediction_df = pd.DataFrame(
        {
            "source_dataset":
                test_df[
                    "source_dataset"
                ].values,

            "time_h":
                test_df[
                    "time_h"
                ].values,

            "true_label":
                test_y.values,

            "predicted_label":
                predicted_y,

            "confidence":
                confidence,

            "correct":
                (
                    test_y.values
                    ==
                    predicted_y
                ),
        }
    )

    # ======================================================
    # CLASS PROBABILITIES
    # ======================================================

    for class_index, class_name in enumerate(
        diagnoser.classes_
    ):

        prediction_df[
            f"prob_{class_name}"
        ] = (
            prediction.probabilities[
                :,
                class_index
            ]
        )

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    importance_df = pd.DataFrame(
        {
            "feature":
                DIAGNOSTIC_FEATURES,

            "importance":
                diagnoser.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df[
        "rank"
    ] = np.arange(
        1,
        len(importance_df) + 1,
    )

    # ======================================================
    # SPLIT SUMMARY
    # ======================================================

    split_rows = []

    for class_name in labels:

        split_rows.append(
            {
                "class":
                    class_name,

                "train_samples":
                    int(
                        (
                            train_y
                            ==
                            class_name
                        ).sum()
                    ),

                "test_samples":
                    int(
                        (
                            test_y
                            ==
                            class_name
                        ).sum()
                    ),
            }
        )

    split_df = pd.DataFrame(
        split_rows
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
    # SAVE TABLES
    # ======================================================

    predictions_file = (
        tables_dir
        / "experiment_037_diagnosis_predictions.csv"
    )

    per_class_file = (
        tables_dir
        / "experiment_037_per_class_metrics.csv"
    )

    confusion_file = (
        tables_dir
        / "experiment_037_confusion_matrix.csv"
    )

    importance_file = (
        tables_dir
        / "experiment_037_feature_importance.csv"
    )

    split_file = (
        tables_dir
        / "experiment_037_train_test_split.csv"
    )

    metrics_file = (
        tables_dir
        / "experiment_037_global_metrics.csv"
    )

    model_file = (
        models_dir
        / "experiment_037_random_forest_diagnoser.joblib"
    )

    prediction_df.to_csv(
        predictions_file,
        index=False,
    )

    per_class_df.to_csv(
        per_class_file,
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        confusion_file
    )

    importance_df.to_csv(
        importance_file,
        index=False,
    )

    split_df.to_csv(
        split_file,
        index=False,
    )

    global_metrics_df = pd.DataFrame(
        [
            {
                "accuracy":
                    accuracy,

                "balanced_accuracy":
                    balanced_accuracy,

                "macro_precision":
                    macro_precision,

                "macro_recall":
                    macro_recall,

                "macro_f1":
                    macro_f1,

                "weighted_precision":
                    weighted_precision,

                "weighted_recall":
                    weighted_recall,

                "weighted_f1":
                    weighted_f1,

                "mean_confidence":
                    float(
                        np.mean(
                            confidence
                        )
                    ),
            }
        ]
    )

    global_metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    joblib.dump(
        diagnoser,
        model_file,
    )

    # ======================================================
    # FIGURE 1 — CONFUSION MATRIX
    # ======================================================

    confusion_figure = (
        figures_dir
        / "experiment_037_diagnosis_confusion_matrix.png"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        aspect="auto",
    )

    plt.xticks(
        range(
            len(labels)
        ),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(
            len(labels)
        ),
        labels,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Random Forest Diagnosis Confusion Matrix"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            plt.text(
                j,
                i,
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        confusion_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — PER-CLASS RECALL
    # ======================================================

    recall_figure = (
        figures_dir
        / "experiment_037_per_class_recall.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        per_class_df[
            "class"
        ],
        per_class_df[
            "recall"
        ],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Diagnostic Class"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Random Forest Per-Class Diagnosis Recall"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        recall_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — TOP FEATURE IMPORTANCE
    # ======================================================

    importance_figure = (
        figures_dir
        / "experiment_037_top_feature_importance.png"
    )

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features[
            "feature"
        ],
        top_features[
            "importance"
        ],
    )

    plt.xlabel(
        "Random Forest Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Top Diagnostic Features"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        importance_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 86)

    print(
        "SENTINEL-XAI - EXPERIMENT 037"
    )

    print(
        "RANDOM FOREST FAULT DIAGNOSIS"
    )

    print("=" * 86)

    print()

    print(
        "Evaluation protocol:"
    )

    print(
        "  Blocked chronological split "
        "within every class"
    )

    print(
        "  First 70% -> training"
    )

    print(
        "  Final 30% -> testing"
    )

    print(
        "  No random sample mixing"
    )

    print()

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_df)}"
    )

    print(
        f"Input features: "
        f"{len(DIAGNOSTIC_FEATURES)}"
    )

    print(
        f"Classes: "
        f"{len(labels)}"
    )

    print()

    print("-" * 86)
    print(
        "GLOBAL DIAGNOSIS METRICS"
    )
    print("-" * 86)

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Mean confidence: "
        f"{np.mean(confidence):.4f}"
    )

    print()

    print("-" * 86)
    print(
        "PER-CLASS TEST RESULTS"
    )
    print("-" * 86)

    for _, row in (
        per_class_df.iterrows()
    ):

        print()

        print(
            row["class"]
        )

        print(
            f"  Precision: "
            f"{row['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{row['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{row['f1']:.4f}"
        )

        print(
            f"  Support:   "
            f"{int(row['support'])}"
        )

    print()

    print("-" * 86)
    print(
        "TOP 10 DIAGNOSTIC FEATURES"
    )
    print("-" * 86)

    print()

    for _, row in (
        importance_df
        .head(10)
        .iterrows()
    ):

        print(
            f"{int(row['rank']):>2}. "
            f"{row['feature']:<42} "
            f"{row['importance']:.6f}"
        )

    print()

    print("=" * 86)

    print(
        "Model saved to:"
    )

    print(
        model_file
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        predictions_file
    )

    print(
        per_class_file
    )

    print(
        confusion_file
    )

    print(
        importance_file
    )

    print(
        metrics_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        confusion_figure
    )

    print(
        recall_figure
    )

    print(
        importance_figure
    )

    print("=" * 86)


if __name__ == "__main__":
    main()