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


from src.diagnosis.diagnostic_features import (
    DIAGNOSTIC_FEATURES,
    FAULT_CLASSES,
)

from src.explainability.shap_explainer import (
    SentinelShapExplainer,
)

from src.explainability.engineering_report import (
    explain_feature,
    humanize_fault,
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
# GET TOP POSITIVE SHAP FEATURES
# ==========================================================

def top_positive_evidence(
    shap_explainer,
    explanation,
    transformed_x,
    position,
    class_name,
    top_k=5,
):

    class_values = (
        shap_explainer.class_values(
            explanation,
            class_name,
        )
    )

    shap_vector = (
        class_values[
            position
        ]
    )

    positive_indices = np.where(
        shap_vector > 0
    )[0]

    positive_indices = (
        positive_indices[
            np.argsort(
                shap_vector[
                    positive_indices
                ]
            )[::-1]
        ]
    )

    positive_indices = (
        positive_indices[
            :top_k
        ]
    )

    rows = []

    for rank, feature_index in enumerate(
        positive_indices,
        start=1,
    ):

        feature = (
            DIAGNOSTIC_FEATURES[
                feature_index
            ]
        )

        feature_value = float(
            transformed_x
            .iloc[
                position
            ][
                feature
            ]
        )

        shap_value = float(
            shap_vector[
                feature_index
            ]
        )

        engineering_text = (
            explain_feature(
                feature=feature,
                value=feature_value,
                shap_value=shap_value,
                diagnosis=class_name,
            )
        )

        rows.append(
            {
                "rank":
                    rank,

                "feature":
                    feature,

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "engineering_explanation":
                    engineering_text,
            }
        )

    return pd.DataFrame(
        rows
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    confidence_threshold = 0.70

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

    (
        train_df,
        test_df,
    ) = blocked_split(
        diagnosis_df,
        train_fraction=0.70,
    )

    test_x = (
        test_df[
            DIAGNOSTIC_FEATURES
        ]
        .copy()
    )

    test_y = (
        test_df[
            "target_label"
        ]
        .to_numpy()
    )

    # ======================================================
    # LOAD FROZEN PHASE-7 MODEL
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

    probabilities = (
        prediction.probabilities
    )

    confidence = (
        prediction.confidence
    )

    raw_predictions = (
        prediction.predicted_labels
    )

    classes = list(
        diagnoser.classes_
    )

    accepted = (
        confidence
        >=
        confidence_threshold
    )

    uncertain = (
        ~accepted
    )

    # ======================================================
    # TOP-1 / TOP-2 CANDIDATES
    # ======================================================

    sorted_indices = np.argsort(
        probabilities,
        axis=1,
    )[
        :,
        ::-1
    ]

    top1_indices = (
        sorted_indices[
            :,
            0
        ]
    )

    top2_indices = (
        sorted_indices[
            :,
            1
        ]
    )

    top1_labels = np.array(
        [
            classes[index]
            for index
            in top1_indices
        ]
    )

    top2_labels = np.array(
        [
            classes[index]
            for index
            in top2_indices
        ]
    )

    top1_probability = (
        probabilities[
            np.arange(
                len(
                    probabilities
                )
            ),
            top1_indices,
        ]
    )

    top2_probability = (
        probabilities[
            np.arange(
                len(
                    probabilities
                )
            ),
            top2_indices,
        ]
    )

    probability_margin = (
        top1_probability
        -
        top2_probability
    )

    # ======================================================
    # SUMMARY TABLE
    # ======================================================

    summary_df = pd.DataFrame(
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
                test_y,

            "raw_prediction":
                raw_predictions,

            "confidence":
                confidence,

            "accepted":
                accepted,

            "final_diagnosis":
                np.where(
                    accepted,
                    raw_predictions,
                    "uncertain",
                ),

            "candidate_1":
                top1_labels,

            "candidate_1_probability":
                top1_probability,

            "candidate_2":
                top2_labels,

            "candidate_2_probability":
                top2_probability,

            "probability_margin":
                probability_margin,
        }
    )

    # ======================================================
    # SHAP EXPLANATIONS
    # ======================================================

    print()
    print(
        "Computing SHAP explanations "
        "for uncertain diagnoses..."
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
    # REPRESENTATIVE UNCERTAIN SAMPLE PER TRUE CLASS
    #
    # True label is used ONLY for offline evaluation/sample
    # selection. It is NOT included in the operational
    # explanation presented to the hypothetical engineer.
    # ======================================================

    uncertain_positions = np.where(
        uncertain
    )[0]

    uncertain_true_classes = []

    for class_name in [
        "nominal",
        *FAULT_CLASSES,
    ]:

        positions = np.where(
            uncertain
            &
            (
                test_y
                ==
                class_name
            )
        )[0]

        if len(
            positions
        ) > 0:

            uncertain_true_classes.append(
                class_name
            )

    reports_dir = (
        project_root
        / "results"
        / "reports"
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

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    representative_rows = []
    evidence_tables = []

    for true_class in (
        uncertain_true_classes
    ):

        positions = np.where(
            uncertain
            &
            (
                test_y
                ==
                true_class
            )
        )[0]

        # --------------------------------------------------
        # Pick the most ambiguous example:
        # smallest top-1 / top-2 probability margin.
        # --------------------------------------------------

        representative_position = (
            positions[
                np.argmin(
                    probability_margin[
                        positions
                    ]
                )
            ]
        )

        candidate_1 = (
            top1_labels[
                representative_position
            ]
        )

        candidate_2 = (
            top2_labels[
                representative_position
            ]
        )

        candidate_1_probability = float(
            top1_probability[
                representative_position
            ]
        )

        candidate_2_probability = float(
            top2_probability[
                representative_position
            ]
        )

        margin = float(
            probability_margin[
                representative_position
            ]
        )

        # ==================================================
        # SHAP EVIDENCE FOR BOTH CANDIDATES
        # ==================================================

        candidate_1_evidence = (
            top_positive_evidence(
                shap_explainer=(
                    shap_explainer
                ),
                explanation=(
                    explanation
                ),
                transformed_x=(
                    transformed_test
                ),
                position=(
                    representative_position
                ),
                class_name=(
                    candidate_1
                ),
                top_k=5,
            )
        )

        candidate_2_evidence = (
            top_positive_evidence(
                shap_explainer=(
                    shap_explainer
                ),
                explanation=(
                    explanation
                ),
                transformed_x=(
                    transformed_test
                ),
                position=(
                    representative_position
                ),
                class_name=(
                    candidate_2
                ),
                top_k=5,
            )
        )

        candidate_1_evidence.insert(
            0,
            "candidate",
            candidate_1,
        )

        candidate_1_evidence.insert(
            0,
            "evaluation_true_class",
            true_class,
        )

        candidate_2_evidence.insert(
            0,
            "candidate",
            candidate_2,
        )

        candidate_2_evidence.insert(
            0,
            "evaluation_true_class",
            true_class,
        )

        evidence_tables.extend(
            [
                candidate_1_evidence,
                candidate_2_evidence,
            ]
        )

        # ==================================================
        # HUMAN-READABLE REPORT
        # ==================================================

        report_lines = [

            "=" * 82,

            "SENTINEL-XAI UNCERTAIN DIAGNOSIS EXPLANATION",

            "=" * 82,

            "",

            "FINAL DECISION: UNCERTAIN",

            "",

            (
                f"Confidence threshold: "
                f"{confidence_threshold:.2f}"
            ),

            "",

            (
                f"Candidate 1: "
                f"{humanize_fault(candidate_1)}"
            ),

            (
                f"Probability: "
                f"{candidate_1_probability:.1%}"
            ),

            "",

            (
                f"Candidate 2: "
                f"{humanize_fault(candidate_2)}"
            ),

            (
                f"Probability: "
                f"{candidate_2_probability:.1%}"
            ),

            "",

            (
                f"Probability margin: "
                f"{margin:.1%}"
            ),

            "",

            "Why SENTINEL did not commit:",

            (
                "The highest class probability did not "
                "reach the required 0.70 confidence "
                "threshold."
            ),

            (
                "The system therefore preserves the "
                "competing diagnoses rather than forcing "
                "a potentially unreliable class label."
            ),

            "",

            "-" * 82,

            (
                f"EVIDENCE SUPPORTING CANDIDATE 1 — "
                f"{humanize_fault(candidate_1)}"
            ),

            "-" * 82,

            "",
        ]

        for _, row in (
            candidate_1_evidence.iterrows()
        ):

            report_lines.append(
                (
                    f"{int(row['rank'])}. "
                    f"{row['engineering_explanation']}"
                )
            )

        report_lines.extend(
            [
                "",
                "-" * 82,

                (
                    f"EVIDENCE SUPPORTING CANDIDATE 2 — "
                    f"{humanize_fault(candidate_2)}"
                ),

                "-" * 82,

                "",
            ]
        )

        for _, row in (
            candidate_2_evidence.iterrows()
        ):

            report_lines.append(
                (
                    f"{int(row['rank'])}. "
                    f"{row['engineering_explanation']}"
                )
            )

        report_lines.extend(
            [
                "",

                "Interpretation:",

                (
                    "This is a contrastive explanation: "
                    "it shows why multiple diagnostic "
                    "hypotheses remain plausible."
                ),

                (
                    "SHAP explains the learned model's "
                    "decision structure and does not prove "
                    "physical causality."
                ),

                "=" * 82,
            ]
        )

        report_text = "\n".join(
            report_lines
        )

        report_path = (
            reports_dir
            /
            (
                "experiment_044_uncertain_"
                f"{true_class}.txt"
            )
        )

        report_path.write_text(
            report_text,
            encoding="utf-8",
        )

        representative_rows.append(
            {
                "evaluation_true_class":
                    true_class,

                "test_position":
                    representative_position,

                "time_h":
                    float(
                        test_df
                        .iloc[
                            representative_position
                        ][
                            "time_h"
                        ]
                    ),

                "candidate_1":
                    candidate_1,

                "candidate_1_probability":
                    candidate_1_probability,

                "candidate_2":
                    candidate_2,

                "candidate_2_probability":
                    candidate_2_probability,

                "probability_margin":
                    margin,

                "report_file":
                    str(
                        report_path
                    ),
            }
        )

    representative_df = pd.DataFrame(
        representative_rows
    )

    if len(
        evidence_tables
    ) > 0:

        evidence_df = pd.concat(
            evidence_tables,
            ignore_index=True,
        )

    else:

        evidence_df = pd.DataFrame()

    # ======================================================
    # ACCEPTED vs UNCERTAIN MARGIN ANALYSIS
    # ======================================================

    mean_accepted_margin = float(
        probability_margin[
            accepted
        ].mean()
    )

    mean_uncertain_margin = float(
        probability_margin[
            uncertain
        ].mean()
    )

    median_accepted_margin = float(
        np.median(
            probability_margin[
                accepted
            ]
        )
    )

    median_uncertain_margin = float(
        np.median(
            probability_margin[
                uncertain
            ]
        )
    )

    # ======================================================
    # UNCERTAIN CLASS COUNTS
    # ======================================================

    class_order = [
        "nominal",
        *FAULT_CLASSES,
    ]

    uncertain_counts = (

        summary_df[
            summary_df[
                "final_diagnosis"
            ]
            ==
            "uncertain"
        ][
            "true_label"
        ]

        .value_counts()

        .reindex(
            class_order,
            fill_value=0,
        )
    )

    uncertain_count_df = pd.DataFrame(
        {
            "true_class":
                uncertain_counts.index,

            "uncertain_samples":
                uncertain_counts.values,
        }
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    summary_file = (
        tables_dir
        / "experiment_044_uncertain_predictions.csv"
    )

    representative_file = (
        tables_dir
        / "experiment_044_uncertain_representatives.csv"
    )

    evidence_file = (
        tables_dir
        / "experiment_044_contrastive_shap_evidence.csv"
    )

    count_file = (
        tables_dir
        / "experiment_044_uncertain_class_counts.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    representative_df.to_csv(
        representative_file,
        index=False,
    )

    evidence_df.to_csv(
        evidence_file,
        index=False,
    )

    uncertain_count_df.to_csv(
        count_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — PROBABILITY MARGIN
    # ======================================================

    margin_figure = (
        figures_dir
        / "experiment_044_probability_margin.png"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            probability_margin[
                accepted
            ],
            probability_margin[
                uncertain
            ],
        ],
        labels=[
            "Accepted",
            "Uncertain",
        ],
    )

    plt.ylabel(
        "Top-1 minus Top-2 Probability"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Diagnostic Ambiguity Margin"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        margin_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — UNCERTAIN SAMPLE DISTRIBUTION
    # ======================================================

    count_figure = (
        figures_dir
        / "experiment_044_uncertain_class_distribution.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        uncertain_count_df[
            "true_class"
        ],
        uncertain_count_df[
            "uncertain_samples"
        ],
    )

    plt.xlabel(
        "Evaluation True Class"
    )

    plt.ylabel(
        "Uncertain Samples"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Distribution of Uncertain Diagnoses"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        count_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # VALIDATION CHECKS
    # ======================================================

    uncertain_count = int(
        uncertain.sum()
    )

    accepted_count = int(
        accepted.sum()
    )

    all_uncertain_below_threshold = bool(
        (
            confidence[
                uncertain
            ]
            <
            confidence_threshold
        ).all()
    )

    all_accepted_at_threshold = bool(
        (
            confidence[
                accepted
            ]
            >=
            confidence_threshold
        ).all()
    )

    candidates_distinct = bool(
        (
            top1_labels
            !=
            top2_labels
        ).all()
    )

    reports_cover_uncertain_classes = (
        len(
            representative_df
        )
        ==
        len(
            uncertain_true_classes
        )
    )

    evidence_for_both_candidates = bool(
        len(
            evidence_df
        )
        >=
        2
        *
        len(
            representative_df
        )
    )

    checks = {

        "Uncertain decisions below confidence threshold":
            all_uncertain_below_threshold,

        "Accepted decisions satisfy confidence threshold":
            all_accepted_at_threshold,

        "Top two diagnostic candidates are distinct":
            candidates_distinct,

        "Every uncertain true class has a report":
            reports_cover_uncertain_classes,

        "Contrastive evidence generated for both candidates":
            evidence_for_both_candidates,
    }

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 94)

    print(
        "SENTINEL-XAI - EXPERIMENT 044"
    )

    print(
        "UNCERTAIN DIAGNOSIS CONTRASTIVE XAI"
    )

    print("=" * 94)

    print()

    print(
        f"Test samples: "
        f"{len(test_x)}"
    )

    print(
        f"Accepted diagnoses: "
        f"{accepted_count}"
    )

    print(
        f"Uncertain diagnoses: "
        f"{uncertain_count}"
    )

    print(
        f"Confidence threshold: "
        f"{confidence_threshold:.2f}"
    )

    print()

    print("-" * 94)

    print(
        "DIAGNOSTIC MARGIN ANALYSIS"
    )

    print("-" * 94)

    print()

    print(
        f"Mean top-1/top-2 margin — accepted: "
        f"{mean_accepted_margin:.4f}"
    )

    print(
        f"Mean top-1/top-2 margin — uncertain: "
        f"{mean_uncertain_margin:.4f}"
    )

    print()

    print(
        f"Median margin — accepted: "
        f"{median_accepted_margin:.4f}"
    )

    print(
        f"Median margin — uncertain: "
        f"{median_uncertain_margin:.4f}"
    )

    print()

    print("-" * 94)

    print(
        "UNCERTAIN DIAGNOSES BY TRUE CLASS"
    )

    print("-" * 94)

    print()

    for _, row in (
        uncertain_count_df.iterrows()
    ):

        print(
            f"{row['true_class']:<34} "
            f"{int(row['uncertain_samples'])}"
        )

    print()

    print("-" * 94)

    print(
        "REPRESENTATIVE CONTRASTIVE EXPLANATIONS"
    )

    print("-" * 94)

    for _, row in (
        representative_df.iterrows()
    ):

        print()

        print(
            f"Evaluation class: "
            f"{row['evaluation_true_class']}"
        )

        print(
            f"  Candidate 1: "
            f"{row['candidate_1']} "
            f"({row['candidate_1_probability']:.4f})"
        )

        print(
            f"  Candidate 2: "
            f"{row['candidate_2']} "
            f"({row['candidate_2_probability']:.4f})"
        )

        print(
            f"  Margin: "
            f"{row['probability_margin']:.4f}"
        )

        print(
            f"  Report: "
            f"{row['report_file']}"
        )

    print()

    print("=" * 94)

    print(
        "VALIDATION CHECKS"
    )

    print("=" * 94)

    passed = 0

    for name, result in (
        checks.items()
    ):

        print()

        print(
            f"{name}: "
            f"{result}"
        )

        if result:

            passed += 1

    print()

    print(
        f"Passed checks: "
        f"{passed}/{len(checks)}"
    )

    print()

    if passed == len(
        checks
    ):

        print(
            "RESULT: UNCERTAIN-DIAGNOSIS "
            "XAI PASSED"
        )

    else:

        print(
            "RESULT: UNCERTAIN-DIAGNOSIS "
            "XAI REQUIRES REVIEW"
        )

    print()

    print(
        "Important:"
    )

    print(
        "True class is used only for offline evaluation "
        "and representative-sample selection."
    )

    print(
        "Operational uncertainty reports expose only "
        "model candidates, probabilities and evidence."
    )

    print()

    print(
        "Tables saved to:"
    )

    print(
        summary_file
    )

    print(
        representative_file
    )

    print(
        evidence_file
    )

    print(
        count_file
    )

    print()

    print(
        "Reports saved under:"
    )

    print(
        reports_dir
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        margin_figure
    )

    print(
        count_figure
    )

    print("=" * 94)


if __name__ == "__main__":
    main()