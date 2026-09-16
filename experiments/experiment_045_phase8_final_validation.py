from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


# ==========================================================
# HELPERS
# ==========================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path
    )


def as_bool(series):

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

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

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # LOAD PHASE-8 RESULTS
    # ======================================================

    fidelity_df = load_csv(
        tables_dir
        / "experiment_041_shap_fidelity.csv"
    )

    engineering_df = load_csv(
        tables_dir
        / "experiment_042_engineering_explanations.csv"
    )

    evidence_df = load_csv(
        tables_dir
        / "experiment_042_evidence_source_composition.csv"
    )

    faithfulness_df = load_csv(
        tables_dir
        / "experiment_043_faithfulness.csv"
    )

    stability_df = load_csv(
        tables_dir
        / "experiment_043_explanation_stability.csv"
    )

    uncertain_predictions = load_csv(
        tables_dir
        / "experiment_044_uncertain_predictions.csv"
    )

    uncertain_representatives = load_csv(
        tables_dir
        / "experiment_044_uncertain_representatives.csv"
    )

    uncertain_counts = load_csv(
        tables_dir
        / "experiment_044_uncertain_class_counts.csv"
    )

    # ======================================================
    # 1. SHAP ADDITIVITY / FIDELITY
    # ======================================================

    mean_fidelity_error = float(
        fidelity_df[
            "absolute_error"
        ].mean()
    )

    max_fidelity_error = float(
        fidelity_df[
            "absolute_error"
        ].max()
    )

    fidelity_pass = (
        max_fidelity_error
        <
        1e-5
    )

    # ======================================================
    # 2. ENGINEERING REPORT COVERAGE
    # ======================================================

    expected_classes = {

        "nominal",

        "solar_array_degradation",

        "battery_degradation",

        "thermal_anomaly",

        "reaction_wheel_degradation",

        "battery_voltage_sensor_drift",

        "telemetry_dropout",
    }

    engineering_classes = set(
        engineering_df[
            "class"
        ]
        .astype(str)
    )

    all_classes_explained = (
        engineering_classes
        ==
        expected_classes
    )

    representative_predictions_correct = bool(
        (
            engineering_df[
                "class"
            ]
            ==
            engineering_df[
                "predicted_label"
            ]
        ).all()
    )

    engineering_pass = (
        all_classes_explained
        and
        representative_predictions_correct
    )

    # ======================================================
    # 3. EVIDENCE COMPOSITION NORMALIZATION
    # ======================================================

    evidence_categories = [

        "rules",

        "physics",

        "ml",

        "telemetry",

        "missing_data",
    ]

    evidence_sums = (
        evidence_df[
            evidence_categories
        ]
        .sum(
            axis=1
        )
    )

    evidence_normalized = bool(
        np.allclose(
            evidence_sums,
            1.0,
            atol=1e-6,
        )
    )

    # ======================================================
    # 4. STABILITY PROFILES AVAILABLE
    #
    # We intentionally do NOT invent a post-hoc numerical
    # stability threshold here.
    # ======================================================

    stability_classes = set(
        stability_df[
            "class"
        ]
        .astype(str)
    )

    all_stability_profiles_present = (
        stability_classes
        ==
        expected_classes
    )

    mean_pairwise_stability = float(
        stability_df[
            "mean_pairwise_top5_jaccard"
        ].mean()
    )

    mean_consensus_overlap = float(
        stability_df[
            "mean_consensus_top5_overlap"
        ].mean()
    )

    minimum_consensus_overlap = float(
        stability_df[
            "mean_consensus_top5_overlap"
        ].min()
    )

    # ======================================================
    # 5. FAITHFULNESS — TOP SHAP vs RANDOM
    # ======================================================

    top_larger = as_bool(
        faithfulness_df[
            "top5_larger_drop"
        ]
    )

    top_flip = as_bool(
        faithfulness_df[
            "top5_prediction_flip"
        ]
    )

    random_flip = as_bool(
        faithfulness_df[
            "random5_prediction_flip"
        ]
    )

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
        top_larger.mean()
    )

    top_flip_rate = float(
        top_flip.mean()
    )

    random_flip_rate = float(
        random_flip.mean()
    )

    mean_drop_pass = (
        mean_top_drop
        >
        mean_random_drop
    )

    # Basic directional criterion:
    # SHAP-selected evidence should beat random evidence
    # on a majority of evaluated samples.
    top_wins_pass = (
        top_wins_fraction
        >
        0.50
    )

    flip_pass = (
        top_flip_rate
        >
        random_flip_rate
    )

    # ======================================================
    # 6. CONFIDENCE-GATE CONSISTENCY
    # ======================================================

    confidence_threshold = 0.70

    accepted = as_bool(
        uncertain_predictions[
            "accepted"
        ]
    )

    confidence_values = (
        uncertain_predictions[
            "confidence"
        ]
        .to_numpy(
            dtype=float
        )
    )

    accepted_gate_correct = bool(
        (
            confidence_values[
                accepted.to_numpy()
            ]
            >=
            confidence_threshold
        ).all()
    )

    uncertain_gate_correct = bool(
        (
            confidence_values[
                ~accepted.to_numpy()
            ]
            <
            confidence_threshold
        ).all()
    )

    confidence_gate_pass = (
        accepted_gate_correct
        and
        uncertain_gate_correct
    )

    # ======================================================
    # 7. UNCERTAIN-CLASS REPORT COVERAGE
    # ======================================================

    nonzero_uncertain_classes = set(
        uncertain_counts.loc[
            uncertain_counts[
                "uncertain_samples"
            ]
            >
            0,
            "true_class",
        ]
        .astype(str)
    )

    represented_uncertain_classes = set(
        uncertain_representatives[
            "evaluation_true_class"
        ]
        .astype(str)
    )

    uncertain_reports_pass = (
        nonzero_uncertain_classes
        ==
        represented_uncertain_classes
    )

    uncertain_report_coverage = (

        len(
            represented_uncertain_classes
        )
        /
        len(
            nonzero_uncertain_classes
        )

        if len(
            nonzero_uncertain_classes
        ) > 0

        else 1.0
    )

    # ======================================================
    # 8. AMBIGUITY-MARGIN SEPARATION
    # ======================================================

    accepted_rows = (
        uncertain_predictions[
            accepted
        ]
    )

    uncertain_rows = (
        uncertain_predictions[
            ~accepted
        ]
    )

    mean_accepted_margin = float(
        accepted_rows[
            "probability_margin"
        ].mean()
    )

    mean_uncertain_margin = float(
        uncertain_rows[
            "probability_margin"
        ].mean()
    )

    margin_separation_pass = (
        mean_accepted_margin
        >
        mean_uncertain_margin
    )

    # ======================================================
    # FINAL VALIDATION CHECKS
    # ======================================================

    checks = [

        {
            "check":
                "SHAP reconstruction fidelity",

            "value":
                max_fidelity_error,

            "criterion":
                "< 1e-5",

            "pass":
                fidelity_pass,
        },

        {
            "check":
                "All seven engineering explanations valid",

            "value":
                int(
                    engineering_pass
                ),

            "criterion":
                "== 1",

            "pass":
                engineering_pass,
        },

        {
            "check":
                "Evidence-source fractions normalized",

            "value":
                float(
                    evidence_sums.min()
                ),

            "criterion":
                "all rows sum to 1",

            "pass":
                evidence_normalized,
        },

        {
            "check":
                "All seven classes have stability profiles",

            "value":
                len(
                    stability_classes
                ),

            "criterion":
                "== 7",

            "pass":
                all_stability_profiles_present,
        },

        {
            "check":
                "SHAP top-5 ablation exceeds random ablation",

            "value":
                mean_top_drop
                -
                mean_random_drop,

            "criterion":
                "> 0",

            "pass":
                mean_drop_pass,
        },

        {
            "check":
                "SHAP evidence beats random on majority",

            "value":
                top_wins_fraction,

            "criterion":
                "> 0.50",

            "pass":
                top_wins_pass,
        },

        {
            "check":
                "SHAP ablation flips more predictions than random",

            "value":
                top_flip_rate
                -
                random_flip_rate,

            "criterion":
                "> 0",

            "pass":
                flip_pass,
        },

        {
            "check":
                "Confidence gate internally consistent",

            "value":
                int(
                    confidence_gate_pass
                ),

            "criterion":
                "== 1",

            "pass":
                confidence_gate_pass,
        },

        {
            "check":
                "Every uncertain class has contrastive XAI",

            "value":
                uncertain_report_coverage,

            "criterion":
                "== 1.0",

            "pass":
                uncertain_reports_pass,
        },

        {
            "check":
                "Accepted diagnoses have larger ambiguity margin",

            "value":
                mean_accepted_margin
                -
                mean_uncertain_margin,

            "criterion":
                "> 0",

            "pass":
                margin_separation_pass,
        },
    ]

    checks_df = pd.DataFrame(
        checks
    )

    global_validation = bool(
        checks_df[
            "pass"
        ].all()
    )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "Mean SHAP reconstruction error",

                "value":
                    mean_fidelity_error,
            },

            {
                "metric":
                    "Maximum SHAP reconstruction error",

                "value":
                    max_fidelity_error,
            },

            {
                "metric":
                    "Mean top-5 SHAP probability drop",

                "value":
                    mean_top_drop,
            },

            {
                "metric":
                    "Mean random-5 probability drop",

                "value":
                    mean_random_drop,
            },

            {
                "metric":
                    "Top-5 SHAP win fraction",

                "value":
                    top_wins_fraction,
            },

            {
                "metric":
                    "Top-5 SHAP prediction flip rate",

                "value":
                    top_flip_rate,
            },

            {
                "metric":
                    "Random-5 prediction flip rate",

                "value":
                    random_flip_rate,
            },

            {
                "metric":
                    "Mean pairwise explanation stability",

                "value":
                    mean_pairwise_stability,
            },

            {
                "metric":
                    "Mean consensus explanation overlap",

                "value":
                    mean_consensus_overlap,
            },

            {
                "metric":
                    "Minimum class consensus overlap",

                "value":
                    minimum_consensus_overlap,
            },

            {
                "metric":
                    "Accepted mean probability margin",

                "value":
                    mean_accepted_margin,
            },

            {
                "metric":
                    "Uncertain mean probability margin",

                "value":
                    mean_uncertain_margin,
            },
        ]
    )

    # ======================================================
    # SAVE TABLES
    # ======================================================

    validation_file = (
        tables_dir
        / "experiment_045_phase8_validation.csv"
    )

    summary_file = (
        tables_dir
        / "experiment_045_phase8_final_summary.csv"
    )

    checks_df.to_csv(
        validation_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ======================================================
    # FINAL FIGURE
    # ======================================================

    figure_file = (
        figures_dir
        / "experiment_045_phase8_final_summary.png"
    )

    metric_names = [

        "SHAP beats\nrandom",

        "SHAP ablation\nflip rate",

        "Mean consensus\nstability",

        "Engineering\ncoverage",

        "Uncertain XAI\ncoverage",
    ]

    metric_values = [

        top_wins_fraction,

        top_flip_rate,

        mean_consensus_overlap,

        len(
            engineering_classes
        )
        /
        7.0,

        uncertain_report_coverage,
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        metric_names,
        metric_values,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score / Fraction"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Final Phase-8 Explainability Summary"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 96)

    print(
        "SENTINEL-XAI - EXPERIMENT 045"
    )

    print(
        "PHASE 8 FINAL EXPLAINABLE-AI VALIDATION"
    )

    print("=" * 96)

    print()

    print(
        "SHAP fidelity:"
    )

    print(
        f"  Mean reconstruction error: "
        f"{mean_fidelity_error:.10f}"
    )

    print(
        f"  Maximum reconstruction error: "
        f"{max_fidelity_error:.10f}"
    )

    print()

    print(
        "Explanation faithfulness:"
    )

    print(
        f"  Mean top-5 SHAP probability drop: "
        f"{mean_top_drop:.4f}"
    )

    print(
        f"  Mean random-5 probability drop: "
        f"{mean_random_drop:.4f}"
    )

    print(
        f"  SHAP top-5 wins fraction: "
        f"{top_wins_fraction:.4f}"
    )

    print(
        f"  SHAP top-5 prediction flip rate: "
        f"{top_flip_rate:.4f}"
    )

    print(
        f"  Random-5 prediction flip rate: "
        f"{random_flip_rate:.4f}"
    )

    print()

    print(
        "Explanation stability:"
    )

    print(
        f"  Mean pairwise top-5 Jaccard: "
        f"{mean_pairwise_stability:.4f}"
    )

    print(
        f"  Mean consensus top-5 overlap: "
        f"{mean_consensus_overlap:.4f}"
    )

    print(
        f"  Minimum class consensus overlap: "
        f"{minimum_consensus_overlap:.4f}"
    )

    print()

    print(
        "Uncertainty explainability:"
    )

    print(
        f"  Mean accepted probability margin: "
        f"{mean_accepted_margin:.4f}"
    )

    print(
        f"  Mean uncertain probability margin: "
        f"{mean_uncertain_margin:.4f}"
    )

    print(
        f"  Uncertain-class explanation coverage: "
        f"{uncertain_report_coverage:.4f}"
    )

    print()

    print("-" * 96)

    print(
        "FINAL PHASE-8 VALIDATION CHECKS"
    )

    print("-" * 96)

    for _, row in (
        checks_df.iterrows()
    ):

        print()

        print(
            row[
                "check"
            ]
        )

        print(
            f"  Value: "
            f"{row['value']}"
        )

        print(
            f"  Criterion: "
            f"{row['criterion']}"
        )

        print(
            f"  PASS: "
            f"{bool(row['pass'])}"
        )

    print()

    print("-" * 96)

    print(
        f"Passed checks: "
        f"{int(checks_df['pass'].sum())}"
        f"/"
        f"{len(checks_df)}"
    )

    print(
        f"GLOBAL PHASE 8 VALIDATION: "
        f"{global_validation}"
    )

    print("-" * 96)

    print()

    print(
        "Important scientific note:"
    )

    print(
        "Explanation stability is reported descriptively "
        "rather than judged against a post-hoc threshold."
    )

    print(
        "SHAP explains model behavior, not physical causality."
    )

    print()

    print(
        "Validation table saved to:"
    )

    print(
        validation_file
    )

    print()

    print(
        "Final summary saved to:"
    )

    print(
        summary_file
    )

    print()

    print(
        "Figure saved to:"
    )

    print(
        figure_file
    )

    print("=" * 96)


if __name__ == "__main__":
    main()