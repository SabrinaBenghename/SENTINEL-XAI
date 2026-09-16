from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================================
# PROJECT
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

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

reports_dir = (
    project_root
    / "results"
    / "reports"
)

tables_dir.mkdir(
    parents=True,
    exist_ok=True,
)

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

reports_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# FROZEN VALIDATED RESULTS
#
# These numbers come from the completed experiments.
# No optimization is performed here.
# ==========================================================

DETECTION_RESULTS = pd.DataFrame(
    [
        {
            "method": "Phase 3 Rule-Based",
            "precision": 1.0000,
            "recall": 0.5634,
            "f1": 0.7207,
            "classification_accuracy": 0.5634,
            "mean_delay_min": 33.7,
        },
        {
            "method": "Phase 4 Model-Based",
            "precision": 1.0000,
            "recall": 0.6488,
            "f1": 0.7870,
            "classification_accuracy": 0.6408,
            "mean_delay_min": 16.7,
        },
        {
            "method": "Phase 5 Isolation Forest",
            "precision": 0.9174,
            "recall": 0.1448,
            "f1": 0.2502,
            "classification_accuracy": np.nan,
            "mean_delay_min": 50.0,
        },
        {
            "method": "Phase 6 Hybrid",
            "precision": 1.0000,
            "recall": 0.7487,
            "f1": 0.8563,
            "classification_accuracy": 0.7451,
            "mean_delay_min": 15.5,
        },
    ]
)


DIAGNOSIS_RESULTS = pd.DataFrame(
    [
        {
            "evaluation": "Phase 7 Development",
            "raw_accuracy": 0.9332,
            "balanced_accuracy": 0.9447,
            "macro_f1": 0.9419,
            "confidence_coverage": 0.6870,
            "accepted_accuracy": 1.0000,
        },
        {
            "evaluation": "Phase 9 Independent Monte Carlo",
            "raw_accuracy": 0.8527,
            "balanced_accuracy": 0.8541,
            "macro_f1": 0.9003,
            "confidence_coverage": 0.7371,
            "accepted_accuracy": 0.9783,
        },
    ]
)


XAI_RESULTS = pd.DataFrame(
    [
        {
            "evaluation": "Phase 8 Development",
            "top5_probability_drop": 0.7026,
            "random5_probability_drop": 0.0145,
            "mean_pairwise_jaccard": 0.6779,
            "consensus_overlap": 0.7536,
        },
        {
            "evaluation": "Phase 9 Independent Monte Carlo",
            "top5_probability_drop": 0.6305,
            "random5_probability_drop": 0.0148,
            "mean_pairwise_jaccard": 0.6876,
            "consensus_overlap": 0.8400,
        },
    ]
)


MONTE_CARLO_RESULTS = {
    "scenarios": 120,
    "hybrid_precision": 0.9970,
    "hybrid_recall": 0.7100,
    "hybrid_f1": 0.8294,
    "hybrid_false_positive_rate": 0.0030,
    "diagnosis_accuracy": 0.8527,
    "diagnosis_macro_f1": 0.9003,
    "confidence_coverage": 0.7371,
    "accepted_accuracy": 0.9783,
    "error_capture_rate": 0.8913,
    "scenario_success_rate": 1.0000,
    "mean_correct_diagnosis_delay_min": 19.45,
    "median_correct_diagnosis_delay_min": 12.55,
    "xai_top5_probability_drop": 0.6305,
    "xai_random5_probability_drop": 0.0148,
    "xai_consensus_overlap": 0.8400,
    "phase8_to_phase9_explanation_overlap": 0.7333,
}


# ==========================================================
# MASTER RESEARCH TABLE
# ==========================================================

def build_master_table():

    rows = []

    # ------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 3,
                "component": "Rule-based detection",
                "metric": "Precision",
                "value": 1.0000,
                "experiment": "013/015",
            },
            {
                "phase": 3,
                "component": "Rule-based detection",
                "metric": "Recall",
                "value": 0.5634,
                "experiment": "013/015",
            },
            {
                "phase": 3,
                "component": "Rule-based detection",
                "metric": "F1",
                "value": 0.7207,
                "experiment": "013/015",
            },
            {
                "phase": 3,
                "component": "Rule-based detection",
                "metric": "Mean detection delay [min]",
                "value": 33.7,
                "experiment": "014/015",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 4
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 4,
                "component": "Model-based detection",
                "metric": "Recall",
                "value": 0.6488,
                "experiment": "023",
            },
            {
                "phase": 4,
                "component": "Model-based detection",
                "metric": "F1",
                "value": 0.7870,
                "experiment": "023",
            },
            {
                "phase": 4,
                "component": "Model-based detection",
                "metric": "Classification accuracy",
                "value": 0.6408,
                "experiment": "023",
            },
            {
                "phase": 4,
                "component": "Model-based detection",
                "metric": "Mean diagnosis delay [min]",
                "value": 16.7,
                "experiment": "023",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 5
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 5,
                "component": "Unsupervised subsystem ML",
                "metric": "Precision",
                "value": 0.9174,
                "experiment": "032",
            },
            {
                "phase": 5,
                "component": "Unsupervised subsystem ML",
                "metric": "Recall",
                "value": 0.1448,
                "experiment": "032",
            },
            {
                "phase": 5,
                "component": "Unsupervised subsystem ML",
                "metric": "F1",
                "value": 0.2502,
                "experiment": "032",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 6
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 6,
                "component": "Hybrid detection",
                "metric": "Precision",
                "value": 1.0000,
                "experiment": "035",
            },
            {
                "phase": 6,
                "component": "Hybrid detection",
                "metric": "Recall",
                "value": 0.7487,
                "experiment": "035",
            },
            {
                "phase": 6,
                "component": "Hybrid detection",
                "metric": "F1",
                "value": 0.8563,
                "experiment": "035",
            },
            {
                "phase": 6,
                "component": "Hybrid detection",
                "metric": "Classification accuracy",
                "value": 0.7451,
                "experiment": "035",
            },
            {
                "phase": 6,
                "component": "Hybrid detection",
                "metric": "Mean diagnosis delay [min]",
                "value": 15.5,
                "experiment": "035",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 7
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 7,
                "component": "Random Forest diagnosis",
                "metric": "Raw accuracy",
                "value": 0.9332,
                "experiment": "037/040",
            },
            {
                "phase": 7,
                "component": "Random Forest diagnosis",
                "metric": "Balanced accuracy",
                "value": 0.9447,
                "experiment": "037/040",
            },
            {
                "phase": 7,
                "component": "Random Forest diagnosis",
                "metric": "Macro F1",
                "value": 0.9419,
                "experiment": "037/040",
            },
            {
                "phase": 7,
                "component": "Confidence-aware diagnosis",
                "metric": "Selective coverage",
                "value": 0.6870,
                "experiment": "039/040",
            },
            {
                "phase": 7,
                "component": "Confidence-aware diagnosis",
                "metric": "Accepted accuracy",
                "value": 1.0000,
                "experiment": "039/040",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 8
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 8,
                "component": "Explainable AI",
                "metric": "SHAP top-5 probability drop",
                "value": 0.7026,
                "experiment": "043/045",
            },
            {
                "phase": 8,
                "component": "Explainable AI",
                "metric": "Random-5 probability drop",
                "value": 0.0145,
                "experiment": "043/045",
            },
            {
                "phase": 8,
                "component": "Explainable AI",
                "metric": "Consensus overlap",
                "value": 0.7536,
                "experiment": "043/045",
            },
        ]
    )

    # ------------------------------------------------------
    # Phase 9
    # ------------------------------------------------------

    rows.extend(
        [
            {
                "phase": 9,
                "component": "Independent Monte Carlo",
                "metric": "Scenarios",
                "value": 120,
                "experiment": "046-052",
            },
            {
                "phase": 9,
                "component": "Independent hybrid detection",
                "metric": "Precision",
                "value": 0.9970,
                "experiment": "048",
            },
            {
                "phase": 9,
                "component": "Independent hybrid detection",
                "metric": "Recall",
                "value": 0.7100,
                "experiment": "048",
            },
            {
                "phase": 9,
                "component": "Independent hybrid detection",
                "metric": "F1",
                "value": 0.8294,
                "experiment": "048",
            },
            {
                "phase": 9,
                "component": "Independent diagnosis",
                "metric": "Raw accuracy",
                "value": 0.8527,
                "experiment": "048",
            },
            {
                "phase": 9,
                "component": "Independent diagnosis",
                "metric": "Macro F1",
                "value": 0.9003,
                "experiment": "048",
            },
            {
                "phase": 9,
                "component": "Independent confidence gate",
                "metric": "Coverage",
                "value": 0.7371,
                "experiment": "050",
            },
            {
                "phase": 9,
                "component": "Independent confidence gate",
                "metric": "Accepted accuracy",
                "value": 0.9783,
                "experiment": "050",
            },
            {
                "phase": 9,
                "component": "Independent confidence gate",
                "metric": "Error-capture rate",
                "value": 0.8913,
                "experiment": "050",
            },
            {
                "phase": 9,
                "component": "Independent XAI",
                "metric": "SHAP top-5 probability drop",
                "value": 0.6305,
                "experiment": "051",
            },
            {
                "phase": 9,
                "component": "Independent XAI",
                "metric": "Random-5 probability drop",
                "value": 0.0148,
                "experiment": "051",
            },
            {
                "phase": 9,
                "component": "Independent XAI",
                "metric": "Consensus overlap",
                "value": 0.8400,
                "experiment": "051",
            },
            {
                "phase": 9,
                "component": "End-to-end system",
                "metric": "Scenario success rate",
                "value": 1.0000,
                "experiment": "048/052",
            },
        ]
    )

    return pd.DataFrame(rows)


# ==========================================================
# MAIN
# ==========================================================

def main():

    master = build_master_table()

    # ======================================================
    # SAVE MASTER TABLES
    # ======================================================

    master_file = (
        tables_dir
        / "experiment_053_master_research_results.csv"
    )

    detection_file = (
        tables_dir
        / "experiment_053_detection_comparison.csv"
    )

    diagnosis_file = (
        tables_dir
        / "experiment_053_diagnosis_generalization.csv"
    )

    xai_file = (
        tables_dir
        / "experiment_053_xai_generalization.csv"
    )

    master.to_csv(
        master_file,
        index=False,
    )

    DETECTION_RESULTS.to_csv(
        detection_file,
        index=False,
    )

    DIAGNOSIS_RESULTS.to_csv(
        diagnosis_file,
        index=False,
    )

    XAI_RESULTS.to_csv(
        xai_file,
        index=False,
    )

    # ======================================================
    # FIGURE 1 — DETECTION METHOD COMPARISON
    # ======================================================

    detection_figure = (
        figures_dir
        / "experiment_053_final_detection_comparison.png"
    )

    metrics = [
        "precision",
        "recall",
        "f1",
    ]

    x = np.arange(
        len(
            DETECTION_RESULTS
        )
    )

    width = 0.24

    plt.figure(
        figsize=(11, 6)
    )

    for index, metric in enumerate(
        metrics
    ):

        plt.bar(
            x
            +
            (
                index - 1
            )
            *
            width,
            DETECTION_RESULTS[
                metric
            ],
            width=width,
            label=metric.replace(
                "_",
                " ",
            ).title(),
        )

    plt.xticks(
        x,
        DETECTION_RESULTS[
            "method"
        ],
        rotation=20,
        ha="right",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Detection Architecture Comparison"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        detection_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — DEVELOPMENT vs MONTE CARLO DIAGNOSIS
    # ======================================================

    diagnosis_figure = (
        figures_dir
        / "experiment_053_diagnosis_generalization.png"
    )

    metrics = [
        "raw_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "confidence_coverage",
        "accepted_accuracy",
    ]

    labels = [
        "Raw Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Confidence Coverage",
        "Accepted Accuracy",
    ]

    x = np.arange(
        len(
            metrics
        )
    )

    width = 0.36

    plt.figure(
        figsize=(11, 6)
    )

    for evaluation_index, (_, row) in enumerate(
        DIAGNOSIS_RESULTS.iterrows()
    ):

        values = [
            row[
                metric
            ]

            for metric in metrics
        ]

        plt.bar(
            x
            +
            (
                evaluation_index
                -
                0.5
            )
            *
            width,
            values,
            width=width,
            label=row[
                "evaluation"
            ],
        )

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right",
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
        "Diagnosis Generalization"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        diagnosis_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 3 — XAI GENERALIZATION
    # ======================================================

    xai_figure = (
        figures_dir
        / "experiment_053_xai_generalization.png"
    )

    metrics = [
        "top5_probability_drop",
        "random5_probability_drop",
        "mean_pairwise_jaccard",
        "consensus_overlap",
    ]

    labels = [
        "SHAP Top-5\nProbability Drop",
        "Random-5\nProbability Drop",
        "Pairwise\nJaccard",
        "Consensus\nOverlap",
    ]

    x = np.arange(
        len(
            metrics
        )
    )

    width = 0.36

    plt.figure(
        figsize=(10, 6)
    )

    for evaluation_index, (_, row) in enumerate(
        XAI_RESULTS.iterrows()
    ):

        values = [
            row[
                metric
            ]

            for metric in metrics
        ]

        plt.bar(
            x
            +
            (
                evaluation_index
                -
                0.5
            )
            *
            width,
            values,
            width=width,
            label=row[
                "evaluation"
            ],
        )

    plt.xticks(
        x,
        labels,
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
        "Explainability Generalization"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        xai_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FINAL TEXT RESEARCH SUMMARY
    # ======================================================

    report_file = (
        reports_dir
        / "experiment_053_final_research_summary.txt"
    )

    report = f"""
SENTINEL-XAI
FINAL RESEARCH RESULTS SUMMARY
============================================================

Research problem
------------------------------------------------------------
Evaluate whether a hybrid combination of rule-based reasoning,
physics/model-based residual monitoring, machine-learning
evidence, supervised diagnosis and explainable AI can improve
spacecraft health monitoring while retaining interpretable
decisions.

MAIN DEVELOPMENT RESULTS
------------------------------------------------------------

Rule-based detector:
  Precision: 1.0000
  Recall:    0.5634
  F1:        0.7207
  Delay:     33.7 min

Model-based detector:
  Precision: 1.0000
  Recall:    0.6488
  F1:        0.7870
  Delay:     16.7 min

Unsupervised ML baseline:
  Precision: 0.9174
  Recall:    0.1448
  F1:        0.2502

Hybrid detector:
  Precision: 1.0000
  Recall:    0.7487
  F1:        0.8563
  Delay:     15.5 min

The hybrid architecture therefore improved detection recall and
F1 over the standalone rule and model-based baselines.

FAULT DIAGNOSIS
------------------------------------------------------------

Development diagnosis accuracy: 0.9332
Development balanced accuracy: 0.9447
Development Macro F1: 0.9419

The supervised diagnosis stage substantially improved fault-type
identification over the direct Phase-6 hybrid classification.

EXPLAINABILITY
------------------------------------------------------------

Development SHAP top-5 probability drop: 0.7026
Development random-5 probability drop: 0.0145

This supports explanation faithfulness: removing features identified
as important by SHAP damaged model confidence far more than removing
random features.

INDEPENDENT MONTE CARLO EVALUATION
------------------------------------------------------------

Scenarios: 120
Retraining: NO
Threshold retuning: NO

Hybrid precision: {MONTE_CARLO_RESULTS["hybrid_precision"]:.4f}
Hybrid recall: {MONTE_CARLO_RESULTS["hybrid_recall"]:.4f}
Hybrid F1: {MONTE_CARLO_RESULTS["hybrid_f1"]:.4f}

Diagnosis accuracy: {MONTE_CARLO_RESULTS["diagnosis_accuracy"]:.4f}
Diagnosis Macro F1: {MONTE_CARLO_RESULTS["diagnosis_macro_f1"]:.4f}

Confidence coverage: {MONTE_CARLO_RESULTS["confidence_coverage"]:.4f}
Accepted accuracy: {MONTE_CARLO_RESULTS["accepted_accuracy"]:.4f}
Error-capture rate: {MONTE_CARLO_RESULTS["error_capture_rate"]:.4f}

Scenario success rate: {MONTE_CARLO_RESULTS["scenario_success_rate"]:.4f}
Mean correct diagnosis delay:
  {MONTE_CARLO_RESULTS["mean_correct_diagnosis_delay_min"]:.2f} min

Monte Carlo SHAP top-5 probability drop:
  {MONTE_CARLO_RESULTS["xai_top5_probability_drop"]:.4f}

Monte Carlo random-5 probability drop:
  {MONTE_CARLO_RESULTS["xai_random5_probability_drop"]:.4f}

Monte Carlo explanation consensus:
  {MONTE_CARLO_RESULTS["xai_consensus_overlap"]:.4f}

Phase-8 to Phase-9 top-5 explanation overlap:
  {MONTE_CARLO_RESULTS["phase8_to_phase9_explanation_overlap"]:.4f}


KEY FINDING
------------------------------------------------------------

The central result is not that one machine-learning model dominates.

The strongest performance was obtained by combining complementary
forms of evidence:

  telemetry
      -> rules
      -> physics/model residuals
      -> unsupervised ML evidence
      -> hybrid detection
      -> supervised diagnosis
      -> confidence gating
      -> explainable diagnosis

The independent Monte Carlo experiments support that this architecture
retains useful detection, diagnosis and explanation behavior under
randomized fault severity, onset time, fault profile and sensor noise.


DOCUMENTED LIMITATION
------------------------------------------------------------

The frozen 0.70 confidence threshold did not fully reproduce the
original Phase-7 selective-diagnosis targets on unseen Monte Carlo
conditions.

Independent performance:

  Accepted accuracy: 0.9783
  Original target:   >= 0.9900

  Error-capture rate: 0.8913
  Original target:    >= 0.9500

  Coverage:           0.7371
  Original target:    >= 0.6000

The independent test set was not used to retune the threshold.

Battery degradation was also the most difficult diagnostic class
under varying operating conditions.


FINAL RESEARCH CONCLUSION
------------------------------------------------------------

SENTINEL-XAI demonstrates a prototype architecture for hybrid,
confidence-aware and explainable spacecraft fault detection and
diagnosis.

The experiments support the hypothesis that combining rules,
physics/model residuals and machine-learning evidence can provide
stronger fault-monitoring performance than the individual methods
used alone, while a supervised diagnosis layer and SHAP-based
explanation system provide interpretable fault identification.

The results apply to the simulated spacecraft and fault scenarios
used in this project and should not be interpreted as flight
qualification or universal spacecraft performance.
"""

    report_file.write_text(
        report.strip() + "\n",
        encoding="utf-8",
    )

    # ======================================================
    # TERMINAL
    # ======================================================

    print()
    print("=" * 96)

    print(
        "SENTINEL-XAI - EXPERIMENT 053"
    )

    print(
        "FINAL RESEARCH RESULTS CONSOLIDATION"
    )

    print("=" * 96)

    print()

    print(
        "Technical system status: FROZEN"
    )

    print(
        "Retraining performed: NO"
    )

    print(
        "Threshold tuning performed: NO"
    )

    print()

    print(
        "Core research result:"
    )

    print()

    print(
        "Phase 3 Rule F1:      0.7207"
    )

    print(
        "Phase 4 Physics F1:   0.7870"
    )

    print(
        "Phase 5 ML F1:        0.2502"
    )

    print(
        "Phase 6 Hybrid F1:    0.8563"
    )

    print()

    print(
        "Phase 7 diagnosis accuracy: 0.9332"
    )

    print(
        "Phase 9 independent accuracy: 0.8527"
    )

    print()

    print(
        "Phase 9 independent hybrid F1: 0.8294"
    )

    print(
        "Independent scenario success: 120/120"
    )

    print()

    print(
        "Independent accepted diagnosis accuracy: "
        "0.9783"
    )

    print()

    print(
        "Monte Carlo SHAP top-5 drop: 0.6305"
    )

    print(
        "Monte Carlo random-5 drop:   0.0148"
    )

    print()

    print("-" * 96)

    print(
        "MASTER TABLE CREATED"
    )

    print("-" * 96)

    print()

    print(
        f"Rows: {len(master)}"
    )

    print()

    print(
        "Tables saved to:"
    )

    print(master_file)
    print(detection_file)
    print(diagnosis_file)
    print(xai_file)

    print()

    print(
        "Figures saved to:"
    )

    print(detection_figure)
    print(diagnosis_figure)
    print(xai_figure)

    print()

    print(
        "Research summary saved to:"
    )

    print(report_file)

    print()

    print(
        "RESULT: PHASE-10 RESEARCH RESULTS "
        "SUCCESSFULLY CONSOLIDATED"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()