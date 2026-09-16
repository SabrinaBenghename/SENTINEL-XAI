from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(path)


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

    # ======================================================
    # LOAD PHASE-6 RESULTS
    # ======================================================

    nominal = load_csv(
        tables_dir
        / "experiment_033_hybrid_nominal_validation.csv"
    )

    comparison = load_csv(
        tables_dir
        / "experiment_034_method_comparison.csv"
    )

    per_fault = load_csv(
        tables_dir
        / "experiment_034_hybrid_per_fault.csv"
    )

    delays = load_csv(
        tables_dir
        / "experiment_034_hybrid_diagnosis_delays.csv"
    )

    sources = load_csv(
        tables_dir
        / "experiment_034_hybrid_source_contributions.csv"
    )

    # ======================================================
    # GET METHOD ROWS
    # ======================================================

    phase4 = comparison[
        comparison["method"]
        ==
        "Phase 4 Physics"
    ].iloc[0]

    phase6 = comparison[
        comparison["method"]
        ==
        "Phase 6 Hybrid"
    ].iloc[0]

    # ======================================================
    # NOMINAL FALSE ALARMS
    # ======================================================

    hybrid_nominal_alarms = int(
        nominal[
            "hybrid_detected"
        ].sum()
    )

    # ======================================================
    # ALL SIX FAULTS CORRECTLY DIAGNOSED?
    # ======================================================

    diagnosed_faults = int(
        delays[
            "detected"
        ].sum()
    )

    # ======================================================
    # MINIMUM PER-FAULT CLASSIFICATION RECALL
    # ======================================================

    minimum_fault_recall = float(
        per_fault[
            "hybrid_classification_recall"
        ].min()
    )

    # ======================================================
    # ML SUPPORT CONTRIBUTION
    # ======================================================

    ml_supported_alarm_samples = 0

    for _, row in sources.iterrows():

        source_name = str(
            row["source_combination"]
        )

        if "ml" in source_name:

            ml_supported_alarm_samples += int(
                row["samples"]
            )

    # ======================================================
    # VALIDATION CHECKS
    # ======================================================

    checks = [

        {
            "check":
                "Healthy hybrid false alarms",

            "value":
                hybrid_nominal_alarms,

            "criterion":
                "== 0",

            "pass":
                hybrid_nominal_alarms
                == 0,
        },

        {
            "check":
                "Hybrid precision",

            "value":
                phase6["precision"],

            "criterion":
                ">= 0.95",

            "pass":
                phase6["precision"]
                >= 0.95,
        },

        {
            "check":
                "Hybrid false-positive rate",

            "value":
                phase6[
                    "false_positive_rate"
                ],

            "criterion":
                "== 0",

            "pass":
                phase6[
                    "false_positive_rate"
                ]
                == 0,
        },

        {
            "check":
                "Recall improves over Phase 4",

            "value":
                phase6["recall"],

            "criterion":
                (
                    f"> "
                    f"{phase4['recall']:.4f}"
                ),

            "pass":
                phase6["recall"]
                >
                phase4["recall"],
        },

        {
            "check":
                "F1 improves over Phase 4",

            "value":
                phase6["f1"],

            "criterion":
                (
                    f"> "
                    f"{phase4['f1']:.4f}"
                ),

            "pass":
                phase6["f1"]
                >
                phase4["f1"],
        },

        {
            "check":
                "Classification improves over Phase 4",

            "value":
                phase6[
                    "classification_accuracy"
                ],

            "criterion":
                (
                    f"> "
                    f"{phase4['classification_accuracy']:.4f}"
                ),

            "pass":
                phase6[
                    "classification_accuracy"
                ]
                >
                phase4[
                    "classification_accuracy"
                ],
        },

        {
            "check":
                "Diagnosis delay no worse than Phase 4",

            "value":
                phase6[
                    "mean_diagnosis_delay_min"
                ],

            "criterion":
                (
                    f"<= "
                    f"{phase4['mean_diagnosis_delay_min']:.1f} min"
                ),

            "pass":
                phase6[
                    "mean_diagnosis_delay_min"
                ]
                <=
                phase4[
                    "mean_diagnosis_delay_min"
                ],
        },

        {
            "check":
                "All six fault types diagnosed",

            "value":
                diagnosed_faults,

            "criterion":
                "== 6",

            "pass":
                diagnosed_faults
                == 6,
        },

        {
            "check":
                "ML contributes supporting evidence",

            "value":
                ml_supported_alarm_samples,

            "criterion":
                "> 0 samples",

            "pass":
                ml_supported_alarm_samples
                > 0,
        },

        {
            "check":
                "Every fault has nonzero classification recall",

            "value":
                minimum_fault_recall,

            "criterion":
                "> 0",

            "pass":
                minimum_fault_recall
                > 0,
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
    # SAVE VALIDATION TABLE
    # ======================================================

    validation_file = (
        tables_dir
        / "experiment_035_phase6_validation.csv"
    )

    checks_df.to_csv(
        validation_file,
        index=False,
    )

    # ======================================================
    # FINAL PERFORMANCE TABLE
    # ======================================================

    final_summary = comparison[
        [
            "method",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "classification_accuracy",
            "mean_diagnosis_delay_min",
        ]
    ].copy()

    summary_file = (
        tables_dir
        / "experiment_035_phase6_final_comparison.csv"
    )

    final_summary.to_csv(
        summary_file,
        index=False,
    )

    # ======================================================
    # FINAL FIGURE
    # ======================================================

    figure_file = (
        figures_dir
        / "experiment_035_phase6_final_comparison.png"
    )

    methods = (
        final_summary[
            "method"
        ]
    )

    x = range(
        len(methods)
    )

    width = 0.25

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        [
            value - width
            for value in x
        ],
        final_summary[
            "precision"
        ],
        width=width,
        label="Precision",
    )

    plt.bar(
        x,
        final_summary[
            "recall"
        ],
        width=width,
        label="Recall",
    )

    plt.bar(
        [
            value + width
            for value in x
        ],
        final_summary[
            "f1"
        ],
        width=width,
        label="F1",
    )

    plt.xticks(
        x,
        methods,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "SENTINEL-XAI - Final Phase-6 Method Comparison"
    )

    plt.grid(True)
    plt.legend()
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
    print("=" * 86)

    print(
        "SENTINEL-XAI - EXPERIMENT 035"
    )

    print(
        "PHASE 6 FINAL HYBRID VALIDATION"
    )

    print("=" * 86)

    print()

    print(
        "Final Phase-6 performance:"
    )

    print()

    print(
        f"Precision: "
        f"{phase6['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{phase6['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{phase6['f1']:.4f}"
    )

    print(
        f"Classification accuracy: "
        f"{phase6['classification_accuracy']:.4f}"
    )

    print(
        f"False-positive rate: "
        f"{phase6['false_positive_rate']:.4f}"
    )

    print(
        f"Mean diagnosis delay: "
        f"{phase6['mean_diagnosis_delay_min']:.1f} min"
    )

    print()

    print(
        f"ML-supported hybrid alarm samples: "
        f"{ml_supported_alarm_samples}"
    )

    print()

    print("-" * 86)

    print(
        "VALIDATION CHECKS"
    )

    print("-" * 86)

    for _, row in checks_df.iterrows():

        print()

        print(
            row["check"]
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

    print("-" * 86)

    print(
        f"Passed checks: "
        f"{int(checks_df['pass'].sum())}"
        f"/"
        f"{len(checks_df)}"
    )

    print(
        f"GLOBAL PHASE 6 VALIDATION: "
        f"{global_validation}"
    )

    print("-" * 86)

    print()

    print(
        "Validation table saved to:"
    )

    print(
        validation_file
    )

    print()

    print(
        "Final comparison saved to:"
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

    print("=" * 86)


if __name__ == "__main__":
    main()