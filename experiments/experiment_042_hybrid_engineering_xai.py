from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.explainability.engineering_report import (
    evidence_category,
    explain_feature,
    humanize_fault,
)


def main():

    # ======================================================
    # LOAD EXPERIMENT-041 LOCAL SHAP RESULTS
    # ======================================================

    local_file = (
        project_root
        / "results"
        / "tables"
        / "experiment_041_shap_local_explanations.csv"
    )

    local_df = pd.read_csv(
        local_file
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

    # ======================================================
    # CLASSES
    # ======================================================

    classes = [
        "nominal",
        "solar_array_degradation",
        "battery_degradation",
        "thermal_anomaly",
        "reaction_wheel_degradation",
        "battery_voltage_sensor_drift",
        "telemetry_dropout",
    ]

    report_rows = []
    category_rows = []

    categories = [
        "rules",
        "physics",
        "ml",
        "telemetry",
        "missing_data",
    ]

    # ======================================================
    # BUILD ONE ENGINEERING REPORT PER CLASS
    # ======================================================

    for class_name in classes:

        class_rows = (
            local_df[
                local_df[
                    "true_label"
                ]
                ==
                class_name
            ]
            .copy()
        )

        if len(
            class_rows
        ) == 0:

            raise RuntimeError(
                f"No SHAP explanation found for "
                f"{class_name}"
            )

        class_rows = (
            class_rows
            .sort_values(
                "shap_value",
                ascending=False,
            )
        )

        supporting = (
            class_rows[
                class_rows[
                    "shap_value"
                ]
                >
                0
            ]
            .head(7)
            .copy()
        )

        confidence = float(
            class_rows[
                "confidence"
            ].iloc[0]
        )

        predicted_label = str(
            class_rows[
                "predicted_label"
            ].iloc[0]
        )

        time_h = float(
            class_rows[
                "time_h"
            ].iloc[0]
        )

        status = (
            "ACCEPTED"
            if confidence >= 0.70
            else "UNCERTAIN"
        )

        diagnosis_name = (
            humanize_fault(
                predicted_label
            )
        )

        # ==================================================
        # SEMANTIC EVIDENCE
        # ==================================================

        evidence_lines = []

        for rank, (_, row) in enumerate(
            supporting.iterrows(),
            start=1,
        ):

            feature = str(
                row[
                    "feature"
                ]
            )

            value = float(
                row[
                    "feature_value"
                ]
            )

            shap_value = float(
                row[
                    "shap_value"
                ]
            )

            explanation = (
                explain_feature(
                    feature=feature,
                    value=value,
                    shap_value=(
                        shap_value
                    ),
                    diagnosis=(
                        predicted_label
                    ),
                )
            )

            evidence_lines.append(
                f"{rank}. {explanation}"
            )

        # ==================================================
        # CATEGORY CONTRIBUTIONS
        # ==================================================

        positive_rows = (
            class_rows[
                class_rows[
                    "shap_value"
                ]
                >
                0
            ]
            .copy()
        )

        positive_rows[
            "category"
        ] = (
            positive_rows[
                "feature"
            ]
            .apply(
                evidence_category
            )
        )

        category_sum = (
            positive_rows
            .groupby(
                "category"
            )[
                "shap_value"
            ]
            .sum()
        )

        total_positive = float(
            category_sum.sum()
        )

        category_record = {
            "class":
                class_name,
        }

        for category in categories:

            value = float(
                category_sum.get(
                    category,
                    0.0,
                )
            )

            if total_positive > 0:

                fraction = (
                    value
                    /
                    total_positive
                )

            else:

                fraction = 0.0

            category_record[
                category
            ] = fraction

        category_rows.append(
            category_record
        )

        # ==================================================
        # HUMAN-READABLE REPORT
        # ==================================================

        report_lines = [

            "=" * 78,

            "SENTINEL-XAI ENGINEERING EXPLANATION",

            "=" * 78,

            "",

            f"Diagnosis: {diagnosis_name}",

            f"Machine label: {predicted_label}",

            f"Confidence: {confidence:.4f}",

            f"Decision status: {status}",

            f"Representative mission time: "
            f"{time_h:.3f} h",

            "",

            "Primary model evidence:",

            "",
        ]

        report_lines.extend(
            evidence_lines
        )

        report_lines.extend(
            [
                "",
                "Evidence-source composition:",
                "",
            ]
        )

        for category in categories:

            fraction = (
                category_record[
                    category
                ]
            )

            report_lines.append(
                f"  {category:<14}: "
                f"{fraction:.1%}"
            )

        report_lines.extend(
            [
                "",
                "Interpretation note:",
                (
                    "SHAP describes how the learned "
                    "diagnostic model used the available "
                    "evidence. It does not establish "
                    "physical causality."
                ),
                "",
                (
                    "Binary rule/physics features are "
                    "interpreted using their actual 0/1 "
                    "value, so absence of a competing "
                    "diagnosis is not incorrectly reported "
                    "as an active detector alarm."
                ),
                "=" * 78,
            ]
        )

        report_text = "\n".join(
            report_lines
        )

        report_path = (
            reports_dir
            /
            (
                "experiment_042_"
                f"{class_name}_explanation.txt"
            )
        )

        report_path.write_text(
            report_text,
            encoding="utf-8",
        )

        report_record = {

            "class":
                class_name,

            "predicted_label":
                predicted_label,

            "confidence":
                confidence,

            "status":
                status,

            "time_h":
                time_h,

            "report_file":
                str(
                    report_path
                ),
        }

        for rank in range(
            1,
            6,
        ):

            if len(
                supporting
            ) >= rank:

                row = (
                    supporting
                    .iloc[
                        rank - 1
                    ]
                )

                report_record[
                    f"feature_{rank}"
                ] = row[
                    "feature"
                ]

                report_record[
                    f"shap_{rank}"
                ] = row[
                    "shap_value"
                ]

            else:

                report_record[
                    f"feature_{rank}"
                ] = ""

                report_record[
                    f"shap_{rank}"
                ] = 0.0

        report_rows.append(
            report_record
        )

    # ======================================================
    # TABLES
    # ======================================================

    reports_df = pd.DataFrame(
        report_rows
    )

    category_df = pd.DataFrame(
        category_rows
    )

    reports_file = (
        tables_dir
        / "experiment_042_engineering_explanations.csv"
    )

    categories_file = (
        tables_dir
        / "experiment_042_evidence_source_composition.csv"
    )

    reports_df.to_csv(
        reports_file,
        index=False,
    )

    category_df.to_csv(
        categories_file,
        index=False,
    )

    # ======================================================
    # FIGURE — EVIDENCE SOURCE COMPOSITION
    # ======================================================

    figure_file = (
        figures_dir
        / "experiment_042_evidence_source_composition.png"
    )

    matrix = (
        category_df[
            categories
        ]
        .to_numpy()
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    plt.xticks(
        range(
            len(categories)
        ),
        categories,
    )

    plt.yticks(
        range(
            len(classes)
        ),
        classes,
    )

    plt.xlabel(
        "Explanation Evidence Source"
    )

    plt.ylabel(
        "Diagnosis"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Hybrid XAI Evidence Composition"
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
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # VALIDATION
    # ======================================================

    all_classes_present = (
        len(
            reports_df
        )
        ==
        7
    )

    all_predictions_correct = bool(
        (
            reports_df[
                "class"
            ]
            ==
            reports_df[
                "predicted_label"
            ]
        ).all()
    )

    all_accepted = bool(
        (
            reports_df[
                "status"
            ]
            ==
            "ACCEPTED"
        ).all()
    )

    all_have_five_features = bool(
        (
            reports_df[
                "feature_5"
            ]
            .astype(str)
            .str.len()
            >
            0
        ).all()
    )

    category_sums = (
        category_df[
            categories
        ]
        .sum(
            axis=1
        )
    )

    category_normalization_ok = bool(
        np.allclose(
            category_sums,
            1.0,
            atol=1e-6,
        )
    )

    checks = {

        "All seven classes explained":
            all_classes_present,

        "Representative predictions correct":
            all_predictions_correct,

        "Representative diagnoses accepted":
            all_accepted,

        "At least five supporting features per report":
            all_have_five_features,

        "Evidence-source fractions normalized":
            category_normalization_ok,
    }

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 92)

    print(
        "SENTINEL-XAI - EXPERIMENT 042"
    )

    print(
        "HYBRID ENGINEERING XAI REPORTS"
    )

    print("=" * 92)

    for _, report in (
        reports_df.iterrows()
    ):

        print()

        print("-" * 92)

        print(
            humanize_fault(
                report[
                    "class"
                ]
            )
        )

        print("-" * 92)

        print(
            f"Prediction: "
            f"{report['predicted_label']}"
        )

        print(
            f"Confidence: "
            f"{report['confidence']:.4f}"
        )

        print(
            f"Status: "
            f"{report['status']}"
        )

        print()

        print(
            "Top SHAP evidence:"
        )

        for rank in range(
            1,
            6,
        ):

            print(
                f"  {rank}. "
                f"{report[f'feature_{rank}']}"
                f" "
                f"({report[f'shap_{rank}']:+.4f})"
            )

        print()

        print(
            f"Engineering report:"
        )

        print(
            report[
                "report_file"
            ]
        )

    print()

    print("=" * 92)

    print(
        "VALIDATION CHECKS"
    )

    print("=" * 92)

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
            "RESULT: HYBRID ENGINEERING "
            "XAI PASSED"
        )

    else:

        print(
            "RESULT: ENGINEERING XAI "
            "REQUIRES REVIEW"
        )

    print()

    print(
        "Tables saved to:"
    )

    print(
        reports_file
    )

    print(
        categories_file
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
        "Figure saved to:"
    )

    print(
        figure_file
    )

    print("=" * 92)


if __name__ == "__main__":
    main()