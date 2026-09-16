from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ==========================================================
# PROJECT PATHS
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

figures_dir = (
    project_root
    / "results"
    / "figures"
)

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# DRAWING HELPERS
# ==========================================================

def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle="",
    fontsize=11,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        fill=False,
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
    )

    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 2,
        )


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
    )

    ax.add_patch(arrow)


# ==========================================================
# FIGURE 1
# COMPLETE SENTINEL-XAI ARCHITECTURE
# ==========================================================

def create_architecture_figure():

    fig, ax = plt.subplots(
        figsize=(16, 9)
    )

    ax.set_xlim(
        0,
        16,
    )

    ax.set_ylim(
        0,
        10,
    )

    ax.axis(
        "off"
    )

    # ------------------------------------------------------
    # TITLE
    # ------------------------------------------------------

    ax.text(
        8,
        9.6,
        "SENTINEL-XAI — Hybrid Explainable Spacecraft Health Monitoring",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # ------------------------------------------------------
    # SPACECRAFT / TELEMETRY
    # ------------------------------------------------------

    add_box(
        ax,
        0.4,
        4.1,
        2.0,
        1.4,
        "Spacecraft",
        "Power • Thermal\nReaction Wheel • Sensors",
    )

    add_box(
        ax,
        3.0,
        4.1,
        2.0,
        1.4,
        "Telemetry",
        "Measured subsystem\nsignals",
    )

    add_arrow(
        ax,
        2.4,
        4.8,
        3.0,
        4.8,
    )

    # ------------------------------------------------------
    # THREE EVIDENCE PATHS
    # ------------------------------------------------------

    add_box(
        ax,
        5.8,
        7.0,
        2.4,
        1.25,
        "Rule-Based",
        "Engineering thresholds\nPhase 3",
    )

    add_box(
        ax,
        5.8,
        4.2,
        2.4,
        1.25,
        "Physics / Models",
        "EKF • residuals\nPhase 4",
    )

    add_box(
        ax,
        5.8,
        1.4,
        2.4,
        1.25,
        "Machine Learning",
        "Isolation Forest\nPhase 5",
    )

    add_arrow(
        ax,
        5.0,
        4.8,
        5.8,
        7.6,
    )

    add_arrow(
        ax,
        5.0,
        4.8,
        5.8,
        4.8,
    )

    add_arrow(
        ax,
        5.0,
        4.8,
        5.8,
        2.0,
    )

    # ------------------------------------------------------
    # HYBRID FUSION
    # ------------------------------------------------------

    add_box(
        ax,
        9.0,
        4.2,
        2.3,
        1.25,
        "Hybrid Fusion",
        "Rules + Physics + ML\nPhase 6",
    )

    add_arrow(
        ax,
        8.2,
        7.6,
        9.0,
        5.1,
    )

    add_arrow(
        ax,
        8.2,
        4.8,
        9.0,
        4.8,
    )

    add_arrow(
        ax,
        8.2,
        2.0,
        9.0,
        4.5,
    )

    # ------------------------------------------------------
    # SUPERVISED DIAGNOSIS
    # ------------------------------------------------------

    add_box(
        ax,
        12.0,
        6.8,
        2.6,
        1.4,
        "Fault Diagnosis",
        "Random Forest\n7 diagnostic classes",
    )

    add_arrow(
        ax,
        11.3,
        4.8,
        12.0,
        7.2,
    )

    # ------------------------------------------------------
    # CONFIDENCE LAYER
    # ------------------------------------------------------

    add_box(
        ax,
        12.0,
        4.0,
        2.6,
        1.4,
        "Confidence Gate",
        "Accept / uncertain\nFrozen threshold = 0.70",
    )

    add_arrow(
        ax,
        13.3,
        6.8,
        13.3,
        5.4,
    )

    # ------------------------------------------------------
    # XAI
    # ------------------------------------------------------

    add_box(
        ax,
        12.0,
        1.2,
        2.6,
        1.4,
        "Explainable AI",
        "SHAP evidence\nEngineering reports",
    )

    add_arrow(
        ax,
        13.3,
        4.0,
        13.3,
        2.6,
    )

    # ------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------

    add_box(
        ax,
        9.0,
        0.1,
        2.3,
        1.2,
        "Health Decision",
        "Fault + confidence\n+ explanation",
    )

    add_arrow(
        ax,
        12.0,
        1.9,
        11.3,
        0.8,
    )

    # ------------------------------------------------------
    # RESEARCH RESULT ANNOTATIONS
    # ------------------------------------------------------

    ax.text(
        10.15,
        6.15,
        "Hybrid F1\n0.8563 development\n0.8294 Monte Carlo",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        15.2,
        7.45,
        "Diagnosis\n93.32% development\n85.27% independent",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        15.25,
        4.65,
        "Accepted accuracy\n97.83%\non unseen data",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.text(
        15.2,
        1.9,
        "SHAP top-5 drop\n0.6305\nRandom: 0.0148",
        ha="center",
        va="center",
        fontsize=9,
    )

    figure_file = (
        figures_dir
        / "experiment_054_sentinel_xai_architecture.png"
    )

    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.show()

    return figure_file


# ==========================================================
# FIGURE 2
# RESEARCH EVOLUTION
# ==========================================================

def create_research_evolution_figure():

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.set_xlim(
        0,
        15,
    )

    ax.set_ylim(
        0,
        8,
    )

    ax.axis(
        "off"
    )

    ax.text(
        7.5,
        7.55,
        "SENTINEL-XAI — Research Evolution",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    boxes = [
        (
            0.4,
            "Baseline",
            "Rules\nF1 = 0.7207",
        ),
        (
            2.8,
            "Physics",
            "Residual monitoring\nF1 = 0.7870",
        ),
        (
            5.2,
            "Unsupervised ML",
            "Isolation Forest\nF1 = 0.2502",
        ),
        (
            7.6,
            "Hybrid",
            "Evidence fusion\nF1 = 0.8563",
        ),
        (
            10.0,
            "Diagnosis",
            "Random Forest\nAccuracy = 93.32%",
        ),
        (
            12.4,
            "Explainability",
            "Confidence + SHAP\nInterpretable output",
        ),
    ]

    y = 4.3

    width = 2.0

    height = 1.6

    for index, (
        x,
        title,
        subtitle,
    ) in enumerate(
        boxes
    ):

        add_box(
            ax,
            x,
            y,
            width,
            height,
            title,
            subtitle,
            fontsize=10,
        )

        if index < len(boxes) - 1:

            next_x = boxes[
                index + 1
            ][0]

            add_arrow(
                ax,
                x + width,
                y + height / 2,
                next_x,
                y + height / 2,
            )

    # ------------------------------------------------------
    # INDEPENDENT VALIDATION
    # ------------------------------------------------------

    add_box(
        ax,
        5.2,
        1.1,
        4.6,
        1.5,
        "Independent Monte Carlo Validation",
        "120 randomized scenarios • no retraining • no threshold retuning",
        fontsize=11,
    )

    add_arrow(
        ax,
        13.4,
        4.3,
        9.8,
        2.0,
    )

    ax.text(
        11.3,
        1.85,
        "Hybrid F1 = 0.8294\nDiagnosis Macro F1 = 0.9003\n120/120 scenario success",
        ha="center",
        va="center",
        fontsize=10,
    )

    figure_file = (
        figures_dir
        / "experiment_054_research_evolution.png"
    )

    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.show()

    return figure_file


# ==========================================================
# FIGURE 3
# FINAL RESEARCH CONTRIBUTION MAP
# ==========================================================

def create_contribution_figure():

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.set_xlim(
        0,
        13,
    )

    ax.set_ylim(
        0,
        9,
    )

    ax.axis(
        "off"
    )

    ax.text(
        6.5,
        8.55,
        "SENTINEL-XAI — Main Research Contributions",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # ------------------------------------------------------
    # CENTER
    # ------------------------------------------------------

    add_box(
        ax,
        4.7,
        3.6,
        3.6,
        1.6,
        "SENTINEL-XAI",
        "Hybrid • Confidence-Aware\nExplainable Fault Diagnosis",
        fontsize=12,
    )

    # ------------------------------------------------------
    # CONTRIBUTIONS
    # ------------------------------------------------------

    add_box(
        ax,
        0.6,
        6.3,
        3.2,
        1.4,
        "1. Hybrid Detection",
        "Rules + physics + ML\noutperform standalone methods",
    )

    add_box(
        ax,
        9.2,
        6.3,
        3.2,
        1.4,
        "2. Fault Diagnosis",
        "7-class supervised\nhealth-state identification",
    )

    add_box(
        ax,
        0.6,
        1.0,
        3.2,
        1.4,
        "3. Explainability",
        "SHAP + physical evidence\n+ engineering reports",
    )

    add_box(
        ax,
        9.2,
        1.0,
        3.2,
        1.4,
        "4. Robust Validation",
        "120 unseen randomized\nMonte Carlo scenarios",
    )

    add_arrow(
        ax,
        3.8,
        6.8,
        5.0,
        5.1,
    )

    add_arrow(
        ax,
        9.2,
        6.8,
        8.0,
        5.1,
    )

    add_arrow(
        ax,
        3.8,
        1.8,
        5.0,
        3.6,
    )

    add_arrow(
        ax,
        9.2,
        1.8,
        8.0,
        3.6,
    )

    # ------------------------------------------------------
    # LIMITATION
    # ------------------------------------------------------

    ax.text(
        6.5,
        0.35,
        "Documented limitation: frozen confidence calibration did not fully transfer "
        "to all unseen Monte Carlo conditions.",
        ha="center",
        va="center",
        fontsize=10,
    )

    figure_file = (
        figures_dir
        / "experiment_054_research_contributions.png"
    )

    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.show()

    return figure_file


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 94)
    print("SENTINEL-XAI - EXPERIMENT 054")
    print("FINAL ARCHITECTURE AND RESEARCH FIGURES")
    print("=" * 94)
    print()

    architecture_file = (
        create_architecture_figure()
    )

    evolution_file = (
        create_research_evolution_figure()
    )

    contribution_file = (
        create_contribution_figure()
    )

    print()
    print(
        "Final architecture figure:"
    )
    print(
        architecture_file
    )

    print()
    print(
        "Research evolution figure:"
    )
    print(
        evolution_file
    )

    print()
    print(
        "Research contribution figure:"
    )
    print(
        contribution_file
    )

    print()

    print(
        "RESULT: FINAL SENTINEL-XAI "
        "ARCHITECTURE FIGURES CREATED"
    )

    print("=" * 94)


if __name__ == "__main__":
    main()