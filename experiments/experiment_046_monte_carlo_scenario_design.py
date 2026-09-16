from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:

    sys.path.insert(
        0,
        str(project_root),
    )


from src.evaluation.monte_carlo_scenarios import (
    FAULT_TYPES,
    MonteCarloScenarioGenerator,
)


def main():

    # ======================================================
    # GENERATE PLAN
    #
    # 20 fresh scenarios per fault:
    #
    # 6 faults x 20 = 120 Monte Carlo scenarios
    # ======================================================

    scenarios_per_fault = 20

    generator = (
        MonteCarloScenarioGenerator(
            random_seed=2026,
        )
    )

    scenarios = (
        generator.generate_balanced(
            scenarios_per_fault=(
                scenarios_per_fault
            )
        )
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    expected_total = (
        len(FAULT_TYPES)
        *
        scenarios_per_fault
    )

    fault_counts = (
        scenarios[
            "fault_type"
        ]
        .value_counts()
        .reindex(
            FAULT_TYPES
        )
    )

    balanced = bool(
        (
            fault_counts
            ==
            scenarios_per_fault
        ).all()
    )

    start_times_valid = bool(
        scenarios[
            "start_time_h"
        ].between(
            1.5,
            3.5,
        ).all()
    )

    severity_valid = bool(
        scenarios[
            "severity"
        ].between(
            0.20,
            0.80,
        ).all()
    )

    noise_valid = bool(
        scenarios[
            "sensor_noise_scale"
        ].between(
            0.75,
            1.50,
        ).all()
    )

    valid_profiles = bool(
        scenarios[
            "profile"
        ]
        .isin(
            [
                "abrupt",
                "gradual",
            ]
        )
        .all()
    )

    gradual = (
        scenarios[
            "profile"
        ]
        ==
        "gradual"
    )

    gradual_ramps_valid = bool(
        scenarios.loc[
            gradual,
            "ramp_duration_h",
        ]
        .between(
            0.20,
            1.00,
        )
        .all()
    )

    abrupt_ramps_valid = bool(
        (
            scenarios.loc[
                ~gradual,
                "ramp_duration_h",
            ]
            ==
            0.0
        ).all()
    )

    seeds_unique = bool(
        scenarios[
            "random_seed"
        ].is_unique
    )

    # ======================================================
    # SAVE
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

    output_file = (
        tables_dir
        / "experiment_046_monte_carlo_scenarios.csv"
    )

    figure_file = (
        figures_dir
        / "experiment_046_monte_carlo_design.png"
    )

    scenarios.to_csv(
        output_file,
        index=False,
    )

    # ======================================================
    # FIGURE
    # ======================================================

    plt.figure(
        figsize=(10, 6)
    )

    for fault_type in FAULT_TYPES:

        subset = scenarios[
            scenarios[
                "fault_type"
            ]
            ==
            fault_type
        ]

        plt.scatter(
            subset[
                "start_time_h"
            ],
            subset[
                "severity"
            ],
            label=fault_type,
        )

    plt.xlabel(
        "Randomized Fault Start Time [h]"
    )

    plt.ylabel(
        "Fault Severity"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Phase-9 Monte Carlo Scenario Design"
    )

    plt.grid(True)
    plt.legend(
        fontsize=8
    )
    plt.tight_layout()

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # CHECKS
    # ======================================================

    checks = {

        "Correct scenario count":
            len(scenarios)
            ==
            expected_total,

        "Balanced fault classes":
            balanced,

        "Fault start times valid":
            start_times_valid,

        "Fault severities valid":
            severity_valid,

        "Sensor-noise scales valid":
            noise_valid,

        "Fault profiles valid":
            valid_profiles,

        "Gradual ramp durations valid":
            gradual_ramps_valid,

        "Abrupt faults have zero ramp":
            abrupt_ramps_valid,

        "Random seeds unique":
            seeds_unique,
    }

    # ======================================================
    # TERMINAL
    # ======================================================

    print()
    print("=" * 92)

    print(
        "SENTINEL-XAI - EXPERIMENT 046"
    )

    print(
        "PHASE 9 MONTE CARLO SCENARIO DESIGN"
    )

    print("=" * 92)

    print()

    print(
        f"Scenarios per fault: "
        f"{scenarios_per_fault}"
    )

    print(
        f"Fault classes: "
        f"{len(FAULT_TYPES)}"
    )

    print(
        f"Total scenarios: "
        f"{len(scenarios)}"
    )

    print()

    print(
        "Randomized dimensions:"
    )

    print(
        "  Fault start time:   "
        "1.5 - 3.5 h"
    )

    print(
        "  Fault severity:     "
        "0.20 - 0.80"
    )

    print(
        "  Fault profile:      "
        "abrupt / gradual"
    )

    print(
        "  Gradual ramp:       "
        "0.20 - 1.00 h"
    )

    print(
        "  Sensor noise scale: "
        "0.75 - 1.50"
    )

    print()

    print("-" * 92)

    print(
        "FAULT DISTRIBUTION"
    )

    print("-" * 92)

    print()

    for fault_type in FAULT_TYPES:

        print(
            f"{fault_type:<36} "
            f"{int(fault_counts[fault_type])}"
        )

    print()

    print("-" * 92)

    print(
        "SCENARIO VALIDATION"
    )

    print("-" * 92)

    passed = 0

    for name, result in checks.items():

        print()

        print(
            f"{name}: {result}"
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
            "RESULT: MONTE CARLO "
            "SCENARIO DESIGN PASSED"
        )

    else:

        print(
            "RESULT: MONTE CARLO "
            "SCENARIO DESIGN REQUIRES REVIEW"
        )

    print()

    print(
        "Scenario table saved to:"
    )

    print(
        output_file
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