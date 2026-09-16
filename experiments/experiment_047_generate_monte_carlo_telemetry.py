from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(
    project_root
) not in sys.path:

    sys.path.insert(
        0,
        str(
            project_root
        ),
    )


from src.evaluation.monte_carlo_scenarios import (
    FAULT_TYPES,
)

from src.evaluation.monte_carlo_runner import (
    generate_monte_carlo_scenario,
    summarize_generated_scenario,
)


def main():

    # ======================================================
    # INPUT SCENARIO PLAN
    # ======================================================

    scenario_file = (

        project_root
        / "results"
        / "tables"
        / "experiment_046_monte_carlo_scenarios.csv"
    )

    if not scenario_file.exists():

        raise FileNotFoundError(
            f"Scenario plan not found:\n"
            f"{scenario_file}"
        )

    scenarios = pd.read_csv(
        scenario_file
    )

    expected_scenarios = int(
        len(
            scenarios
        )
    )

    # ======================================================
    # OUTPUT DIRECTORIES
    # ======================================================

    data_dir = (
        project_root
        / "data"
        / "monte_carlo"
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

    data_dir.mkdir(
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

    # ======================================================
    # REMOVE STALE MONTE CARLO FILES FROM PREVIOUS RUNS
    # ======================================================

    for stale_file in (
        data_dir.glob(
            "scenario_*.csv"
        )
    ):

        stale_file.unlink()

    # ======================================================
    # GENERATE ALL SCENARIOS
    # ======================================================

    summaries = []

    failures = []

    print()
    print(
        "Generating fresh Monte Carlo telemetry..."
    )
    print()

    for number, (_, scenario) in enumerate(
        scenarios.iterrows(),
        start=1,
    ):

        scenario_id = int(
            scenario[
                "scenario_id"
            ]
        )

        fault_type = str(
            scenario[
                "fault_type"
            ]
        )

        output_file = (

            data_dir
            /
            (
                f"scenario_{scenario_id:04d}_"
                f"{fault_type}.csv"
            )
        )

        try:

            df = (
                generate_monte_carlo_scenario(

                    scenario=scenario,

                    duration_hours=6.0,

                    dt_s=60.0,
                )
            )

            df.to_csv(
                output_file,
                index=False,
            )

            summary = (
                summarize_generated_scenario(

                    df=df,

                    output_file=(
                        output_file
                    ),
                )
            )

            summaries.append(
                summary
            )

            print(
                f"[{number:03d}/"
                f"{expected_scenarios:03d}] "
                f"Scenario {scenario_id:04d} "
                f"{fault_type:<34} "
                f"OK"
            )

        except Exception as exc:

            failures.append(
                {
                    "scenario_id":
                        scenario_id,

                    "fault_type":
                        fault_type,

                    "error":
                        repr(
                            exc
                        ),
                }
            )

            print(
                f"[{number:03d}/"
                f"{expected_scenarios:03d}] "
                f"Scenario {scenario_id:04d} "
                f"{fault_type:<34} "
                f"FAILED"
            )

            print(
                f"    {exc}"
            )

    # ======================================================
    # DATAFRAMES
    # ======================================================

    summary_df = pd.DataFrame(
        summaries
    )

    failure_df = pd.DataFrame(
        failures,
        columns=[
            "scenario_id",
            "fault_type",
            "error",
        ],
    )

    # ======================================================
    # FAILURE LOG
    # ======================================================

    failure_file = (

        tables_dir
        /
        "experiment_047_generation_failures.csv"
    )

    failure_df.to_csv(
        failure_file,
        index=False,
    )

    # ======================================================
    # STOP CLEANLY IF EVERYTHING FAILED
    # ======================================================

    if summary_df.empty:

        print()
        print("=" * 96)

        print(
            "SENTINEL-XAI - EXPERIMENT 047"
        )

        print(
            "MONTE CARLO TELEMETRY GENERATION FAILED"
        )

        print("=" * 96)

        print()

        print(
            f"Planned scenarios: "
            f"{expected_scenarios}"
        )

        print(
            "Generated scenarios: 0"
        )

        print(
            f"Generation failures: "
            f"{len(failures)}"
        )

        print()

        print(
            "Failure log saved to:"
        )

        print(
            failure_file
        )

        print("=" * 96)

        return

    # ======================================================
    # SAVE MANIFEST
    # ======================================================

    manifest_file = (

        tables_dir
        /
        "experiment_047_monte_carlo_manifest.csv"
    )

    summary_df.to_csv(
        manifest_file,
        index=False,
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    all_generated = bool(
        len(
            summary_df
        )
        ==
        expected_scenarios
    )

    no_failures = bool(
        len(
            failures
        )
        ==
        0
    )

    correct_sample_count = bool(
        (
            summary_df[
                "rows"
            ]
            ==
            361
        ).all()
    )

    all_have_fault_samples = bool(
        (
            summary_df[
                "active_fault_samples"
            ]
            >
            0
        ).all()
    )

    unique_scenario_ids = bool(
        summary_df[
            "scenario_id"
        ].is_unique
    )

    no_prefault_missing = bool(
        (
            summary_df[
                "pre_fault_missing_cells"
            ]
            ==
            0
        ).all()
    )

    # ======================================================
    # DROPOUT VALIDATION
    # ======================================================

    dropout_rows = (
        summary_df[
            summary_df[
                "fault_type"
            ]
            ==
            "telemetry_dropout"
        ]
    )

    dropout_missing_valid = bool(
        len(
            dropout_rows
        )
        >
        0
        and
        (
            dropout_rows[
                "active_fault_missing_cells"
            ]
            >
            0
        ).all()
    )

    # ======================================================
    # SENSOR-DRIFT VALIDATION
    # ======================================================

    drift_rows = (
        summary_df[
            summary_df[
                "fault_type"
            ]
            ==
            "battery_voltage_sensor_drift"
        ]
    )

    drift_bias_valid = bool(
        len(
            drift_rows
        )
        >
        0
        and
        (
            drift_rows[
                "mean_active_battery_voltage_residual_v"
            ]
            >
            0.0
        ).all()
    )

    # ======================================================
    # BALANCED DISTRIBUTION
    # ======================================================

    generated_fault_counts = (

        summary_df[
            "fault_type"
        ]

        .value_counts()

        .reindex(
            FAULT_TYPES,
            fill_value=0,
        )
    )

    expected_per_fault = int(
        expected_scenarios
        /
        len(
            FAULT_TYPES
        )
    )

    balanced_generation = bool(
        (
            generated_fault_counts
            ==
            expected_per_fault
        ).all()
    )

    checks = {

        "All planned scenarios generated":
            all_generated,

        "No generation failures":
            no_failures,

        "Every scenario has 361 samples":
            correct_sample_count,

        "Every scenario contains active fault samples":
            all_have_fault_samples,

        "Scenario IDs remain unique":
            unique_scenario_ids,

        "No missing telemetry before faults":
            no_prefault_missing,

        "Dropout scenarios create missing telemetry":
            dropout_missing_valid,

        "Sensor-drift scenarios develop positive bias":
            drift_bias_valid,

        "Generated fault classes remain balanced":
            balanced_generation,
    }

    # ======================================================
    # NOISE VALIDATION STATISTICS
    # ======================================================

    valid_noise = (
        summary_df[
            [
                "requested_noise_scale",
                "realized_battery_voltage_noise_scale",
            ]
        ]
        .dropna()
    )

    noise_error = (

        valid_noise[
            "realized_battery_voltage_noise_scale"
        ]

        -

        valid_noise[
            "requested_noise_scale"
        ]
    )

    mean_absolute_noise_scale_error = float(
        np.abs(
            noise_error
        ).mean()
    )

    noise_scale_correlation = float(
        valid_noise.corr()
        .iloc[
            0,
            1
        ]
    )

    # ======================================================
    # FIGURE 1 — REQUESTED vs REALIZED SENSOR NOISE
    # ======================================================

    noise_figure = (

        figures_dir
        /
        "experiment_047_noise_scale_validation.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        valid_noise[
            "requested_noise_scale"
        ],
        valid_noise[
            "realized_battery_voltage_noise_scale"
        ],
    )

    minimum = float(
        min(
            valid_noise[
                "requested_noise_scale"
            ].min(),

            valid_noise[
                "realized_battery_voltage_noise_scale"
            ].min(),
        )
    )

    maximum = float(
        max(
            valid_noise[
                "requested_noise_scale"
            ].max(),

            valid_noise[
                "realized_battery_voltage_noise_scale"
            ].max(),
        )
    )

    plt.plot(
        [
            minimum,
            maximum,
        ],
        [
            minimum,
            maximum,
        ],
        linestyle="--",
        label=(
            "Ideal requested = realized"
        ),
    )

    plt.xlabel(
        "Requested Sensor-Noise Scale"
    )

    plt.ylabel(
        "Realized Battery-Voltage Noise Scale"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo Sensor-Noise Validation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        noise_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # FIGURE 2 — FAULT EXPOSURE
    # ======================================================

    active_figure = (

        figures_dir
        /
        "experiment_047_active_fault_samples.png"
    )

    active_groups = [

        summary_df.loc[
            summary_df[
                "fault_type"
            ]
            ==
            fault_type,
            "active_fault_samples",
        ]
        .to_numpy()

        for fault_type
        in FAULT_TYPES
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.boxplot(
        active_groups,
        tick_labels=FAULT_TYPES,
    )

    plt.ylabel(
        "Active Fault Samples"
    )

    plt.xlabel(
        "Fault Type"
    )

    plt.title(
        "SENTINEL-XAI - "
        "Monte Carlo Fault Exposure"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        active_figure,
        dpi=200,
    )

    plt.show()

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 96)

    print(
        "SENTINEL-XAI - EXPERIMENT 047"
    )

    print(
        "MONTE CARLO TELEMETRY GENERATION"
    )

    print("=" * 96)

    print()

    print(
        f"Planned scenarios: "
        f"{expected_scenarios}"
    )

    print(
        f"Generated scenarios: "
        f"{len(summary_df)}"
    )

    print(
        f"Generation failures: "
        f"{len(failures)}"
    )

    print()

    print(
        "Generated fault distribution:"
    )

    print()

    for fault_type in (
        FAULT_TYPES
    ):

        print(
            f"  {fault_type:<36} "
            f"{int(generated_fault_counts[fault_type])}"
        )

    print()

    print("-" * 96)

    print(
        "RANDOMIZED SENSOR-NOISE CHECK"
    )

    print("-" * 96)

    print()

    print(
        f"Requested / realized scale correlation: "
        f"{noise_scale_correlation:.4f}"
    )

    print(
        f"Mean absolute realized-scale error: "
        f"{mean_absolute_noise_scale_error:.4f}"
    )

    print()

    print(
        "Small requested/realized differences are "
        "expected because noise scale is estimated "
        "from a finite pre-fault sample window."
    )

    print()

    print("-" * 96)

    print(
        "GENERATION VALIDATION"
    )

    print("-" * 96)

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
            "RESULT: MONTE CARLO TELEMETRY "
            "GENERATION PASSED"
        )

    else:

        print(
            "RESULT: MONTE CARLO TELEMETRY "
            "GENERATION REQUIRES REVIEW"
        )

    print()

    print(
        "Monte Carlo datasets saved under:"
    )

    print(
        data_dir
    )

    print()

    print(
        "Manifest saved to:"
    )

    print(
        manifest_file
    )

    print()

    print(
        "Failure log saved to:"
    )

    print(
        failure_file
    )

    print()

    print(
        "Figures saved to:"
    )

    print(
        noise_figure
    )

    print(
        active_figure
    )

    print("=" * 96)


if __name__ == "__main__":
    main()