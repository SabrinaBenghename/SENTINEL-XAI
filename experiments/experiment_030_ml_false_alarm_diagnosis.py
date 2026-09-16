from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.ml.feature_pipeline import (
    build_ml_features,
)

from src.ml.subsystem_isolation_forest import (
    SUBSYSTEM_FEATURES,
)

from src.ml.persistent_anomaly_detector import (
    PersistentAnomalyFilter,
)


# ==========================================================
# SCORE ONE DATASET
# ==========================================================

def score_dataset(
    df,
    detector,
):

    features = build_ml_features(
        df
    )

    result = pd.DataFrame(
        {
            "time_h":
                df["time_h"].values,
        }
    )

    raw_columns = []

    for subsystem, columns in (
        SUBSYSTEM_FEATURES.items()
    ):

        x = features[
            columns
        ]

        missing = (
            x.isna()
            .any(axis=1)
            .to_numpy()
        )

        valid = (
            ~missing
        )

        scores = np.full(
            len(x),
            np.nan,
        )

        if valid.any():

            scores[
                valid
            ] = (
                -detector.models[
                    subsystem
                ].score_samples(
                    x.loc[
                        valid
                    ]
                )
            )

        threshold = (
            detector.thresholds[
                subsystem
            ]
        )

        anomaly = np.zeros(
            len(x),
            dtype=bool,
        )

        anomaly[
            valid
        ] = (
            scores[
                valid
            ]
            >
            threshold
        )

        anomaly[
            missing
        ] = True

        result[
            f"{subsystem}_score"
        ] = scores

        result[
            f"{subsystem}_threshold"
        ] = threshold

        result[
            f"{subsystem}_anomaly"
        ] = anomaly

        raw_columns.append(
            f"{subsystem}_anomaly"
        )

    # ======================================================
    # GLOBAL RAW ANOMALY
    # ======================================================

    result[
        "raw_anomaly"
    ] = result[
        raw_columns
    ].any(
        axis=1
    )

    # ======================================================
    # PERSISTENCE
    # ======================================================

    persistence = (
        PersistentAnomalyFilter(
            required_samples=3,
        )
    )

    persistent = []
    counter = []

    for value in result[
        "raw_anomaly"
    ]:

        state = persistence.update(
            bool(value)
        )

        persistent.append(
            state.persistent_anomaly
        )

        counter.append(
            state.consecutive_count
        )

    result[
        "persistence_count"
    ] = counter

    result[
        "persistent_anomaly"
    ] = persistent

    return result


# ==========================================================
# MAIN
# ==========================================================

def main():

    model_file = (
        project_root
        / "results"
        / "models"
        / "experiment_028_subsystem_iforest.joblib"
    )

    detector = joblib.load(
        model_file
    )

    # ======================================================
    # ORIGINAL 6-HOUR HEALTHY DATASET
    # ======================================================

    nominal_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    nominal_df = pd.read_csv(
        nominal_file
    )

    nominal_result = score_dataset(
        nominal_df,
        detector,
    )

    # ======================================================
    # PRE-FAULT REGION OF ONE REPRESENTATIVE FAULT DATASET
    # ======================================================

    solar_file = (
        project_root
        / "data"
        / "faults"
        / "experiment_006_solar_degradation.csv"
    )

    solar_df = pd.read_csv(
        solar_file
    )

    solar_prefault_df = (
        solar_df[
            solar_df["time_h"] < 2.0
        ]
        .copy()
    )

    prefault_result = score_dataset(
        solar_prefault_df,
        detector,
    )

    # ======================================================
    # COUNTS
    # ======================================================

    rows = []

    for name, table in [
        (
            "6h_nominal_full",
            nominal_result,
        ),
        (
            "solar_prefault_0_2h",
            prefault_result,
        ),
    ]:

        row = {
            "dataset":
                name,

            "samples":
                len(table),

            "raw_alarm_samples":
                int(
                    table[
                        "raw_anomaly"
                    ].sum()
                ),

            "persistent_alarm_samples":
                int(
                    table[
                        "persistent_anomaly"
                    ].sum()
                ),
        }

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            row[
                f"{subsystem}_raw_alarms"
            ] = int(
                table[
                    f"{subsystem}_anomaly"
                ].sum()
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    # ======================================================
    # FIRST / LAST PERSISTENT ALARM
    # ======================================================

    persistent_rows = (
        nominal_result[
            nominal_result[
                "persistent_anomaly"
            ]
        ]
    )

    # ======================================================
    # SAVE
    # ======================================================

    output_dir = (
        project_root
        / "results"
        / "tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_dir
        / "experiment_030_ml_false_alarm_summary.csv"
    )

    detail_file = (
        output_dir
        / "experiment_030_6h_nominal_detail.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    nominal_result.to_csv(
        detail_file,
        index=False,
    )

    # ======================================================
    # TERMINAL OUTPUT
    # ======================================================

    print()
    print("=" * 80)

    print(
        "SENTINEL-XAI - EXPERIMENT 030"
    )

    print(
        "ML FALSE-ALARM DIAGNOSIS"
    )

    print("=" * 80)

    print()

    for _, row in summary.iterrows():

        print(
            row["dataset"]
        )

        print(
            f"  Samples: "
            f"{int(row['samples'])}"
        )

        print(
            f"  Raw alarm samples: "
            f"{int(row['raw_alarm_samples'])}"
        )

        print(
            f"  Persistent alarm samples: "
            f"{int(row['persistent_alarm_samples'])}"
        )

        print()

        print(
            "  Subsystem raw alarms:"
        )

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            print(
                f"    {subsystem:<10}: "
                f"{int(row[f'{subsystem}_raw_alarms'])}"
            )

        print()

    print("-" * 80)

    if len(
        persistent_rows
    ) > 0:

        first_time = float(
            persistent_rows[
                "time_h"
            ].iloc[0]
        )

        last_time = float(
            persistent_rows[
                "time_h"
            ].iloc[-1]
        )

        print(
            f"First persistent alarm in "
            f"6h nominal run: "
            f"{first_time:.3f} h"
        )

        print(
            f"Last persistent alarm in "
            f"6h nominal run: "
            f"{last_time:.3f} h"
        )

    else:

        print(
            "No persistent alarms in "
            "6h nominal run."
        )

    print()

    print(
        "Subsystem thresholds:"
    )

    for subsystem, threshold in (
        detector.thresholds.items()
    ):

        print(
            f"  {subsystem:<10}: "
            f"{threshold:.6f}"
        )

    print()

    print(
        "Summary saved to:"
    )

    print(
        summary_file
    )

    print()

    print(
        "Detailed nominal scores saved to:"
    )

    print(
        detail_file
    )

    print("=" * 80)


if __name__ == "__main__":
    main()