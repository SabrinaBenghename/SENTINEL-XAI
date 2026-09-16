from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd


from src.simulation.spacecraft_simulator import (
    run_simulation,
)

from src.faults.fault_manager import (
    FaultConfig,
)

from src.faults.sensor_faults import (
    apply_battery_voltage_drift,
    apply_telemetry_dropout,
)


# ==========================================================
# CANONICAL FAULT NAMES
#
# IMPORTANT:
# FaultManager expects the canonical SENTINEL fault names.
#
# spacecraft_simulator.py is responsible for translating
# those faults internally into solar_degradation,
# wheel_degradation, etc.
# ==========================================================

SIMULATOR_FAULT_NAMES = {

    "solar_array_degradation":
        "solar_array_degradation",

    "battery_degradation":
        "battery_degradation",

    "thermal_anomaly":
        "thermal_anomaly",

    "reaction_wheel_degradation":
        "reaction_wheel_degradation",
}


PHYSICAL_FAULTS = set(
    SIMULATOR_FAULT_NAMES.keys()
)


SENSOR_FAULTS = {

    "battery_voltage_sensor_drift",

    "telemetry_dropout",
}


# ==========================================================
# MEASURED / TRUE CHANNEL PAIRS
# ==========================================================

MEASUREMENT_PAIRS = [

    (
        "battery_soc_measured",
        "battery_soc_true",
    ),

    (
        "battery_voltage_measured",
        "battery_voltage_true",
    ),

    (
        "battery_current_measured",
        "battery_current_true",
    ),

    (
        "battery_temp_measured",
        "battery_temp_true",
    ),

    (
        "electronics_temp_measured",
        "electronics_temp_true",
    ),

    (
        "wheel_speed_measured",
        "wheel_speed_true",
    ),

    (
        "wheel_current_measured",
        "wheel_current_true",
    ),

    (
        "wheel_temp_measured",
        "wheel_temp_true",
    ),
]


MEASURED_CHANNELS = [

    measured

    for measured, _
    in MEASUREMENT_PAIRS
]


# ==========================================================
# REQUIRED SIMULATOR COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [

    "time_s",
    "time_h",

] + [

    column

    for pair in MEASUREMENT_PAIRS
    for column in pair
]


# ==========================================================
# NORMALIZE SIMULATOR OUTPUT
# ==========================================================

def simulation_to_dataframe(
    simulation_output,
) -> pd.DataFrame:

    if isinstance(
        simulation_output,
        pd.DataFrame,
    ):

        df = (
            simulation_output
            .copy()
            .reset_index(
                drop=True
            )
        )

    elif isinstance(
        simulation_output,
        (list, tuple),
    ):

        if len(
            simulation_output
        ) == 0:

            raise ValueError(
                "run_simulation returned no telemetry samples."
            )

        first = simulation_output[0]

        if isinstance(
            first,
            dict,
        ):

            df = pd.DataFrame(
                simulation_output
            )

        elif is_dataclass(
            first
        ):

            df = pd.DataFrame(
                [
                    asdict(record)
                    for record
                    in simulation_output
                ]
            )

        elif hasattr(
            first,
            "__dict__",
        ):

            df = pd.DataFrame(
                [
                    vars(record)
                    for record
                    in simulation_output
                ]
            )

        else:

            try:

                df = pd.DataFrame(
                    simulation_output
                )

            except Exception as exc:

                raise TypeError(
                    "Could not convert run_simulation "
                    "output into a pandas DataFrame."
                ) from exc

        df = df.reset_index(
            drop=True
        )

    else:

        raise TypeError(
            "Unexpected run_simulation return type: "
            f"{type(simulation_output).__name__}"
        )

    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Simulator telemetry is missing required columns: "
            f"{missing_columns}"
        )

    return df


# ==========================================================
# FAULT SCHEDULE
# ==========================================================

def calculate_fault_state(
    time_h: float,
    start_time_h: float,
    requested_severity: float,
    profile: str,
    ramp_duration_h: float,
):

    if time_h < start_time_h:

        return (
            False,
            0.0,
            0.0,
        )

    if profile == "abrupt":

        return (
            True,
            1.0,
            requested_severity,
        )

    if ramp_duration_h <= 0.0:

        progress = 1.0

    else:

        progress = (
            time_h
            -
            start_time_h
        ) / ramp_duration_h

        progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

    active = bool(
        progress > 0.0
    )

    effective_severity = float(
        requested_severity
        *
        progress
    )

    return (
        active,
        progress,
        effective_severity,
    )


# ==========================================================
# RANDOMIZED SENSOR NOISE
# ==========================================================

def scale_sensor_noise(
    df: pd.DataFrame,
    noise_scale: float,
) -> pd.DataFrame:

    """
    z_MC = x + scale * (z - x)

    True telemetry is used here only for offline Monte Carlo
    generation and never becomes a diagnosis input feature.
    """

    result = df.copy()

    for measured, truth in (
        MEASUREMENT_PAIRS
    ):

        residual = (
            result[
                measured
            ]
            -
            result[
                truth
            ]
        )

        result[
            measured
        ] = (
            result[
                truth
            ]
            +
            float(
                noise_scale
            )
            *
            residual
        )

    return result


# ==========================================================
# SENSOR FAULT INJECTION
# ==========================================================

def inject_sensor_fault(
    df: pd.DataFrame,
    scenario,
) -> pd.DataFrame:

    result = df.copy()

    fault_type = str(
        scenario[
            "fault_type"
        ]
    )

    for index in range(
        len(
            result
        )
    ):

        time_h = float(
            result.at[
                index,
                "time_h",
            ]
        )

        (
            active,
            _,
            effective_severity,
        ) = calculate_fault_state(

            time_h=time_h,

            start_time_h=float(
                scenario[
                    "start_time_h"
                ]
            ),

            requested_severity=float(
                scenario[
                    "severity"
                ]
            ),

            profile=str(
                scenario[
                    "profile"
                ]
            ),

            ramp_duration_h=float(
                scenario[
                    "ramp_duration_h"
                ]
            ),
        )

        if not active:
            continue

        # ==================================================
        # BATTERY-VOLTAGE SENSOR DRIFT
        # ==================================================

        if (
            fault_type
            ==
            "battery_voltage_sensor_drift"
        ):

            current_voltage = result.at[
                index,
                "battery_voltage_measured",
            ]

            if not pd.isna(
                current_voltage
            ):

                result.at[
                    index,
                    "battery_voltage_measured",
                ] = (
                    apply_battery_voltage_drift(

                        measured_voltage_v=float(
                            current_voltage
                        ),

                        severity=float(
                            effective_severity
                        ),
                    )
                )

        # ==================================================
        # TELEMETRY DROPOUT
        # ==================================================

        elif (
            fault_type
            ==
            "telemetry_dropout"
        ):

            for channel in (
                MEASURED_CHANNELS
            ):

                value = result.at[
                    index,
                    channel,
                ]

                result.at[
                    index,
                    channel,
                ] = (
                    apply_telemetry_dropout(

                        value=value,

                        severity=float(
                            effective_severity
                        ),
                    )
                )

        else:

            raise ValueError(
                "Unsupported sensor fault: "
                f"{fault_type}"
            )

    return result


# ==========================================================
# CANONICAL GROUND-TRUTH METADATA
# ==========================================================

def add_ground_truth_metadata(
    df: pd.DataFrame,
    scenario,
) -> pd.DataFrame:

    result = df.copy()

    active_values = []
    progress_values = []
    severity_values = []
    label_values = []

    for time_h in result[
        "time_h"
    ]:

        (
            active,
            progress,
            effective_severity,
        ) = calculate_fault_state(

            time_h=float(
                time_h
            ),

            start_time_h=float(
                scenario[
                    "start_time_h"
                ]
            ),

            requested_severity=float(
                scenario[
                    "severity"
                ]
            ),

            profile=str(
                scenario[
                    "profile"
                ]
            ),

            ramp_duration_h=float(
                scenario[
                    "ramp_duration_h"
                ]
            ),
        )

        active_values.append(
            bool(
                active
            )
        )

        progress_values.append(
            float(
                progress
            )
        )

        severity_values.append(
            float(
                effective_severity
            )
        )

        if active:

            label_values.append(
                str(
                    scenario[
                        "fault_type"
                    ]
                )
            )

        else:

            label_values.append(
                "nominal"
            )

    # ======================================================
    # EVALUATION GROUND TRUTH
    # ======================================================

    result[
        "fault_active"
    ] = active_values

    result[
        "fault_progress"
    ] = progress_values

    result[
        "fault_requested_severity"
    ] = float(
        scenario[
            "severity"
        ]
    )

    result[
        "fault_effective_severity"
    ] = severity_values

    result[
        "fault_label"
    ] = label_values

    # ======================================================
    # SCENARIO METADATA
    # ======================================================

    result[
        "scenario_id"
    ] = int(
        scenario[
            "scenario_id"
        ]
    )

    result[
        "scenario_fault_type"
    ] = str(
        scenario[
            "fault_type"
        ]
    )

    result[
        "scenario_start_time_h"
    ] = float(
        scenario[
            "start_time_h"
        ]
    )

    result[
        "scenario_severity"
    ] = float(
        scenario[
            "severity"
        ]
    )

    result[
        "scenario_profile"
    ] = str(
        scenario[
            "profile"
        ]
    )

    result[
        "scenario_ramp_duration_h"
    ] = float(
        scenario[
            "ramp_duration_h"
        ]
    )

    result[
        "scenario_sensor_noise_scale"
    ] = float(
        scenario[
            "sensor_noise_scale"
        ]
    )

    result[
        "scenario_random_seed"
    ] = int(
        scenario[
            "random_seed"
        ]
    )

    return result


# ==========================================================
# GENERATE ONE MONTE CARLO SCENARIO
# ==========================================================

def generate_monte_carlo_scenario(
    scenario,
    duration_hours: float = 6.0,
    dt_s: float = 60.0,
) -> pd.DataFrame:

    fault_type = str(
        scenario[
            "fault_type"
        ]
    )

    sensor_seed = int(
        scenario[
            "random_seed"
        ]
    )

    noise_scale = float(
        scenario[
            "sensor_noise_scale"
        ]
    )

    # ======================================================
    # PHYSICAL FAULTS
    # ======================================================

    if fault_type in PHYSICAL_FAULTS:

        # --------------------------------------------------
        # IMPORTANT FIX:
        #
        # Send canonical fault names to FaultManager.
        #
        # Examples:
        #
        # solar_array_degradation
        # reaction_wheel_degradation
        #
        # spacecraft_simulator.py handles the internal
        # subsystem parameter mapping.
        # --------------------------------------------------

        internal_fault_name = (
            SIMULATOR_FAULT_NAMES[
                fault_type
            ]
        )

        fault = FaultConfig(

            fault_type=(
                internal_fault_name
            ),

            start_time_h=float(
                scenario[
                    "start_time_h"
                ]
            ),

            severity=float(
                scenario[
                    "severity"
                ]
            ),

            profile=str(
                scenario[
                    "profile"
                ]
            ),

            ramp_duration_h=float(
                scenario[
                    "ramp_duration_h"
                ]
            ),
        )

        simulation_output = (
            run_simulation(

                duration_hours=(
                    duration_hours
                ),

                dt_s=dt_s,

                sensor_seed=(
                    sensor_seed
                ),

                faults=[
                    fault
                ],
            )
        )

        df = simulation_to_dataframe(
            simulation_output
        )

    # ======================================================
    # SENSOR FAULTS
    # ======================================================

    elif fault_type in SENSOR_FAULTS:

        internal_fault_name = (
            "postprocessed_sensor_fault"
        )

        simulation_output = (
            run_simulation(

                duration_hours=(
                    duration_hours
                ),

                dt_s=dt_s,

                sensor_seed=(
                    sensor_seed
                ),

                faults=None,
            )
        )

        df = simulation_to_dataframe(
            simulation_output
        )

    else:

        raise ValueError(
            "Unsupported Monte Carlo fault type: "
            f"{fault_type}"
        )

    # ======================================================
    # RANDOMIZED SENSOR NOISE
    # ======================================================

    df = scale_sensor_noise(

        df=df,

        noise_scale=(
            noise_scale
        ),
    )

    # ======================================================
    # SENSOR-FAULT POSTPROCESSING
    # ======================================================

    if fault_type in SENSOR_FAULTS:

        df = inject_sensor_fault(

            df=df,

            scenario=scenario,
        )

    # ======================================================
    # CANONICAL EVALUATION LABELS
    # ======================================================

    df = add_ground_truth_metadata(

        df=df,

        scenario=scenario,
    )

    df[
        "scenario_internal_fault_name"
    ] = internal_fault_name

    return df


# ==========================================================
# SCENARIO SUMMARY
# ==========================================================

def summarize_generated_scenario(
    df: pd.DataFrame,
    output_file: Path,
):

    scenario_id = int(
        df[
            "scenario_id"
        ].iloc[0]
    )

    fault_type = str(
        df[
            "scenario_fault_type"
        ].iloc[0]
    )

    start_time_h = float(
        df[
            "scenario_start_time_h"
        ].iloc[0]
    )

    requested_noise_scale = float(
        df[
            "scenario_sensor_noise_scale"
        ].iloc[0]
    )

    # ======================================================
    # PRE-FAULT SEGMENT
    # ======================================================

    pre_fault = (
        df[
            df[
                "time_h"
            ]
            <
            start_time_h
        ]
    )

    # ======================================================
    # ACTIVE FAULT SEGMENT
    # ======================================================

    active_fault = (
        df[
            df[
                "fault_active"
            ]
            .astype(bool)
        ]
    )

    # ======================================================
    # REALIZED BATTERY-VOLTAGE NOISE
    # ======================================================

    if len(
        pre_fault
    ) > 1:

        voltage_residual = (
            pre_fault[
                "battery_voltage_measured"
            ]
            -
            pre_fault[
                "battery_voltage_true"
            ]
        )

        realized_voltage_std = float(
            voltage_residual.std(
                ddof=1
            )
        )

        realized_noise_scale = float(
            realized_voltage_std
            /
            0.03
        )

    else:

        realized_voltage_std = float(
            "nan"
        )

        realized_noise_scale = float(
            "nan"
        )

    # ======================================================
    # MISSING VALUES
    # ======================================================

    pre_fault_missing = int(
        pre_fault[
            MEASURED_CHANNELS
        ]
        .isna()
        .sum()
        .sum()
    )

    active_missing = int(
        active_fault[
            MEASURED_CHANNELS
        ]
        .isna()
        .sum()
        .sum()
    )

    # ======================================================
    # ACTIVE BATTERY-VOLTAGE RESIDUAL
    # ======================================================

    if len(
        active_fault
    ) > 0:

        active_voltage_residual = (
            active_fault[
                "battery_voltage_measured"
            ]
            -
            active_fault[
                "battery_voltage_true"
            ]
        )

        mean_active_voltage_residual = float(
            active_voltage_residual.mean()
        )

    else:

        mean_active_voltage_residual = float(
            "nan"
        )

    return {

        "scenario_id":
            scenario_id,

        "fault_type":
            fault_type,

        "rows":
            int(
                len(
                    df
                )
            ),

        "fault_start_time_h":
            start_time_h,

        "requested_severity":
            float(
                df[
                    "scenario_severity"
                ].iloc[0]
            ),

        "profile":
            str(
                df[
                    "scenario_profile"
                ].iloc[0]
            ),

        "ramp_duration_h":
            float(
                df[
                    "scenario_ramp_duration_h"
                ].iloc[0]
            ),

        "requested_noise_scale":
            requested_noise_scale,

        "realized_battery_voltage_noise_std_v":
            realized_voltage_std,

        "realized_battery_voltage_noise_scale":
            realized_noise_scale,

        "pre_fault_samples":
            int(
                len(
                    pre_fault
                )
            ),

        "active_fault_samples":
            int(
                len(
                    active_fault
                )
            ),

        "pre_fault_missing_cells":
            pre_fault_missing,

        "active_fault_missing_cells":
            active_missing,

        "mean_active_battery_voltage_residual_v":
            mean_active_voltage_residual,

        "internal_fault_name":
            str(
                df[
                    "scenario_internal_fault_name"
                ].iloc[0]
            ),

        "output_file":
            str(
                output_file
            ),
    }