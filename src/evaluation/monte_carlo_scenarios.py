from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


FAULT_TYPES = [

    "solar_array_degradation",

    "battery_degradation",

    "thermal_anomaly",

    "reaction_wheel_degradation",

    "battery_voltage_sensor_drift",

    "telemetry_dropout",
]


@dataclass
class MonteCarloScenario:

    scenario_id: int

    fault_type: str

    start_time_h: float

    severity: float

    profile: str

    ramp_duration_h: float

    sensor_noise_scale: float

    random_seed: int


class MonteCarloScenarioGenerator:

    def __init__(
        self,
        random_seed: int = 2026,
    ):

        self.rng = (
            np.random.default_rng(
                random_seed
            )
        )

    # ======================================================
    # ONE RANDOM SCENARIO
    # ======================================================

    def generate_one(
        self,
        scenario_id: int,
        fault_type: str,
    ) -> MonteCarloScenario:

        if fault_type not in FAULT_TYPES:

            raise ValueError(
                f"Unknown fault type: {fault_type}"
            )

        # --------------------------------------------------
        # Fault begins somewhere between 1.5 h and 3.5 h.
        # This avoids always teaching/testing the system
        # that faults start at exactly 2 h.
        # --------------------------------------------------

        start_time_h = float(
            self.rng.uniform(
                1.5,
                3.5,
            )
        )

        # --------------------------------------------------
        # Fault severity
        #
        # Broad enough to include milder and stronger
        # versions than the deterministic development cases.
        # --------------------------------------------------

        severity = float(
            self.rng.uniform(
                0.20,
                0.80,
            )
        )

        # --------------------------------------------------
        # Abrupt vs gradual
        # --------------------------------------------------

        profile = str(
            self.rng.choice(
                [
                    "abrupt",
                    "gradual",
                ],
                p=[
                    0.35,
                    0.65,
                ],
            )
        )

        if profile == "gradual":

            ramp_duration_h = float(
                self.rng.uniform(
                    0.20,
                    1.00,
                )
            )

        else:

            ramp_duration_h = 0.0

        # --------------------------------------------------
        # Sensor noise scale
        #
        # 1.0 = nominal sensor noise.
        # Values above/below nominal create distribution
        # shift without becoming unrealistic.
        # --------------------------------------------------

        sensor_noise_scale = float(
            self.rng.uniform(
                0.75,
                1.50,
            )
        )

        random_seed = int(
            self.rng.integers(
                1,
                2_147_000_000,
            )
        )

        return MonteCarloScenario(

            scenario_id=scenario_id,

            fault_type=fault_type,

            start_time_h=start_time_h,

            severity=severity,

            profile=profile,

            ramp_duration_h=ramp_duration_h,

            sensor_noise_scale=(
                sensor_noise_scale
            ),

            random_seed=random_seed,
        )

    # ======================================================
    # BALANCED MONTE CARLO SET
    # ======================================================

    def generate_balanced(
        self,
        scenarios_per_fault: int,
    ) -> pd.DataFrame:

        scenarios = []

        scenario_id = 1

        for fault_type in FAULT_TYPES:

            for _ in range(
                scenarios_per_fault
            ):

                scenario = (
                    self.generate_one(
                        scenario_id=(
                            scenario_id
                        ),
                        fault_type=(
                            fault_type
                        ),
                    )
                )

                scenarios.append(
                    asdict(
                        scenario
                    )
                )

                scenario_id += 1

        return pd.DataFrame(
            scenarios
        )