from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultConfig:
    """
    Definition of one scheduled spacecraft fault.
    """

    fault_type: str

    # Mission time when fault begins
    start_time_h: float

    # Requested fault magnitude: 0.0 -> 1.0
    severity: float

    # "abrupt" or "gradual"
    profile: str = "abrupt"

    # Only used for gradual faults
    ramp_duration_h: float = 0.5


@dataclass
class FaultState:
    """
    Fault condition at one instant in time.
    """

    fault_type: str
    active: bool

    requested_severity: float
    progress: float
    effective_severity: float

    label: str


class FaultManager:
    """
    Controls fault timing and severity progression.

    Important:
    This class does NOT modify spacecraft physics.

    It only says:
        - whether a fault is active
        - how far it has developed
        - its effective severity

    Individual subsystem models will later use this
    information to modify their physics.
    """

    def __init__(
        self,
        faults: list[FaultConfig] | None = None,
    ):
        if faults is None:
            faults = []

        self.faults = faults

        self._validate_faults()

    def _validate_faults(self):

        for fault in self.faults:

            if not 0.0 <= fault.severity <= 1.0:
                raise ValueError(
                    "Fault severity must be between "
                    "0.0 and 1.0."
                )

            if fault.start_time_h < 0.0:
                raise ValueError(
                    "Fault start time cannot be negative."
                )

            if fault.profile not in {
                "abrupt",
                "gradual",
            }:
                raise ValueError(
                    "Fault profile must be "
                    "'abrupt' or 'gradual'."
                )

            if (
                fault.profile == "gradual"
                and fault.ramp_duration_h <= 0.0
            ):
                raise ValueError(
                    "Gradual faults require a positive "
                    "ramp duration."
                )

    def get_state(
        self,
        fault: FaultConfig,
        time_s: float,
    ) -> FaultState:

        time_h = time_s / 3600.0

        # ------------------------------------------
        # Before fault begins
        # ------------------------------------------

        if time_h < fault.start_time_h:

            return FaultState(
                fault_type=fault.fault_type,
                active=False,
                requested_severity=fault.severity,
                progress=0.0,
                effective_severity=0.0,
                label="nominal",
            )

        # ------------------------------------------
        # Abrupt fault
        # ------------------------------------------

        if fault.profile == "abrupt":

            progress = 1.0

        # ------------------------------------------
        # Gradual fault
        # ------------------------------------------

        else:

            elapsed_h = (
                time_h
                - fault.start_time_h
            )

            progress = (
                elapsed_h
                / fault.ramp_duration_h
            )

            progress = max(
                0.0,
                min(1.0, progress),
            )

        effective_severity = (
            fault.severity
            * progress
        )

        active = (
            effective_severity > 0.0
        )

        if active:
            label = fault.fault_type
        else:
            label = "nominal"

        return FaultState(
            fault_type=fault.fault_type,
            active=active,
            requested_severity=fault.severity,
            progress=progress,
            effective_severity=effective_severity,
            label=label,
        )

    def get_all_states(
        self,
        time_s: float,
    ) -> list[FaultState]:

        return [
            self.get_state(
                fault,
                time_s,
            )
            for fault in self.faults
        ]


# ==========================================================
# SIMPLE TEST
# ==========================================================

if __name__ == "__main__":

    solar_fault = FaultConfig(
        fault_type="solar_array_degradation",
        start_time_h=2.0,
        severity=0.30,
        profile="gradual",
        ramp_duration_h=0.5,
    )

    manager = FaultManager(
        faults=[
            solar_fault,
        ]
    )

    test_times_h = [
        0.0,
        1.5,
        2.0,
        2.1,
        2.25,
        2.5,
        3.0,
    ]

    print()
    print("=" * 70)
    print("SENTINEL-XAI - FAULT MANAGER TEST")
    print("=" * 70)

    print(
        "Fault: solar_array_degradation"
    )

    print(
        "Requested severity: 30 %"
    )

    print(
        "Start time: 2.0 h"
    )

    print(
        "Ramp duration: 0.5 h"
    )

    print()
    print(
        f"{'Time [h]':<12}"
        f"{'Active':<10}"
        f"{'Progress':<12}"
        f"{'Effective severity':<20}"
        f"{'Label'}"
    )

    print("-" * 70)

    for time_h in test_times_h:

        state = manager.get_state(
            solar_fault,
            time_s=time_h * 3600.0,
        )

        print(
            f"{time_h:<12.2f}"
            f"{str(state.active):<10}"
            f"{state.progress:<12.2f}"
            f"{state.effective_severity * 100:<20.1f}"
            f"{state.label}"
        )

    print("=" * 70)