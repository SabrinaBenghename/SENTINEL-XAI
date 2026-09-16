from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PersistentAnomalyResult:

    raw_anomaly: bool
    persistent_anomaly: bool
    consecutive_count: int


class PersistentAnomalyFilter:
    """
    Temporal persistence layer for SENTINEL-XAI.

    A fault alarm is produced only when the raw anomaly
    condition remains active for several consecutive samples.
    """

    def __init__(
        self,
        required_samples: int = 3,
    ):

        if required_samples < 1:

            raise ValueError(
                "required_samples must be >= 1"
            )

        self.required_samples = (
            required_samples
        )

        self.counter = 0

    def reset(self):

        self.counter = 0

    def update(
        self,
        raw_anomaly: bool,
    ) -> PersistentAnomalyResult:

        if raw_anomaly:

            self.counter += 1

        else:

            self.counter = 0

        persistent = (
            self.counter
            >= self.required_samples
        )

        return PersistentAnomalyResult(

            raw_anomaly=bool(
                raw_anomaly
            ),

            persistent_anomaly=bool(
                persistent
            ),

            consecutive_count=int(
                self.counter
            ),
        )