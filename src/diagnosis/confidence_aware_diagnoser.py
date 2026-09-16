from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConfidenceAwarePrediction:

    raw_labels: np.ndarray
    final_labels: np.ndarray

    confidence: np.ndarray
    accepted: np.ndarray

    probabilities: np.ndarray


class ConfidenceAwareDiagnoser:
    """
    SENTINEL-XAI confidence-aware diagnosis layer.

    The underlying classifier always produces its most
    likely fault class.

    The confidence-aware layer accepts that diagnosis only
    when confidence is sufficiently high.

    Otherwise:

        final diagnosis = "uncertain"

    This avoids forcing SENTINEL-XAI to claim a specific
    fault type when the diagnostic evidence is ambiguous.
    """

    def __init__(
        self,
        diagnoser,
        confidence_threshold: float = 0.70,
    ):

        if not (
            0.0
            <
            confidence_threshold
            <=
            1.0
        ):

            raise ValueError(
                "confidence_threshold must be in (0, 1]."
            )

        self.diagnoser = diagnoser

        self.confidence_threshold = float(
            confidence_threshold
        )

    def predict(
        self,
        x: pd.DataFrame,
    ) -> ConfidenceAwarePrediction:

        prediction = (
            self.diagnoser.predict(
                x
            )
        )

        raw_labels = (
            prediction.predicted_labels
        )

        confidence = (
            prediction.confidence
        )

        accepted = (
            confidence
            >=
            self.confidence_threshold
        )

        final_labels = np.where(
            accepted,
            raw_labels,
            "uncertain",
        )

        return ConfidenceAwarePrediction(

            raw_labels=raw_labels,

            final_labels=final_labels,

            confidence=confidence,

            accepted=accepted,

            probabilities=(
                prediction.probabilities
            ),
        )