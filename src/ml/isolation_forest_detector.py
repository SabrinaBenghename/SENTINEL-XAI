from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestPrediction:
    anomaly_scores: np.ndarray
    is_anomaly: np.ndarray


class IsolationForestAnomalyDetector:
    """
    SENTINEL-XAI unsupervised anomaly detector.

    Training:
        Healthy spacecraft telemetry only.

    Output:
        Larger anomaly score = more abnormal.

    No fault labels are used during training.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
    ):

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )

        self.threshold = None
        self.is_fitted = False

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(
        self,
        x: pd.DataFrame,
    ):

        self.model.fit(
            x
        )

        self.is_fitted = True

        return self

    # ======================================================
    # ANOMALY SCORE
    # ======================================================

    def anomaly_score(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:

        if not self.is_fitted:

            raise RuntimeError(
                "Isolation Forest must be fitted "
                "before anomaly scores are computed."
            )

        # sklearn:
        # lower score_samples() = more abnormal.
        #
        # We invert the sign so SENTINEL uses:
        #
        # higher anomaly score = more abnormal.
        scores = -self.model.score_samples(
            x
        )

        return scores

    # ======================================================
    # THRESHOLD CALIBRATION
    # ======================================================

    def calibrate_threshold(
        self,
        calibration_x: pd.DataFrame,
        quantile: float = 0.99,
    ) -> float:

        if not (
            0.0
            <
            quantile
            <
            1.0
        ):

            raise ValueError(
                "quantile must be between 0 and 1."
            )

        scores = self.anomaly_score(
            calibration_x
        )

        self.threshold = float(
            np.quantile(
                scores,
                quantile,
            )
        )

        return self.threshold

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        x: pd.DataFrame,
    ) -> IsolationForestPrediction:

        if self.threshold is None:

            raise RuntimeError(
                "Anomaly threshold has not been calibrated."
            )

        scores = self.anomaly_score(
            x
        )

        anomalies = (
            scores
            >
            self.threshold
        )

        return IsolationForestPrediction(
            anomaly_scores=scores,
            is_anomaly=anomalies,
        )