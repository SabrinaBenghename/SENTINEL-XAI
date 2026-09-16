from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DiagnosisPrediction:

    predicted_labels: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray


class RandomForestDiagnoser:
    """
    SENTINEL-XAI dedicated fault-diagnosis classifier.

    Input:
        Diagnostic evidence from telemetry, rules,
        model-based residuals and subsystem ML scores.

    Output:
        One of seven classes:
            nominal
            solar_array_degradation
            battery_degradation
            thermal_anomaly
            reaction_wheel_degradation
            battery_voltage_sensor_drift
            telemetry_dropout

    Missing telemetry is median-imputed numerically.
    Explicit missing-data indicator features already exist
    in the diagnostic feature vector.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
    ):

        self.pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ):

        self.pipeline.fit(
            x,
            y,
        )

        self.is_fitted = True

        return self

    def predict(
        self,
        x: pd.DataFrame,
    ) -> DiagnosisPrediction:

        if not self.is_fitted:

            raise RuntimeError(
                "Diagnoser must be fitted before prediction."
            )

        labels = self.pipeline.predict(
            x
        )

        probabilities = (
            self.pipeline.predict_proba(
                x
            )
        )

        confidence = np.max(
            probabilities,
            axis=1,
        )

        return DiagnosisPrediction(
            predicted_labels=labels,
            confidence=confidence,
            probabilities=probabilities,
        )

    @property
    def classes_(self):

        return (
            self.pipeline[
                "classifier"
            ].classes_
        )

    @property
    def feature_importances_(self):

        return (
            self.pipeline[
                "classifier"
            ].feature_importances_
        )