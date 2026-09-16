from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ==========================================================
# SUBSYSTEM FEATURE GROUPS
# ==========================================================

SUBSYSTEM_FEATURES = {

    "power": [
        "in_sunlight",
        "solar_power_w",
        "load_power_w",
        "battery_current_measured",
        "battery_soc_rate",
    ],

    "battery": [
        "battery_soc_measured",
        "battery_voltage_measured",
        "battery_current_measured",
        "battery_soc_rate",
        "battery_voltage_rate",
    ],

    "thermal": [
        "battery_temp_measured",
        "electronics_temp_measured",
        "battery_temp_rate",
        "electronics_temp_rate",
        "battery_current_measured",
        "in_sunlight",
    ],

    "wheel": [
        "wheel_target_speed_rpm",
        "wheel_speed_measured",
        "wheel_current_measured",
        "wheel_temp_measured",
        "wheel_speed_rate",
        "wheel_current_rate",
        "wheel_temp_rate",
        "wheel_tracking_error_rpm",
    ],
}


@dataclass
class SubsystemPrediction:

    subsystem_scores: dict
    subsystem_anomalies: dict

    maximum_score_ratio: np.ndarray
    raw_anomaly: np.ndarray


class SubsystemIsolationForest:

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
    ):

        self.n_estimators = n_estimators
        self.random_state = random_state

        self.models = {}
        self.thresholds = {}

        for index, subsystem in enumerate(
            SUBSYSTEM_FEATURES
        ):

            self.models[subsystem] = (
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination="auto",
                    random_state=(
                        random_state
                        + index
                    ),
                    n_jobs=-1,
                )
            )

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(
        self,
        feature_df: pd.DataFrame,
    ):

        for subsystem, columns in (
            SUBSYSTEM_FEATURES.items()
        ):

            x = feature_df[
                columns
            ]

            self.models[
                subsystem
            ].fit(
                x
            )

        return self

    # ======================================================
    # SCORE ONE SUBSYSTEM
    # ======================================================

    def subsystem_score(
        self,
        feature_df,
        subsystem,
    ):

        columns = (
            SUBSYSTEM_FEATURES[
                subsystem
            ]
        )

        return -self.models[
            subsystem
        ].score_samples(
            feature_df[
                columns
            ]
        )

    # ======================================================
    # CALIBRATION
    # ======================================================

    def calibrate(
        self,
        calibration_df,
        quantile=0.99,
    ):

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            scores = (
                self.subsystem_score(
                    calibration_df,
                    subsystem,
                )
            )

            self.thresholds[
                subsystem
            ] = float(
                np.quantile(
                    scores,
                    quantile,
                )
            )

        return self.thresholds

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        feature_df,
    ):

        if not self.thresholds:

            raise RuntimeError(
                "Detector has not been calibrated."
            )

        subsystem_scores = {}
        subsystem_anomalies = {}

        score_ratios = []

        for subsystem in (
            SUBSYSTEM_FEATURES
        ):

            scores = (
                self.subsystem_score(
                    feature_df,
                    subsystem,
                )
            )

            threshold = (
                self.thresholds[
                    subsystem
                ]
            )

            anomalies = (
                scores > threshold
            )

            subsystem_scores[
                subsystem
            ] = scores

            subsystem_anomalies[
                subsystem
            ] = anomalies

            score_ratios.append(
                scores
                /
                threshold
            )

        ratio_matrix = np.vstack(
            score_ratios
        )

        maximum_score_ratio = (
            np.max(
                ratio_matrix,
                axis=0,
            )
        )

        raw_anomaly = (
            maximum_score_ratio > 1.0
        )

        return SubsystemPrediction(
            subsystem_scores=(
                subsystem_scores
            ),

            subsystem_anomalies=(
                subsystem_anomalies
            ),

            maximum_score_ratio=(
                maximum_score_ratio
            ),

            raw_anomaly=(
                raw_anomaly
            ),
        )