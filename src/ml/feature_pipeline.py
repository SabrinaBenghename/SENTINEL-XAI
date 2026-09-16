from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ==========================================================
# FEATURE NAMES
# ==========================================================

BASE_FEATURES = [

    "in_sunlight",

    "solar_power_w",
    "load_power_w",

    "battery_soc_measured",
    "battery_voltage_measured",
    "battery_current_measured",

    "battery_temp_measured",
    "electronics_temp_measured",

    "wheel_target_speed_rpm",
    "wheel_speed_measured",
    "wheel_current_measured",
    "wheel_temp_measured",
]


DERIVED_FEATURES = [

    "battery_soc_rate",
    "battery_voltage_rate",
    "battery_temp_rate",
    "electronics_temp_rate",

    "wheel_speed_rate",
    "wheel_current_rate",
    "wheel_temp_rate",

    "wheel_tracking_error_rpm",
]


ALL_FEATURES = (
    BASE_FEATURES
    + DERIVED_FEATURES
)


# ==========================================================
# SPLIT CONTAINER
# ==========================================================

@dataclass
class NominalMLSplit:

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


# ==========================================================
# RATE HELPER
# ==========================================================

def calculate_rate(
    values: pd.Series,
    time_s: pd.Series,
) -> pd.Series:

    value_difference = (
        values.diff()
    )

    time_difference = (
        time_s.diff()
    )

    rate = (
        value_difference
        / time_difference
    )

    # First sample has no previous measurement.
    rate.iloc[0] = 0.0

    # Protect against invalid divisions.
    rate = rate.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return rate


# ==========================================================
# FEATURE BUILDER
# ==========================================================

def build_ml_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    feature_df = pd.DataFrame(
        index=df.index
    )

    # ======================================================
    # RAW OBSERVABLE TELEMETRY
    # ======================================================

    for column in BASE_FEATURES:

        if column not in df.columns:

            raise KeyError(
                f"Required telemetry column missing: "
                f"{column}"
            )

        feature_df[column] = (
            df[column]
        )

    # ======================================================
    # TEMPORAL FEATURES
    # ======================================================

    time_s = (
        df["time_s"]
    )

    feature_df[
        "battery_soc_rate"
    ] = calculate_rate(
        df["battery_soc_measured"],
        time_s,
    )

    feature_df[
        "battery_voltage_rate"
    ] = calculate_rate(
        df["battery_voltage_measured"],
        time_s,
    )

    feature_df[
        "battery_temp_rate"
    ] = calculate_rate(
        df["battery_temp_measured"],
        time_s,
    )

    feature_df[
        "electronics_temp_rate"
    ] = calculate_rate(
        df["electronics_temp_measured"],
        time_s,
    )

    feature_df[
        "wheel_speed_rate"
    ] = calculate_rate(
        df["wheel_speed_measured"],
        time_s,
    )

    feature_df[
        "wheel_current_rate"
    ] = calculate_rate(
        df["wheel_current_measured"],
        time_s,
    )

    feature_df[
        "wheel_temp_rate"
    ] = calculate_rate(
        df["wheel_temp_measured"],
        time_s,
    )

    # ======================================================
    # SIMPLE COMMAND-TRACKING FEATURE
    # ======================================================

    feature_df[
        "wheel_tracking_error_rpm"
    ] = (
        df["wheel_speed_measured"]
        - df["wheel_target_speed_rpm"]
    )

    # ======================================================
    # PRESERVE TIME FOR ANALYSIS
    #
    # Time itself is NOT used as an ML feature.
    # ======================================================

    feature_df.insert(
        0,
        "time_h",
        df["time_h"],
    )

    return feature_df


# ==========================================================
# NOMINAL SPLIT
# ==========================================================

def split_nominal_dataset(
    feature_df: pd.DataFrame,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> NominalMLSplit:

    number_of_samples = len(
        feature_df
    )

    train_end = int(
        number_of_samples
        * train_fraction
    )

    calibration_end = int(
        number_of_samples
        * (
            train_fraction
            + calibration_fraction
        )
    )

    train = (
        feature_df
        .iloc[
            :train_end
        ]
        .copy()
    )

    calibration = (
        feature_df
        .iloc[
            train_end:
            calibration_end
        ]
        .copy()
    )

    test = (
        feature_df
        .iloc[
            calibration_end:
        ]
        .copy()
    )

    return NominalMLSplit(
        train=train,
        calibration=calibration,
        test=test,
    )


# ==========================================================
# ML MATRIX
# ==========================================================

def feature_matrix(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        feature_df[
            ALL_FEATURES
        ]
        .copy()
    )