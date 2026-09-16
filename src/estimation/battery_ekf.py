from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BatteryEKFState:
    time_s: float

    soc_predicted: float
    soc_estimated: float

    voltage_predicted_v: float
    voltage_measured_v: float

    innovation_v: float

    covariance_predicted: float
    covariance_updated: float

    kalman_gain: float


class BatteryEKF:
    """
    Scalar EKF for battery state-of-charge estimation.

    State:
        x = SOC fraction [0, 1]

    Process model:
        SOC(k+1) =
            SOC(k)
            - P_battery * dt / battery_capacity

    Measurement model:
        V = 26 + 4 * SOC

    Although the current measurement model is linear,
    the implementation follows the EKF prediction /
    innovation / correction structure so it can later
    be extended to nonlinear battery models.
    """

    def __init__(
        self,
        initial_soc: float = 0.85,
        initial_covariance: float = 0.01,
        battery_capacity_wh: float = 1120.0,
        process_noise: float = 1e-6,
        voltage_noise_std_v: float = 0.03,
    ):

        self.soc = float(initial_soc)

        self.P = float(
            initial_covariance
        )

        self.battery_capacity_wh = float(
            battery_capacity_wh
        )

        self.Q = float(
            process_noise
        )

        self.R = float(
            voltage_noise_std_v ** 2
        )

    # ======================================================
    # BATTERY VOLTAGE MODEL
    # ======================================================

    @staticmethod
    def voltage_model(
        soc: float,
    ) -> float:

        return (
            26.0
            + 4.0 * soc
        )

    # ======================================================
    # EKF STEP
    # ======================================================

    def step(
        self,
        time_s: float,
        dt_s: float,
        battery_current_a: float,
        battery_voltage_measured_v: float,
    ) -> BatteryEKFState:

        # ==================================================
        # 1. ESTIMATE BATTERY POWER
        #
        # P = V * I
        #
        # Positive current:
        # battery discharging
        #
        # Negative current:
        # battery charging
        # ==================================================

        battery_power_w = (
            battery_voltage_measured_v
            * battery_current_a
        )

        # ==================================================
        # 2. PREDICTION
        # ==================================================

        soc_change = (
            -battery_power_w
            * dt_s
            / 3600.0
            / self.battery_capacity_wh
        )

        soc_predicted = (
            self.soc
            + soc_change
        )

        soc_predicted = float(
            np.clip(
                soc_predicted,
                0.0,
                1.0,
            )
        )

        # State-transition Jacobian
        F = 1.0

        covariance_predicted = (
            F
            * self.P
            * F
            + self.Q
        )

        # ==================================================
        # 3. PREDICT MEASUREMENT
        # ==================================================

        voltage_predicted_v = (
            self.voltage_model(
                soc_predicted
            )
        )

        # Measurement Jacobian:
        #
        # dV / dSOC = 4
        H = 4.0

        # ==================================================
        # 4. INNOVATION
        #
        # residual = measurement - prediction
        # ==================================================

        innovation_v = (
            battery_voltage_measured_v
            - voltage_predicted_v
        )

        innovation_covariance = (
            H
            * covariance_predicted
            * H
            + self.R
        )

        # ==================================================
        # 5. KALMAN GAIN
        # ==================================================

        kalman_gain = (
            covariance_predicted
            * H
            / innovation_covariance
        )

        # ==================================================
        # 6. CORRECTION
        # ==================================================

        soc_updated = (
            soc_predicted
            + kalman_gain
            * innovation_v
        )

        soc_updated = float(
            np.clip(
                soc_updated,
                0.0,
                1.0,
            )
        )

        covariance_updated = (
            1.0
            - kalman_gain
            * H
        ) * covariance_predicted

        # ==================================================
        # 7. SAVE FILTER STATE
        # ==================================================

        self.soc = (
            soc_updated
        )

        self.P = (
            covariance_updated
        )

        return BatteryEKFState(
            time_s=time_s,

            soc_predicted=soc_predicted,

            soc_estimated=soc_updated,

            voltage_predicted_v=(
                voltage_predicted_v
            ),

            voltage_measured_v=(
                battery_voltage_measured_v
            ),

            innovation_v=(
                innovation_v
            ),

            covariance_predicted=(
                covariance_predicted
            ),

            covariance_updated=(
                covariance_updated
            ),

            kalman_gain=(
                kalman_gain
            ),
        )


# ==========================================================
# NOMINAL VALIDATION
# ==========================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    output_file = (
        project_root
        / "results"
        / "tables"
        / "battery_ekf_nominal.csv"
    )

    df = pd.read_csv(
        input_file
    )

    # Intentionally start with an imperfect SOC estimate.
    #
    # True simulator starts near 90%.
    # EKF starts at 85%.
    #
    # We want to see whether measurements make it converge.
    ekf = BatteryEKF(
        initial_soc=0.85,
        initial_covariance=0.01,
        battery_capacity_wh=1120.0,
        process_noise=1e-6,
        voltage_noise_std_v=0.03,
    )

    records = []

    for index, row in df.iterrows():

        if index == 0:
            dt_s = 60.0
        else:
            dt_s = (
                row["time_s"]
                - df.iloc[
                    index - 1
                ]["time_s"]
            )

        state = ekf.step(
            time_s=row["time_s"],

            dt_s=dt_s,

            battery_current_a=(
                row[
                    "battery_current_measured"
                ]
            ),

            battery_voltage_measured_v=(
                row[
                    "battery_voltage_measured"
                ]
            ),
        )

        true_soc = (
            row["battery_soc_true"]
            / 100.0
        )

        soc_error = (
            state.soc_estimated
            - true_soc
        )

        records.append(
            {
                "time_s":
                    row["time_s"],

                "time_h":
                    row["time_h"],

                "soc_true":
                    true_soc,

                "soc_predicted":
                    state.soc_predicted,

                "soc_estimated":
                    state.soc_estimated,

                "soc_error":
                    soc_error,

                "voltage_predicted_v":
                    state.voltage_predicted_v,

                "voltage_measured_v":
                    state.voltage_measured_v,

                "innovation_v":
                    state.innovation_v,

                "kalman_gain":
                    state.kalman_gain,

                "covariance":
                    state.covariance_updated,
            }
        )

    result_df = pd.DataFrame(
        records
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    # ======================================================
    # METRICS
    # ======================================================

    rmse_soc = float(
        np.sqrt(
            np.mean(
                result_df[
                    "soc_error"
                ] ** 2
            )
        )
    )

    mae_soc = float(
        np.mean(
            np.abs(
                result_df[
                    "soc_error"
                ]
            )
        )
    )

    mean_innovation = float(
        result_df[
            "innovation_v"
        ].mean()
    )

    innovation_std = float(
        result_df[
            "innovation_v"
        ].std()
    )

    initial_error = float(
        abs(
            result_df[
                "soc_error"
            ].iloc[0]
        )
    )

    final_error = float(
        abs(
            result_df[
                "soc_error"
            ].iloc[-1]
        )
    )

    print()
    print("=" * 72)

    print(
        "SENTINEL-XAI - BATTERY EKF NOMINAL TEST"
    )

    print("=" * 72)

    print(
        f"Initial SOC estimation error: "
        f"{initial_error * 100.0:.3f} percentage points"
    )

    print(
        f"Final SOC estimation error: "
        f"{final_error * 100.0:.3f} percentage points"
    )

    print()

    print(
        f"SOC RMSE: "
        f"{rmse_soc * 100.0:.3f} percentage points"
    )

    print(
        f"SOC MAE: "
        f"{mae_soc * 100.0:.3f} percentage points"
    )

    print()

    print(
        f"Mean voltage innovation: "
        f"{mean_innovation:.5f} V"
    )

    print(
        f"Voltage innovation std: "
        f"{innovation_std:.5f} V"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        output_file
    )

    print("=" * 72)


if __name__ == "__main__":
    main()