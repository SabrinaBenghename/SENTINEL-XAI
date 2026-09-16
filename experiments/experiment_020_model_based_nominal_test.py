from pathlib import Path
import sys

import pandas as pd


project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(
        0,
        str(project_root),
    )


from src.detection.model_based_detector import (
    ModelBasedDetector,
)


def main():

    input_file = (
        project_root
        / "data"
        / "nominal"
        / "nominal_spacecraft_telemetry.csv"
    )

    df = pd.read_csv(
        input_file
    )

    detector = ModelBasedDetector()

    detections = []

    for index, row in df.iterrows():

        if index == 0:
            dt_s = 60.0
        else:
            dt_s = (
                row["time_s"]
                - df.iloc[index - 1]["time_s"]
            )

        result = detector.detect(
            row=row,
            dt_s=dt_s,
        )

        if result.detected:

            detections.append(
                {
                    "time_h":
                        row["time_h"],

                    "fault":
                        result.predicted_fault,
                }
            )

    print()
    print("=" * 72)
    print("SENTINEL-XAI - EXPERIMENT 020")
    print("MODEL-BASED DETECTOR NOMINAL TEST")
    print("=" * 72)

    print(
        f"Nominal telemetry samples: "
        f"{len(df)}"
    )

    print(
        f"False alarms: "
        f"{len(detections)}"
    )

    if len(detections) == 0:

        print(
            "RESULT: NOMINAL TEST PASSED"
        )

    else:

        print(
            "RESULT: FALSE ALARMS DETECTED"
        )

        print()
        print("First detections:")

        for detection in detections[:10]:

            print(
                detection
            )

    print("=" * 72)


if __name__ == "__main__":
    main()