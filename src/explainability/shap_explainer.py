from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class SentinelShapExplainer:

    def __init__(
        self,
        diagnoser,
        feature_names,
    ):

        self.diagnoser = diagnoser

        self.feature_names = list(
            feature_names
        )

        # --------------------------------------------------
        # Extract preprocessing + classifier from the
        # frozen Phase-7 sklearn pipeline.
        # --------------------------------------------------

        self.imputer = (
            diagnoser.pipeline[
                "imputer"
            ]
        )

        self.classifier = (
            diagnoser.pipeline[
                "classifier"
            ]
        )

        self.classes = list(
            self.classifier.classes_
        )

        # --------------------------------------------------
        # TreeExplainer is exact/efficient for Random Forest.
        # --------------------------------------------------

        self.explainer = (
            shap.TreeExplainer(
                self.classifier
            )
        )

    # ======================================================
    # PREPROCESS
    # ======================================================

    def transform(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:

        transformed = (
            self.imputer.transform(
                x
            )
        )

        return pd.DataFrame(
            transformed,
            columns=self.feature_names,
            index=x.index,
        )

    # ======================================================
    # EXPLAIN
    # ======================================================

    def explain(
        self,
        x: pd.DataFrame,
    ):

        transformed = (
            self.transform(
                x
            )
        )

        explanation = (
            self.explainer(
                transformed
            )
        )

        return (
            transformed,
            explanation,
        )

    # ======================================================
    # CLASS INDEX
    # ======================================================

    def class_index(
        self,
        class_name,
    ):

        return self.classes.index(
            class_name
        )

    # ======================================================
    # EXTRACT CLASS-SPECIFIC SHAP VALUES
    # ======================================================

    def class_values(
        self,
        explanation,
        class_name,
    ):

        values = (
            explanation.values
        )

        class_index = (
            self.class_index(
                class_name
            )
        )

        # SHAP 0.52 multiclass tree output:
        #
        # samples x features x classes

        if values.ndim == 3:

            return values[
                :,
                :,
                class_index,
            ]

        # Safety fallback.

        if values.ndim == 2:

            return values

        raise ValueError(
            f"Unexpected SHAP value shape: "
            f"{values.shape}"
        )

    # ======================================================
    # BASE VALUE FOR ONE SAMPLE / CLASS
    # ======================================================

    def base_value(
        self,
        explanation,
        sample_index,
        class_name,
    ):

        base_values = (
            explanation.base_values
        )

        class_index = (
            self.class_index(
                class_name
            )
        )

        if np.ndim(
            base_values
        ) == 2:

            return float(
                base_values[
                    sample_index,
                    class_index,
                ]
            )

        if np.ndim(
            base_values
        ) == 1:

            if len(
                base_values
            ) == len(
                self.classes
            ):

                return float(
                    base_values[
                        class_index
                    ]
                )

            return float(
                base_values[
                    sample_index
                ]
            )

        return float(
            base_values
        )