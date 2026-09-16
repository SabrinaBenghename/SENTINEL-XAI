from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridDecision:

    detected: bool
    predicted_fault: str
    confidence: float

    evidence: list[str]
    active_sources: list[str]

    ml_support: bool
    ml_max_score_ratio: float
    ml_dominant_subsystem: str


class HybridFusionEngine:
    """
    SENTINEL-XAI Phase-6 hybrid decision layer.

    Combines:

        Phase 3:
            rule-based detector

        Phase 4:
            refined model-based detector

        Phase 5:
            subsystem Isolation Forest

    Fusion philosophy:

        Physics/model evidence:
            primary

        Deterministic rules:
            trusted complementary evidence

        ML:
            supporting evidence only

    ML is not allowed to create a hard spacecraft
    fault alarm by itself because Phase 5 showed
    insufficient standalone reliability.
    """

    SENSOR_DRIFT = (
        "battery_voltage_sensor_drift"
    )

    BATTERY_DEGRADATION = (
        "battery_degradation"
    )

    TELEMETRY_DROPOUT = (
        "telemetry_dropout"
    )

    def __init__(
        self,
        ml_support_ratio: float = 1.0,
    ):

        self.ml_support_ratio = (
            ml_support_ratio
        )

    # ======================================================
    # ML SUPPORT
    # ======================================================

    def _analyze_ml(
        self,
        ml_persistent_anomaly,
        ml_subsystem_ratios,
    ):

        if not ml_subsystem_ratios:

            return (
                False,
                0.0,
                "none",
            )

        dominant_subsystem = max(
            ml_subsystem_ratios,
            key=ml_subsystem_ratios.get,
        )

        maximum_ratio = float(
            ml_subsystem_ratios[
                dominant_subsystem
            ]
        )

        ml_support = (
            bool(
                ml_persistent_anomaly
            )
            and
            maximum_ratio
            >=
            self.ml_support_ratio
        )

        return (
            ml_support,
            maximum_ratio,
            dominant_subsystem,
        )

    # ======================================================
    # FUSION
    # ======================================================

    def fuse(
        self,
        rule_result,
        model_result,
        ml_persistent_anomaly,
        ml_subsystem_ratios,
    ) -> HybridDecision:

        (
            ml_support,
            ml_max_ratio,
            dominant_subsystem,
        ) = self._analyze_ml(
            ml_persistent_anomaly=(
                ml_persistent_anomaly
            ),
            ml_subsystem_ratios=(
                ml_subsystem_ratios
            ),
        )

        rule_fault = (
            bool(
                rule_result.detected
            )
            and
            rule_result.predicted_fault
            != "nominal"
        )

        model_fault = (
            bool(
                model_result.detected
            )
            and
            model_result.predicted_fault
            != "nominal"
        )

        evidence = []
        sources = []

        # ==================================================
        # 1. TELEMETRY DROPOUT
        #
        # Missing telemetry is strong deterministic evidence.
        # ==================================================

        if (
            (
                rule_fault
                and
                rule_result.predicted_fault
                ==
                self.TELEMETRY_DROPOUT
            )
            or
            (
                model_fault
                and
                model_result.predicted_fault
                ==
                self.TELEMETRY_DROPOUT
            )
        ):

            if rule_fault:

                sources.append(
                    "rules"
                )

                evidence.extend(
                    rule_result.evidence
                )

            if model_fault:

                sources.append(
                    "physics"
                )

                evidence.extend(
                    model_result.evidence
                )

            if ml_support:

                sources.append(
                    "ml"
                )

                evidence.append(
                    (
                        "ML also reports persistent "
                        f"{dominant_subsystem} anomaly "
                        f"(ratio={ml_max_ratio:.3f})"
                    )
                )

            return HybridDecision(

                detected=True,

                predicted_fault=(
                    self.TELEMETRY_DROPOUT
                ),

                confidence=0.99,

                evidence=evidence,

                active_sources=sources,

                ml_support=ml_support,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 2. RULES AND PHYSICS AGREE
        # ==================================================

        if (
            rule_fault
            and
            model_fault
            and
            rule_result.predicted_fault
            ==
            model_result.predicted_fault
        ):

            predicted_fault = (
                model_result.predicted_fault
            )

            sources.extend(
                [
                    "rules",
                    "physics",
                ]
            )

            evidence.append(
                (
                    "Rule-based and model-based "
                    "detectors agree"
                )
            )

            evidence.extend(
                rule_result.evidence
            )

            evidence.extend(
                model_result.evidence
            )

            confidence = 0.98

            if ml_support:

                sources.append(
                    "ml"
                )

                evidence.append(
                    (
                        "ML provides supporting "
                        f"{dominant_subsystem} anomaly "
                        f"evidence "
                        f"(ratio={ml_max_ratio:.3f})"
                    )
                )

                confidence = 0.99

            return HybridDecision(

                detected=True,

                predicted_fault=(
                    predicted_fault
                ),

                confidence=(
                    confidence
                ),

                evidence=evidence,

                active_sources=sources,

                ml_support=ml_support,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 3. RULE / MODEL DISAGREEMENT
        # ==================================================

        if (
            rule_fault
            and
            model_fault
        ):

            # ----------------------------------------------
            # Sensor drift special case.
            #
            # Phase 3 showed very strong sensor-drift
            # discrimination, while battery degradation
            # and voltage drift can look similar to the
            # battery EKF.
            # ----------------------------------------------

            if (
                rule_result.predicted_fault
                ==
                self.SENSOR_DRIFT
            ):

                predicted_fault = (
                    self.SENSOR_DRIFT
                )

                selected_source = (
                    "rules"
                )

                confidence = 0.94

            else:

                # Physics/model detector is the general
                # primary diagnostic source.

                predicted_fault = (
                    model_result.predicted_fault
                )

                selected_source = (
                    "physics"
                )

                confidence = 0.90

            sources.extend(
                [
                    "rules",
                    "physics",
                ]
            )

            evidence.append(
                (
                    "Detector disagreement: "
                    f"rules={rule_result.predicted_fault}, "
                    f"physics={model_result.predicted_fault}"
                )
            )

            evidence.append(
                (
                    f"Fusion selected "
                    f"{selected_source} diagnosis"
                )
            )

            if ml_support:

                sources.append(
                    "ml"
                )

                evidence.append(
                    (
                        "ML provides supporting "
                        f"{dominant_subsystem} anomaly "
                        f"evidence "
                        f"(ratio={ml_max_ratio:.3f})"
                    )
                )

                confidence = min(
                    0.99,
                    confidence + 0.03,
                )

            return HybridDecision(

                detected=True,

                predicted_fault=(
                    predicted_fault
                ),

                confidence=confidence,

                evidence=evidence,

                active_sources=sources,

                ml_support=ml_support,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 4. PHYSICS ONLY
        # ==================================================

        if model_fault:

            sources.append(
                "physics"
            )

            evidence.extend(
                model_result.evidence
            )

            confidence = max(
                0.88,
                float(
                    model_result.confidence
                ),
            )

            if ml_support:

                sources.append(
                    "ml"
                )

                evidence.append(
                    (
                        "ML supports physics alarm: "
                        f"{dominant_subsystem} ratio "
                        f"{ml_max_ratio:.3f}"
                    )
                )

                confidence = min(
                    0.99,
                    confidence + 0.03,
                )

            return HybridDecision(

                detected=True,

                predicted_fault=(
                    model_result.predicted_fault
                ),

                confidence=confidence,

                evidence=evidence,

                active_sources=sources,

                ml_support=ml_support,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 5. RULES ONLY
        # ==================================================

        if rule_fault:

            sources.append(
                "rules"
            )

            evidence.extend(
                rule_result.evidence
            )

            confidence = max(
                0.86,
                float(
                    rule_result.confidence
                ),
            )

            if (
                rule_result.predicted_fault
                ==
                self.SENSOR_DRIFT
            ):

                confidence = max(
                    confidence,
                    0.94,
                )

            if ml_support:

                sources.append(
                    "ml"
                )

                evidence.append(
                    (
                        "ML supports rule-based alarm: "
                        f"{dominant_subsystem} ratio "
                        f"{ml_max_ratio:.3f}"
                    )
                )

                confidence = min(
                    0.99,
                    confidence + 0.03,
                )

            return HybridDecision(

                detected=True,

                predicted_fault=(
                    rule_result.predicted_fault
                ),

                confidence=confidence,

                evidence=evidence,

                active_sources=sources,

                ml_support=ml_support,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 6. ML-ONLY ANOMALY
        #
        # IMPORTANT:
        #
        # Phase-5 ML cannot independently produce a hard
        # spacecraft fault alarm.
        # ==================================================

        if ml_support:

            evidence.append(
                (
                    "Persistent ML anomaly observed "
                    f"in {dominant_subsystem} subsystem "
                    f"(ratio={ml_max_ratio:.3f}), "
                    "but rules and physics remain nominal"
                )
            )

            evidence.append(
                (
                    "ML evidence is supporting-only; "
                    "hard fault alarm suppressed"
                )
            )

            return HybridDecision(

                detected=False,

                predicted_fault="nominal",

                confidence=0.80,

                evidence=evidence,

                active_sources=[
                    "ml_support_only"
                ],

                ml_support=True,

                ml_max_score_ratio=(
                    ml_max_ratio
                ),

                ml_dominant_subsystem=(
                    dominant_subsystem
                ),
            )

        # ==================================================
        # 7. ALL SYSTEMS NOMINAL
        # ==================================================

        return HybridDecision(

            detected=False,

            predicted_fault="nominal",

            confidence=1.0,

            evidence=[
                (
                    "Rules and physics remain nominal; "
                    "no persistent ML support"
                )
            ],

            active_sources=[],

            ml_support=False,

            ml_max_score_ratio=(
                ml_max_ratio
            ),

            ml_dominant_subsystem=(
                dominant_subsystem
            ),
        )