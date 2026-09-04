import numpy as np
from typing import Dict, Tuple, Any


class RiskAssessmentService:
    """Calculates composite risk score (0-100) and assigns discrete risk levels

    Based on the SIH thermal risk scoring methodology in scripts/create_risk_level.py.
    """

    @staticmethod
    def calculate_risk(
        max_frp: float,
        total_detections: int,
        persistence_percent: float,
        night_detection_ratio: float,
        detections_per_active_day: float,
        context_flags: Dict[str, int]
    ) -> Tuple[float, str, Dict[str, float]]:
        """Compute the weighted multi-factor thermal risk score.

        Returns:
            (risk_score, risk_level, component_breakdown)
        """
        # 1. Thermal Intensity (30% weight) - Scale typical FRP (0 to 60+ MW)
        frp_component = min(max(float(max_frp), 0.0) / 50.0 * 100.0, 100.0)

        # 2. Event Frequency (15% weight) - Scale total detections (1 to 30+)
        freq_component = min(max(float(total_detections), 1.0) / 20.0 * 100.0, 100.0)

        # 3. Persistence (20% weight) - Scale persistence % (0 to 10%)
        pers_component = min(max(float(persistence_percent), 0.0) / 5.0 * 100.0, 100.0)

        # 4. Night Activity (10% weight) - Ratio 0.0 to 1.0
        night_component = min(max(float(night_detection_ratio), 0.0) * 100.0, 100.0)

        # 5. Activity Density (10% weight) - Detections per active day
        density_component = min(max(float(detections_per_active_day), 1.0) / 4.0 * 100.0, 100.0)

        # 6. Infrastructure / Context Score (15% weight)
        ctx_sum = 0.0
        ctx_sum += context_flags.get("industrial_presence", 0) * 15.0
        ctx_sum += context_flags.get("nearby_mining", 0) * 15.0
        ctx_sum += context_flags.get("nearby_power_infrastructure", 0) * 12.0
        ctx_sum += context_flags.get("nearby_railway", 0) * 8.0
        ctx_sum += context_flags.get("nearby_highway", 0) * 5.0
        ctx_sum += context_flags.get("facility_presence", 0) * 8.0
        ctx_sum += context_flags.get("mixed_industrial_infrastructure", 0) * 12.0
        ctx_sum += context_flags.get("context_industrial", 0) * 10.0
        ctx_sum += context_flags.get("context_mining", 0) * 10.0
        ctx_component = min(ctx_sum, 100.0)

        # Weighted combination
        raw_score = (
            (frp_component * 0.30)
            + (freq_component * 0.15)
            + (pers_component * 0.20)
            + (night_component * 0.10)
            + (density_component * 0.10)
            + (ctx_component * 0.15)
        )

        risk_score = round(float(np.clip(raw_score, 0.0, 100.0)), 2)

        # Assign discrete risk level
        if risk_score < 25.0:
            risk_level = "LOW"
        elif risk_score < 50.0:
            risk_level = "MODERATE"
        elif risk_score < 75.0:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        breakdown = {
            "thermal_intensity_score": round(frp_component, 2),
            "event_frequency_score": round(freq_component, 2),
            "persistence_score": round(pers_component, 2),
            "night_activity_score": round(night_component, 2),
            "activity_density_score": round(density_component, 2),
            "infrastructure_context_score": round(ctx_component, 2)
        }

        return risk_score, risk_level, breakdown
