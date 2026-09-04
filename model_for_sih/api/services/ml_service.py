import os
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from api.schemas import ThermalEventInput, PredictionResponse, BatchPredictionResponse
from api.services.risk_service import RiskAssessmentService
from api.config import BASE_DIR


class MLModelService:
    """Manages Random Forest model inference, spatial context resolution, and feature extraction."""

    def __init__(self, model_path: Path, feature_path: Path):
        self.model_path = model_path
        self.feature_path = feature_path
        self.worldcover_dir = BASE_DIR / "data" / "raw" / "worldcover" / "india"
        self.dataset_path = BASE_DIR / "data" / "processed" / "ml" / "fire_type_dataset.csv"
        
        self.model = None
        self.features: List[str] = []
        self.classes: List[str] = []
        self._historical_df: Optional[pd.DataFrame] = None
        
        self.load()

    def load(self):
        """Loads the serialized model, feature columns, and historical dataset index."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature file not found at {self.feature_path}")

        # Load trained Random Forest model
        self.model = joblib.load(str(self.model_path))

        # Load feature order
        with open(self.feature_path, "r", encoding="utf-8") as f:
            self.features = [line.strip() for line in f if line.strip()]

        # Model target classes
        self.classes = [str(cls) for cls in self.model.classes_]

        # Load historical dataset index if available
        if self.dataset_path.exists():
            try:
                df = pd.read_csv(self.dataset_path)
                if "event_cluster_id" in df.columns:
                    df["event_cluster_id"] = df["event_cluster_id"].astype(int)
                    self._historical_df = df.set_index("event_cluster_id")
            except Exception as e:
                print(f"Warning: unable to index historical dataset: {e}")

        print(f"✓ ML Model loaded successfully: {len(self.classes)} classes, {len(self.features)} features")

    def _get_worldcover_class(self, lat: float, lon: float) -> int:
        """Looks up the real ESA WorldCover landcover classification code for given coordinates."""
        lat_tile = math.floor(lat / 3.0) * 3
        lon_tile = math.floor(lon / 3.0) * 3
        lat_str = f"N{int(lat_tile):02d}" if lat_tile >= 0 else f"S{abs(int(lat_tile)):02d}"
        lon_str = f"E{int(lon_tile):03d}" if lon_tile >= 0 else f"W{abs(int(lon_tile)):03d}"

        tif_path = self.worldcover_dir / f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"

        if tif_path.exists():
            try:
                with rasterio.open(tif_path) as src:
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                        if val != src.nodata and val in [10, 20, 30, 40, 50, 60, 70, 80]:
                            return int(val)
            except Exception:
                pass

        # Geographic biome fallback for coordinates across India
        # Forest belts: Western Ghats, Northeast, Himalayan foothills, Central Indian forests
        if (lat >= 26.0 and lon >= 88.0) or (8.0 <= lat <= 16.0 and 74.0 <= lon <= 77.5) or (19.0 <= lat <= 24.0 and 80.0 <= lon <= 85.0):
            return 10  # Tree cover / Forest
        # Agricultural heartlands (Indo-Gangetic plains, Punjab, Haryana, UP, Bihar)
        elif (24.0 <= lat <= 32.0 and 74.0 <= lon <= 88.0):
            return 40  # Cropland
        # Mining belts (Jharkhand, Odisha, Chhattisgarh mining clusters)
        elif (21.5 <= lat <= 24.5 and 83.5 <= lon <= 87.0):
            return 60  # Barren / Mining / Open land
        # Arid regions (Rajasthan / Gujarat open lands)
        elif (lon <= 73.0 and lat >= 23.0):
            return 60  # Barren

        return 40  # Default rural/cropland

    def build_feature_dict(self, event: ThermalEventInput) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Transforms high-level or raw satellite input into the exact 41 numerical features."""
        lat = float(event.latitude)
        lon = float(event.longitude)

        # 0. Check if this is a historical event cluster in our pre-indexed dataset
        if event.event_id and event.event_id.startswith("hist_") and self._historical_df is not None:
            try:
                cluster_id = int(event.event_id.replace("hist_", ""))
                if cluster_id in self._historical_df.index:
                    row = self._historical_df.loc[cluster_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    
                    feature_map = {col: float(row.get(col, 0.0)) for col in self.features}
                    summary = {
                        "avg_frp_mw": round(float(row.get("avg_frp", 15.0)), 2),
                        "max_frp_mw": round(float(row.get("max_frp", 15.0)), 2),
                        "bright_ti4_k": round(float(row.get("avg_bright_ti4", 335.0)), 1),
                        "bright_ti5_k": round(float(row.get("avg_bright_ti5", 300.0)), 1),
                        "total_detections": int(row.get("total_detections", 1)),
                        "worldcover_class": int(row.get("worldcover_class", 10)),
                        "fire_type": str(row.get("fire_type", "UNKNOWN_THERMAL_EVENT")),
                        "acq_date": event.acq_date or str(row.get("first_detection", "")),
                        "satellite": event.satellite or "VIIRS Archive"
                    }
                    return feature_map, summary
            except Exception:
                pass

        # 1. Resolve FRP metrics
        base_frp = event.frp if event.frp is not None else 15.0
        avg_frp = event.avg_frp if event.avg_frp is not None else base_frp
        max_frp = event.max_frp if event.max_frp is not None else avg_frp
        min_frp = event.min_frp if event.min_frp is not None else avg_frp
        frp_std = event.frp_std if event.frp_std is not None else 0.0

        # 2. Brightness temperatures
        avg_ti4 = event.avg_bright_ti4 if event.avg_bright_ti4 is not None else (event.bright_ti4 or 335.0)
        max_ti4 = event.max_bright_ti4 if event.max_bright_ti4 is not None else (event.bright_ti4 or 345.0)
        avg_ti5 = event.avg_bright_ti5 if event.avg_bright_ti5 is not None else (event.bright_ti5 or 300.0)

        # 3. Detection & Persistence metrics
        total_detections = max(event.total_detections or 1, 1)
        detection_days = max(event.detection_days or 1, 1)
        observation_days = max(event.observation_days or 365, 1)

        night_detections = event.night_detections if event.night_detections is not None else (1 if event.daynight == "N" else 0)
        if event.night_detection_ratio is not None:
            night_detection_ratio = float(event.night_detection_ratio)
        else:
            night_detection_ratio = night_detections / total_detections

        persistence_percent = (detection_days / observation_days) * 100.0
        detections_per_active_day = total_detections / detection_days

        # 4. Derived Thermal Features
        frp_range = max_frp - min_frp
        frp_variability = frp_std / max(avg_frp, 1e-6)
        detections_per_obs_day = total_detections / observation_days
        night_activity_intensity = night_detections / total_detections
        high_intensity_indicator = 1 if max_frp >= 25.0 else 0
        persistent_indicator = 1 if persistence_percent >= 1.0 else 0

        # 5. Resolve Real Landcover from GeoTIFF or provided input
        if event.worldcover_class is not None:
            worldcover_class = int(event.worldcover_class)
        else:
            worldcover_class = self._get_worldcover_class(lat, lon)

        # ESA WorldCover: 10: Tree, 20: Shrub, 30: Grass, 40: Crop, 50: Built, 60: Bare
        is_natural = 1 if worldcover_class in [10, 20, 30] else 0
        is_agri = 1 if worldcover_class == 40 else 0
        is_urban = 1 if worldcover_class == 50 else 0
        is_water = 1 if worldcover_class == 80 else 0
        is_barren = 1 if worldcover_class in [60, 70] else 0

        landcover_natural = event.landcover_natural if event.landcover_natural is not None else is_natural
        landcover_agricultural = event.landcover_agricultural if event.landcover_agricultural is not None else is_agri
        landcover_urban = event.landcover_urban if event.landcover_urban is not None else is_urban
        landcover_water = event.landcover_water if event.landcover_water is not None else is_water
        landcover_barren = event.landcover_barren if event.landcover_barren is not None else is_barren
        landcover_valid = 1

        # 6. Context & Infrastructure flags
        # Only mark industrial if explicitly built-up / industrial
        is_industrial_area = 1 if (landcover_urban == 1 and (21.0 <= lat <= 23.5 or 12.0 <= lat <= 13.5 or 18.5 <= lat <= 20.0)) else 0
        is_mining_area = 1 if (21.0 <= lat <= 24.5 and 82.0 <= lon <= 87.0 and (landcover_barren == 1 or worldcover_class in [30, 60])) else 0

        context_agricultural = event.context_agricultural if event.context_agricultural is not None else (1 if landcover_agricultural == 1 else 0)
        context_industrial = event.context_industrial if event.context_industrial is not None else (1 if is_industrial_area else 0)
        context_mining = event.context_mining if event.context_mining is not None else (1 if is_mining_area else 0)
        context_rural = event.context_rural if event.context_rural is not None else (1 if (landcover_agricultural == 1 or landcover_natural == 1) and not is_industrial_area else 0)

        industrial_presence = event.industrial_presence if event.industrial_presence is not None else (1 if (context_industrial == 1 or is_industrial_area) else 0)
        infrastructure_presence = event.infrastructure_presence if event.infrastructure_presence is not None else (1 if (is_industrial_area or is_mining_area) else 0)
        facility_presence = event.facility_presence if event.facility_presence is not None else (1 if is_industrial_area else 0)
        nearby_highway = event.nearby_highway if event.nearby_highway is not None else (1 if landcover_urban == 1 else 0)
        nearby_railway = event.nearby_railway if event.nearby_railway is not None else (1 if is_industrial_area else 0)
        nearby_power = event.nearby_power_infrastructure if event.nearby_power_infrastructure is not None else (1 if (is_industrial_area or is_mining_area) else 0)
        nearby_mining = event.nearby_mining if event.nearby_mining is not None else (1 if is_mining_area else 0)
        mixed_industrial_infrastructure = event.mixed_industrial_infrastructure if event.mixed_industrial_infrastructure is not None else (1 if (industrial_presence and infrastructure_presence) else 0)

        # Assemble the full feature mapping
        feature_map = {
            "latitude": lat,
            "longitude": lon,
            "total_detections": float(total_detections),
            "detection_days": float(detection_days),
            "avg_frp": float(avg_frp),
            "max_frp": float(max_frp),
            "min_frp": float(min_frp),
            "frp_std": float(frp_std),
            "avg_bright_ti4": float(avg_ti4),
            "max_bright_ti4": float(max_ti4),
            "avg_bright_ti5": float(avg_ti5),
            "night_detections": float(night_detections),
            "night_detection_ratio": float(night_detection_ratio),
            "observation_days": float(observation_days),
            "persistence_percent": float(persistence_percent),
            "detections_per_active_day": float(detections_per_active_day),
            "frp_range": float(frp_range),
            "frp_variability": float(frp_variability),
            "detections_per_observation_day": float(detections_per_obs_day),
            "night_activity_intensity": float(night_activity_intensity),
            "high_intensity_indicator": int(high_intensity_indicator),
            "persistent_indicator": int(persistent_indicator),
            "landcover_natural": int(landcover_natural),
            "landcover_agricultural": int(landcover_agricultural),
            "landcover_urban": int(landcover_urban),
            "landcover_water": int(landcover_water),
            "landcover_barren": int(landcover_barren),
            "landcover_valid": int(landcover_valid),
            "worldcover_class": int(worldcover_class),
            "context_agricultural": int(context_agricultural),
            "context_industrial": int(context_industrial),
            "context_mining": int(context_mining),
            "context_rural": int(context_rural),
            "industrial_presence": int(industrial_presence),
            "infrastructure_presence": int(infrastructure_presence),
            "facility_presence": int(facility_presence),
            "nearby_highway": int(nearby_highway),
            "nearby_railway": int(nearby_railway),
            "nearby_power_infrastructure": int(nearby_power),
            "nearby_mining": int(nearby_mining),
            "mixed_industrial_infrastructure": int(mixed_industrial_infrastructure)
        }

        summary = {
            "avg_frp_mw": round(avg_frp, 2),
            "max_frp_mw": round(max_frp, 2),
            "bright_ti4_k": round(avg_ti4, 1),
            "bright_ti5_k": round(avg_ti5, 1),
            "night_detection_pct": round(night_detection_ratio * 100.0, 1),
            "total_detections": total_detections,
            "worldcover_class": worldcover_class,
            "acq_date": event.acq_date,
            "acq_time": event.acq_time,
            "satellite": event.satellite
        }

        return feature_map, summary

    def predict_single(self, event: ThermalEventInput) -> PredictionResponse:
        """Runs inference for a single thermal event."""
        feature_map, summary = self.build_feature_dict(event)

        # Build single-row DataFrame aligned with feature list
        row_data = [feature_map.get(col, 0.0) for col in self.features]
        df_input = pd.DataFrame([row_data], columns=self.features)

        # Model inference
        predicted_class = str(self.model.predict(df_input)[0])
        probabilities = self.model.predict_proba(df_input)[0]

        # Class probabilities mapping
        class_probs = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(self.model.classes_, probabilities)
        }
        confidence = float(max(probabilities))

        # Risk score calculation
        context_flags = {
            "industrial_presence": int(feature_map["industrial_presence"]),
            "nearby_mining": int(feature_map["nearby_mining"]),
            "nearby_power_infrastructure": int(feature_map["nearby_power_infrastructure"]),
            "nearby_railway": int(feature_map["nearby_railway"]),
            "nearby_highway": int(feature_map["nearby_highway"]),
            "facility_presence": int(feature_map["facility_presence"]),
            "mixed_industrial_infrastructure": int(feature_map["mixed_industrial_infrastructure"]),
            "context_industrial": int(feature_map["context_industrial"]),
            "context_mining": int(feature_map["context_mining"])
        }

        risk_score, risk_level, risk_breakdown = RiskAssessmentService.calculate_risk(
            max_frp=feature_map["max_frp"],
            total_detections=int(feature_map["total_detections"]),
            persistence_percent=feature_map["persistence_percent"],
            night_detection_ratio=feature_map["night_detection_ratio"],
            detections_per_active_day=feature_map["detections_per_active_day"],
            context_flags=context_flags
        )

        summary["risk_breakdown"] = risk_breakdown

        return PredictionResponse(
            event_id=event.event_id,
            latitude=event.latitude,
            longitude=event.longitude,
            predicted_fire_type=predicted_class,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary
        )

    def predict_batch(self, events: List[ThermalEventInput]) -> BatchPredictionResponse:
        """Runs high-performance vectorized inference over multiple thermal events."""
        if not events:
            return BatchPredictionResponse(total_events=0, results=[])

        feature_maps = []
        summaries = []

        for event in events:
            fmap, summ = self.build_feature_dict(event)
            feature_maps.append(fmap)
            summaries.append(summ)

        # Construct batch DataFrame
        rows = [[fmap.get(col, 0.0) for col in self.features] for fmap in feature_maps]
        df_batch = pd.DataFrame(rows, columns=self.features)

        # Batch Model inference
        predictions = self.model.predict(df_batch)
        probabilities = self.model.predict_proba(df_batch)

        results = []
        for i, event in enumerate(events):
            pred_class = str(predictions[i])
            probs = probabilities[i]
            confidence = float(max(probs))
            class_probs = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(self.model.classes_, probs)
            }

            fmap = feature_maps[i]
            context_flags = {
                "industrial_presence": int(fmap["industrial_presence"]),
                "nearby_mining": int(fmap["nearby_mining"]),
                "nearby_power_infrastructure": int(fmap["nearby_power_infrastructure"]),
                "nearby_railway": int(fmap["nearby_railway"]),
                "nearby_highway": int(fmap["nearby_highway"]),
                "facility_presence": int(fmap["facility_presence"]),
                "mixed_industrial_infrastructure": int(fmap["mixed_industrial_infrastructure"]),
                "context_industrial": int(fmap["context_industrial"]),
                "context_mining": int(fmap["context_mining"])
            }

            risk_score, risk_level, risk_breakdown = RiskAssessmentService.calculate_risk(
                max_frp=fmap["max_frp"],
                total_detections=int(fmap["total_detections"]),
                persistence_percent=fmap["persistence_percent"],
                night_detection_ratio=fmap["night_detection_ratio"],
                detections_per_active_day=fmap["detections_per_active_day"],
                context_flags=context_flags
            )

            summ = summaries[i]
            summ["risk_breakdown"] = risk_breakdown

            results.append(
                PredictionResponse(
                    event_id=event.event_id,
                    latitude=event.latitude,
                    longitude=event.longitude,
                    predicted_fire_type=pred_class,
                    confidence=round(confidence, 4),
                    class_probabilities=class_probs,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    summary=summ
                )
            )

        return BatchPredictionResponse(total_events=len(results), results=results)
