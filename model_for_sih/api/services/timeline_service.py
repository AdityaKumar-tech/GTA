import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from api.config import BASE_DIR
from api.services.risk_service import RiskAssessmentService
from api.services.landmask_service import LandmaskService


class TimelineArchiveService:
    """Provides historical and time-sliced thermal event datasets

    Spanning 6-month and 1-year historical archives across India with pre-computed
    fire types and risk assessments.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self._load_dataset()

    def _load_dataset(self):
        """Loads and pre-indexes the 174,000+ historical events dataset, filtering out ocean points."""
        feat_path = BASE_DIR / "data" / "processed" / "ml" / "fire_type_dataset.csv"
        events_path = BASE_DIR / "data" / "processed" / "events" / "firms_thermal_events.csv"

        if not feat_path.exists() or not events_path.exists():
            print("Warning: Historical datasets not found in data/processed/")
            return

        try:
            print("Loading historical thermal archive for timeline service...")
            df_feat = pd.read_csv(feat_path)
            df_events = pd.read_csv(
                events_path,
                usecols=[
                    "event_cluster_id",
                    "first_detection",
                    "last_detection",
                    "persistence_percent",
                    "night_detection_ratio",
                    "detections_per_active_day"
                ]
            )

            df = df_feat.merge(df_events, on="event_cluster_id", how="left")
            df["first_detection"] = pd.to_datetime(df["first_detection"], errors="coerce")
            df["last_detection"] = pd.to_datetime(df["last_detection"], errors="coerce")

            # Remove all ocean points and confirmed water body points
            initial_count = len(df)
            landmask = LandmaskService.get_instance()
            df = landmask.filter_dataframe(df, lat_col="latitude", lon_col="longitude", remove_water_class=True)
            removed_ocean = initial_count - len(df)

            self.df = df
            print(f"✓ Timeline archive loaded: {len(self.df):,} historical thermal events ({removed_ocean:,} ocean points filtered out)")
        except Exception as e:
            print(f"Error loading timeline archive: {e}")
            self.df = None

    def get_timeline_events(
        self,
        time_preset: str = "180d",
        fire_type_filter: str = "ALL",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = 50000
    ) -> Dict[str, Any]:
        """Filters historical events by time range and fire type, returning GeoJSON."""
        if self.df is None or self.df.empty:
            return {
                "type": "FeatureCollection",
                "count": 0,
                "preset": time_preset,
                "summary": {},
                "features": []
            }

        df_filtered = self.df.copy()

        # Date Filtering
        if start_date and end_date:
            try:
                s_dt = pd.to_datetime(start_date)
                e_dt = pd.to_datetime(end_date)
                df_filtered = df_filtered[
                    (df_filtered["first_detection"] >= s_dt) & (df_filtered["first_detection"] <= e_dt)
                ]
            except Exception:
                pass
        elif time_preset != "all":
            max_dt = df_filtered["first_detection"].max()
            days_map = {
                "24h": 1,
                "7d": 7,
                "30d": 30,
                "90d": 90,
                "180d": 180,
                "365d": 365
            }
            days = days_map.get(time_preset, 30)
            min_dt = max_dt - pd.Timedelta(days=days)
            df_filtered = df_filtered[df_filtered["first_detection"] >= min_dt]

        # Fire Type Filtering
        if fire_type_filter and fire_type_filter.upper() != "ALL":
            target_filter = fire_type_filter.upper()
            if target_filter == "WILDFIRE" or target_filter == "FOREST":
                df_filtered = df_filtered[df_filtered["fire_type"].isin(["FOREST_WILDFIRE", "GRASSLAND_FIRE"])]
            elif target_filter == "CROP" or target_filter == "AGRICULTURAL":
                df_filtered = df_filtered[df_filtered["fire_type"] == "CROP_RESIDUE_FIRE"]
            elif target_filter == "INDUSTRIAL":
                df_filtered = df_filtered[df_filtered["fire_type"].isin(["INDUSTRIAL_FIRE", "FACILITY_RELATED_FIRE"])]
            elif target_filter == "MINING":
                df_filtered = df_filtered[df_filtered["fire_type"] == "MINING_RELATED_FIRE"]
            else:
                df_filtered = df_filtered[df_filtered["fire_type"] == target_filter]

        # Compute summary distribution before sampling
        type_counts = df_filtered["fire_type"].value_counts().to_dict()

        total_matching = len(df_filtered)
        if limit and limit > 0 and total_matching > limit:
            df_sampled = df_filtered.sample(n=limit, random_state=42)
        else:
            df_sampled = df_filtered

        # Build GeoJSON Features
        features = []
        for _, row in df_sampled.iterrows():
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                fire_type = str(row.get("fire_type", "UNKNOWN_THERMAL_EVENT"))
                avg_frp = float(row.get("avg_frp", 15.0))
                max_frp = float(row.get("max_frp", avg_frp))
                detections = int(row.get("total_detections", 1))
                first_det = str(row["first_detection"].strftime("%Y-%m-%d")) if pd.notna(row["first_detection"]) else ""
                last_det = str(row["last_detection"].strftime("%Y-%m-%d")) if pd.notna(row["last_detection"]) else ""

                # Compute quick risk score
                pers = float(row.get("persistence_percent", 1.0))
                night_r = float(row.get("night_detection_ratio", 0.0))
                dens = float(row.get("detections_per_active_day", 1.0))

                ctx_flags = {
                    "industrial_presence": int(row.get("industrial_presence", 0)),
                    "nearby_mining": int(row.get("nearby_mining", 0)),
                    "nearby_power_infrastructure": int(row.get("nearby_power_infrastructure", 0)),
                    "nearby_railway": int(row.get("nearby_railway", 0)),
                    "nearby_highway": int(row.get("nearby_highway", 0)),
                    "facility_presence": int(row.get("facility_presence", 0)),
                    "mixed_industrial_infrastructure": int(row.get("mixed_industrial_infrastructure", 0)),
                    "context_industrial": int(row.get("context_industrial", 0)),
                    "context_mining": int(row.get("context_mining", 0))
                }

                risk_score, risk_level, _ = RiskAssessmentService.calculate_risk(
                    max_frp=max_frp,
                    total_detections=detections,
                    persistence_percent=pers,
                    night_detection_ratio=night_r,
                    detections_per_active_day=dens,
                    context_flags=ctx_flags
                )

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "event_id": f"hist_{row.get('event_cluster_id', 0)}",
                        "latitude": lat,
                        "longitude": lon,
                        "fire_type": fire_type,
                        "frp": round(max_frp, 2),
                        "avg_frp": round(avg_frp, 2),
                        "bright_ti4": round(float(row.get("avg_bright_ti4", 335.0)), 1),
                        "bright_ti5": round(float(row.get("avg_bright_ti5", 300.0)), 1),
                        "total_detections": detections,
                        "acq_date": first_det,
                        "last_detection": last_det,
                        "daynight": "N" if night_r > 0.5 else "D",
                        "satellite": "VIIRS Archive",
                        "risk_score": risk_score,
                        "risk_level": risk_level
                    }
                })
            except Exception:
                continue

        return {
            "type": "FeatureCollection",
            "preset": time_preset,
            "filter": fire_type_filter,
            "total_matches": total_matching,
            "count": len(features),
            "summary": {
                "fire_type_distribution": type_counts,
                "date_span": f"{df_filtered['first_detection'].min().date() if not df_filtered.empty else ''} to {df_filtered['first_detection'].max().date() if not df_filtered.empty else ''}"
            },
            "features": features
        }
