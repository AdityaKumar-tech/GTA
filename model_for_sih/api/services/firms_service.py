import io
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from api.config import FIRMS_MAP_KEY, INDIA_BBOX, FIRMS_SOURCES, BASE_DIR
from api.schemas import ThermalEventInput


class FIRMSLiveService:
    """Service to fetch real-time thermal detections from NASA FIRMS API

    Includes in-memory caching, multi-satellite aggregation (NOAA-20, NOAA-21, SNPP),
    and automated AI fire classification & risk assessment for all live points.
    """

    def __init__(self, map_key: str = FIRMS_MAP_KEY, cache_ttl_seconds: int = 900, ml_service=None):
        self.map_key = map_key
        self.cache_ttl = cache_ttl_seconds
        self.ml_service = ml_service
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0

    def set_ml_service(self, ml_service):
        self.ml_service = ml_service

    def fetch_live_hotspots(
        self,
        source: str = "ALL",
        day_range: int = 1,
        fire_type: str = "ALL",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetches active VIIRS hotspot points across India and evaluates fire classifications."""
        current_time = time.time()
        cache_key = f"{source}_{day_range}"

        # Check if full dataset is in cache
        if (
            not force_refresh
            and cache_key in self._cache
            and (current_time - self._cache_timestamp) < self.cache_ttl
        ):
            full_geojson = self._cache[cache_key]
        else:
            today_str = datetime.utcnow().strftime("%Y-%m-%d")

            # If source is "ALL", fetch from all 3 VIIRS satellites in parallel and merge
            if source.upper() in ["ALL", "VIIRS_ALL"]:
                sources_to_fetch = FIRMS_SOURCES
            else:
                sources_to_fetch = [source]

            all_dfs = []

            def fetch_single_source(src):
                return self._fetch_raw_source_df(src, day_range)

            with ThreadPoolExecutor(max_workers=len(sources_to_fetch)) as executor:
                results = list(executor.map(fetch_single_source, sources_to_fetch))

            for df in results:
                if df is not None and not df.empty:
                    all_dfs.append(df)

            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                # Deduplicate nearby readings from the same instrument
                if "latitude" in combined_df.columns and "longitude" in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
                
                full_geojson = self._dataframe_to_geojson(combined_df, "VIIRS_COMBINED", today_str)
            else:
                # Fallback to local dataset if NASA API is momentarily unreachable
                full_geojson = self._load_local_fallback_geojson(today_str)

            # Store in cache
            self._cache[cache_key] = full_geojson
            self._cache_timestamp = current_time
            print(f"✓ Loaded & AI-classified {full_geojson['count']} active thermal hotspots across India")

        # Apply fire_type filter if requested
        if fire_type and fire_type.upper() != "ALL":
            target = fire_type.upper()
            filtered_features = []
            for feat in full_geojson.get("features", []):
                p_type = feat.get("properties", {}).get("fire_type", "").upper()
                if target in ["WILDFIRE", "FOREST"] and p_type in ["FOREST_WILDFIRE", "GRASSLAND_FIRE"]:
                    filtered_features.append(feat)
                elif target in ["CROP", "AGRICULTURAL"] and p_type == "CROP_RESIDUE_FIRE":
                    filtered_features.append(feat)
                elif target == "INDUSTRIAL" and p_type in ["INDUSTRIAL_FIRE", "FACILITY_RELATED_FIRE"]:
                    filtered_features.append(feat)
                elif target == "MINING" and p_type == "MINING_RELATED_FIRE":
                    filtered_features.append(feat)
                elif p_type == target:
                    filtered_features.append(feat)

            return {
                "type": "FeatureCollection",
                "source": full_geojson.get("source", source),
                "date": full_geojson.get("date", ""),
                "count": len(filtered_features),
                "total_matches": full_geojson.get("count", len(full_geojson.get("features", []))),
                "features": filtered_features
            }

        return full_geojson

    def _fetch_raw_source_df(self, source: str, day_range: int) -> Optional[pd.DataFrame]:
        """Queries NASA FIRMS Area API for a single source."""
        west = INDIA_BBOX["west"]
        south = INDIA_BBOX["south"]
        east = INDIA_BBOX["east"]
        north = INDIA_BBOX["north"]

        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{self.map_key}/{source}/{west},{south},{east},{north}/{day_range}"
        )

        headers = {
            "User-Agent": "SIH-Thermal-Monitor/1.0 (Research Application)"
        }

        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200 and response.text.strip():
                if "latitude" in response.text:
                    df = pd.read_csv(io.StringIO(response.text))
                    df["satellite_source"] = source
                    return df
            return None
        except Exception as e:
            print(f"NASA FIRMS fetch warning for {source}: {e}")
            return None

    def _dataframe_to_geojson(self, df: pd.DataFrame, source: str, date_str: str) -> Dict[str, Any]:
        """Converts raw FIRMS DataFrame into standard GeoJSON with AI fire classifications."""
        raw_events: List[ThermalEventInput] = []
        raw_rows: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            try:
                lat = float(row.get("latitude", 0.0))
                lon = float(row.get("longitude", 0.0))
                frp = float(row.get("frp", 10.0))
                ti4 = float(row.get("bright_ti4", 330.0))
                ti5 = float(row.get("bright_ti5", 300.0))
                acq_date = str(row.get("acq_date", date_str))
                acq_time = str(row.get("acq_time", "0000")).zfill(4)
                daynight = str(row.get("daynight", "D")).upper()
                satellite = str(row.get("satellite", row.get("satellite_source", "VIIRS")))
                confidence = str(row.get("confidence", "nominal"))

                event_input = ThermalEventInput(
                    event_id=f"live_{idx}",
                    latitude=lat,
                    longitude=lon,
                    frp=frp,
                    bright_ti4=ti4,
                    bright_ti5=ti5,
                    acq_date=acq_date,
                    acq_time=acq_time,
                    daynight=daynight,
                    satellite=satellite
                )
                raw_events.append(event_input)
                raw_rows.append({
                    "lat": lat,
                    "lon": lon,
                    "frp": frp,
                    "ti4": ti4,
                    "ti5": ti5,
                    "acq_date": acq_date,
                    "acq_time": acq_time,
                    "daynight": daynight,
                    "satellite": satellite,
                    "confidence_str": confidence
                })
            except Exception:
                continue

        # AI Classification batch execution
        predictions_map = {}
        if self.ml_service and raw_events:
            try:
                batch_res = self.ml_service.predict_batch(raw_events)
                for res in batch_res.results:
                    predictions_map[res.event_id] = res
            except Exception as e:
                print(f"Warning: batch classification error in FIRMSLiveService: {e}")

        features = []
        for idx, r in enumerate(raw_rows):
            event_id = f"live_{idx}"
            pred = predictions_map.get(event_id)

            fire_type = pred.predicted_fire_type if pred else "UNKNOWN_THERMAL_EVENT"
            risk_level = pred.risk_level if pred else "MODERATE"
            risk_score = pred.risk_score if pred else 45.0
            ml_confidence = pred.confidence if pred else 0.75

            acq_time_fmt = f"{r['acq_time'][:2]}:{r['acq_time'][2:]} UTC" if len(r['acq_time']) == 4 else r['acq_time']

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["lon"], r["lat"]]
                },
                "properties": {
                    "event_id": event_id,
                    "latitude": r["lat"],
                    "longitude": r["lon"],
                    "fire_type": fire_type,
                    "frp": round(r["frp"], 2),
                    "bright_ti4": round(r["ti4"], 1),
                    "bright_ti5": round(r["ti5"], 1),
                    "acq_date": r["acq_date"],
                    "acq_time": acq_time_fmt,
                    "daynight": r["daynight"],
                    "satellite": r["satellite"],
                    "confidence": ml_confidence,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "source": source
                }
            })

        return {
            "type": "FeatureCollection",
            "source": source,
            "date": date_str,
            "count": len(features),
            "features": features
        }

    def _load_local_fallback_geojson(self, date_str: str) -> Dict[str, Any]:
        """Fallback to sample local thermal events if NASA API is offline."""
        fallback_file = BASE_DIR / "data" / "processed" / "events" / "firms_thermal_events.csv"
        if fallback_file.exists():
            try:
                df = pd.read_csv(fallback_file).head(150)
                df = df.rename(columns={"center_latitude": "latitude", "center_longitude": "longitude", "avg_frp": "frp"})
                return self._dataframe_to_geojson(df, "LOCAL_CACHE_FALLBACK", date_str)
            except Exception:
                pass

        return {
            "type": "FeatureCollection",
            "source": "EMPTY",
            "date": date_str,
            "count": 0,
            "features": []
        }
