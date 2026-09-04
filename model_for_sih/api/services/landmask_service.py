import json
from pathlib import Path
from typing import Optional, List
import pandas as pd
from shapely.geometry import shape, Point
from shapely.prepared import prep
from api.config import BASE_DIR


class LandmaskService:
    """Service to detect and filter out oceanic points using high-resolution 10m landmass boundaries."""

    _instance: Optional["LandmaskService"] = None
    _prep_land = None
    _land_geom = None

    @classmethod
    def get_instance(cls) -> "LandmaskService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        geojson_path = BASE_DIR / "data" / "processed" / "context" / "south_asia_landmass.geojson"
        if geojson_path.exists():
            try:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._land_geom = shape(data["features"][0]["geometry"])
                self._prep_land = prep(self._land_geom)
                print(f"✓ LandmaskService initialized with 10m South Asia coastline boundary")
            except Exception as e:
                print(f"Warning: Failed to load south_asia_landmass.geojson: {e}")
                self._prep_land = None
        else:
            print(f"Warning: Landmask geojson not found at {geojson_path}")
            self._prep_land = None

    def is_land(self, lat: float, lon: float) -> bool:
        """Returns True if coordinate is on land, False if in ocean."""
        if self._prep_land is None:
            # Fallback boundary check: within mainland India bounding coordinates
            return 8.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5
        return bool(self._prep_land.contains(Point(lon, lat)))

    def is_ocean(self, lat: float, lon: float) -> bool:
        """Returns True if coordinate is located in the ocean/sea."""
        return not self.is_land(lat, lon)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        remove_water_class: bool = True
    ) -> pd.DataFrame:
        """Filters a DataFrame to remove all oceanic and waterbody detections."""
        if df.empty or lat_col not in df.columns or lon_col not in df.columns:
            return df

        if self._prep_land is not None:
            is_land_mask = [
                self._prep_land.contains(Point(lon, lat))
                for lat, lon in zip(df[lat_col], df[lon_col])
            ]
            land_series = pd.Series(is_land_mask, index=df.index)
        else:
            # Fallback
            land_series = (df[lat_col] >= 8.0) & (df[lat_col] <= 37.5) & (df[lon_col] >= 68.0) & (df[lon_col] <= 97.5)

        if remove_water_class and "landcover_water" in df.columns:
            # Also exclude confirmed inland water bodies / reservoirs
            clean_mask = land_series & (df["landcover_water"] != 1)
        elif remove_water_class and "worldcover_class" in df.columns:
            clean_mask = land_series & (df["worldcover_class"] != 80)
        else:
            clean_mask = land_series

        return df[clean_mask]
