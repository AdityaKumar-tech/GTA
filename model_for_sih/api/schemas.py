from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ThermalEventInput(BaseModel):
    """Input payload representing a single thermal event or raw NASA FIRMS hotspot."""
    event_id: Optional[str] = Field(None, description="Optional unique identifier for the event/point")
    
    # Coordinates
    latitude: float = Field(..., description="Latitude coordinate", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Longitude coordinate", ge=-180.0, le=180.0)
    
    # Thermal Metrics
    frp: Optional[float] = Field(None, description="Fire Radiative Power (MW) from single detection")
    avg_frp: Optional[float] = Field(None, description="Mean Fire Radiative Power (MW)")
    max_frp: Optional[float] = Field(None, description="Peak Fire Radiative Power (MW)")
    min_frp: Optional[float] = Field(None, description="Minimum Fire Radiative Power (MW)")
    frp_std: Optional[float] = Field(0.0, description="Standard deviation of FRP")
    
    # Brightness Temperatures (Kelvin)
    bright_ti4: Optional[float] = Field(None, description="VIIRS Band I-4 brightness temp (K)")
    avg_bright_ti4: Optional[float] = Field(330.0, description="Average VIIRS Band I-4 brightness temp (K)")
    max_bright_ti4: Optional[float] = Field(340.0, description="Max VIIRS Band I-4 brightness temp (K)")
    bright_ti5: Optional[float] = Field(None, description="VIIRS Band I-5 brightness temp (K)")
    avg_bright_ti5: Optional[float] = Field(300.0, description="Average VIIRS Band I-5 brightness temp (K)")
    
    # Detection & Persistence
    total_detections: Optional[int] = Field(1, description="Total hotspot detections in this cluster")
    detection_days: Optional[int] = Field(1, description="Number of active detection days")
    observation_days: Optional[int] = Field(365, description="Monitoring window in days")
    night_detections: Optional[int] = Field(0, description="Number of nighttime detections")
    night_detection_ratio: Optional[float] = Field(None, description="Ratio of night to total detections (0.0 to 1.0)")
    daynight: Optional[str] = Field("D", description="'D' for Day, 'N' for Night")
    
    # Land Cover (defaults to None so real ESA WorldCover tile lookup executes dynamically)
    worldcover_class: Optional[int] = Field(None, description="ESA WorldCover class code (10: Tree, 40: Crop, 50: Urban, 60: Bare)")
    landcover_natural: Optional[int] = Field(None, description="1 if natural landcover (forest, shrub), 0 otherwise")
    landcover_agricultural: Optional[int] = Field(None, description="1 if cropland/agriculture, 0 otherwise")
    landcover_urban: Optional[int] = Field(None, description="1 if urban/built-up, 0 otherwise")
    landcover_water: Optional[int] = Field(None, description="1 if water body, 0 otherwise")
    landcover_barren: Optional[int] = Field(None, description="1 if bare/barren, 0 otherwise")
    landcover_valid: Optional[int] = Field(1, description="1 if valid landcover available, 0 otherwise")
    
    # Context & Infrastructure Flags (defaults to None)
    context_agricultural: Optional[int] = Field(None, description="1 if agricultural context")
    context_industrial: Optional[int] = Field(None, description="1 if industrial context")
    context_mining: Optional[int] = Field(None, description="1 if mining context")
    context_rural: Optional[int] = Field(None, description="1 if rural context")
    industrial_presence: Optional[int] = Field(None, description="1 if industry present nearby")
    infrastructure_presence: Optional[int] = Field(None, description="1 if infrastructure present nearby")
    facility_presence: Optional[int] = Field(None, description="1 if factory/facility present nearby")
    nearby_highway: Optional[int] = Field(None, description="1 if near highway")
    nearby_railway: Optional[int] = Field(None, description="1 if near railway")
    nearby_power_infrastructure: Optional[int] = Field(None, description="1 if near power lines/substations")
    nearby_mining: Optional[int] = Field(None, description="1 if near mining site")
    mixed_industrial_infrastructure: Optional[int] = Field(None, description="1 if both industrial and infrastructure are present")

    # Optional Pre-known or Archive Metadata
    fire_type: Optional[str] = Field(None, description="Pre-classified or archive fire type if known")
    acq_date: Optional[str] = Field(None, description="Acquisition date (YYYY-MM-DD)")
    acq_time: Optional[str] = Field(None, description="Acquisition time (HHMM)")
    satellite: Optional[str] = Field(None, description="Satellite name (e.g. NOAA-20, SNPP)")


class BatchThermalEventInput(BaseModel):
    """Container for batch predictions."""
    events: List[ThermalEventInput]


class PredictionResponse(BaseModel):
    """Response containing ML fire classification and risk score."""
    event_id: Optional[str] = None
    latitude: float
    longitude: float
    predicted_fire_type: str
    confidence: float
    class_probabilities: Dict[str, float]
    risk_score: float
    risk_level: str
    summary: Dict[str, Any]


class BatchPredictionResponse(BaseModel):
    """Response for batch inference."""
    total_events: int
    results: List[PredictionResponse]


class GeoJSONPoint(BaseModel):
    """GeoJSON Feature object for MapLibre rendering."""
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class FIRMSLiveResponse(BaseModel):
    """GeoJSON FeatureCollection response containing live thermal hotspots."""
    type: str = "FeatureCollection"
    source: str
    date: str
    count: int
    features: List[Dict[str, Any]]


class ModelMetadataResponse(BaseModel):
    """Metadata response describing the loaded model."""
    model_type: str
    total_features: int
    features: List[str]
    classes: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    feature_count: int
    version: str = "1.0.0"
