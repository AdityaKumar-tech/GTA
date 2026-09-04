from fastapi import APIRouter, Request, Query
from typing import Optional, List
from api.schemas import FIRMSLiveResponse
from api.config import FIRMS_SOURCES

router = APIRouter(prefix="/firms", tags=["NASA FIRMS Live Ingestion"])


@router.get("/live", response_model=FIRMSLiveResponse)
def get_live_hotspots(
    request: Request,
    source: str = Query("ALL", description="Satellite source ('ALL' for all VIIRS satellites combined)"),
    day_range: int = Query(1, ge=1, le=5, description="Number of days to fetch (1 to 5)"),
    fire_type: str = Query("ALL", description="Filter fire type: 'ALL', 'INDUSTRIAL_FIRE', 'FOREST_WILDFIRE', etc."),
    force_refresh: bool = Query(False, description="Bypass cache and query NASA directly")
):
    """Fetches real-time thermal detections over India from NASA FIRMS as GeoJSON.

    Directly compatible with MapLibre GL map layers.
    """
    firms_service = getattr(request.app.state, "firms_service", None)
    if firms_service is None:
        return FIRMSLiveResponse(
            type="FeatureCollection",
            source=source,
            date="",
            count=0,
            features=[]
        )

    geojson_data = firms_service.fetch_live_hotspots(
        source=source,
        day_range=day_range,
        fire_type=fire_type,
        force_refresh=force_refresh
    )
    return geojson_data


@router.get("/sources", response_model=List[str])
def list_firms_sources():
    """Returns available NASA FIRMS satellite instrument sources."""
    return FIRMS_SOURCES
