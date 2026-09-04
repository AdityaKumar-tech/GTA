from fastapi import APIRouter, Request, Query
from typing import Optional, Dict, Any

router = APIRouter(prefix="/timeline", tags=["Timeline & Historical Archive"])


@router.get("", response_model=Dict[str, Any])
def get_timeline_events(
    request: Request,
    preset: str = Query("30d", description="Time preset: '24h', '7d', '30d', 'all'"),
    fire_type: str = Query("ALL", description="Filter fire type: 'ALL', or any specific fire type"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: Optional[int] = Query(50000, ge=1, le=50000, description="Maximum points to return")
):
    """Fetches historical thermal events and predictions across a 6-month timeline in India."""
    timeline_service = getattr(request.app.state, "timeline_service", None)
    if timeline_service is None:
        return {
            "type": "FeatureCollection",
            "preset": preset,
            "count": 0,
            "summary": {},
            "features": []
        }

    return timeline_service.get_timeline_events(
        time_preset=preset,
        fire_type_filter=fire_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
