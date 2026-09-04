from fastapi import APIRouter, Request, HTTPException
from api.schemas import (
    ThermalEventInput,
    BatchThermalEventInput,
    PredictionResponse,
    BatchPredictionResponse
)

router = APIRouter(prefix="/predict", tags=["Fire Classification & Risk Inference"])


@router.post("", response_model=PredictionResponse)
def predict_single_event(event: ThermalEventInput, request: Request):
    """Classifies the fire type and evaluates multi-factor risk for a single hotspot or event."""
    ml_service = getattr(request.app.state, "ml_service", None)
    if ml_service is None or ml_service.model is None:
        raise HTTPException(status_code=503, detail="ML Model service is not loaded.")

    try:
        result = ml_service.predict_single(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch_events(batch_input: BatchThermalEventInput, request: Request):
    """High-performance vectorized batch prediction for multiple hotspots."""
    ml_service = getattr(request.app.state, "ml_service", None)
    if ml_service is None or ml_service.model is None:
        raise HTTPException(status_code=503, detail="ML Model service is not loaded.")

    try:
        result = ml_service.predict_batch(batch_input.events)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
