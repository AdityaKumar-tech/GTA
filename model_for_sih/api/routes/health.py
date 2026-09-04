from fastapi import APIRouter, Request
from api.schemas import HealthResponse, ModelMetadataResponse

router = APIRouter(tags=["Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    """Liveness probe verifying API status and model availability in memory."""
    ml_service = getattr(request.app.state, "ml_service", None)
    is_loaded = ml_service is not None and ml_service.model is not None
    feature_count = len(ml_service.features) if is_loaded else 0

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        feature_count=feature_count,
        version="1.0.0"
    )


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_metadata(request: Request):
    """Returns model metadata, target classes, and 41 expected feature names."""
    ml_service = request.app.state.ml_service
    return ModelMetadataResponse(
        model_type="RandomForestClassifier",
        total_features=len(ml_service.features),
        features=ml_service.features,
        classes=ml_service.classes
    )
