from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import MODEL_PATH, FEATURE_PATH, FIRMS_MAP_KEY, CORS_ORIGINS
from api.services.ml_service import MLModelService
from api.services.firms_service import FIRMSLiveService
from api.services.timeline_service import TimelineArchiveService

from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router
from api.routes.firms import router as firms_router
from api.routes.timeline import router as timeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that loads the ML model and initializes services on startup."""
    print("=" * 70)
    print("STARTING THERMAL CLASSIFICATION & RISK SERVICE")
    print("=" * 70)

    # 1. Load ML Model Service
    try:
        app.state.ml_service = MLModelService(
            model_path=MODEL_PATH,
            feature_path=FEATURE_PATH
        )
    except Exception as e:
        print(f"✗ Failed to load ML Model: {e}")
        app.state.ml_service = None

    # 2. Initialize FIRMS Live Fetcher Service
    app.state.firms_service = FIRMSLiveService(
        map_key=FIRMS_MAP_KEY,
        cache_ttl_seconds=900,
        ml_service=app.state.ml_service
    )

    # 3. Initialize Timeline & Historical Archive Service
    app.state.timeline_service = TimelineArchiveService()

    print("✓ All background services initialized successfully")
    print("=" * 70)
    yield
    print("Shutting down Thermal Event API...")


app = FastAPI(
    title="Industrial & Thermal Event Fire Classification API",
    description=(
        "Production-grade FastAPI backend for SIH. "
        "Provides real-time NASA FIRMS hotspot ingestion, "
        "Random Forest fire type classification, and composite risk scoring."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for React Vite frontend and Express backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(firms_router, prefix="/api/v1")
app.include_router(timeline_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return JSONResponse({
        "service": "Thermal Event Fire Classification & Risk Assessment API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "live_hotspots_url": "/api/v1/firms/live",
        "predict_url": "/api/v1/predict"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
