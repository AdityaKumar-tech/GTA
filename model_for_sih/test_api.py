import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from main import app

client = TestClient(app)


def test_root():
    print("\n[TEST 1] Testing Root Endpoint GET / ...")
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "online"
    print("✓ Root endpoint returned status 'online'")


def test_health():
    print("\n[TEST 2] Testing Health Endpoint GET /api/v1/health ...")
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["model_loaded"] is True, "Model should be loaded"
    assert data["feature_count"] == 41, f"Expected 41 features, got {data['feature_count']}"
    print(f"✓ Health check passed (Features: {data['feature_count']}, Status: {data['status']})")


def test_metadata():
    print("\n[TEST 3] Testing Metadata Endpoint GET /api/v1/metadata ...")
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert len(data["features"]) == 41
    assert len(data["classes"]) > 0
    print(f"✓ Metadata check passed (Classes: {len(data['classes'])}, Total Features: {data['total_features']})")


def test_single_predict_industrial():
    print("\n[TEST 4] Testing Single Prediction POST /api/v1/predict (Industrial Event) ...")
    payload = {
        "event_id": "test-industrial-001",
        "latitude": 11.274,
        "longitude": 78.686,
        "frp": 32.6,
        "avg_frp": 14.8,
        "max_frp": 32.6,
        "min_frp": 4.2,
        "frp_std": 8.7,
        "avg_bright_ti4": 342.8,
        "max_bright_ti4": 361.4,
        "avg_bright_ti5": 304.6,
        "total_detections": 18,
        "detection_days": 8,
        "observation_days": 366,
        "night_detections": 12,
        "daynight": "N",
        "landcover_urban": 1,
        "landcover_natural": 0,
        "landcover_agricultural": 0,
        "worldcover_class": 50,
        "context_industrial": 1,
        "industrial_presence": 1,
        "infrastructure_presence": 1,
        "facility_presence": 1,
        "nearby_highway": 1,
        "nearby_railway": 1,
        "nearby_power_infrastructure": 1,
        "nearby_mining": 0,
        "mixed_industrial_infrastructure": 1
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["predicted_fire_type"] == "INDUSTRIAL_FIRE", f"Expected INDUSTRIAL_FIRE, got {data['predicted_fire_type']}"
    assert data["confidence"] > 0.5, f"Expected confidence > 0.5, got {data['confidence']}"
    assert "risk_score" in data
    assert "risk_level" in data
    print(f"✓ Industrial prediction: {data['predicted_fire_type']} (Confidence: {data['confidence']*100:.2f}%, Risk: {data['risk_level']} - {data['risk_score']}/100)")


def test_single_predict_natural_hotspot():
    print("\n[TEST 5] Testing Single Prediction POST /api/v1/predict (Raw FIRMS Forest Hotspot) ...")
    payload = {
        "event_id": "raw-firms-forest-002",
        "latitude": 30.3165,
        "longitude": 78.0322,
        "frp": 45.0,
        "bright_ti4": 355.0,
        "bright_ti5": 305.0,
        "daynight": "D",
        "landcover_natural": 1,
        "landcover_urban": 0,
        "landcover_agricultural": 0,
        "worldcover_class": 10,
        "context_industrial": 0,
        "industrial_presence": 0,
        "infrastructure_presence": 0,
        "facility_presence": 0,
        "nearby_highway": 0,
        "nearby_power_infrastructure": 0
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predicted_fire_type" in data
    assert "risk_score" in data
    print(f"✓ Forest prediction: {data['predicted_fire_type']} (Confidence: {data['confidence']*100:.2f}%, Risk: {data['risk_level']} - {data['risk_score']}/100)")


def test_batch_predict():
    print("\n[TEST 6] Testing Batch Prediction POST /api/v1/predict/batch ...")
    payload = {
        "events": [
            {
                "event_id": "batch-1",
                "latitude": 11.274,
                "longitude": 78.686,
                "frp": 25.0,
                "context_industrial": 1,
                "industrial_presence": 1
            },
            {
                "event_id": "batch-2",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "frp": 12.0,
                "landcover_agricultural": 1,
                "worldcover_class": 40
            }
        ]
    }

    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["total_events"] == 2
    assert len(data["results"]) == 2
    print(f"✓ Batch prediction processed {data['total_events']} events successfully")


def test_firms_sources():
    print("\n[TEST 7] Testing FIRMS Sources Endpoint GET /api/v1/firms/sources ...")
    response = client.get("/api/v1/firms/sources")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    sources = response.json()
    assert len(sources) > 0
    print(f"✓ FIRMS sources available: {sources}")


def test_firms_live_endpoint():
    print("\n[TEST 8] Testing FIRMS Live Endpoint GET /api/v1/firms/live ...")
    response = client.get("/api/v1/firms/live")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    print(f"✓ FIRMS live endpoint responded with {data['count']} GeoJSON hotspot features (Source: {data['source']})")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FASTAPI ENDPOINT AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    
    with TestClient(app) as client:
        test_root()
        test_health()
        test_metadata()
        test_single_predict_industrial()
        test_single_predict_natural_hotspot()
        test_batch_predict()
        test_firms_sources()
        test_firms_live_endpoint()

    print("\n" + "=" * 70)
    print("🎉 ALL 8 FASTAPI API TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
