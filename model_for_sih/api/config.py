import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "models" / "fire_type_random_forest.joblib"
FEATURE_PATH = BASE_DIR / "models" / "fire_type_features.txt"

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "023b1ee52c69d905eaedee65df7bd7cf")

# India bounding box: [west, south, east, north]
INDIA_BBOX = {
    "west": 68.0,
    "south": 6.0,
    "east": 97.5,
    "north": 37.5
}

# Supported VIIRS sources for FIRMS
FIRMS_SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT"
]

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:5000",
    "*"
]
