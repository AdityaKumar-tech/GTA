import os
from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fire_type_random_forest.joblib"
FEATURE_PATH = BASE_DIR / "models" / "fire_type_features.txt"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("FIRE TYPE MODEL - SYNTHETIC PREDICTION TEST")
print("=" * 70)

print("\n[1] Loading trained model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(str(MODEL_PATH))

print("✓ Model loaded successfully")


# ============================================================
# LOAD FEATURE ORDER
# ============================================================

print("\n[2] Loading feature list...")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_PATH}"
    )

with open(FEATURE_PATH, "r") as f:
    features = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"✓ {len(features)} features loaded"
)


# ============================================================
# SYNTHETIC THERMAL EVENT
# ============================================================

print("\n[3] Creating synthetic thermal event...")


synthetic_event = {

    "latitude": 11.274,
    "longitude": 78.686,

    "total_detections": 18,
    "detection_days": 8,

    "avg_frp": 14.8,
    "max_frp": 32.6,
    "min_frp": 4.2,
    "frp_std": 8.7,

    "avg_bright_ti4": 342.8,
    "max_bright_ti4": 361.4,
    "avg_bright_ti5": 304.6,

    "night_detections": 12,
    "night_detection_ratio": 0.67,

    "observation_days": 366,
    "persistence_percent": 2.19,

    "detections_per_active_day": 2.25,

    "frp_range": 28.4,
    "frp_variability": 0.59,

    "detections_per_observation_day": 0.049,

    "night_activity_intensity": 0.67,

    "high_intensity_indicator": 1,
    "persistent_indicator": 1,

    "landcover_natural": 0,
    "landcover_agricultural": 0,
    "landcover_urban": 1,
    "landcover_water": 0,
    "landcover_barren": 0,

    "landcover_valid": 1,
    "worldcover_class": 50,

    "context_agricultural": 0,
    "context_industrial": 1,
    "context_mining": 0,
    "context_rural": 0,

    "industrial_presence": 1,
    "infrastructure_presence": 1,
    "facility_presence": 1,

    "nearby_highway": 1,
    "nearby_railway": 1,
    "nearby_power_infrastructure": 1,
    "nearby_mining": 0,

    "mixed_industrial_infrastructure": 1
}


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

print("\n[4] Preparing model input...")

input_data = {}

for feature in features:

    input_data[feature] = synthetic_event.get(
        feature,
        0
    )


X = pd.DataFrame(
    [input_data],
    columns=features
)


print(
    f"✓ Input shape: {X.shape}"
)


# ============================================================
# PREDICT
# ============================================================

print("\n[5] Running model prediction...")

prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

classes = model.classes_


# ============================================================
# CONFIDENCE
# ============================================================

probability_dict = {
    str(cls): float(probability)
    for cls, probability
    in zip(classes, probabilities)
}

confidence = max(
    probability_dict.values()
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                    PREDICTION RESULT")
print("=" * 70)

print()

print(
    f"Latitude          : "
    f"{synthetic_event['latitude']}"
)

print(
    f"Longitude         : "
    f"{synthetic_event['longitude']}"
)

print(
    f"Average FRP       : "
    f"{synthetic_event['avg_frp']} MW"
)

print(
    f"Maximum FRP       : "
    f"{synthetic_event['max_frp']} MW"
)

print(
    f"Night Detection   : "
    f"{synthetic_event['night_detection_ratio'] * 100:.1f}%"
)

print()

print(
    f"PREDICTED FIRE TYPE : "
    f"{prediction}"
)

print(
    f"CONFIDENCE          : "
    f"{confidence * 100:.2f}%"
)


# ============================================================
# PROBABILITIES
# ============================================================

print("\n")
print("-" * 70)
print("CLASS PROBABILITIES")
print("-" * 70)

for class_name, probability in sorted(
    probability_dict.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{class_name:<35}"
        f"{probability * 100:>8.2f}%"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)

if prediction:

    print("✓ MODEL IS WORKING SUCCESSFULLY")

else:

    print("✗ MODEL PREDICTION FAILED")

print("=" * 70)