import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 6B: PREPARE ML FEATURE MATRIX
# ============================================================

print("=" * 75)
print("STEP 6B: PREPARING ML FEATURE MATRIX")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\features\thermal_event_context.csv"
)

OUTPUT_FILE = (
    r"data\processed\ml\model_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/8] Loading enriched dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Records loaded: {len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

print()
print("[2/8] Checking required columns...")

required_columns = [

    "event_cluster_id",

    # Spatial
    "center_latitude",
    "center_longitude",

    # Thermal
    "total_detections",
    "detection_days",
    "avg_frp",
    "max_frp",
    "min_frp",
    "frp_std",
    "avg_bright_ti4",
    "max_bright_ti4",
    "avg_bright_ti5",

    # Temporal/activity
    "night_detections",
    "night_detection_ratio",
    "observation_days",
    "persistence_percent",
    "detections_per_active_day",

    # Thermal classification
    "thermal_behavior",

    # Land cover
    "worldcover_class",
    "landcover_group",

    # Context
    "context_category",
    "industrial_presence",
    "infrastructure_presence",
    "facility_presence",
    "nearby_highway",
    "nearby_railway",
    "nearby_power_infrastructure",
    "nearby_mining",
    "mixed_industrial_infrastructure"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )

print(
    f"✓ All {len(required_columns)} required columns available"
)


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

print()
print("[3/8] Creating numerical feature matrix...")


features = pd.DataFrame()


# ============================================================
# SPATIAL FEATURES
# ============================================================

features["latitude"] = pd.to_numeric(
    df["center_latitude"],
    errors="coerce"
)

features["longitude"] = pd.to_numeric(
    df["center_longitude"],
    errors="coerce"
)


# ============================================================
# THERMAL FEATURES
# ============================================================

thermal_columns = [

    "total_detections",
    "detection_days",
    "avg_frp",
    "max_frp",
    "min_frp",
    "frp_std",
    "avg_bright_ti4",
    "max_bright_ti4",
    "avg_bright_ti5",

    "night_detections",
    "night_detection_ratio",

    "observation_days",
    "persistence_percent",

    "detections_per_active_day"
]


for column in thermal_columns:

    features[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

print()
print("[4/8] Creating derived thermal features...")


# FRP range

features["frp_range"] = (
    features["max_frp"]
    -
    features["min_frp"]
)


# FRP variability relative to mean

features["frp_variability"] = (

    features["frp_std"]

    /

    features["avg_frp"].replace(
        0,
        np.nan
    )

)


# Detection density

features["detections_per_observation_day"] = (

    features["total_detections"]

    /

    features["observation_days"].replace(
        0,
        np.nan
    )

)


# Night activity intensity

features["night_activity_intensity"] = (

    features["night_detections"]

    /

    features["total_detections"].replace(
        0,
        np.nan
    )

)


# High intensity indicator

frp_threshold = features["max_frp"].quantile(
    0.75
)

features["high_intensity_indicator"] = (
    features["max_frp"] >= frp_threshold
).astype(int)


# Persistent event indicator

features["persistent_indicator"] = (
    features["persistence_percent"] >= 1.0
).astype(int)


# ============================================================
# THERMAL BEHAVIOR ENCODING
# ============================================================

print()
print("[5/8] Encoding thermal behavior...")


behavior = (
    df["thermal_behavior"]
    .astype(str)
    .str.upper()
)


features["behavior_high_intensity"] = (
    behavior.str.contains(
        "HIGH_INTENSITY",
        na=False
    )
).astype(int)


features["behavior_persistent"] = (
    behavior.str.contains(
        "PERSISTENT",
        na=False
    )
).astype(int)


features["behavior_frequent"] = (
    behavior.str.contains(
        "FREQUENT",
        na=False
    )
).astype(int)


features["behavior_intermittent"] = (
    behavior.str.contains(
        "INTERMITTENT",
        na=False
    )
).astype(int)


# ============================================================
# LAND COVER ENCODING
# ============================================================

print()
print("[6/8] Encoding land-cover information...")


landcover = (
    df["landcover_group"]
    .astype(str)
    .str.upper()
)


features["landcover_natural"] = (
    landcover == "NATURAL"
).astype(int)


features["landcover_agricultural"] = (
    landcover == "AGRICULTURAL"
).astype(int)


features["landcover_urban"] = (
    landcover == "URBAN"
).astype(int)


features["landcover_water"] = (
    landcover == "WATER"
).astype(int)


features["landcover_barren"] = (
    landcover == "BARREN"
).astype(int)


features["landcover_valid"] = (
    df["landcover_valid"]
).astype(int)


# Keep WorldCover class where available

features["worldcover_class"] = pd.to_numeric(
    df["worldcover_class"],
    errors="coerce"
)


# ============================================================
# CONTEXT / INFRASTRUCTURE FEATURES
# ============================================================

print()
print("[7/8] Adding infrastructure and industry features...")


context = (
    df["context_category"]
    .astype(str)
    .str.upper()
)


features["context_agricultural"] = (
    context == "AGRICULTURAL"
).astype(int)


features["context_industrial"] = (
    context == "INDUSTRIAL"
).astype(int)


features["context_mining"] = (
    context == "MINING"
).astype(int)


features["context_rural"] = (
    context == "RURAL"
).astype(int)


features["industrial_presence"] = (
    pd.to_numeric(
        df["industrial_presence"],
        errors="coerce"
    )
)


features["infrastructure_presence"] = (
    pd.to_numeric(
        df["infrastructure_presence"],
        errors="coerce"
    )
)


features["facility_presence"] = (
    pd.to_numeric(
        df["facility_presence"],
        errors="coerce"
    )
)


features["nearby_highway"] = (
    pd.to_numeric(
        df["nearby_highway"],
        errors="coerce"
    )
)


features["nearby_railway"] = (
    pd.to_numeric(
        df["nearby_railway"],
        errors="coerce"
    )
)


features["nearby_power_infrastructure"] = (
    pd.to_numeric(
        df["nearby_power_infrastructure"],
        errors="coerce"
    )
)


features["nearby_mining"] = (
    pd.to_numeric(
        df["nearby_mining"],
        errors="coerce"
    )
)


features["mixed_industrial_infrastructure"] = (
    pd.to_numeric(
        df["mixed_industrial_infrastructure"],
        errors="coerce"
    )
)


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

print()
print("Cleaning numerical values...")


features = features.replace(
    [np.inf, -np.inf],
    np.nan
)


# WorldCover class missing means
# land-cover category is still unavailable.
# Use -1 rather than pretending it is a real class.

features["worldcover_class"] = (
    features["worldcover_class"]
    .fillna(-1)
)


# Remaining numerical missing values

features = features.fillna(0)


# ============================================================
# ADD EVENT ID
# ============================================================

features.insert(
    0,
    "event_cluster_id",
    df["event_cluster_id"]
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("Validating feature matrix...")


print(
    f"Rows       : {len(features):,}"
)

print(
    f"Features   : {len(features.columns):,}"
)

print(
    f"Missing    : {features.isna().sum().sum():,}"
)

print(
    f"Infinite   : "
    f"{np.isinf(features.select_dtypes(include=np.number)).sum().sum():,}"
)


# ============================================================
# SAVE
# ============================================================

print()
print("Saving ML feature matrix...")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("✓ STEP 6B COMPLETE")
print("=" * 75)

print()
print(
    f"Input events  : {len(df):,}"
)

print(
    f"Output events : {len(features):,}"
)

print(
    f"ML features   : {len(features.columns) - 1:,}"
)

print()
print("Output:")
print(
    os.path.abspath(OUTPUT_FILE)
)

print()
print("=" * 75)
print("NEXT STEP → TARGET / RISK LABEL GENERATION")
print("=" * 75)