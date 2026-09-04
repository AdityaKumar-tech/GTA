import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================

print("=" * 75)
print("STEP 6: INDIA-WIDE THERMAL EVENT FEATURE ENGINEERING")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    r"data\processed\events\firms_thermal_events.csv"
)

OUTPUT_FILE = (
    r"data\processed\ml\model_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/7] Loading enriched thermal-event dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Records loaded: {len(df):,}"
)

print(
    f"Columns available: {len(df.columns)}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


def numeric_column(candidates, default=0):

    column = find_column(candidates)

    if column is None:

        return pd.Series(
            default,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

print()
print("[2/7] Identifying model features...")


LAT_COL = find_column([
    "center_latitude",
    "latitude",
    "lat"
])

LON_COL = find_column([
    "center_longitude",
    "longitude",
    "lon"
])

FRP_COL = find_column([
    "mean_frp",
    "max_frp",
    "frp_mean",
    "frp_max",
    "frp"
])

PERSISTENCE_COL = find_column([
    "persistence_days",
    "persistence",
    "event_duration_days",
    "duration_days"
])

DETECTION_COL = find_column([
    "detection_count",
    "detections",
    "frequency",
    "event_frequency"
])

BEHAVIOR_COL = find_column([
    "thermal_behavior"
])

LANDCOVER_COL = find_column([
    "landcover",
    "land_cover",
    "worldcover_class",
    "worldcover_label",
    "landcover_label",
    "land_cover_class"
])


print(
    f"Latitude             : {LAT_COL}"
)

print(
    f"Longitude            : {LON_COL}"
)

print(
    f"FRP                  : {FRP_COL}"
)

print(
    f"Persistence           : {PERSISTENCE_COL}"
)

print(
    f"Detection count       : {DETECTION_COL}"
)

print(
    f"Thermal behavior      : {BEHAVIOR_COL}"
)

print(
    f"Land cover            : {LANDCOVER_COL}"
)


# ============================================================
# CREATE BASE FEATURES
# ============================================================

print()
print("[3/7] Creating thermal and spatial features...")


features = pd.DataFrame(
    index=df.index
)


# ------------------------------------------------------------
# Spatial
# ------------------------------------------------------------

features["latitude"] = numeric_column(
    ["center_latitude", "latitude", "lat"],
    default=20
)

features["longitude"] = numeric_column(
    ["center_longitude", "longitude", "lon"],
    default=78
)


# ------------------------------------------------------------
# Thermal
# ------------------------------------------------------------

features["mean_frp"] = numeric_column(
    ["mean_frp", "frp_mean", "frp"],
    default=0
)

features["max_frp"] = numeric_column(
    ["max_frp", "frp_max", "frp"],
    default=0
)

features["persistence_days"] = numeric_column(
    [
        "persistence_days",
        "persistence",
        "event_duration_days",
        "duration_days"
    ],
    default=0
)

features["detection_count"] = numeric_column(
    [
        "detection_count",
        "detections",
        "frequency",
        "event_frequency"
    ],
    default=1
)


# ============================================================
# DERIVED THERMAL FEATURES
# ============================================================

print()
print("[4/7] Creating derived thermal features...")


# ------------------------------------------------------------
# FRP per detection
# ------------------------------------------------------------

features["frp_per_detection"] = (
    features["mean_frp"]
    /
    features["detection_count"].replace(
        0,
        1
    )
)


# ------------------------------------------------------------
# Persistence intensity
# ------------------------------------------------------------

features["thermal_activity_rate"] = (

    features["detection_count"]
    /
    features["persistence_days"].replace(
        0,
        1
    )

)


# ------------------------------------------------------------
# High intensity indicator
# ------------------------------------------------------------

frp_threshold = (
    features["max_frp"]
    .quantile(0.75)
)


features["high_frp_event"] = (
    features["max_frp"]
    >= frp_threshold
).astype(np.int8)


# ------------------------------------------------------------
# Persistent event
# ------------------------------------------------------------

features["persistent_event"] = (
    features["persistence_days"]
    >= 3
).astype(np.int8)


# ------------------------------------------------------------
# Frequent event
# ------------------------------------------------------------

frequency_threshold = (
    features["detection_count"]
    .quantile(0.75)
)


features["frequent_event"] = (
    features["detection_count"]
    >= frequency_threshold
).astype(np.int8)


# ============================================================
# THERMAL BEHAVIOR
# ============================================================

print()
print("Encoding thermal behavior...")


if BEHAVIOR_COL:

    behavior = (
        df[BEHAVIOR_COL]
        .astype(str)
        .str.upper()
    )

else:

    behavior = pd.Series(
        "UNKNOWN",
        index=df.index
    )


features["behavior_high_intensity"] = (
    behavior.str.contains(
        "HIGH_INTENSITY",
        na=False
    )
).astype(np.int8)


features["behavior_persistent"] = (
    behavior.str.contains(
        "PERSISTENT",
        na=False
    )
).astype(np.int8)


features["behavior_frequent"] = (
    behavior.str.contains(
        "FREQUENT",
        na=False
    )
).astype(np.int8)


features["behavior_intermittent"] = (
    behavior.str.contains(
        "INTERMITTENT",
        na=False
    )
).astype(np.int8)


# ============================================================
# LAND COVER FEATURES
# ============================================================

print()
print("Encoding land-cover features...")


if LANDCOVER_COL:

    landcover = (
        df[LANDCOVER_COL]
        .astype(str)
        .str.upper()
    )

else:

    landcover = pd.Series(
        "UNKNOWN",
        index=df.index
    )


landcover_classes = {

    "forest": [
        "TREE",
        "FOREST"
    ],

    "cropland": [
        "CROP",
        "AGRICULT"
    ],

    "built_up": [
        "BUILT",
        "URBAN"
    ],

    "grassland": [
        "GRASS"
    ],

    "shrubland": [
        "SHRUB"
    ],

    "barren": [
        "BARE",
        "SPARSE"
    ],

    "water": [
        "WATER"
    ],

    "wetland": [
        "WETLAND"
    ]
}


for feature_name, keywords in landcover_classes.items():

    pattern = "|".join(keywords)

    features[
        f"landcover_{feature_name}"
    ] = (
        landcover.str.contains(
            pattern,
            na=False
        )
    ).astype(np.int8)


# ============================================================
# CONTEXT FEATURES
# ============================================================

print()
print("Encoding infrastructure and industry context...")


CONTEXT_COL = find_column([
    "context_category"
])

INDUSTRY_COL = find_column([
    "nearby_industry_type"
])

INFRA_COL = find_column([
    "nearby_infrastructure_type"
])

FACILITY_COL = find_column([
    "nearby_facility_type"
])


if CONTEXT_COL:

    context = (
        df[CONTEXT_COL]
        .astype(str)
        .str.upper()
    )

else:

    context = pd.Series(
        "UNKNOWN",
        index=df.index
    )


if INDUSTRY_COL:

    industry = (
        df[INDUSTRY_COL]
        .astype(str)
        .str.upper()
    )

else:

    industry = pd.Series(
        "NONE",
        index=df.index
    )


if INFRA_COL:

    infrastructure = (
        df[INFRA_COL]
        .astype(str)
        .str.upper()
    )

else:

    infrastructure = pd.Series(
        "NONE",
        index=df.index
    )


if FACILITY_COL:

    facility = (
        df[FACILITY_COL]
        .astype(str)
        .str.upper()
    )

else:

    facility = pd.Series(
        "NONE",
        index=df.index
    )


# ------------------------------------------------------------
# Context categories
# ------------------------------------------------------------

features["context_agricultural"] = (
    context == "AGRICULTURAL"
).astype(np.int8)


features["context_forest"] = (
    context == "FOREST"
).astype(np.int8)


features["context_urban"] = (
    context == "URBAN"
).astype(np.int8)


features["context_industrial"] = (
    context == "INDUSTRIAL"
).astype(np.int8)


features["context_mining"] = (
    context == "MINING"
).astype(np.int8)


features["context_rural"] = (
    context == "RURAL"
).astype(np.int8)


# ============================================================
# INDUSTRY FEATURES
# ============================================================

features["industry_present"] = (
    industry != "NONE"
).astype(np.int8)


features["industry_mining"] = (
    industry == "MINING"
).astype(np.int8)


features["industry_steel"] = (
    industry == "STEEL"
).astype(np.int8)


features["industry_cement"] = (
    industry == "CEMENT"
).astype(np.int8)


features["industry_chemical"] = (
    industry == "CHEMICAL"
).astype(np.int8)


features["industry_textile"] = (
    industry == "TEXTILE"
).astype(np.int8)


features["industry_food"] = (
    industry == "FOOD_PROCESSING"
).astype(np.int8)


# ============================================================
# INFRASTRUCTURE FEATURES
# ============================================================

features["infrastructure_present"] = (
    infrastructure != "NONE"
).astype(np.int8)


features["near_highway"] = (
    infrastructure == "HIGHWAY"
).astype(np.int8)


features["near_railway"] = (
    infrastructure == "RAILWAY"
).astype(np.int8)


features["near_power"] = (
    infrastructure.isin([
        "POWER_LINE",
        "POWER_SUBSTATION"
    ])
).astype(np.int8)


features["near_rural_road"] = (
    infrastructure == "RURAL_ROAD"
).astype(np.int8)


features["near_mining_road"] = (
    infrastructure == "MINING_ROAD"
).astype(np.int8)


# ============================================================
# FACILITY FEATURES
# ============================================================

features["facility_present"] = (
    facility != "NONE"
).astype(np.int8)


features["factory_present"] = (
    facility == "FACTORY"
).astype(np.int8)


features["warehouse_present"] = (
    facility == "WAREHOUSE"
).astype(np.int8)


features["power_plant_present"] = (
    facility == "POWER_PLANT"
).astype(np.int8)


features["cold_storage_present"] = (
    facility == "COLD_STORAGE"
).astype(np.int8)


# ============================================================
# COMBINED FEATURES
# ============================================================

print()
print("Creating combined risk-related features...")


features["industrial_infrastructure"] = (

    (
        features["industry_present"]
        == 1
    )

    &

    (
        features["infrastructure_present"]
        == 1
    )

).astype(np.int8)


features["high_intensity_industrial"] = (

    (
        features["high_frp_event"]
        == 1
    )

    &

    (
        features["industry_present"]
        == 1
    )

).astype(np.int8)


features["persistent_industrial"] = (

    (
        features["persistent_event"]
        == 1
    )

    &

    (
        features["industry_present"]
        == 1
    )

).astype(np.int8)


features["forest_thermal_event"] = (

    (
        features["landcover_forest"]
        == 1
    )

    &

    (
        features["high_frp_event"]
        == 1
    )

).astype(np.int8)


features["agricultural_thermal_event"] = (

    (
        features["landcover_cropland"]
        == 1
    )

    &

    (
        features["frequent_event"]
        == 1
    )

).astype(np.int8)


# ============================================================
# CLEAN FEATURES
# ============================================================

print()
print("Cleaning feature matrix...")


features = features.replace(
    [np.inf, -np.inf],
    np.nan
)


features = features.fillna(0)


# ============================================================
# CHECK
# ============================================================

print()
print("Feature matrix summary:")
print(
    f"Rows    : {len(features):,}"
)

print(
    f"Features: {len(features.columns)}"
)


missing_values = (
    features.isna()
    .sum()
    .sum()
)


print(
    f"Missing values: {missing_values:,}"
)


# ============================================================
# PRESERVE EVENT ID IF AVAILABLE
# ============================================================

EVENT_ID_COL = find_column([
    "event_id",
    "cluster_id",
    "thermal_event_id"
])


if EVENT_ID_COL:

    features.insert(
        0,
        "event_id",
        df[EVENT_ID_COL].values
    )


# ============================================================
# SAVE
# ============================================================

print()
print("Saving ML feature dataset...")


os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
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
print("✓ STEP 6 COMPLETE")
print("=" * 75)

print()

print(
    f"Input records  : {len(df):,}"
)

print(
    f"Output records : {len(features):,}"
)

print(
    f"ML features    : {len(features.columns)}"
)

print()

print("Output:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()

print("=" * 75)
print("NEXT STEP → TARGET / LABEL GENERATION")
print("=" * 75)

print()
print(
    "We will now construct the target variable"
)

print(
    "for supervised ML training."
)

print("=" * 75)