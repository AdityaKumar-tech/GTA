import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 7: CREATE ACTUAL TRAINING DATASET
# ============================================================

print("=" * 75)
print("STEP 7: CREATING ACTUAL FIRE-TYPE TRAINING DATASET")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\ml\fire_type_dataset.csv"
)

OUTPUT_FILE = (
    r"data\processed\ml\fire_type_training.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/8] Loading fire-type dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Records loaded: {len(df):,}"
)


# ============================================================
# CHECK TARGET
# ============================================================

print()
print("[2/8] Checking target variable...")

if "fire_type" not in df.columns:
    raise ValueError(
        "fire_type column is missing."
    )

print()
print("Fire types:")
print(
    df["fire_type"].value_counts().to_string()
)


# ============================================================
# REMOVE INVALID TARGETS
# ============================================================

print()
print("[3/8] Removing invalid target records...")

before = len(df)

df = df[
    df["fire_type"].notna()
].copy()

df = df[
    df["fire_type"].astype(str).str.strip() != ""
].copy()

removed = before - len(df)

print(
    f"Removed: {removed:,}"
)

print(
    f"Valid records: {len(df):,}"
)


# ============================================================
# FEATURES
# ============================================================

print()
print("[4/8] Selecting training features...")


# ------------------------------------------------------------
# These are legitimate predictors.
# ------------------------------------------------------------

feature_columns = [

    # Spatial
    "latitude",
    "longitude",

    # Detection characteristics
    "total_detections",
    "detection_days",

    # Thermal intensity
    "avg_frp",
    "max_frp",
    "min_frp",
    "frp_std",

    # Brightness temperature
    "avg_bright_ti4",
    "max_bright_ti4",
    "avg_bright_ti5",

    # Night activity
    "night_detections",
    "night_detection_ratio",

    # Persistence
    "observation_days",
    "persistence_percent",

    # Detection activity
    "detections_per_active_day",

    # Derived thermal features
    "frp_range",
    "frp_variability",
    "detections_per_observation_day",
    "night_activity_intensity",

    # Thermal intensity indicators
    "high_intensity_indicator",
    "persistent_indicator",

    # Land cover
    "landcover_natural",
    "landcover_agricultural",
    "landcover_urban",
    "landcover_water",
    "landcover_barren",
    "landcover_valid",
    "worldcover_class",

    # Context
    "context_agricultural",
    "context_industrial",
    "context_mining",
    "context_rural",

    # Infrastructure
    "industrial_presence",
    "infrastructure_presence",
    "facility_presence",
    "nearby_highway",
    "nearby_railway",
    "nearby_power_infrastructure",
    "nearby_mining",
    "mixed_industrial_infrastructure"
]


# ------------------------------------------------------------
# Verify features
# ------------------------------------------------------------

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing training features:\n"
        + "\n".join(missing_features)
    )


print(
    f"Selected features: {len(feature_columns)}"
)


# ============================================================
# CREATE X
# ============================================================

print()
print("[5/8] Creating feature matrix X...")

X = df[
    feature_columns
].copy()


# Convert everything to numeric

for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Missing numerical values

missing_before = (
    X.isna()
    .sum()
    .sum()
)

print(
    f"Missing values before cleaning: "
    f"{missing_before:,}"
)


X = X.fillna(0)


# ============================================================
# CREATE Y
# ============================================================

print()
print("[6/8] Creating target vector y...")

y = (
    df["fire_type"]
    .astype(str)
    .str.strip()
)


# ============================================================
# COMBINE
# ============================================================

training_df = X.copy()

training_df["fire_type"] = y


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("[7/8] Validating training dataset...")

print(
    f"Training records : {len(training_df):,}"
)

print(
    f"Input features   : {len(X.columns):,}"
)

print(
    f"Target classes    : {y.nunique():,}"
)

print(
    f"Missing values    : "
    f"{training_df.isna().sum().sum():,}"
)

print(
    f"Infinite values   : "
    f"{np.isinf(
        training_df
        .select_dtypes(include=np.number)
    ).sum().sum():,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Target distribution")
print("-" * 75)

distribution = (
    y.value_counts()
)

for label, count in distribution.items():

    percentage = (
        count
        /
        len(y)
        *
        100
    )

    print(
        f"{label:35} "
        f"{count:8,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SAVE
# ============================================================

print()
print("[8/8] Saving training dataset...")

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("✓ STEP 7 COMPLETE")
print("=" * 75)

print()
print(
    f"Records       : {len(training_df):,}"
)

print(
    f"Features      : {len(X.columns):,}"
)

print(
    f"Target        : fire_type"
)

print(
    f"Classes       : {y.nunique()}"
)

print()
print("Output:")
print(
    os.path.abspath(OUTPUT_FILE)
)

print()
print("=" * 75)
print("NEXT STEP → TRAIN / TEST SPLIT")
print("=" * 75)
