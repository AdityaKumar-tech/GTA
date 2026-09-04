import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 6C: RISK LABEL GENERATION
# ============================================================

print("=" * 75)
print("STEP 6C: THERMAL EVENT RISK LABEL GENERATION")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\ml\model_features.csv"
)

OUTPUT_FILE = (
    r"data\processed\ml\training_dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

print()
print("[1/7] Loading ML feature dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Events loaded: {len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

print()
print("[2/7] Checking required columns...")

required = [
    "avg_frp",
    "max_frp",
    "frp_std",
    "total_detections",
    "detection_days",
    "persistence_percent",
    "night_detection_ratio",
    "detections_per_active_day",
    "industrial_presence",
    "infrastructure_presence",
    "facility_presence",
    "nearby_highway",
    "nearby_railway",
    "nearby_power_infrastructure",
    "nearby_mining",
    "mixed_industrial_infrastructure",
    "landcover_natural",
    "landcover_agricultural",
    "landcover_urban",
    "context_industrial",
    "context_mining"
]

for column in required:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )

print(
    f"✓ All {len(required)} required columns available"
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def percentile_score(series, low=0.0, high=100.0):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    ranks = values.rank(
        pct=True,
        method="average"
    )

    return (
        ranks * (high - low) + low
    )


# ============================================================
# CREATE COMPONENT SCORES
# ============================================================

print()
print("[3/7] Calculating thermal risk components...")


# ------------------------------------------------------------
# 1. THERMAL INTENSITY
# ------------------------------------------------------------

frp_score = percentile_score(
    df["max_frp"]
)


# ------------------------------------------------------------
# 2. EVENT FREQUENCY
# ------------------------------------------------------------

frequency_score = percentile_score(
    df["total_detections"]
)


# ------------------------------------------------------------
# 3. PERSISTENCE
# ------------------------------------------------------------

persistence_score = percentile_score(
    df["persistence_percent"]
)


# ------------------------------------------------------------
# 4. NIGHT ACTIVITY
# ------------------------------------------------------------

night_score = (
    pd.to_numeric(
        df["night_detection_ratio"],
        errors="coerce"
    )
    .fillna(0)
    * 100
)


# ------------------------------------------------------------
# 5. ACTIVITY DENSITY
# ------------------------------------------------------------

activity_score = percentile_score(
    df["detections_per_active_day"]
)


# ============================================================
# CONTEXT SCORE
# ============================================================

print()
print("[4/7] Calculating infrastructure/context risk...")


context_score = np.zeros(
    len(df)
)


# Industrial presence

context_score += (
    df["industrial_presence"]
    .fillna(0)
    * 15
)


# Mining

context_score += (
    df["nearby_mining"]
    .fillna(0)
    * 15
)


# Power infrastructure

context_score += (
    df["nearby_power_infrastructure"]
    .fillna(0)
    * 12
)


# Railway

context_score += (
    df["nearby_railway"]
    .fillna(0)
    * 8
)


# Highway

context_score += (
    df["nearby_highway"]
    .fillna(0)
    * 5
)


# Facility

context_score += (
    df["facility_presence"]
    .fillna(0)
    * 8
)


# Mixed industrial/infrastructure

context_score += (
    df["mixed_industrial_infrastructure"]
    .fillna(0)
    * 12
)


# Industrial context

context_score += (
    df["context_industrial"]
    .fillna(0)
    * 10
)


# Mining context

context_score += (
    df["context_mining"]
    .fillna(0)
    * 10
)


# Cap score

context_score = np.clip(
    context_score,
    0,
    100
)


# ============================================================
# COMBINE SCORES
# ============================================================

print()
print("[5/7] Creating final risk score...")


risk_score = (

    frp_score * 0.30

    +

    frequency_score * 0.15

    +

    persistence_score * 0.20

    +

    night_score * 0.10

    +

    activity_score * 0.10

    +

    context_score * 0.15

)


risk_score = np.clip(
    risk_score,
    0,
    100
)


df["risk_score"] = (
    risk_score.round(2)
)


# ============================================================
# RISK LEVEL
# ============================================================

print()
print("[6/7] Assigning risk levels...")


def assign_risk(score):

    if score < 25:
        return "LOW"

    elif score < 50:
        return "MODERATE"

    elif score < 75:
        return "HIGH"

    else:
        return "CRITICAL"


df["risk_level"] = (
    df["risk_score"]
    .apply(assign_risk)
)


# ============================================================
# DISTRIBUTION
# ============================================================

print()
print("Risk distribution:")
print("-" * 50)

distribution = (
    df["risk_level"]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MODERATE",
            "HIGH",
            "CRITICAL"
        ],
        fill_value=0
    )
)

for level, count in distribution.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{level:10} : "
        f"{count:8,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SAVE
# ============================================================

print()
print("[7/7] Saving training dataset...")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("✓ STEP 6C COMPLETE")
print("=" * 75)

print()
print(
    f"Events          : {len(df):,}"
)

print(
    f"Features        : {len(df.columns):,}"
)

print(
    f"Risk score      : 0–100"
)

print(
    f"Risk classes    : {df['risk_level'].nunique()}"
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