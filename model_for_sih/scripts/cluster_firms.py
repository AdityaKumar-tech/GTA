import os
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 2: SPATIAL-TEMPORAL FIRMS EVENT CLUSTERING
# ============================================================

print("=" * 70)
print("STEP 2: INDIA-WIDE FIRMS THERMAL EVENT CLUSTERING")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    r"data\raw\firms\india"
    r"\india_firms_viirs_2025-07-30_to_2026-07-30.csv"
)

OUTPUT_DIR = r"data\processed\events"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "firms_thermal_events.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Spatial radius approximately 1 km
SPATIAL_EPS_KM = 1.0

# Temporal radius
TEMPORAL_EPS_DAYS = 3

# Minimum number of detections required
MIN_SAMPLES = 2


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n[1/7] Loading FIRMS dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Detections loaded: {len(df):,}")


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

print("\n[2/7] Checking required columns...")

required_columns = [
    "latitude",
    "longitude",
    "acq_date",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns: {missing}"
    )

print("✓ Required columns available")


# ------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------

print("\n[3/7] Cleaning detections...")

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
)

if "frp" in df.columns:
    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce"
    )

if "bright_ti4" in df.columns:
    df["bright_ti4"] = pd.to_numeric(
        df["bright_ti4"],
        errors="coerce"
    )

if "bright_ti5" in df.columns:
    df["bright_ti5"] = pd.to_numeric(
        df["bright_ti5"],
        errors="coerce"
    )


before = len(df)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "acq_date"
    ]
)

print(
    f"Removed invalid records: "
    f"{before - len(df):,}"
)

print(
    f"Valid detections: {len(df):,}"
)


# ------------------------------------------------------------
# CREATE NUMERIC SPATIAL-TEMPORAL FEATURES
# ------------------------------------------------------------

print("\n[4/7] Creating spatial-temporal coordinates...")


# Convert latitude/longitude to approximate kilometres.

lat_mean = df["latitude"].mean()

lat_scale = 111.32

lon_scale = (
    111.32 *
    np.cos(
        np.radians(lat_mean)
    )
)

df["x_km"] = (
    df["longitude"] *
    lon_scale
)

df["y_km"] = (
    df["latitude"] *
    lat_scale
)


# Convert date to days from earliest observation

df["date_days"] = (
    df["acq_date"] -
    df["acq_date"].min()
).dt.total_seconds() / 86400.0


# ------------------------------------------------------------
# SCALE TIME
# ------------------------------------------------------------

# We want:
#
# 1 spatial DBSCAN unit ≈ 1 km
#
# and
#
# TEMPORAL_EPS_DAYS ≈ 3 days
#
# Therefore scale temporal dimension so that
# 3 days has approximately the same influence as 1 km.

df["time_scaled"] = (
    df["date_days"] /
    TEMPORAL_EPS_DAYS
)


# ------------------------------------------------------------
# BUILD CLUSTERING MATRIX
# ------------------------------------------------------------

X = df[
    [
        "x_km",
        "y_km",
        "time_scaled"
    ]
].values


# ------------------------------------------------------------
# CLUSTER
# ------------------------------------------------------------

print("\n[5/7] Running DBSCAN clustering...")

print(
    f"Spatial radius : "
    f"{SPATIAL_EPS_KM} km"
)

print(
    f"Temporal radius: "
    f"{TEMPORAL_EPS_DAYS} days"
)

print(
    f"Minimum samples: "
    f"{MIN_SAMPLES}"
)


# Scale spatial coordinates so EPS corresponds
# approximately to the desired spatial radius.

X_scaled = X.copy()

X_scaled[:, 0] /= SPATIAL_EPS_KM
X_scaled[:, 1] /= SPATIAL_EPS_KM


clusterer = DBSCAN(
    eps=1.0,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    n_jobs=-1
)

labels = clusterer.fit_predict(
    X_scaled
)

df["event_cluster_id"] = labels


# ------------------------------------------------------------
# REMOVE NOISE
# ------------------------------------------------------------

noise_count = int(
    (df["event_cluster_id"] == -1).sum()
)

clustered = df[
    df["event_cluster_id"] != -1
].copy()

print(
    f"\nNoise detections: {noise_count:,}"
)

print(
    f"Clustered detections: {len(clustered):,}"
)

print(
    f"Candidate events: "
    f"{clustered['event_cluster_id'].nunique():,}"
)


# ------------------------------------------------------------
# CREATE EVENT-LEVEL FEATURES
# ------------------------------------------------------------

print("\n[6/7] Creating event-level features...")


grouped = clustered.groupby(
    "event_cluster_id"
)


events = grouped.agg(
    total_detections=(
        "event_cluster_id",
        "size"
    ),

    detection_days=(
        "acq_date",
        lambda x: x.dt.date.nunique()
    ),

    first_detection=(
        "acq_date",
        "min"
    ),

    last_detection=(
        "acq_date",
        "max"
    ),

    center_latitude=(
        "latitude",
        "mean"
    ),

    center_longitude=(
        "longitude",
        "mean"
    )
).reset_index()


# ------------------------------------------------------------
# FRP FEATURES
# ------------------------------------------------------------

if "frp" in clustered.columns:

    frp_features = grouped["frp"].agg(
        avg_frp="mean",
        max_frp="max",
        min_frp="min",
        frp_std="std"
    ).reset_index()

    events = events.merge(
        frp_features,
        on="event_cluster_id",
        how="left"
    )


# ------------------------------------------------------------
# BRIGHTNESS FEATURES
# ------------------------------------------------------------

if "bright_ti4" in clustered.columns:

    ti4_features = grouped["bright_ti4"].agg(
        avg_bright_ti4="mean",
        max_bright_ti4="max"
    ).reset_index()

    events = events.merge(
        ti4_features,
        on="event_cluster_id",
        how="left"
    )


if "bright_ti5" in clustered.columns:

    ti5_features = grouped["bright_ti5"].agg(
        avg_bright_ti5="mean"
    ).reset_index()

    events = events.merge(
        ti5_features,
        on="event_cluster_id",
        how="left"
    )


# ------------------------------------------------------------
# DAY/NIGHT FEATURES
# ------------------------------------------------------------

if "daynight" in clustered.columns:

    night_counts = (
        clustered["daynight"]
        .astype(str)
        .str.upper()
        .eq("N")
        .groupby(
            clustered["event_cluster_id"]
        )
        .sum()
        .reset_index(
            name="night_detections"
        )
    )

    events = events.merge(
        night_counts,
        on="event_cluster_id",
        how="left"
    )

    events["night_detection_ratio"] = (
        events["night_detections"] /
        events["total_detections"]
    )


# ------------------------------------------------------------
# PERSISTENCE
# ------------------------------------------------------------

print("\nCalculating persistence...")

global_start = df["acq_date"].min()
global_end = df["acq_date"].max()

observation_days = (
    global_end - global_start
).days + 1

events["observation_days"] = (
    observation_days
)

events["persistence_percent"] = (
    events["detection_days"] /
    events["observation_days"]
) * 100.0


# ------------------------------------------------------------
# DETECTIONS PER ACTIVE DAY
# ------------------------------------------------------------

events["detections_per_active_day"] = (
    events["total_detections"] /
    events["detection_days"]
)


# ------------------------------------------------------------
# THERMAL BEHAVIOR
# ------------------------------------------------------------

def classify_behavior(row):

    persistence = row[
        "persistence_percent"
    ]

    detections = row[
        "total_detections"
    ]

    avg_frp = row.get(
        "avg_frp",
        np.nan
    )

    max_frp = row.get(
        "max_frp",
        np.nan
    )

    if persistence >= 20:
        return "PERSISTENT_THERMAL_SOURCE"

    if max_frp >= 10:
        return "HIGH_INTENSITY_EVENT"

    if detections >= 20:
        return "FREQUENT_THERMAL_ACTIVITY"

    return "INTERMITTENT_THERMAL_ACTIVITY"


events["thermal_behavior"] = events.apply(
    classify_behavior,
    axis=1
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print("\n[7/7] Saving event dataset...")

events.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("✓ STEP 2 COMPLETE")
print("=" * 70)

print()

print(
    f"Input detections : "
    f"{len(df):,}"
)

print(
    f"Candidate events : "
    f"{len(events):,}"
)

print(
    f"Noise detections : "
    f"{noise_count:,}"
)

print()

print("Output:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()

print("Thermal behavior distribution:")
print(
    events[
        "thermal_behavior"
    ]
    .value_counts()
    .to_string()
)

print()

print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "\nStep 3 → Add land-cover/context information"
)

print("=" * 70)