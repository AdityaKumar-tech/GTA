import os
import glob

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import Window

# ============================================================
# STEP 3B: EXTRACT LAND-COVER FEATURES
# ============================================================

print("=" * 70)
print("STEP 3B: LAND-COVER FEATURE EXTRACTION")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_FILE = (
    r"data\processed\events\firms_thermal_events.csv"
)

WORLD_COVER_DIR = (
    r"data\raw\worldcover\india"
)

OUTPUT_DIR = (
    r"data\processed\landcover"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "thermal_events_landcover.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# LOAD EVENTS
# ------------------------------------------------------------

print()
print("[1/5] Loading thermal events...")

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Thermal events: {len(df):,}"
)

# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required = [
    "event_cluster_id",
    "center_latitude",
    "center_longitude"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )

# ------------------------------------------------------------
# LOAD WORLD COVER TILES
# ------------------------------------------------------------

print()
print("[2/5] Loading WorldCover tiles...")

files = glob.glob(
    os.path.join(
        WORLD_COVER_DIR,
        "*.tif"
    )
)

if not files:

    raise FileNotFoundError(
        "No WorldCover GeoTIFF files found."
    )

print(
    f"WorldCover tiles found: {len(files)}"
)

datasets = []

for path in files:

    try:

        src = rasterio.open(path)

        datasets.append(src)

        print(
            f"✓ {os.path.basename(path)}"
        )

        print(
            f"  CRS: {src.crs}"
        )

        print(
            f"  Bounds: {src.bounds}"
        )

    except Exception as e:

        print(
            f"Could not open {path}: {e}"
        )

# ------------------------------------------------------------
# WORLDCOVER CLASS NAMES
# ------------------------------------------------------------

class_names = {

    10: "TREE_COVER",

    20: "SHRUBLAND",

    30: "GRASSLAND",

    40: "CROPLAND",

    50: "BUILT_UP",

    60: "BARE_SPARSE_VEGETATION",

    70: "SNOW_ICE",

    80: "PERMANENT_WATER",

    90: "HERBACEOUS_WETLAND",

    95: "MANGROVE",

    100: "MOSS_LICHEN"
}

# ------------------------------------------------------------
# POINT LOOKUP
# ------------------------------------------------------------

def get_landcover(lat, lon):

    for src in datasets:

        bounds = src.bounds

        if (
            bounds.left <= lon <= bounds.right
            and
            bounds.bottom <= lat <= bounds.top
        ):

            try:

                row, col = src.index(
                    lon,
                    lat
                )

                if (
                    row < 0
                    or col < 0
                    or row >= src.height
                    or col >= src.width
                ):
                    continue

                value = src.read(
                    1,
                    window=Window(
                        col,
                        row,
                        1,
                        1
                    )
                )[0, 0]

                if value == src.nodata:

                    return np.nan

                return int(value)

            except Exception:

                return np.nan

    return np.nan


# ------------------------------------------------------------
# EXTRACT
# ------------------------------------------------------------

print()
print("[3/5] Extracting land-cover classes...")

landcover_values = []

total = len(df)

for i, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    lat = getattr(
        row,
        "center_latitude"
    )

    lon = getattr(
        row,
        "center_longitude"
    )

    value = get_landcover(
        lat,
        lon
    )

    landcover_values.append(value)

    if (
        i <= 10
        or i % 10000 == 0
        or i == total
    ):

        print(
            f"[{i:,}/{total:,}] "
            f"Land-cover extraction"
        )

# ------------------------------------------------------------
# ADD FEATURES
# ------------------------------------------------------------

df["worldcover_class"] = (
    landcover_values
)

df["worldcover_name"] = (
    df["worldcover_class"]
    .map(class_names)
    .fillna("UNKNOWN")
)

# ------------------------------------------------------------
# LAND-COVER GROUP
# ------------------------------------------------------------

def classify_landcover(name):

    if name == "TREE_COVER":
        return "NATURAL"

    if name in [
        "SHRUBLAND",
        "GRASSLAND",
        "HERBACEOUS_WETLAND",
        "MANGROVE",
        "MOSS_LICHEN"
    ]:
        return "NATURAL"

    if name == "CROPLAND":
        return "AGRICULTURAL"

    if name == "BUILT_UP":
        return "BUILT_ENVIRONMENT"

    if name == "BARE_SPARSE_VEGETATION":
        return "BARE_LAND"

    if name == "PERMANENT_WATER":
        return "WATER"

    if name == "SNOW_ICE":
        return "SNOW_ICE"

    return "UNKNOWN"


df["landcover_group"] = (
    df["worldcover_name"]
    .apply(classify_landcover)
)

# ------------------------------------------------------------
# VALIDITY
# ------------------------------------------------------------

df["landcover_valid"] = (
    df["worldcover_class"]
    .notna()
    .astype(int)
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print()
print("[4/5] Saving land-cover dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("[5/5] Summary")

print("=" * 70)

print(
    f"Events processed : {len(df):,}"
)

print(
    f"Valid land-cover: "
    f"{df['landcover_valid'].sum():,}"
)

print(
    f"Unknown         : "
    f"{(df['landcover_valid'] == 0).sum():,}"
)

print()
print("Land-cover distribution:")
print(
    df["worldcover_name"]
    .value_counts()
    .to_string()
)

print()
print("Land-cover groups:")
print(
    df["landcover_group"]
    .value_counts()
    .to_string()
)

print()
print("Output:")
print(
    os.path.abspath(OUTPUT_FILE)
)

# ------------------------------------------------------------
# CLOSE RASTERS
# ------------------------------------------------------------

for src in datasets:
    src.close()

print()
print("=" * 70)
print("✓ STEP 3B COMPLETE")
print("=" * 70)

print()
print(
    "Next → Step 3C: "
    "OpenStreetMap infrastructure enrichment"
)

print("=" * 70)