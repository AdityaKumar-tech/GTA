import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# STEP 1: INDIA-WIDE FIRMS DATA DOWNLOAD
# ============================================================

print("=" * 70)
print("STEP 1: INDIA-WIDE NASA FIRMS DATA DOWNLOAD")
print("=" * 70)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise RuntimeError(
        "\nFIRMS_MAP_KEY environment variable is not set.\n"
        "Set it in PowerShell before running this script."
    )

# India approximate bounding box
WEST = 68.0
SOUTH = 6.0
EAST = 97.5
NORTH = 37.5

# ------------------------------------------------------------
# IMPORTANT:
# Start with a manageable period.
# Once this works, increase the period.
# ------------------------------------------------------------

START_DATE = "2025-07-30"
END_DATE = "2026-07-30"

# VIIRS source
SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
]

OUTPUT_DIR = r"data\raw\firms\india"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "india_firms_viirs_2025-07-30_to_2026-07-30.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def daterange_chunks(start_date, end_date, chunk_days=5):
    current = start_date

    while current <= end_date:
        chunk_end = min(
            current + timedelta(days=chunk_days - 1),
            end_date
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)


def download_chunk(source, start_date, end_date):

    day_range = (end_date - start_date).days + 1

    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/{MAP_KEY}/"
        f"{source}/"
        f"{WEST},{SOUTH},{EAST},{NORTH}/"
        f"{day_range}/"
        f"{start_date.strftime('%Y-%m-%d')}"
    )

    print()
    print("-" * 70)
    print(f"Source : {source}")
    print(
        f"Dates  : "
        f"{start_date.strftime('%Y-%m-%d')} → "
        f"{end_date.strftime('%Y-%m-%d')}"
    )

    try:

        response = requests.get(
            url,
            timeout=120
        )

        if response.status_code != 200:

            print(
                f"ERROR: HTTP {response.status_code}"
            )

            print(response.text[:500])

            return None

        if not response.text.strip():

            print("No data returned.")

            return None

        from io import StringIO

        df = pd.read_csv(
            StringIO(response.text)
        )

        if df.empty:

            print("No detections.")

            return None

        print(
            f"Detections received: {len(df):,}"
        )

        return df

    except Exception as e:

        print(
            f"Download error: {e}"
        )

        return None


# ------------------------------------------------------------
# DATE RANGE
# ------------------------------------------------------------

start_date = datetime.strptime(
    START_DATE,
    "%Y-%m-%d"
).date()

end_date = datetime.strptime(
    END_DATE,
    "%Y-%m-%d"
).date()

# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

print()
print("Configuration")
print("-" * 70)

print(f"India bbox : {WEST}, {SOUTH}, {EAST}, {NORTH}")
print(f"Start date : {START_DATE}")
print(f"End date   : {END_DATE}")

print()
print("Sources:")
for source in SOURCES:
    print(f"  - {source}")

print()
print("Starting download...")

all_data = []

total_requests = 0

for source in SOURCES:

    chunks = list(
        daterange_chunks(
            start_date,
            end_date,
            chunk_days=5
        )
    )

    print()
    print("=" * 70)
    print(
        f"{source}: {len(chunks)} chunks"
    )
    print("=" * 70)

    for index, (chunk_start, chunk_end) in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\n[{index}/{len(chunks)}]"
        )

        df = download_chunk(
            source,
            chunk_start,
            chunk_end
        )

        total_requests += 1

        if df is not None:

            df["firms_source"] = source

            all_data.append(df)

        # Avoid hammering the API
        time.sleep(1)


# ------------------------------------------------------------
# CHECK RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

print(
    f"API requests: {total_requests}"
)

if not all_data:

    raise RuntimeError(
        "\nNo FIRMS data was downloaded."
    )

df = pd.concat(
    all_data,
    ignore_index=True
)

print(
    f"Raw detections: {len(df):,}"
)

# ------------------------------------------------------------
# CLEAN COLUMN TYPES
# ------------------------------------------------------------

print()
print("Cleaning data...")

if "latitude" in df.columns:

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

if "longitude" in df.columns:

    df["longitude"] = pd.to_numeric(
        df["longitude"],
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

# ------------------------------------------------------------
# REMOVE INVALID COORDINATES
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
)

df = df[
    (df["latitude"] >= SOUTH)
    & (df["latitude"] <= NORTH)
    & (df["longitude"] >= WEST)
    & (df["longitude"] <= EAST)
]

print(
    f"Removed invalid coordinates: "
    f"{before - len(df):,}"
)

# ------------------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------------------

before = len(df)

duplicate_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
]

duplicate_columns = [
    col
    for col in duplicate_columns
    if col in df.columns
]

if duplicate_columns:

    df = df.drop_duplicates(
        subset=duplicate_columns
    )

print(
    f"Removed duplicates: "
    f"{before - len(df):,}"
)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

if "acq_date" in df.columns:

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    df = df.sort_values(
        "acq_date"
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print()
print("Saving dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("✓ FIRMS INDIA DATASET CREATED")
print("=" * 70)

print()
print(f"Output:")
print(os.path.abspath(OUTPUT_FILE))

print()
print(f"Total detections: {len(df):,}")

if "satellite" in df.columns:

    print()
    print("Satellite distribution:")
    print(
        df["satellite"]
        .value_counts()
        .to_string()
    )

if "daynight" in df.columns:

    print()
    print("Day/night distribution:")
    print(
        df["daynight"]
        .value_counts()
        .to_string()
    )

if "frp" in df.columns:

    print()
    print("FRP statistics:")
    print(
        df["frp"]
        .describe()
        .to_string()
    )

if "acq_date" in df.columns:

    print()
    print("Date range:")

    print(
        df["acq_date"].min(),
        "→",
        df["acq_date"].max()
    )

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "\nStep 2 → Create spatial thermal-event candidates"
)

print(
    "\nThe next module will cluster/group individual "
    "FIRMS detections into persistent thermal sources "
    "and candidate fire events."
)

print("=" * 70)