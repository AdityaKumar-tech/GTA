import os
import math
import requests
from tqdm import tqdm

# ============================================================
# STEP 3A: DOWNLOAD ESA WORLDCOVER TILES FOR INDIA
# ============================================================

print("=" * 70)
print("STEP 3A: ESA WORLDCOVER INDIA DATA")
print("=" * 70)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

WEST = 68.0
SOUTH = 6.0
EAST = 97.5
NORTH = 37.5

YEAR = 2021
VERSION = "v200"

OUTPUT_DIR = r"data\raw\worldcover\india"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    f"{VERSION}/{YEAR}/map/"
)

# ------------------------------------------------------------
# WORLD COVER TILE NAME
# ------------------------------------------------------------

def latitude_code(lat):
    if lat >= 0:
        return f"N{int(lat):02d}"
    return f"S{abs(int(lat)):02d}"


def longitude_code(lon):
    if lon >= 0:
        return f"E{int(lon):03d}"
    return f"W{abs(int(lon)):03d}"


# ------------------------------------------------------------
# CREATE REQUIRED TILE LIST
# ------------------------------------------------------------

tiles = []

lat_start = math.floor(SOUTH / 3) * 3
lat_end = math.ceil(NORTH / 3) * 3

lon_start = math.floor(WEST / 3) * 3
lon_end = math.ceil(EAST / 3) * 3

for lat in range(lat_start, lat_end, 3):

    for lon in range(lon_start, lon_end, 3):

        # Tile lower-left coordinate
        tile = (
            latitude_code(lat)
            + longitude_code(lon)
        )

        tiles.append(tile)

print()
print("India bounding box:")
print(f"Longitude: {WEST} → {EAST}")
print(f"Latitude : {SOUTH} → {NORTH}")

print()
print(f"Potential tiles: {len(tiles)}")

# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

downloaded = 0
skipped = 0
failed = 0

for index, tile in enumerate(tiles, start=1):

    filename = (
        f"ESA_WorldCover_10m_{YEAR}_{VERSION}_"
        f"{tile}_Map.tif"
    )

    url = BASE_URL + filename

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print()
    print("-" * 70)
    print(f"[{index}/{len(tiles)}] {tile}")

    # Already downloaded
    if os.path.exists(output_path):

        size_mb = (
            os.path.getsize(output_path)
            / (1024 * 1024)
        )

        print(
            f"Already exists: {size_mb:.1f} MB"
        )

        skipped += 1
        continue

    try:

        print("Downloading...")

        response = requests.get(
            url,
            stream=True,
            timeout=180
        )

        if response.status_code != 200:

            print(
                f"ERROR HTTP {response.status_code}"
            )

            failed += 1
            continue

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        with open(
            output_path,
            "wb"
        ) as f:

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=tile
            ) as pbar:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)
                        pbar.update(len(chunk))

        downloaded += 1

        print("✓ Download complete")

    except Exception as e:

        print(
            f"Download failed: {e}"
        )

        failed += 1

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("WORLD COVER DOWNLOAD COMPLETE")
print("=" * 70)

print()
print(f"Tiles identified : {len(tiles)}")
print(f"Downloaded       : {downloaded}")
print(f"Already existed  : {skipped}")
print(f"Failed           : {failed}")

print()
print("Output directory:")
print(os.path.abspath(OUTPUT_DIR))

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print()
print(
    "Step 3B → Extract land-cover features "
    "for every FIRMS thermal event."
)

print("=" * 70)