import os
import time
import json
import math
import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from shapely.strtree import STRtree


# ============================================================
# STEP 3C: OPENSTREETMAP INFRASTRUCTURE ENRICHMENT
# INDIA-WIDE
# ============================================================

print("=" * 70)
print("STEP 3C: OPENSTREETMAP INFRASTRUCTURE ENRICHMENT")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\events\firms_thermal_events.csv"
)

OUTPUT_DIR = (
    r"data\processed\context"
)

CACHE_DIR = os.path.join(
    OUTPUT_DIR,
    "osm_cache"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "firms_thermal_events_osm.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# INDIA BOUNDING BOX
# ============================================================

WEST = 68.0
SOUTH = 6.0
EAST = 97.5
NORTH = 37.5


# ============================================================
# GRID CONFIGURATION
# ============================================================

# Approximately 1 degree cells.
#
# India:
# longitude: 68 → 97.5
# latitude : 6 → 37.5
#
# approximately 30 x 32 cells

GRID_SIZE = 1.0


# ============================================================
# OSM / OVERPASS SERVERS
# ============================================================

OVERPASS_URLS = [

    "https://overpass-api.de/api/interpreter",

    "https://overpass.private.coffee/api/interpreter",

    "https://overpass.kumi.systems/api/interpreter",

]


# ============================================================
# REQUEST CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = 180

MAX_RETRIES = 4

SLEEP_BETWEEN_REQUESTS = 2


HEADERS = {

    "User-Agent":
        "SIH-Industrial-Thermal-Monitor/1.0 "
        "(India geospatial research project)",

    "Accept":
        "application/json",

    "Content-Type":
        "text/plain",

}


# ============================================================
# LOAD THERMAL EVENTS
# ============================================================

print()
print("[1/8] Loading thermal events...")
print("-" * 70)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\nThermal event file not found:\n{INPUT_FILE}"
    )


events = pd.read_csv(INPUT_FILE)


print(
    f"Events loaded: {len(events):,}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [

    "event_cluster_id",

    "center_latitude",

    "center_longitude",

]


missing = [

    c for c in required_columns
    if c not in events.columns

]


if missing:

    raise ValueError(
        f"\nMissing required columns:\n{missing}"
    )


# ============================================================
# CLEAN COORDINATES
# ============================================================

events["center_latitude"] = pd.to_numeric(
    events["center_latitude"],
    errors="coerce"
)

events["center_longitude"] = pd.to_numeric(
    events["center_longitude"],
    errors="coerce"
)


events = events.dropna(
    subset=[
        "center_latitude",
        "center_longitude"
    ]
).copy()


events = events[
    (events["center_latitude"] >= SOUTH)
    &
    (events["center_latitude"] <= NORTH)
    &
    (events["center_longitude"] >= WEST)
    &
    (events["center_longitude"] <= EAST)
].copy()


events = events.reset_index(drop=True)


print(
    f"Valid events: {len(events):,}"
)


# ============================================================
# CREATE INDIA GRID
# ============================================================

print()
print("[2/8] Creating India grid...")
print("-" * 70)


grid = []


cell_id = 0

lat = SOUTH


while lat < NORTH:

    lon = WEST

    lat_max = min(
        lat + GRID_SIZE,
        NORTH
    )

    while lon < EAST:

        lon_max = min(
            lon + GRID_SIZE,
            EAST
        )

        grid.append({

            "cell_id": cell_id,

            "west": lon,

            "south": lat,

            "east": lon_max,

            "north": lat_max,

        })

        cell_id += 1

        lon += GRID_SIZE

    lat += GRID_SIZE


print(
    f"India grid cells: {len(grid)}"
)


# ============================================================
# ASSIGN EVENTS TO GRID
# ============================================================

def get_cell_id(lat, lon):

    col = int(
        math.floor(
            (lon - WEST) / GRID_SIZE
        )
    )

    row = int(
        math.floor(
            (lat - SOUTH) / GRID_SIZE
        )
    )

    cells_per_row = int(
        math.ceil(
            (EAST - WEST) / GRID_SIZE
        )
    )

    cell = (
        row * cells_per_row
        + col
    )

    if cell < 0 or cell >= len(grid):

        return None

    return cell


events["osm_cell_id"] = [

    get_cell_id(lat, lon)

    for lat, lon in zip(
        events["center_latitude"],
        events["center_longitude"]
    )

]


# ============================================================
# OSM QUERY
# ============================================================

def build_query(cell):

    west = cell["west"]

    south = cell["south"]

    east = cell["east"]

    north = cell["north"]


    query = f"""

[out:json][timeout:120];

(
    /*
    INDUSTRIAL
    */

    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "industrial"
    ];


    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "landuse"="industrial"
    ];


    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "man_made"="works"
    ];


    /*
    POWER
    */

    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "power"
    ];


    /*
    RAILWAY
    */

    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "railway"
    ];


    /*
    ROADS
    */

    way(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "highway"
    ];


    /*
    NATURAL FEATURES
    */

    nwr(
        {south},
        {west},
        {north},
        {east}
    )
    [
        "natural"
    ];

);

out center tags;

"""

    return query


# ============================================================
# DOWNLOAD OSM CELL
# ============================================================

def download_osm_cell(cell):

    cell_id = cell["cell_id"]

    cache_file = os.path.join(
        CACHE_DIR,
        f"osm_cell_{cell_id}.json"
    )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if os.path.exists(cache_file):

        try:

            with open(
                cache_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            print(
                f"  ✓ Cache found: "
                f"{len(data.get('elements', []))} features"
            )

            return data

        except Exception:

            print(
                "  Cache corrupted. Re-downloading."
            )


    query = build_query(cell)


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        server_index = (
            (cell_id + attempt - 1)
            % len(OVERPASS_URLS)
        )

        url = OVERPASS_URLS[
            server_index
        ]


        print(
            f"  Attempt {attempt}/{MAX_RETRIES} "
            f"→ {url}"
        )


        try:

            response = requests.post(

                url,

                data=query,

                headers=HEADERS,

                timeout=REQUEST_TIMEOUT

            )


            print(
                f"  HTTP status: "
                f"{response.status_code}"
            )


            if response.status_code == 200:

                data = response.json()


                with open(
                    cache_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        data,
                        f
                    )


                print(
                    f"  ✓ Downloaded "
                    f"{len(data.get('elements', []))} "
                    f"OSM features"
                )


                time.sleep(
                    SLEEP_BETWEEN_REQUESTS
                )


                return data


            elif response.status_code in [
                429,
                502,
                503,
                504
            ]:

                print(
                    "  Server busy."
                )


            else:

                print(
                    "  OSM error:",
                    response.status_code
                )

                print(
                    response.text[:300]
                )


        except requests.exceptions.Timeout:

            print(
                "  Request timed out."
            )


        except requests.exceptions.ConnectionError as e:

            print(
                "  Connection error:",
                str(e)[:200]
            )


        except Exception as e:

            print(
                "  Unexpected error:",
                str(e)[:200]
            )


        wait_time = (
            2 ** attempt
        )


        print(
            f"  Waiting {wait_time} seconds..."
        )


        time.sleep(
            wait_time
        )


    print(
        "  ✗ Cell download failed."
    )

    return None


# ============================================================
# PARSE OSM ELEMENTS
# ============================================================

def parse_osm_elements(data):

    features = []


    if data is None:

        return features


    elements = data.get(
        "elements",
        []
    )


    for element in elements:

        tags = element.get(
            "tags",
            {}
        )


        lat = None

        lon = None


        # Node

        if element.get("type") == "node":

            lat = element.get("lat")

            lon = element.get("lon")


        # Way / relation center

        elif "center" in element:

            center = element["center"]

            lat = center.get("lat")

            lon = center.get("lon")


        if lat is None or lon is None:

            continue


        # ----------------------------------------------------
        # CLASSIFY OSM FEATURE
        # ----------------------------------------------------

        category = "OTHER"


        power = tags.get(
            "power",
            ""
        ).lower()


        railway = tags.get(
            "railway",
            ""
        ).lower()


        highway = tags.get(
            "highway",
            ""
        ).lower()


        landuse = tags.get(
            "landuse",
            ""
        ).lower()


        natural = tags.get(
            "natural",
            ""
        ).lower()


        industrial = tags.get(
            "industrial",
            ""
        ).lower()


        man_made = tags.get(
            "man_made",
            ""
        ).lower()


        if (
            power
        ):

            category = "POWER"


        elif (
            railway
        ):

            category = "RAILWAY"


        elif (
            highway
        ):

            category = "ROAD"


        elif (
            landuse == "industrial"
            or industrial
            or man_made == "works"
        ):

            category = "INDUSTRIAL"


        elif (
            natural
        ):

            category = "NATURAL"


        features.append({

            "lat": float(lat),

            "lon": float(lon),

            "category": category,

            "power_type": power,

            "railway_type": railway,

            "highway_type": highway,

            "landuse_type": landuse,

            "natural_type": natural,

            "industrial_type": industrial,

            "man_made_type": man_made,

        })


    return features


# ============================================================
# DOWNLOAD ALL REQUIRED OSM CELLS
# ============================================================

print()
print("[3/8] Downloading OSM infrastructure...")
print("-" * 70)


required_cells = sorted(
    events["osm_cell_id"]
    .dropna()
    .astype(int)
    .unique()
)


print(
    f"Cells containing thermal events: "
    f"{len(required_cells)}"
)


all_features = []


for position, cell_id in enumerate(
    required_cells,
    start=1
):

    cell = grid[cell_id]


    print()
    print(
        f"[{position}/{len(required_cells)}] "
        f"OSM cell {cell_id}"
    )

    print(
        f"  Bounds: "
        f"{cell['west']:.2f},"
        f"{cell['south']:.2f} → "
        f"{cell['east']:.2f},"
        f"{cell['north']:.2f}"
    )


    data = download_osm_cell(
        cell
    )


    if data is None:

        continue


    features = parse_osm_elements(
        data
    )


    print(
        f"  Parsed features: "
        f"{len(features):,}"
    )


    all_features.extend(
        features
    )


print()
print(
    f"Total OSM features collected: "
    f"{len(all_features):,}"
)


# ============================================================
# CHECK OSM DATA
# ============================================================

if not all_features:

    raise RuntimeError(
        """
No OSM features were collected.

The Overpass API may be temporarily unavailable.
Run this script again; cached cells will be reused.
"""
    )


osm = pd.DataFrame(
    all_features
)


# Remove duplicate geometries

osm = osm.drop_duplicates(
    subset=[
        "lat",
        "lon",
        "category"
    ]
).reset_index(
    drop=True
)


print(
    f"Unique OSM features: "
    f"{len(osm):,}"
)


print()
print(
    "OSM category distribution:"
)

print(
    osm["category"]
    .value_counts()
    .to_string()
)


# ============================================================
# CREATE GEOMETRIES
# ============================================================

print()
print("[4/8] Creating spatial geometries...")
print("-" * 70)


osm_geometry = [

    Point(
        lon,
        lat
    )

    for lat, lon in zip(
        osm["lat"],
        osm["lon"]
    )

]


osm_gdf = gpd.GeoDataFrame(
    osm,
    geometry=osm_geometry,
    crs="EPSG:4326"
)


event_geometry = [

    Point(
        lon,
        lat
    )

    for lat, lon in zip(
        events["center_latitude"],
        events["center_longitude"]
    )

]


events_gdf = gpd.GeoDataFrame(
    events,
    geometry=event_geometry,
    crs="EPSG:4326"
)


# ============================================================
# CREATE LOCAL METRIC PROJECTION
# ============================================================

print()
print("[5/8] Preparing metric spatial index...")
print("-" * 70)


# India spans multiple UTM zones.
#
# For distance calculations across the entire country,
# use an azimuthal/equal-area style projection.
#
# EPSG:6933 gives metre-based global distances suitable
# for neighbourhood calculations.

osm_metric = osm_gdf.to_crs(
    "EPSG:6933"
)

events_metric = events_gdf.to_crs(
    "EPSG:6933"
)


# ============================================================
# SPATIAL INDEX
# ============================================================

print(
    "Building OSM spatial index..."
)


osm_tree = STRtree(
    osm_metric.geometry.values
)


osm_geometries = osm_metric.geometry.values


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_event_context(
    point,
    osm_df,
    osm_geometries,
    tree
):

    # Search within 2 km

    buffer_distance = 2000

    possible_indices = tree.query(
        point.buffer(
            buffer_distance
        )
    )


    nearby = []


    for idx in possible_indices:

        geom = osm_geometries[idx]

        distance = point.distance(
            geom
        )


        if distance <= buffer_distance:

            nearby.append(
                (
                    idx,
                    distance
                )
            )


    # --------------------------------------------------------
    # INITIAL COUNTS
    # --------------------------------------------------------

    counts_500 = {

        "INDUSTRIAL": 0,
        "POWER": 0,
        "RAILWAY": 0,
        "ROAD": 0,
        "NATURAL": 0,

    }


    counts_1000 = {

        "INDUSTRIAL": 0,
        "POWER": 0,
        "RAILWAY": 0,
        "ROAD": 0,
        "NATURAL": 0,

    }


    counts_2000 = {

        "INDUSTRIAL": 0,
        "POWER": 0,
        "RAILWAY": 0,
        "ROAD": 0,
        "NATURAL": 0,

    }


    # --------------------------------------------------------
    # COUNT FEATURES
    # --------------------------------------------------------

    for idx, distance in nearby:

        category = osm_df.iloc[
            idx
        ]["category"]


        if category not in counts_500:

            continue


        if distance <= 500:

            counts_500[
                category
            ] += 1


        if distance <= 1000:

            counts_1000[
                category
            ] += 1


        if distance <= 2000:

            counts_2000[
                category
            ] += 1


    # --------------------------------------------------------
    # NEAREST FEATURE
    # --------------------------------------------------------

    if nearby:

        nearest_idx, nearest_distance = min(
            nearby,
            key=lambda x: x[1]
        )


        nearest_category = (
            osm_df.iloc[
                nearest_idx
            ]["category"]
        )

    else:

        nearest_distance = -1

        nearest_category = "NONE"


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    industrial_near = (
        counts_2000["INDUSTRIAL"] > 0
    )


    power_near = (
        counts_2000["POWER"] > 0
    )


    railway_near = (
        counts_2000["RAILWAY"] > 0
    )


    road_near = (
        counts_2000["ROAD"] > 0
    )


    if industrial_near:

        context = "INDUSTRIAL"


    elif power_near:

        context = "POWER_INFRASTRUCTURE"


    elif railway_near:

        context = "RAILWAY"


    elif road_near:

        context = "ROAD_ADJACENT"


    else:

        context = "MIXED_OTHER"


    # --------------------------------------------------------
    # FEATURE DICTIONARY
    # --------------------------------------------------------

    result = {

        # Nearest

        "nearest_osm_distance_km":
            nearest_distance / 1000
            if nearest_distance >= 0
            else -1,

        "nearest_osm_category":
            nearest_category,


        # 500 m

        "industrial_count_0_5km":
            counts_500["INDUSTRIAL"],

        "power_count_0_5km":
            counts_500["POWER"],

        "railway_count_0_5km":
            counts_500["RAILWAY"],

        "road_count_0_5km":
            counts_500["ROAD"],

        "natural_count_0_5km":
            counts_500["NATURAL"],


        # 1 km

        "industrial_count_1_0km":
            counts_1000["INDUSTRIAL"],

        "power_count_1_0km":
            counts_1000["POWER"],

        "railway_count_1_0km":
            counts_1000["RAILWAY"],

        "road_count_1_0km":
            counts_1000["ROAD"],

        "natural_count_1_0km":
            counts_1000["NATURAL"],


        # 2 km

        "industrial_count_2_0km":
            counts_2000["INDUSTRIAL"],

        "power_count_2_0km":
            counts_2000["POWER"],

        "railway_count_2_0km":
            counts_2000["RAILWAY"],

        "road_count_2_0km":
            counts_2000["ROAD"],

        "natural_count_2_0km":
            counts_2000["NATURAL"],


        # Boolean context

        "near_industrial":
            int(industrial_near),

        "near_power_infrastructure":
            int(power_near),

        "near_railway":
            int(railway_near),

        "near_major_road":
            int(road_near),


        "context_type":
            context,

    }


    return result


# ============================================================
# PROCESS EVENTS
# ============================================================

print()
print("[6/8] Calculating OSM contextual features...")
print("-" * 70)


results = []


total_events = len(
    events_metric
)


for i, point in enumerate(
    events_metric.geometry,
    start=1
):

    if i % 1000 == 0:

        print(
            f"Processed "
            f"{i:,}/{total_events:,}"
        )


    result = calculate_event_context(

        point,

        osm_metric,

        osm_geometries,

        osm_tree

    )


    results.append(
        result
    )


context_df = pd.DataFrame(
    results
)


# ============================================================
# MERGE FEATURES
# ============================================================

print()
print("[7/8] Merging OSM features...")
print("-" * 70)


final_df = pd.concat(

    [

        events.reset_index(
            drop=True
        ),

        context_df.reset_index(
            drop=True
        )

    ],

    axis=1

)


# Remove geometry

if "geometry" in final_df.columns:

    final_df = final_df.drop(
        columns=["geometry"]
    )


# Remove helper grid column

if "osm_cell_id" in final_df.columns:

    final_df = final_df.drop(
        columns=["osm_cell_id"]
    )


# ============================================================
# SAVE
# ============================================================

print()
print("[8/8] Saving enriched dataset...")
print("-" * 70)


final_df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("✓ STEP 3C COMPLETE")
print("=" * 70)

print()

print(
    f"Input thermal events : "
    f"{len(events):,}"
)

print(
    f"Output thermal events: "
    f"{len(final_df):,}"
)

print()

print(
    "OSM features collected:"
)

print(
    f"{len(osm):,}"
)

print()

print(
    "Context distribution:"
)

print(
    final_df[
        "context_type"
    ]
    .value_counts()
    .to_string()
)

print()

print(
    "Output:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()

print("=" * 70)

print(
    "NEXT STEP"
)

print("=" * 70)

print()

print(
    "Step 4 → Sentinel-2 spectral feature extraction"
)

print()

print(
    "The Sentinel-2 module will add spectral "
    "information around each thermal event."
)

print("=" * 70)