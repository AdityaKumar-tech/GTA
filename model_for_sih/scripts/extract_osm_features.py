import os
import math
import osmium
import pandas as pd
import numpy as np
from collections import defaultdict


# ============================================================
# STEP 4: FAST INDIA-WIDE OSM FEATURE EXTRACTION
# ============================================================

print("=" * 70)
print("STEP 4: FAST INDIA-WIDE OSM FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_FILE = (
    r"data\processed\landcover\thermal_events_landcover.csv"
)

OSM_FILE = (
    r"data\raw\osm\india-260831.osm.pbf"
)

OUTPUT_DIR = (
    r"data\processed\features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "thermal_event_landcover_osm.csv"
)


# ============================================================
# GRID CONFIGURATION
# ============================================================

# Approximately 1 km grid
GRID_SIZE_KM = 1.0

# Approximate degrees per km
LAT_KM = 1 / 111.0


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(EVENT_FILE):

    raise FileNotFoundError(
        f"\nEvent dataset not found:\n{EVENT_FILE}"
    )


if not os.path.exists(OSM_FILE):

    raise FileNotFoundError(
        f"\nOSM PBF not found:\n{OSM_FILE}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# STEP 1 — LOAD EVENTS
# ============================================================

print()
print("[1/6] Loading thermal events...")

events = pd.read_csv(
    EVENT_FILE
)

print(
    f"Thermal events loaded: {len(events):,}"
)


# ============================================================
# FIND COORDINATE COLUMNS
# ============================================================

LAT_COLUMN = None
LON_COLUMN = None

possible_lat = [
    "center_latitude",
    "latitude",
    "lat"
]

possible_lon = [
    "center_longitude",
    "longitude",
    "lon"
]


for column in possible_lat:

    if column in events.columns:

        LAT_COLUMN = column
        break


for column in possible_lon:

    if column in events.columns:

        LON_COLUMN = column
        break


if LAT_COLUMN is None:

    raise ValueError(
        "Latitude column not found."
    )


if LON_COLUMN is None:

    raise ValueError(
        "Longitude column not found."
    )


print(
    f"Latitude column : {LAT_COLUMN}"
)

print(
    f"Longitude column: {LON_COLUMN}"
)


# ============================================================
# CLEAN COORDINATES
# ============================================================

events[LAT_COLUMN] = pd.to_numeric(
    events[LAT_COLUMN],
    errors="coerce"
)

events[LON_COLUMN] = pd.to_numeric(
    events[LON_COLUMN],
    errors="coerce"
)


events = events.dropna(
    subset=[
        LAT_COLUMN,
        LON_COLUMN
    ]
).copy()


print(
    f"Valid events: {len(events):,}"
)


# ============================================================
# STEP 2 — CREATE EVENT GRID
# ============================================================

print()
print("[2/6] Creating spatial grid...")


def get_grid_id(lat, lon):

    lat_index = math.floor(
        lat / LAT_KM
    )

    lon_km = (
        111.0 *
        math.cos(
            math.radians(lat)
        )
    )

    if lon_km <= 0:

        lon_km = 111.0


    lon_step = 1.0 / lon_km

    lon_index = math.floor(
        lon / lon_step
    )

    return (
        lat_index,
        lon_index
    )


events["grid_id"] = [
    get_grid_id(
        lat,
        lon
    )
    for lat, lon in zip(
        events[LAT_COLUMN],
        events[LON_COLUMN]
    )
]


unique_grids = events[
    "grid_id"
].unique()


print(
    f"Thermal-event grid cells: "
    f"{len(unique_grids):,}"
)


# ============================================================
# CREATE SET FOR FAST LOOKUP
# ============================================================

event_grid_set = set(
    unique_grids
)


# ============================================================
# OSM FEATURE STORAGE
# ============================================================

grid_features = defaultdict(
    lambda: {
        "road_count": 0,
        "major_road_count": 0,
        "railway_count": 0,
        "building_count": 0,
        "industrial_count": 0,
        "residential_count": 0,
        "power_count": 0,
        "waterway_count": 0
    }
)


# ============================================================
# HELPER
# ============================================================

def get_way_grid(way):

    try:

        if not way.nodes:

            return None


        node = way.nodes[0]

        lat = node.lat
        lon = node.lon


        if lat is None or lon is None:

            return None


        grid = get_grid_id(
            lat,
            lon
        )


        if grid not in event_grid_set:

            return None


        return grid

    except Exception:

        return None


# ============================================================
# OSMIUM HANDLER
# ============================================================

class IndiaOSMHandler(
    osmium.SimpleHandler
):

    def node(
        self,
        n
    ):

        tags = n.tags

        grid = None

        try:

            if n.location.valid():

                grid = get_grid_id(
                    n.location.lat,
                    n.location.lon
                )

        except Exception:

            return


        if grid not in event_grid_set:

            return


        features = grid_features[
            grid
        ]


        # ----------------------------------------------------
        # POWER
        # ----------------------------------------------------

        if "power" in tags:

            features[
                "power_count"
            ] += 1


        # ----------------------------------------------------
        # BUILDING
        # ----------------------------------------------------

        if "building" in tags:

            features[
                "building_count"
            ] += 1


    def way(
        self,
        w
    ):

        tags = w.tags

        grid = get_way_grid(
            w
        )


        if grid is None:

            return


        features = grid_features[
            grid
        ]


        # ----------------------------------------------------
        # HIGHWAYS
        # ----------------------------------------------------

        highway = tags.get(
            "highway"
        )


        if highway:

            features[
                "road_count"
            ] += 1


            major_roads = {
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link",
                "primary",
                "primary_link"
            }


            if highway in major_roads:

                features[
                    "major_road_count"
                ] += 1


        # ----------------------------------------------------
        # RAILWAYS
        # ----------------------------------------------------

        railway = tags.get(
            "railway"
        )


        if railway:

            features[
                "railway_count"
            ] += 1


        # ----------------------------------------------------
        # BUILDINGS
        # ----------------------------------------------------

        if "building" in tags:

            features[
                "building_count"
            ] += 1


        # ----------------------------------------------------
        # LANDUSE
        # ----------------------------------------------------

        landuse = tags.get(
            "landuse"
        )


        if landuse == "industrial":

            features[
                "industrial_count"
            ] += 1


        if landuse == "residential":

            features[
                "residential_count"
            ] += 1


        # ----------------------------------------------------
        # POWER
        # ----------------------------------------------------

        if "power" in tags:

            features[
                "power_count"
            ] += 1


        # ----------------------------------------------------
        # WATERWAY
        # ----------------------------------------------------

        if "waterway" in tags:

            features[
                "waterway_count"
            ] += 1


# ============================================================
# STEP 3 — PROCESS PBF
# ============================================================

print()
print("[3/6] Processing India OSM PBF...")

print()
print("IMPORTANT:")
print(
    "The PBF is scanned ONCE."
)

print(
    "No individual OSM request is made "
    "for each thermal event."
)

print()
print(
    "This may take time because the India "
    "PBF itself is very large."
)

print()


handler = IndiaOSMHandler()


handler.apply_file(
    OSM_FILE,
    locations=True
)


print()
print("✓ OSM PBF processing complete")


# ============================================================
# STEP 4 — CONVERT FEATURES TO DATAFRAME
# ============================================================

print()
print("[4/6] Creating OSM grid feature table...")


feature_rows = []


for grid_id, values in grid_features.items():

    row = {

        "grid_id": grid_id,

        "road_count":
            values["road_count"],

        "major_road_count":
            values["major_road_count"],

        "railway_count":
            values["railway_count"],

        "building_count":
            values["building_count"],

        "industrial_count":
            values["industrial_count"],

        "residential_count":
            values["residential_count"],

        "power_count":
            values["power_count"],

        "waterway_count":
            values["waterway_count"]

    }

    feature_rows.append(
        row
    )


osm_features = pd.DataFrame(
    feature_rows
)


print(
    f"Grid cells with OSM features: "
    f"{len(osm_features):,}"
)


# ============================================================
# STEP 5 — JOIN OSM TO EVENTS
# ============================================================

print()
print("[5/6] Joining OSM features to thermal events...")


events = events.merge(
    osm_features,
    on="grid_id",
    how="left"
)


# ============================================================
# FILL EMPTY OSM CELLS
# ============================================================

osm_columns = [

    "road_count",
    "major_road_count",
    "railway_count",
    "building_count",
    "industrial_count",
    "residential_count",
    "power_count",
    "waterway_count"

]


for column in osm_columns:

    events[column] = (
        events[column]
        .fillna(0)
        .astype(np.int32)
    )


# ============================================================
# REMOVE INTERNAL GRID COLUMN?
# ============================================================

# Keep it because it is useful for spatial modelling
# and debugging.


# ============================================================
# STEP 6 — SAVE
# ============================================================

print()
print("[6/6] Saving final OSM-enriched dataset...")


events.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("✓ STEP 4 COMPLETE")
print("=" * 70)

print()

print(
    f"Thermal events: "
    f"{len(events):,}"
)

print(
    f"OSM grid cells: "
    f"{len(osm_features):,}"
)

print()

print("OSM FEATURES:")

for column in osm_columns:

    print(
        f"  {column}"
    )

print()

print("Output:")

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()
print("=" * 70)

print(
    "NEXT STEP → FEATURE ENGINEERING"
)

print("=" * 70)