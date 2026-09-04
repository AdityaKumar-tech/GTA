import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 5
# SYNTHETIC INFRASTRUCTURE + INDUSTRY CONTEXT
# ============================================================

print("=" * 75)
print("STEP 5: SYNTHETIC INFRASTRUCTURE & INDUSTRY CONTEXT")
print("=" * 75)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    r"data\processed\landcover\thermal_events_landcover.csv"
)

OUTPUT_FILE = (
    r"data\processed\features\thermal_event_context.csv"
)

SEED = 42

rng = np.random.default_rng(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/8] Loading dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

original_count = len(df)

print(
    f"Records loaded: {original_count:,}"
)


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

def find_column(candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


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


LANDCOVER_COL = find_column([
    "landcover",
    "land_cover",
    "worldcover_class",
    "worldcover_label",
    "landcover_label",
    "land_cover_class"
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


FREQUENCY_COL = find_column([
    "detection_count",
    "detections",
    "frequency",
    "event_frequency"
])


BEHAVIOR_COL = find_column([
    "thermal_behavior"
])


print()
print("Detected columns:")

print(
    f"Latitude        : {LAT_COL}"
)

print(
    f"Longitude       : {LON_COL}"
)

print(
    f"Land cover      : {LANDCOVER_COL}"
)

print(
    f"FRP             : {FRP_COL}"
)

print(
    f"Persistence     : {PERSISTENCE_COL}"
)

print(
    f"Frequency       : {FREQUENCY_COL}"
)

print(
    f"Thermal behavior: {BEHAVIOR_COL}"
)


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

print()
print("[2/8] Preparing thermal features...")


def numeric_series(column):

    if column is None:

        return pd.Series(
            np.zeros(len(df)),
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


frp = numeric_series(FRP_COL)

persistence = numeric_series(
    PERSISTENCE_COL
)

frequency = numeric_series(
    FREQUENCY_COL
)


# ============================================================
# NORMALIZATION
# ============================================================

def percentile_score(series):

    if series.nunique() <= 1:

        return pd.Series(
            0.5,
            index=series.index
        )

    ranks = series.rank(
        pct=True
    )

    return ranks.fillna(0.5)


frp_score = percentile_score(
    frp
)

persistence_score = percentile_score(
    persistence
)

frequency_score = percentile_score(
    frequency
)


# ============================================================
# LAND-COVER GROUP
# ============================================================

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


# ============================================================
# CLASSIFY LAND COVER
# ============================================================

print()
print("[3/8] Classifying land-cover context...")


land_group = pd.Series(
    "OTHER",
    index=df.index
)


land_group[
    landcover.str.contains(
        "TREE|FOREST",
        na=False
    )
] = "FOREST"


land_group[
    landcover.str.contains(
        "CROP|AGRICULT",
        na=False
    )
] = "CROPLAND"


land_group[
    landcover.str.contains(
        "BUILT|URBAN",
        na=False
    )
] = "BUILT_UP"


land_group[
    landcover.str.contains(
        "GRASS",
        na=False
    )
] = "GRASSLAND"


land_group[
    landcover.str.contains(
        "SHRUB",
        na=False
    )
] = "SHRUBLAND"


land_group[
    landcover.str.contains(
        "BARE|SPARSE",
        na=False
    )
] = "BARREN"


land_group[
    landcover.str.contains(
        "WATER",
        na=False
    )
] = "WATER"


land_group[
    landcover.str.contains(
        "WETLAND",
        na=False
    )
] = "WETLAND"


# ============================================================
# SPATIAL INFORMATION
# ============================================================

print()
print("[4/8] Preparing spatial information...")


if LAT_COL:

    lat = pd.to_numeric(
        df[LAT_COL],
        errors="coerce"
    ).fillna(20)

else:

    lat = pd.Series(
        20,
        index=df.index
    )


if LON_COL:

    lon = pd.to_numeric(
        df[LON_COL],
        errors="coerce"
    ).fillna(78)

else:

    lon = pd.Series(
        78,
        index=df.index
    )


# Coarse spatial cells

df["_geo_lat"] = np.floor(
    lat / 0.5
)

df["_geo_lon"] = np.floor(
    lon / 0.5
)

df["_geo_cell"] = (
    df["_geo_lat"].astype(str)
    + "_"
    + df["_geo_lon"].astype(str)
)


# ============================================================
# CONTEXT SCORES
# ============================================================

industrial_score = (
    0.45 * (
        land_group == "BUILT_UP"
    ).astype(float)

    +

    0.20 * (
        land_group == "BARREN"
    ).astype(float)

    +

    0.15 * frp_score

    +

    0.10 * frequency_score

    +

    0.10 * persistence_score
)


agricultural_score = (
    0.70 * (
        land_group == "CROPLAND"
    ).astype(float)

    +

    0.15 * frequency_score

    +

    0.15 * persistence_score
)


forest_score = (
    0.75 * (
        land_group == "FOREST"
    ).astype(float)

    +

    0.15 * persistence_score

    +

    0.10 * frp_score
)


urban_score = (
    0.70 * (
        land_group == "BUILT_UP"
    ).astype(float)

    +

    0.15 * frequency_score

    +

    0.15 * frp_score
)


mining_score = (
    0.55 * (
        land_group == "BARREN"
    ).astype(float)

    +

    0.20 * frp_score

    +

    0.15 * persistence_score

    +

    0.10 * frequency_score
)


# ============================================================
# PRIMARY CONTEXT
# ============================================================

print()
print("[5/8] Generating primary context...")


context = []


for i in df.index:

    scores = {

        "AGRICULTURAL":
            agricultural_score.loc[i],

        "FOREST":
            forest_score.loc[i],

        "URBAN":
            urban_score.loc[i],

        "INDUSTRIAL":
            industrial_score.loc[i],

        "MINING":
            mining_score.loc[i],

        "RURAL":
            0.15
    }


    # Controlled randomness

    for key in scores:

        scores[key] += rng.uniform(
            0,
            0.12
        )


    selected = max(
        scores,
        key=scores.get
    )


    context.append(
        selected
    )


df["context_category"] = context


# ============================================================
# INDUSTRY TYPE
# ============================================================

print()
print("[6/8] Generating industry types...")


industry_values = []


for i in df.index:

    context_value = (
        df.loc[
            i,
            "context_category"
        ]
    )


    if context_value == "AGRICULTURAL":

        choices = [

            "FOOD_PROCESSING",
            "SUGAR",
            "TEXTILE",
            "RICE_MILL",
            "OIL_PROCESSING"

        ]

        probabilities = [

            0.35,
            0.18,
            0.18,
            0.19,
            0.10

        ]


    elif context_value == "INDUSTRIAL":

        choices = [

            "STEEL",
            "CEMENT",
            "CHEMICAL",
            "TEXTILE",
            "FOOD_PROCESSING",
            "ENGINEERING"

        ]

        probabilities = [

            0.18,
            0.17,
            0.15,
            0.15,
            0.18,
            0.17

        ]


    elif context_value == "MINING":

        choices = [

            "MINING",
            "CEMENT",
            "MINERAL_PROCESSING"

        ]

        probabilities = [

            0.55,
            0.25,
            0.20

        ]


    elif context_value == "URBAN":

        choices = [

            "NONE",
            "FOOD_PROCESSING",
            "TEXTILE",
            "ENGINEERING",
            "CHEMICAL"

        ]

        probabilities = [

            0.35,
            0.18,
            0.17,
            0.18,
            0.12

        ]


    elif context_value == "FOREST":

        choices = [

            "NONE",
            "PAPER",
            "WOOD_PROCESSING"

        ]

        probabilities = [

            0.70,
            0.15,
            0.15

        ]


    else:

        choices = [

            "NONE",
            "FOOD_PROCESSING",
            "MINING"

        ]

        probabilities = [

            0.70,
            0.15,
            0.15

        ]


    industry_values.append(
        rng.choice(
            choices,
            p=probabilities
        )
    )


df["nearby_industry_type"] = (
    industry_values
)


# ============================================================
# INFRASTRUCTURE TYPE
# ============================================================

print()
print("[7/8] Generating infrastructure types...")


infrastructure_values = []


for i in df.index:

    context_value = (
        df.loc[
            i,
            "context_category"
        ]
    )


    if context_value == "URBAN":

        choices = [

            "HIGHWAY",
            "RAILWAY",
            "RESIDENTIAL",
            "WAREHOUSE",
            "POWER_SUBSTATION"

        ]


    elif context_value == "INDUSTRIAL":

        choices = [

            "HIGHWAY",
            "RAILWAY",
            "POWER_LINE",
            "POWER_SUBSTATION",
            "WAREHOUSE"

        ]


    elif context_value == "AGRICULTURAL":

        choices = [

            "RURAL_ROAD",
            "HIGHWAY",
            "POWER_LINE",
            "RAILWAY",
            "NONE"

        ]


    elif context_value == "MINING":

        choices = [

            "MINING_ROAD",
            "RAILWAY",
            "POWER_LINE",
            "HIGHWAY"

        ]


    elif context_value == "FOREST":

        choices = [

            "FOREST_ROAD",
            "HIGHWAY",
            "POWER_LINE",
            "NONE"

        ]


    else:

        choices = [

            "RURAL_ROAD",
            "HIGHWAY",
            "POWER_LINE",
            "NONE"

        ]


    infrastructure_values.append(
        rng.choice(
            choices
        )
    )


df["nearby_infrastructure_type"] = (
    infrastructure_values
)


# ============================================================
# FACILITY TYPE
# ============================================================

print()
print("Generating facility information...")


facility_values = []


for i in df.index:

    context_value = (
        df.loc[
            i,
            "context_category"
        ]
    )


    if context_value == "INDUSTRIAL":

        choices = [

            "FACTORY",
            "WAREHOUSE",
            "POWER_PLANT",
            "INDUSTRIAL_ESTABLISHMENT"

        ]


    elif context_value == "MINING":

        choices = [

            "MINE",
            "MINERAL_PROCESSING",
            "STORAGE_YARD"

        ]


    elif context_value == "AGRICULTURAL":

        choices = [

            "WAREHOUSE",
            "FOOD_PROCESSING_UNIT",
            "COLD_STORAGE",
            "NONE"

        ]


    elif context_value == "URBAN":

        choices = [

            "WAREHOUSE",
            "FACTORY",
            "LANDFILL",
            "NONE"

        ]


    elif context_value == "FOREST":

        choices = [

            "NONE",
            "TIMBER_STORAGE"

        ]


    else:

        choices = [

            "NONE",
            "WAREHOUSE",
            "LANDFILL"

        ]


    facility_values.append(
        rng.choice(
            choices
        )
    )


df["nearby_facility_type"] = (
    facility_values
)


# ============================================================
# BINARY FEATURES
# ============================================================

df["industrial_presence"] = (
    df["nearby_industry_type"] != "NONE"
).astype(np.int8)


df["infrastructure_presence"] = (
    df["nearby_infrastructure_type"] != "NONE"
).astype(np.int8)


df["facility_presence"] = (
    df["nearby_facility_type"] != "NONE"
).astype(np.int8)


# ============================================================
# SPECIFIC INFRASTRUCTURE FLAGS
# ============================================================

df["nearby_highway"] = (
    df[
        "nearby_infrastructure_type"
    ].isin([
        "HIGHWAY"
    ])
).astype(np.int8)


df["nearby_railway"] = (
    df[
        "nearby_infrastructure_type"
    ].isin([
        "RAILWAY"
    ])
).astype(np.int8)


df["nearby_power_infrastructure"] = (
    df[
        "nearby_infrastructure_type"
    ].isin([
        "POWER_LINE",
        "POWER_SUBSTATION"
    ])
).astype(np.int8)


df["nearby_mining"] = (
    df[
        "nearby_industry_type"
    ].isin([
        "MINING"
    ])
).astype(np.int8)


# ============================================================
# COMBINED CONTEXT
# ============================================================

df["mixed_industrial_infrastructure"] = (

    (
        df["industrial_presence"]
        == 1
    )

    &

    (
        df["infrastructure_presence"]
        == 1
    )

).astype(np.int8)


# ============================================================
# REMOVE INTERNAL SPATIAL COLUMNS
# ============================================================

df.drop(
    columns=[
        "_geo_lat",
        "_geo_lon",
        "_geo_cell"
    ],
    inplace=True
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("[8/8] Validating dataset...")


if len(df) != original_count:

    raise RuntimeError(
        "ERROR: Number of records changed!"
    )


required_columns = [

    "context_category",

    "nearby_industry_type",

    "nearby_infrastructure_type",

    "nearby_facility_type",

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

        raise RuntimeError(
            f"Missing output column: {column}"
        )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("✓ SYNTHETIC CONTEXT DATASET CREATED")
print("=" * 75)

print()

print(
    f"Original records : {original_count:,}"
)

print(
    f"Output records   : {len(df):,}"
)

print()

print("Context distribution:")

print(
    df[
        "context_category"
    ]
    .value_counts()
    .to_string()
)

print()

print("Industry distribution:")

print(
    df[
        "nearby_industry_type"
    ]
    .value_counts()
    .to_string()
)

print()

print("Infrastructure distribution:")

print(
    df[
        "nearby_infrastructure_type"
    ]
    .value_counts()
    .to_string()
)

print()

print("Facility distribution:")

print(
    df[
        "nearby_facility_type"
    ]
    .value_counts()
    .to_string()
)

print()

print("Binary feature distribution:")

print(
    "Industrial presence :",
    int(
        df["industrial_presence"].sum()
    )
)

print(
    "Infrastructure      :",
    int(
        df["infrastructure_presence"].sum()
    )
)

print(
    "Facility presence   :",
    int(
        df["facility_presence"].sum()
    )
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
print("NEXT STEP → FEATURE ENGINEERING")
print("=" * 75)