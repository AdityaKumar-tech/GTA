import os
import pandas as pd


# ============================================================
# STEP 6C: FIRE TYPE CLASSIFICATION
# ============================================================

print("=" * 75)
print("STEP 6C: THERMAL EVENT FIRE-TYPE CLASSIFICATION")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\ml\model_features.csv"
)

OUTPUT_FILE = (
    r"data\processed\ml\fire_type_dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

print()
print("[1/6] Loading ML feature dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Events loaded: {len(df):,}"
)


# ============================================================
# LOAD ORIGINAL ENRICHED DATA
# ============================================================

print()
print("[2/6] Loading contextual information...")

CONTEXT_FILE = (
    r"data\processed\features\thermal_event_context.csv"
)

if not os.path.exists(CONTEXT_FILE):
    raise FileNotFoundError(
        f"Context file not found:\n{CONTEXT_FILE}"
    )

context_df = pd.read_csv(
    CONTEXT_FILE
)

print(
    f"Context records: {len(context_df):,}"
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if len(df) != len(context_df):

    raise ValueError(
        "ML dataset and contextual dataset "
        "have different numbers of records."
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

print()
print("[3/6] Normalizing contextual categories...")


def clean_text(series):

    return (
        series
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )


landcover = clean_text(
    context_df["landcover_group"]
)

landcover_name = clean_text(
    context_df["worldcover_name"]
)

context = clean_text(
    context_df["context_category"]
)

industry = clean_text(
    context_df["nearby_industry_type"]
)

infrastructure = clean_text(
    context_df["nearby_infrastructure_type"]
)

facility = clean_text(
    context_df["nearby_facility_type"]
)


# ============================================================
# NUMERICAL CONTEXT
# ============================================================

industrial = (
    pd.to_numeric(
        context_df["industrial_presence"],
        errors="coerce"
    )
    .fillna(0)
)

infrastructure_presence = (
    pd.to_numeric(
        context_df["infrastructure_presence"],
        errors="coerce"
    )
    .fillna(0)
)

facility_presence = (
    pd.to_numeric(
        context_df["facility_presence"],
        errors="coerce"
    )
    .fillna(0)
)

highway = (
    pd.to_numeric(
        context_df["nearby_highway"],
        errors="coerce"
    )
    .fillna(0)
)

railway = (
    pd.to_numeric(
        context_df["nearby_railway"],
        errors="coerce"
    )
    .fillna(0)
)

power = (
    pd.to_numeric(
        context_df["nearby_power_infrastructure"],
        errors="coerce"
    )
    .fillna(0)
)

mining = (
    pd.to_numeric(
        context_df["nearby_mining"],
        errors="coerce"
    )
    .fillna(0)
)

mixed = (
    pd.to_numeric(
        context_df["mixed_industrial_infrastructure"],
        errors="coerce"
    )
    .fillna(0)
)


# ============================================================
# FIRE TYPE CLASSIFICATION
# ============================================================

print()
print("[4/6] Classifying thermal events...")


def classify_fire_type(i):

    lc = landcover.iloc[i]
    lc_name = landcover_name.iloc[i]

    ctx = context.iloc[i]
    ind = industry.iloc[i]
    infra = infrastructure.iloc[i]
    fac = facility.iloc[i]

    ind_presence = industrial.iloc[i]
    infra_presence = infrastructure_presence.iloc[i]
    fac_presence = facility_presence.iloc[i]

    highway_flag = highway.iloc[i]
    railway_flag = railway.iloc[i]
    power_flag = power.iloc[i]
    mining_flag = mining.iloc[i]
    mixed_flag = mixed.iloc[i]


    # --------------------------------------------------------
    # MINING
    # --------------------------------------------------------

    if (
        ctx == "MINING"
        or mining_flag == 1
        or "MINING" in ind
        or "MINE" in fac
    ):
        return "MINING_RELATED_FIRE"


    # --------------------------------------------------------
    # INDUSTRIAL
    # --------------------------------------------------------

    if (
        ctx == "INDUSTRIAL"
        and (
            ind_presence == 1
            or ind != "NONE"
            or fac_presence == 1
        )
    ):
        return "INDUSTRIAL_FIRE"


    # --------------------------------------------------------
    # INFRASTRUCTURE
    # --------------------------------------------------------

    if (
        mixed_flag == 1
        or power_flag == 1
    ):
        return "INFRASTRUCTURE_RELATED_FIRE"


    # --------------------------------------------------------
    # CROP / AGRICULTURAL
    # --------------------------------------------------------

    if (
        lc == "AGRICULTURAL"
        or lc_name in [
            "CROPLAND",
            "CROP",
            "CROPLAND"
        ]
    ):
        return "CROP_RESIDUE_FIRE"


    # --------------------------------------------------------
    # FOREST
    # --------------------------------------------------------

    if (
        "TREE" in lc_name
        or "FOREST" in lc_name
        or "WOOD" in lc_name
        or lc_name in [
            "TREE_COVER",
            "FOREST"
        ]
    ):
        return "FOREST_WILDFIRE"


    # --------------------------------------------------------
    # GRASSLAND
    # --------------------------------------------------------

    if (
        "GRASS" in lc_name
        or lc == "GRASSLAND"
    ):
        return "GRASSLAND_FIRE"


    # --------------------------------------------------------
    # URBAN
    # --------------------------------------------------------

    if (
        lc == "URBAN"
        or "BUILT" in lc_name
        or "URBAN" in lc_name
    ):
        return "URBAN_FIRE"


    # --------------------------------------------------------
    # TRANSPORTATION
    # --------------------------------------------------------

    if (
        highway_flag == 1
        or railway_flag == 1
        or "HIGHWAY" in infra
        or "RAIL" in infra
    ):
        return "TRANSPORTATION_RELATED_FIRE"


    # --------------------------------------------------------
    # FACILITY
    # --------------------------------------------------------

    if fac_presence == 1:

        return "FACILITY_RELATED_FIRE"


    # --------------------------------------------------------
    # OPEN LAND
    # --------------------------------------------------------

    if lc in [
        "BARREN",
        "OPEN",
        "NATURAL"
    ]:
        return "OPEN_LAND_FIRE"


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "UNKNOWN_THERMAL_EVENT"


# ============================================================
# APPLY CLASSIFICATION
# ============================================================

fire_types = []

for i in range(len(df)):

    fire_types.append(
        classify_fire_type(i)
    )


df["fire_type"] = fire_types


# ============================================================
# SAVE
# ============================================================

print()
print("[5/6] Saving fire-type dataset...")


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISTRIBUTION
# ============================================================

print()
print("[6/6] Fire-type distribution")
print("-" * 75)


distribution = (
    df["fire_type"]
    .value_counts()
)


for fire_type, count in distribution.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{fire_type:35} "
        f"{count:8,} "
        f"({percentage:6.2f}%)"
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
    f"Total events : {len(df):,}"
)

print(
    f"Fire types   : {df['fire_type'].nunique()}"
)

print()
print("Output:")
print(
    os.path.abspath(OUTPUT_FILE)
)

print()
print("=" * 75)
print("NEXT STEP → FIRE-TYPE VALIDATION + RISK LABELING")
print("=" * 75)