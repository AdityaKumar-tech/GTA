import os
import pandas as pd


# ============================================================
# STEP 6A: INSPECT FINAL ENRICHED DATASET
# ============================================================

print("=" * 75)
print("STEP 6A: INSPECTING THERMAL EVENT DATASET")
print("=" * 75)


# ------------------------------------------------------------
# CHANGE THIS PATH IF YOUR FILE HAS A DIFFERENT NAME
# ------------------------------------------------------------

INPUT_FILE = (
    r"data\processed\features\thermal_event_context.csv"
)


# ------------------------------------------------------------
# CHECK FILE
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nFile not found:\n{INPUT_FILE}\n"
    )


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print()
print("[1/5] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Records: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ------------------------------------------------------------
# COLUMN LIST
# ------------------------------------------------------------

print()
print("[2/5] Columns")
print("-" * 75)

for i, column in enumerate(df.columns, start=1):
    print(
        f"{i:3}. {column}"
    )


# ------------------------------------------------------------
# DATA TYPES
# ------------------------------------------------------------

print()
print("[3/5] Data types")
print("-" * 75)

print(
    df.dtypes.to_string()
)


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print()
print("[4/5] Missing values")
print("-" * 75)

missing = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing.to_string()
)


# ------------------------------------------------------------
# SAMPLE
# ------------------------------------------------------------

print()
print("[5/5] First 5 records")
print("-" * 75)

print(
    df.head().to_string()
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 75)
print("✓ INSPECTION COMPLETE")
print("=" * 75)