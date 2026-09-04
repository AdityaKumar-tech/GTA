import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# STEP 8: TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("=" * 75)
print("STEP 8: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    r"data\processed\ml\fire_type_training.csv"
)

OUTPUT_DIR = (
    r"data\processed\ml\splits"
)


TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

VALIDATION_FILE = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/7] Loading training dataset...")

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Training dataset not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print(
    f"Total records: {len(df):,}"
)


# ============================================================
# CHECK TARGET
# ============================================================

print()
print("[2/7] Checking target variable...")

if "fire_type" not in df.columns:

    raise ValueError(
        "fire_type column is missing."
    )


print(
    f"Fire-type classes: "
    f"{df['fire_type'].nunique()}"
)


print()
print(
    df["fire_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# REMOVE EXTREMELY RARE CLASSES
# ============================================================

print()
print("[3/7] Checking class sizes...")

class_counts = (
    df["fire_type"]
    .value_counts()
)

rare_classes = (
    class_counts[
        class_counts < 4
    ]
    .index
    .tolist()
)


if rare_classes:

    print(
        "Warning: extremely rare classes detected:"
    )

    for cls in rare_classes:

        print(
            f"  {cls}: "
            f"{class_counts[cls]}"
        )

    print()
    print(
        "These classes cannot reliably be "
        "split across train/validation/test."
    )

    print(
        "They will be removed for this prototype."
    )

    df = df[
        ~df["fire_type"].isin(
            rare_classes
        )
    ].copy()


print(
    f"Records after class filtering: "
    f"{len(df):,}"
)


# ============================================================
# CREATE X AND Y
# ============================================================

print()
print("[4/7] Separating features and target...")

X = df.drop(
    columns=["fire_type"]
)

y = df["fire_type"]


print(
    f"Features: {X.shape[1]}"
)

print(
    f"Records : {X.shape[0]:,}"
)


# ============================================================
# FIRST SPLIT
# ============================================================

print()
print("[5/7] Creating training and temporary sets...")


X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)


# ============================================================
# SECOND SPLIT
# ============================================================

print()
print("[6/7] Creating validation and test sets...")


X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
)


# ============================================================
# REBUILD DATASETS
# ============================================================

train_df = X_train.copy()

train_df["fire_type"] = y_train.values


validation_df = X_validation.copy()

validation_df["fire_type"] = (
    y_validation.values
)


test_df = X_test.copy()

test_df["fire_type"] = y_test.values


# ============================================================
# SAVE
# ============================================================

print()
print("[7/7] Saving datasets...")


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


train_df.to_csv(
    TRAIN_FILE,
    index=False
)


validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)


test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 75)
print("SPLIT RESULTS")
print("=" * 75)


print()
print(
    f"Training   : {len(train_df):,} "
    f"({len(train_df)/len(df)*100:.2f}%)"
)

print(
    f"Validation : {len(validation_df):,} "
    f"({len(validation_df)/len(df)*100:.2f}%)"
)

print(
    f"Testing    : {len(test_df):,} "
    f"({len(test_df)/len(df)*100:.2f}%)"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Class distribution")
print("-" * 75)


distribution = pd.DataFrame({
    "Train": train_df["fire_type"].value_counts(),
    "Validation": validation_df["fire_type"].value_counts(),
    "Test": test_df["fire_type"].value_counts()
}).fillna(0).astype(int)


print(
    distribution.to_string()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("✓ STEP 8 COMPLETE")
print("=" * 75)

print()
print("Generated files:")

print(
    os.path.abspath(TRAIN_FILE)
)

print(
    os.path.abspath(VALIDATION_FILE)
)

print(
    os.path.abspath(TEST_FILE)
)

print()
print("=" * 75)
print("NEXT STEP → TRAIN FIRST FIRE-TYPE CLASSIFIER")
print("=" * 75)