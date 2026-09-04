import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 9: TRAIN FIRE-TYPE CLASSIFICATION MODEL
# ============================================================

print("=" * 75)
print("STEP 9: TRAINING FIRE-TYPE CLASSIFICATION MODEL")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = (
    r"data\processed\ml\splits\train.csv"
)

VALIDATION_FILE = (
    r"data\processed\ml\splits\validation.csv"
)

TEST_FILE = (
    r"data\processed\ml\splits\test.csv"
)

MODEL_DIR = (
    r"models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "fire_type_random_forest.joblib"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "fire_type_features.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1/8] Loading datasets...")


for file in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE
]:

    if not os.path.exists(file):

        raise FileNotFoundError(
            f"File not found:\n{file}"
        )


train_df = pd.read_csv(
    TRAIN_FILE
)

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"Training records   : {len(train_df):,}"
)

print(
    f"Validation records : {len(validation_df):,}"
)

print(
    f"Test records       : {len(test_df):,}"
)


# ============================================================
# TARGET
# ============================================================

print()
print("[2/8] Separating features and target...")


TARGET = "fire_type"


if TARGET not in train_df.columns:

    raise ValueError(
        "fire_type column missing."
    )


X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]


X_validation = validation_df.drop(
    columns=[TARGET]
)

y_validation = validation_df[TARGET]


X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]


# ============================================================
# REMOVE ID COLUMN
# ============================================================

if "event_cluster_id" in X_train.columns:

    X_train = X_train.drop(
        columns=["event_cluster_id"]
    )

    X_validation = X_validation.drop(
        columns=["event_cluster_id"]
    )

    X_test = X_test.drop(
        columns=["event_cluster_id"]
    )


print(
    f"Features used: {X_train.shape[1]}"
)


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

print()
print("[3/8] Preparing numerical features...")


for column in X_train.columns:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_validation[column] = pd.to_numeric(
        X_validation[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )


# Replace invalid values

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values

X_train = X_train.fillna(0)

X_validation = X_validation.fillna(0)

X_test = X_test.fillna(0)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("[4/8] Training Random Forest...")


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=4,

    min_samples_leaf=2,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "✓ Random Forest training complete"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("[5/8] Evaluating on validation set...")


validation_predictions = model.predict(
    X_validation
)


validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_precision = precision_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="weighted",
    zero_division=0
)


print()
print("VALIDATION RESULTS")
print("-" * 50)

print(
    f"Accuracy  : {validation_accuracy:.4f}"
)

print(
    f"Precision : {validation_precision:.4f}"
)

print(
    f"Recall    : {validation_recall:.4f}"
)

print(
    f"F1 Score  : {validation_f1:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("[6/8] Evaluating on test set...")


test_predictions = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print()
print("=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print()
print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("CONFUSION MATRIX")
print("=" * 75)

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(
    cm_df.to_string()
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("[7/8] Calculating feature importance...")


importance = pd.DataFrame({

    "feature":
        X_train.columns,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print()
print("TOP 20 FEATURES")
print("-" * 75)

print(
    importance
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("[8/8] Saving trained model...")


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_FILE
)


with open(
    FEATURE_FILE,
    "w"
) as file:

    for feature in X_train.columns:

        file.write(
            feature + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("✓ STEP 9 COMPLETE")
print("=" * 75)

print()
print(
    f"Model              : Random Forest"
)

print(
    f"Training records   : {len(X_train):,}"
)

print(
    f"Validation records : {len(X_validation):,}"
)

print(
    f"Test records       : {len(X_test):,}"
)

print(
    f"Features           : {X_train.shape[1]}"
)

print(
    f"Classes            : {len(model.classes_)}"
)

print()
print("Saved model:")
print(
    os.path.abspath(MODEL_FILE)
)

print()
print("=" * 75)
print("NEXT STEP → MODEL ANALYSIS + IMPROVEMENT")
print("=" * 75)