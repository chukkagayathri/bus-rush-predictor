# preprocess.py
# Preprocess bus_data.csv for ML training
# Bus Rush Predictor Project

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ---- LOAD DATA ----
print("Loading dataset...")
df = pd.read_csv("apsrtc_bus_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nOriginal data sample:")
print(df.head(3))

# ---- UNDERSTAND DATA TYPES ----
print(f"\nData types before encoding:")
print(df.dtypes)

# ---- LABEL ENCODING ----
# ML models only understand numbers — not text
# We convert: "Monday" → 0, "Tuesday" → 1 etc.

print("\nEncoding categorical columns...")

# Create encoders dictionary — save for later use in Flask app
encoders = {}

categorical_columns = [
    "route_id",
    "day_of_week",
    "weather",
    "rush_level"   # this is our target variable
]

df_encoded = df.copy()  # work on a copy

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nData after encoding:")
print(df_encoded.head(3))
print(f"\nData types after encoding:")
print(df_encoded.dtypes)

# ---- DEFINE FEATURES AND TARGET ----
# Features = what we feed the model (inputs)
# Target = what model predicts (output)

feature_columns = [
    "route_id",      # encoded
    "day_of_week",   # encoded
    "hour",          # already numeric
    "weather",       # encoded
    "is_holiday",    # already 0/1
    "is_weekend",    # already 0/1
    "capacity",      # already numeric
    "passenger_count" # already numeric
]

target_column = "rush_level"

X = df_encoded[feature_columns]
y = df_encoded[target_column]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature sample:")
print(X.head(3))
print(f"\nTarget sample:")
print(y.head(3))

print(f"\nTarget distribution:")
for label, encoded in zip(
    encoders["rush_level"].classes_,
    range(len(encoders["rush_level"].classes_))
):
    count = (y == encoded).sum()
    pct = count/len(y)*100
    print(f"  {label} ({encoded}): {count} ({pct:.1f}%)")

# ---- TRAIN TEST SPLIT ----
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y    # ensures same rush level ratio in both splits
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

print(f"\nTrain target distribution:")
print(pd.Series(y_train).value_counts())

print(f"\nTest target distribution:")
print(pd.Series(y_test).value_counts())

# ---- SAVE PROCESSED DATA ----
print("\nSaving processed data...")
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Save encoders — needed in Flask app later
joblib.dump(encoders, "encoders.pkl")
print("Encoders saved as encoders.pkl")

# Save feature column names — needed in Flask app later
joblib.dump(feature_columns, "feature_columns.pkl")
print("Feature columns saved as feature_columns.pkl")

print("\nPreprocessing complete!")
print("Files saved:")
print("  X_train.csv, X_test.csv")
print("  y_train.csv, y_test.csv")
print("  encoders.pkl")
print("  feature_columns.pkl")