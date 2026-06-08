# day12_verify.py
# Verify all saved model files are working correctly

import os
import joblib
import pandas as pd

print("Verifying saved files...")
print("=" * 50)

# Check all files exist
files_needed = [
    "apsrtc_bus_data.csv",
    "rush_model.pkl",
    "encoders.pkl",
    "feature_columns.pkl",
    "X_train.csv",
    "X_test.csv"
]

for f in files_needed:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "✅" if exists else "❌"
    print(f"{status} {f} ({size:,} bytes)")

# Stop if critical files are missing
if not all(os.path.exists(f) for f in files_needed[1:]):
    print("\n❌ Required model files are missing!")
    exit()

# Load saved files
print("\nLoading model...")
model = joblib.load("rush_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print(f"Model type: {type(model).__name__}")
print(f"Number of trees: {model.n_estimators}")
print(f"Feature columns: {feature_columns}")
print(f"Encoder keys: {list(encoders.keys())}")
print(f"Rush labels: {encoders['rush_level'].classes_}")

# Test prediction
print("\nTest prediction")
print("-" * 50)

sample = pd.DataFrame([{
    "route_id": encoders["route_id"].transform(["G01"])[0],
    "day_of_week": encoders["day_of_week"].transform(["Monday"])[0],
    "hour": 8,
    "weather": encoders["weather"].transform(["Rainy"])[0],
    "is_holiday": 0,
    "is_weekend": 0,
    "capacity": 50,
    "passenger_count": 47
}])

pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

rush = encoders["rush_level"].inverse_transform([pred])[0]
confidence = max(proba) * 100

print("Input:")
print("  Route: G01")
print("  Day: Monday")
print("  Hour: 08:00")
print("  Weather: Rainy")
print("  Capacity: 50")
print("  Passengers: 47")

print("\nPrediction:")
print(f"  Rush Level : {rush}")
print(f"  Confidence : {confidence:.1f}%")

print("\nClass Probabilities:")
for label, prob in zip(encoders["rush_level"].classes_, proba):
    print(f"  {label:8} : {prob*100:.1f}%")

print("\n✅ All files verified successfully!")