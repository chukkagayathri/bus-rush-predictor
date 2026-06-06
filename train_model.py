# train_model.py
# Train ML models for Bus Rush Predictor
# Bus Rush Predictor Project

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---- LOAD PREPROCESSED DATA ----
print("Loading preprocessed data...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

encoders = joblib.load("encoders.pkl")

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ---- MODEL 1: DECISION TREE ----
print("\n" + "="*50)
print("TRAINING DECISION TREE")
print("="*50)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test, dt_pred,
    target_names=encoders["rush_level"].classes_
))

# ---- MODEL 2: RANDOM FOREST ----
print("\n" + "="*50)
print("TRAINING RANDOM FOREST")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # use all CPU cores
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test, rf_pred,
    target_names=encoders["rush_level"].classes_
))

# ---- COMPARE MODELS ----
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Decision Tree : {dt_accuracy*100:.2f}%")
print(f"Random Forest : {rf_accuracy*100:.2f}%")
print(f"Improvement   : +{(rf_accuracy-dt_accuracy)*100:.2f}%")

# ---- FEATURE IMPORTANCE ----
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)
feature_names = X_train.columns.tolist()
importances = rf_model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(
    feat_imp["feature"],
    feat_imp["importance"],
    color="#3498DB",
    edgecolor="black"
)
plt.xlabel("Importance Score")
plt.title("Feature Importance — Random Forest\nBus Rush Predictor",
          fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("chart5_feature_importance.png", dpi=150)
plt.show()
print("\nFeature importance chart saved!")

# ---- CONFUSION MATRIX ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rush_labels = encoders["rush_level"].classes_

for ax, (model, pred, name) in zip(
    axes,
    [(dt_model, dt_pred, "Decision Tree"),
     (rf_model, rf_pred, "Random Forest")]
):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=rush_labels
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}\nAccuracy: {accuracy_score(y_test, pred)*100:.2f}%",
                 fontweight="bold")

plt.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart6_confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix chart saved!")

# ---- SAVE BEST MODEL ----
print("\n" + "="*50)
print("SAVING BEST MODEL")
print("="*50)
joblib.dump(rf_model, "rush_model.pkl")
print(f"Random Forest model saved as rush_model.pkl")
print(f"Final accuracy: {rf_accuracy*100:.2f}%")

# ---- SAMPLE PREDICTION ----
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

# Load encoders for sample prediction
route_enc = encoders["route_id"]
day_enc = encoders["day_of_week"]
weather_enc = encoders["weather"]
rush_enc = encoders["rush_level"]

def predict_rush(route, day, hour, weather, is_holiday, is_weekend, capacity, passengers):
    sample = pd.DataFrame([{
        "route_id": route_enc.transform([route])[0],
        "day_of_week": day_enc.transform([day])[0],
        "hour": hour,
        "weather": weather_enc.transform([weather])[0],
        "is_holiday": int(is_holiday),
        "is_weekend": int(is_weekend),
        "capacity": capacity,
        "passenger_count": passengers
    }])
    pred = rf_model.predict(sample)[0]
    proba = rf_model.predict_proba(sample)[0]
    rush = rush_enc.inverse_transform([pred])[0]
    confidence = max(proba) * 100
    return rush, confidence

# Test samples
samples = [
    ("G01", "Monday", 8, "Rainy", False, False, 50, 47),
    ("G02", "Sunday", 14, "Sunny", False, True, 45, 12),
    ("G03", "Friday", 18, "Cloudy", False, False, 55, 48),
    ("G04", "Wednesday", 11, "Sunny", False, False, 25, 8),
]

for route, day, hour, weather, holiday, weekend, cap, pax in samples:
    rush, conf = predict_rush(
        route, day, hour, weather, holiday, weekend, cap, pax
    )
    print(f"Route:{route} | {day} {hour}:00 | {weather} | "
          f"Passengers:{pax}/{cap} → {rush} ({conf:.1f}% confident)")