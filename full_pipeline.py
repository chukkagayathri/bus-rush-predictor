# full_pipeline.py
# APSRTC Bus Rush Predictor — Complete Pipeline
# Includes passenger count between stops feature
# Author: Gayatri Chukka | CSE AI/ML

import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

print("=" * 65)
print("   APSRTC BUS RUSH PREDICTOR — FULL PIPELINE")
print("   With Passenger Count Between Stops Feature")
print("=" * 65)

random.seed(42)
np.random.seed(42)

# ── ROUTE CONFIGURATION WITH STOPS ─────────────────────────
ROUTES = {
    "Guntur-Brodipet": {
        "capacity": 52,
        "stops": ["Guntur Bus Stand", "Brodipet X Roads",
                  "Arundelpet", "Kothapet", "Brodipet Terminal"],
        "zone": "Guntur City"
    },
    "Guntur-Nallapadu": {
        "capacity": 48,
        "stops": ["Guntur Bus Stand", "Nallapadu",
                  "Namburu", "Mangalagiri", "Nallapadu Terminal"],
        "zone": "Guntur-Vijayawada Corridor"
    },
    "Guntur-Amaravati": {
        "capacity": 60,
        "stops": ["Guntur Bus Stand", "Tadepalle",
                  "Undavalli", "Amaravati Capital", "Amaravati Terminal"],
        "zone": "Capital Region"
    },
    "Guntur-Mangalagiri": {
        "capacity": 52,
        "stops": ["Guntur Bus Stand", "Namburu",
                  "Mangalagiri", "Pedakakani", "Vijayawada"],
        "zone": "Guntur-Vijayawada Corridor"
    },
    "Guntur-Tenali": {
        "capacity": 55,
        "stops": ["Guntur Bus Stand", "Pedanandipadu",
                  "Kollipara", "Tenali Junction", "Tenali Bus Stand"],
        "zone": "Guntur-Tenali Corridor"
    },
    "Guntur-Ponnur": {
        "capacity": 45,
        "stops": ["Guntur Bus Stand", "Chilakaluripet Road",
                  "Bapatla", "Ponnur Junction", "Ponnur"],
        "zone": "Coastal Belt"
    },
    "Guntur-Sattenapalli": {
        "capacity": 50,
        "stops": ["Guntur Bus Stand", "Piduguralla",
                  "Nadikudi", "Rentachintala", "Sattenapalli"],
        "zone": "Guntur-Nalgonda Corridor"
    },
    "Guntur-Narasaraopet": {
        "capacity": 55,
        "stops": ["Guntur Bus Stand", "Macherla Road",
                  "Dachepalli", "Bellamkonda", "Narasaraopet"],
        "zone": "Guntur-Palnadu Corridor"
    },
    "Tenali-Repalle": {
        "capacity": 42,
        "stops": ["Tenali Bus Stand", "Nizampatnam Road",
                  "Bapatla", "Chirala", "Repalle"],
        "zone": "Tenali Coastal"
    },
    "Tenali-Bapatla": {
        "capacity": 45,
        "stops": ["Tenali Bus Stand", "Vemuru",
                  "Karumanchi", "Bapatla"],
        "zone": "Tenali Coastal"
    },
    "Tenali-Chirala": {
        "capacity": 48,
        "stops": ["Tenali Bus Stand", "Bapatla",
                  "Inkollu", "Vetapalem", "Chirala"],
        "zone": "Tenali-Chirala Corridor"
    },
    "Tenali-Guntur": {
        "capacity": 55,
        "stops": ["Tenali Bus Stand", "Kollipara",
                  "Pedanandipadu", "Guntur Bus Stand"],
        "zone": "Guntur-Tenali Corridor"
    },
    "Tenali-Vijayawada": {
        "capacity": 60,
        "stops": ["Tenali Bus Stand", "Kollipara",
                  "Guntur Bus Stand", "Mangalagiri", "Vijayawada"],
        "zone": "Express Corridor"
    },
    "Tenali-Ongole": {
        "capacity": 50,
        "stops": ["Tenali Bus Stand", "Chirala",
                  "Vetapalem", "Singarayakonda", "Ongole"],
        "zone": "South Coastal Corridor"
    },
}

DAYS = ["Monday","Tuesday","Wednesday","Thursday",
        "Friday","Saturday","Sunday"]
WEATHER_OPTIONS = ["Sunny","Rainy","Cloudy","Foggy"]
WEATHER_WEIGHTS = [0.50, 0.22, 0.22, 0.06]

def is_peak(hour):
    return (7 <= hour <= 10) or (17 <= hour <= 20)

def is_wknd(day):
    return day in ["Saturday","Sunday"]

def gen_passengers(route_id, hour, day, weather, is_holiday):
    cap  = ROUTES[route_id]["capacity"]
    base = cap * 0.42
    if is_peak(hour):            base *= 1.85
    if is_wknd(day):             base *= 0.62
    if is_holiday:               base *= 1.35
    if weather == "Rainy":       base *= 1.28
    elif weather == "Foggy":     base *= 1.12
    if "Amaravati" in route_id:  base *= 1.25
    if "Vijayawada" in route_id: base *= 1.20
    if route_id in ["Guntur-Sattenapalli","Guntur-Narasaraopet",
                    "Tenali-Repalle"] and day in ["Wednesday","Saturday"]:
        base *= 1.30
    variation  = random.uniform(0.82, 1.18)
    passengers = int(base * variation)
    return max(0, min(passengers, cap))

def rush_label(passengers, capacity):
    pct = passengers / capacity * 100
    if pct < 40:   return "Low"
    elif pct < 75: return "Medium"
    else:          return "High"

# ── GENERATE PASSENGER FLOW BETWEEN STOPS ──────────────────
def generate_stop_flow(route_id, hour, day, weather,
                       is_holiday, total_passengers):
    """
    Generate realistic passenger counts between every stop pair.
    This is the Origin-Destination (OD) Matrix for a single trip.
    Returns a dict: {(from_stop, to_stop): passenger_count}
    """
    stops    = ROUTES[route_id]["stops"]
    capacity = ROUTES[route_id]["capacity"]
    n        = len(stops)
    flow     = {}

    for i in range(n - 1):
        for j in range(i + 1, n):
            distance      = j - i
            # Longer trips carry more passengers at peak
            dist_factor   = (0.55 + distance * 0.18) if is_peak(hour) \
                            else (0.38 + distance * 0.13)
            # Earlier stops have more boardings
            origin_factor = max(1.0 - (i * 0.14), 0.40)
            pax = int(total_passengers * dist_factor
                      * origin_factor * random.uniform(0.82, 1.18))
            pax = max(1, min(pax, capacity))
            flow[(stops[i], stops[j])] = pax

    return flow

# ── STEP 1: GENERATE DATASET ──────────────────────────────
print("\n📊 STEP 1: Generating dataset...")

data = []
for _ in range(8000):
    route_id    = random.choice(list(ROUTES.keys()))
    cap         = ROUTES[route_id]["capacity"]
    day         = random.choice(DAYS)
    hour        = random.randint(5, 23)
    weather     = random.choices(WEATHER_OPTIONS,
                                  weights=WEATHER_WEIGHTS)[0]
    is_holiday  = random.random() < 0.10
    weekend     = is_wknd(day)
    passengers  = gen_passengers(route_id, hour, day,
                                  weather, is_holiday)
    rush        = rush_label(passengers, cap)

    # Passenger flow for the busiest stop-to-stop segment
    flow        = generate_stop_flow(route_id, hour, day,
                                      weather, is_holiday, passengers)
    max_seg_pax = max(flow.values()) if flow else passengers
    min_seg_pax = min(flow.values()) if flow else 0

    data.append({
        "route_id":        route_id,
        "day_of_week":     day,
        "hour":            hour,
        "weather":         weather,
        "is_holiday":      int(is_holiday),
        "is_weekend":      int(weekend),
        "capacity":        cap,
        "passenger_count": passengers,
        "max_segment_pax": max_seg_pax,
        "min_segment_pax": min_seg_pax,
        "rush_level":      rush
    })

df = pd.DataFrame(data)
df.to_csv("apsrtc_bus_data.csv", index=False)

print(f"  ✅ apsrtc_bus_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")
print("  Rush distribution:")
for label, count in df["rush_level"].value_counts().items():
    print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")

# ── STEP 2: PREPROCESS ────────────────────────────────────
print("\n⚙️  STEP 2: Preprocessing...")

feature_columns = [
    "route_id","day_of_week","hour",
    "weather","is_holiday","is_weekend","capacity"
]

encoders   = {}
df_encoded = df.copy()
for col in ["route_id","day_of_week","weather","rush_level"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col]   = le

print("  Encoded values:")
for col in ["route_id","day_of_week","weather","rush_level"]:
    le = encoders[col]
    m  = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    print(f"    {col}: {m}")

X = df_encoded[feature_columns]
y = df_encoded["rush_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# Save splits
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",   index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",   index=False)

joblib.dump(encoders,        "encoders.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(ROUTES,          "routes_config.pkl")
print("  ✅ encoders.pkl, feature_columns.pkl, routes_config.pkl saved")
print("  ✅ X_train.csv, X_test.csv, y_train.csv, y_test.csv saved")

# ── STEP 3: TRAIN ─────────────────────────────────────────
print("\n🤖 STEP 3: Training models...")

dt = DecisionTreeClassifier(max_depth=10,
                             min_samples_split=5,
                             random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
print(f"  Decision Tree : {dt_acc*100:.2f}%")

rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                             min_samples_split=5,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"  Random Forest : {rf_acc*100:.2f}%")

print("\n  Classification Report:")
print(classification_report(
    y_test, rf_pred,
    target_names=encoders["rush_level"].classes_
))

feat_imp = pd.DataFrame({
    "feature":    feature_columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print("  Feature Importance:")
print(feat_imp.to_string(index=False))

joblib.dump(rf, "rush_model.pkl")
print(f"\n  ✅ rush_model.pkl saved (accuracy: {rf_acc*100:.2f}%)")

# ── STEP 4: CHARTS ────────────────────────────────────────
print("\n📈 STEP 4: Generating charts...")

COLORS = {"Low":"#27AE60","Medium":"#F39C12","High":"#E74C3C"}

# Chart 1 — Rush Distribution
fig, ax = plt.subplots(figsize=(8,5))
rc   = df["rush_level"].value_counts()
bars = ax.bar(rc.index, rc.values,
              color=[COLORS[x] for x in rc.index],
              edgecolor="black", linewidth=0.6, width=0.55)
ax.set_title("Bus Rush Level Distribution\nAPSRTC Guntur & Tenali",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Rush Level"); ax.set_ylabel("Number of Trips")
ax.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, rc.values):
    ax.text(bar.get_x()+bar.get_width()/2,
            val+40, str(val), ha="center",
            fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig("chart1_rush_distribution.png", dpi=150)
plt.close()
print("  ✅ chart1_rush_distribution.png")

# Chart 2 — Passengers by Hour
fig, ax = plt.subplots(figsize=(13,5))
ha = df.groupby("hour")["passenger_count"].mean()
ax.plot(ha.index, ha.values, color="#2980B9",
        linewidth=2.5, marker="o", markersize=5, zorder=3)
ax.axvspan(7,  10, alpha=0.12, color="red",    label="Morning peak")
ax.axvspan(17, 20, alpha=0.12, color="orange", label="Evening peak")
ax.set_title("Average Passengers by Hour — APSRTC",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Hour"); ax.set_ylabel("Avg Passengers")
ax.set_xticks(range(5,24)); ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart2_passengers_by_hour.png", dpi=150)
plt.close()
print("  ✅ chart2_passengers_by_hour.png")

# Chart 3 — Rush by Route
fig, ax = plt.subplots(figsize=(15,6))
rr = df.groupby("route_id")["rush_level"].value_counts(
    normalize=True).mul(100).unstack(fill_value=0)
for col in ["Low","Medium","High"]:
    if col not in rr.columns: rr[col] = 0
rr[["Low","Medium","High"]].plot(
    kind="bar", stacked=True,
    color=["#27AE60","#F39C12","#E74C3C"],
    edgecolor="black", linewidth=0.5, ax=ax)
ax.set_title("Rush Level Distribution by Route (%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Route"); ax.set_ylabel("Percentage (%)")
plt.xticks(rotation=25, ha="right", fontsize=9)
ax.legend(title="Rush Level")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart3_rush_by_route.png", dpi=150)
plt.close()
print("  ✅ chart3_rush_by_route.png")

# Chart 4 — Feature Importance
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(feat_imp["feature"], feat_imp["importance"],
        color="#2980B9", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance — Random Forest",
             fontsize=13, fontweight="bold")
ax.invert_yaxis(); ax.spines[["top","right"]].set_visible(False)
for i,(feat,imp) in enumerate(
        zip(feat_imp["feature"], feat_imp["importance"])):
    ax.text(imp+0.002, i, f"{imp:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("chart4_feature_importance.png", dpi=150)
plt.close()
print("  ✅ chart4_feature_importance.png")

# Chart 5 — Confusion Matrix
fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax,(pred,name,acc) in zip(axes,[
    (dt_pred,"Decision Tree",dt_acc),
    (rf_pred,"Random Forest",rf_acc)]):
    cm   = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoders["rush_level"].classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy: {acc*100:.2f}%", fontweight="bold")
plt.suptitle("Confusion Matrix Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart5_confusion_matrix.png", dpi=150)
plt.close()
print("  ✅ chart5_confusion_matrix.png")

# Chart 6 — Passenger Segment Analysis
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].hist(df["max_segment_pax"], bins=20,
             color="#E74C3C", edgecolor="black", alpha=0.8)
axes[0].set_title("Most Crowded Segment\nPassenger Distribution",
                   fontweight="bold")
axes[0].set_xlabel("Passengers in Busiest Segment")
axes[0].set_ylabel("Frequency")
axes[1].hist(df["min_segment_pax"], bins=20,
             color="#27AE60", edgecolor="black", alpha=0.8)
axes[1].set_title("Least Crowded Segment\nPassenger Distribution",
                   fontweight="bold")
axes[1].set_xlabel("Passengers in Least Crowded Segment")
plt.suptitle("Passenger Count Between Stops — Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart6_passenger_segment_analysis.png", dpi=150)
plt.close()
print("  ✅ chart6_passenger_segment_analysis.png")

# Chart 7 — Best Travel Time Heatmap
fig, ax = plt.subplots(figsize=(14,6))
pivot = df.groupby(["hour","rush_level"]).size().unstack(fill_value=0)
for col in ["Low","Medium","High"]:
    if col not in pivot.columns: pivot[col] = 0
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
low_pct   = pivot_pct.get("Low", pd.Series(0, index=pivot_pct.index))
ax.fill_between(low_pct.index, low_pct.values,
                color="#27AE60", alpha=0.6, label="% Low rush")
ax.axvspan(7,10,  alpha=0.15, color="red",    label="Morning peak")
ax.axvspan(17,20, alpha=0.15, color="orange", label="Evening peak")
ax.set_title("Best Travel Time — Probability of Low Rush by Hour",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("% Low Rush Probability")
ax.set_xticks(range(5,24)); ax.legend()
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("chart7_best_travel_time.png", dpi=150)
plt.close()
print("  ✅ chart7_best_travel_time.png")

# ── STEP 5: VERIFY ────────────────────────────────────────
print("\n✅ STEP 5: Verifying files...")
files = [
    "apsrtc_bus_data.csv","rush_model.pkl","encoders.pkl",
    "feature_columns.pkl","routes_config.pkl",
    "X_train.csv","X_test.csv","y_train.csv","y_test.csv",
    "chart1_rush_distribution.png","chart2_passengers_by_hour.png",
    "chart3_rush_by_route.png","chart4_feature_importance.png",
    "chart5_confusion_matrix.png","chart6_passenger_segment_analysis.png",
    "chart7_best_travel_time.png",
]
all_ok = True
for f in files:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) if exists else 0
    print(f"  {'✅' if exists else '❌'} {f:45} ({size:>10,} bytes)")
    if not exists: all_ok = False

print()
print("=" * 65)
print("  🎉 PIPELINE COMPLETE")
print(f"  Model accuracy : {rf_acc*100:.2f}%")
print(f"  Dataset rows   : {len(df)}")
print(f"  Total routes   : {len(ROUTES)}")
print("=" * 65)
print("  Next: python app.py → http://localhost:5000")