# model_improvements.py
# Improve model predictions and add confidence scores
# Bus Rush Predictor Project

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---- LOAD SAVED MODEL AND ENCODERS ----
print("Loading model and encoders...")
rf_model = joblib.load("rush_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

route_enc = encoders["route_id"]
day_enc = encoders["day_of_week"]
weather_enc = encoders["weather"]
rush_enc = encoders["rush_level"]

print(f"Model loaded successfully")
print(f"Rush level classes: {rush_enc.classes_}")

# ---- PREDICT WITH CONFIDENCE ----
def predict_rush_detailed(route, day, hour, weather,
                           is_holiday, is_weekend, capacity, passengers):
    """
    Predict rush level with confidence score.
    Returns: rush_level, confidence, all_probabilities
    """
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

    # All probabilities
    all_proba = {
        label: round(p * 100, 1)
        for label, p in zip(rush_enc.classes_, proba)
    }

    return rush, confidence, all_proba


# ---- BEST TRAVEL TIME FEATURE ----
def find_best_travel_time(route, day, weather,
                           is_holiday, is_weekend, capacity):
    """
    Find the best (least crowded) time to travel on a given route.
    Loops through all operating hours and finds the lowest predicted rush.
    """
    OPERATING_HOURS = list(range(5, 24))

    results = []
    for hour in OPERATING_HOURS:
        # For best time prediction, use average load (40% capacity)
        avg_passengers = int(capacity * 0.4)
        rush, confidence, all_proba = predict_rush_detailed(
            route, day, hour, weather,
            is_holiday, is_weekend, capacity, avg_passengers
        )
        results.append({
            "hour": hour,
            "rush_level": rush,
            "low_probability": all_proba.get("Low", 0),
            "confidence": confidence
        })

    results_df = pd.DataFrame(results)

    # Best time = hour with highest probability of Low rush
    best_idx = results_df["low_probability"].idxmax()
    best_hour = results_df.loc[best_idx, "hour"]
    best_rush = results_df.loc[best_idx, "rush_level"]
    best_low_prob = results_df.loc[best_idx, "low_probability"]

    return best_hour, best_rush, best_low_prob, results_df


# ---- TEST DETAILED PREDICTION ----
print("\n" + "="*50)
print("DETAILED PREDICTIONS WITH CONFIDENCE")
print("="*50)

test_cases = [
    ("G01", "Monday", 8, "Rainy", False, False, 50, 47),
    ("G02", "Sunday", 14, "Sunny", False, True, 45, 12),
    ("G03", "Friday", 18, "Cloudy", False, False, 55, 48),
]

for route, day, hour, weather, holiday, weekend, cap, pax in test_cases:
    rush, conf, proba = predict_rush_detailed(
        route, day, hour, weather, holiday, weekend, cap, pax
    )
    print(f"\nRoute {route} | {day} {hour}:00 | "
          f"{weather} | {pax}/{cap} passengers")
    print(f"  Prediction : {rush}")
    print(f"  Confidence : {conf:.1f}%")
    print(f"  All probabilities:")
    for label, p in proba.items():
        bar = "█" * int(p/5)
        print(f"    {label:8}: {p:5.1f}% {bar}")


# ---- TEST BEST TRAVEL TIME ----
print("\n" + "="*50)
print("BEST TRAVEL TIME FINDER")
print("="*50)

routes_to_check = ["G01", "G02", "G03"]
for route in routes_to_check:
    best_hour, best_rush, best_prob, all_results = find_best_travel_time(
        route=route,
        day="Monday",
        weather="Sunny",
        is_holiday=False,
        is_weekend=False,
        capacity=50
    )
    print(f"\nRoute {route} on Monday (Sunny):")
    print(f"  Best time to travel: {best_hour}:00")
    print(f"  Expected rush: {best_rush}")
    print(f"  Chance of Low rush: {best_prob:.1f}%")

    # Show full hourly breakdown
    print(f"  Hourly breakdown:")
    for _, row in all_results.iterrows():
        indicator = "← BEST" if row["hour"] == best_hour else ""
        print(f"    {int(row['hour']):02d}:00 — "
              f"{row['rush_level']:8} "
              f"(Low prob: {row['low_probability']:.1f}%) {indicator}")


# ---- VISUALISE HOURLY PREDICTIONS ----
plt.figure(figsize=(14, 6))
colors = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}

for i, route in enumerate(["G01", "G02", "G03"]):
    _, _, _, results = find_best_travel_time(
        route, "Monday", "Sunny", False, False, 50
    )
    plt.subplot(1, 3, i+1)
    bar_colors = [colors[r] for r in results["rush_level"]]
    plt.bar(results["hour"], results["low_probability"],
            color=bar_colors, edgecolor="black", linewidth=0.5)
    plt.title(f"Route {route}\nLow Rush Probability by Hour",
              fontweight="bold", fontsize=10)
    plt.xlabel("Hour")
    plt.ylabel("Low Rush Probability (%)")
    plt.xticks(range(5, 24, 2))
    plt.ylim(0, 100)

plt.suptitle("Best Travel Time Analysis — Monday Sunny",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart7_best_travel_time.png", dpi=150)
plt.show()
print("\nChart saved: chart7_best_travel_time.png")

# ---- SAVE PREDICTION FUNCTIONS ----
# Save as a separate module for Flask app to import
print("\nAll improvements complete!")
print("predict_rush_detailed() and find_best_travel_time()")
print("are ready to be imported into Flask app")