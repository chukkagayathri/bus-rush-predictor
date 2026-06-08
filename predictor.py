# predictor.py
# Standalone prediction module
# Flask app will import this file
# Bus Rush Predictor Project

import pandas as pd
import joblib

# Load everything once when module is imported
# (not every time a prediction is made — much faster)
model = joblib.load("rush_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

route_enc = encoders["route_id"]
day_enc = encoders["day_of_week"]
weather_enc = encoders["weather"]
rush_enc = encoders["rush_level"]

# Available options (for Flask form dropdowns)
AVAILABLE_ROUTES = list(route_enc.classes_)
AVAILABLE_DAYS = list(day_enc.classes_)
AVAILABLE_WEATHER = list(weather_enc.classes_)
OPERATING_HOURS = list(range(5, 24))

# Route capacities
ROUTE_CAPACITIES = {
    "G01": 50,
    "G02": 45,
    "G03": 55,
    "G04": 50,
    "G05": 60,
    "G06": 45,
    "G07": 55,
    "G08": 50
}


def encode_input(route, day, hour, weather,
                 is_holiday, is_weekend, capacity, passengers):
    """Encode raw input into model-ready format."""
    return pd.DataFrame([{
        "route_id": route_enc.transform([route])[0],
        "day_of_week": day_enc.transform([day])[0],
        "hour": int(hour),
        "weather": weather_enc.transform([weather])[0],
        "is_holiday": int(is_holiday),
        "is_weekend": int(is_weekend),
        "capacity": int(capacity),
        "passenger_count": int(passengers)
    }])


def predict_rush(route, day, hour, weather,
                 is_holiday, is_weekend, capacity, passengers):
    """
    Predict rush level for given inputs.

    Returns dict with:
    - rush_level: "Low" / "Medium" / "High"
    - confidence: percentage (float)
    - all_probabilities: dict of all class probabilities
    - color: hex color for UI
    - emoji: visual indicator
    """
    sample = encode_input(
        route, day, hour, weather,
        is_holiday, is_weekend, capacity, passengers
    )

    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    rush = rush_enc.inverse_transform([pred])[0]
    confidence = round(max(proba) * 100, 1)

    all_proba = {
        label: round(p * 100, 1)
        for label, p in zip(rush_enc.classes_, proba)
    }

    color_map = {
        "Low": "#2ECC71",    # green
        "Medium": "#F39C12", # orange
        "High": "#E74C3C"    # red
    }
    emoji_map = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🔴"
    }

    return {
        "rush_level": rush,
        "confidence": confidence,
        "all_probabilities": all_proba,
        "color": color_map[rush],
        "emoji": emoji_map[rush]
    }


def find_best_time(route, day, weather, is_holiday, is_weekend):
    """Find hour with lowest predicted rush for given route/day/weather."""
    capacity = ROUTE_CAPACITIES.get(route, 50)
    avg_passengers = int(capacity * 0.4)

    best_hour = 5
    best_low_prob = -1

    for hour in OPERATING_HOURS:
        result = predict_rush(
            route, day, hour, weather,
            is_holiday, is_weekend, capacity, avg_passengers
        )
        low_prob = result["all_probabilities"].get("Low", 0)
        if low_prob > best_low_prob:
            best_low_prob = low_prob
            best_hour = hour
            best_rush = result["rush_level"]

    return {
        "best_hour": best_hour,
        "rush_level": best_rush,
        "low_probability": best_low_prob
    }


# ---- TEST ----
if __name__ == "__main__":
    print("Testing predictor module...")

    result = predict_rush(
    "G01", "Monday", 8, "Rainy",
    False, False, 50, 47
)
    print(f"\nPrediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    best = find_best_time("G01", "Monday", "Sunny", False, False)
    print(f"\nBest travel time:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    print("\nPredictor module ready for Flask!")