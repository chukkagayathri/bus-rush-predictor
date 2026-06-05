# generate_dataset.py
# Generates realistic APSRTC bus rush dataset for Bus Rush Predictor
# Author: Gayatri Chukka — CSE AI/ML

import random
import pandas as pd
from datetime import datetime, timedelta

# ---- CONFIGURATION ----
NUM_ROWS = 5000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---- DEFINE REALISTIC DATA ----

ROUTES = {
    "G01": {"name": "Guntur - Amaravati",      "capacity": 50},
    "G02": {"name": "Guntur - Tenali",         "capacity": 45},
    "G03": {"name": "Guntur - Mangalagiri",    "capacity": 55},
    "G04": {"name": "Guntur - Sattenapalli",   "capacity": 50},
    "G05": {"name": "Guntur - Ponnur",         "capacity": 60},
    "G06": {"name": "Guntur - Chilakaluripet", "capacity": 55},
    "G07": {"name": "Guntur - Narasaraopet",   "capacity": 50},
    "G08": {"name": "Guntur - Vijayawada",     "capacity": 60},
}

DISTRICTS = [
    "Guntur",
    "Palnadu",
    "Bapatla",
    "NTR",
    "Krishna"
]

WEATHER_OPTIONS = ["Sunny", "Rainy", "Cloudy", "Foggy"]
WEATHER_WEIGHTS = [0.55, 0.20, 0.20, 0.05]

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# ---- HELPER FUNCTIONS ----

def is_peak_hour(hour):
    return (7 <= hour <= 10) or (17 <= hour <= 20)

def is_weekend(day):
    return day in ["Saturday", "Sunday"]

def generate_passengers(route_id, hour, day, weather, is_holiday):
    """
    Generate realistic passenger count based on conditions.
    """
    capacity = ROUTES[route_id]["capacity"]
    base = capacity * 0.4

    # Peak hour boost
    if is_peak_hour(hour):
        base *= 1.8

    # Weekend reduction
    if is_weekend(day):
        base *= 0.6

    # Holiday boost
    if is_holiday:
        base *= 1.3

    # Weather effect
    if weather == "Rainy":
        base *= 1.25
    elif weather == "Foggy":
        base *= 1.10

    # Route-specific patterns
    if route_id in ["G01", "G08"]:  # Amaravati & Vijayawada routes
        if hour in [9, 10, 18, 19]:
            base *= 1.4

    if route_id == "G05":  # Ponnur route - slightly less busy
        base *= 0.7

    # Random variation (±15%)
    variation = random.uniform(0.85, 1.15)
    passengers = int(base * variation)

    # Cannot exceed capacity
    passengers = min(passengers, capacity)
    passengers = max(passengers, 0)

    return passengers

def get_rush_label(passengers, capacity):
    pct = (passengers / capacity) * 100

    if pct < 40:
        return "Low"
    elif pct < 75:
        return "Medium"
    else:
        return "High"

# ---- GENERATE DATA ----

print("Generating APSRTC bus rush dataset...")

data = []

for i in range(NUM_ROWS):

    # Random district
    district = random.choice(DISTRICTS)

    # Random route
    route_id = random.choice(list(ROUTES.keys()))
    capacity = ROUTES[route_id]["capacity"]

    # Random day
    day = random.choice(DAYS_OF_WEEK)

    # Random hour (5 AM to 11 PM)
    hour = random.randint(5, 23)

    # Random weather
    weather = random.choices(
        WEATHER_OPTIONS,
        weights=WEATHER_WEIGHTS
    )[0]

    # Holiday (10% chance)
    is_holiday = random.random() < 0.10

    # Weekend
    weekend = is_weekend(day)

    # Passenger count
    passengers = generate_passengers(
        route_id,
        hour,
        day,
        weather,
        is_holiday
    )

    # Rush label
    rush_level = get_rush_label(
        passengers,
        capacity
    )

    # Add row
    data.append({
        "district": district,
        "route_id": route_id,
        "day_of_week": day,
        "hour": hour,
        "weather": weather,
        "is_holiday": int(is_holiday),
        "is_weekend": int(weekend),
        "capacity": capacity,
        "passenger_count": passengers,
        "rush_level": rush_level
    })

# ---- CREATE DATAFRAME ----

df = pd.DataFrame(data)

# ---- VERIFY DATA ----

print(f"\nDataset shape: {df.shape}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nRush level distribution:")
print(df["rush_level"].value_counts())
print(df["rush_level"].value_counts(normalize=True).round(3))

print(f"\nRoute distribution:")
print(df["route_id"].value_counts())

print(f"\nDistrict distribution:")
print(df["district"].value_counts())

print(f"\nBasic statistics:")
print(df[["passenger_count", "capacity", "hour"]].describe())

print(f"\nNull values check:")
print(df.isnull().sum())

# ---- SAVE TO CSV ----

df.to_csv("apsrtc_bus_data.csv", index=False)

print(f"\nDataset saved as apsrtc_bus_data.csv")
print(f"Total rows: {len(df)}")

# ---- QUICK SANITY CHECK ----

print("\nSanity checks:")

print(
    f"Max passengers <= capacity always: "
    f"{(df['passenger_count'] <= df['capacity']).all()}"
)

print(
    f"No negative passengers: "
    f"{(df['passenger_count'] >= 0).all()}"
)

print(
    f"All rush levels valid: "
    f"{df['rush_level'].isin(['Low', 'Medium', 'High']).all()}"
)