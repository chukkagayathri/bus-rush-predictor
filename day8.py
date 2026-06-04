# Day 8 — Pandas Deep Dive on bus_data.csv
import pandas as pd

df = pd.read_csv("bus_data.csv")

# ---- BASIC EXPLORATION ----
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nHead:")
print(df.head(10))

# ---- INDEXING AND SELECTING ----
# Select single column
print("\nAll rush levels:")
print(df["rush_level"].head())

# Select multiple columns
print("\nRoute and rush:")
print(df[["route_id", "rush_level"]].head())

# Select rows by condition
high_rush = df[df["rush_level"] == "High"]
print(f"\nHigh rush trips: {len(high_rush)}")

# Multiple conditions
peak_rainy = df[
    (df["hour"].between(7, 10)) &
    (df["weather"] == "Rainy")
]
print(f"\nPeak hour rainy trips: {len(peak_rainy)}")

# ---- GROUPBY ----
print("\nAverage passengers by route:")
print(df.groupby("route_id")["passenger_count"].mean().round(1))

print("\nRush level count by day:")
print(df.groupby("day_of_week")["rush_level"]
      .value_counts().unstack(fill_value=0))

print("\nHighest rush hours:")
print(df.groupby("hour")["passenger_count"]
      .mean().sort_values(ascending=False).head(5))

# ---- VALUE COUNTS ----
print("\nWeather distribution:")
print(df["weather"].value_counts())

print("\nDay distribution:")
print(df["day_of_week"].value_counts())

# ---- STATISTICS ----
print("\nPassenger stats per route:")
print(df.groupby("route_id")["passenger_count"].agg([
    "min", "max", "mean", "std"
]).round(1))

# ---- DATA TYPES ----
print("\nColumn data types:")
print(df.dtypes)

# ---- MISSING VALUES ----
print("\nMissing values per column:")
print(df.isnull().sum())

# ---- SORTING ----
print("\nTop 10 busiest trips:")
print(df.nlargest(10, "passenger_count")[[
    "route_id", "day_of_week", "hour",
    "weather", "passenger_count", "rush_level"
]])