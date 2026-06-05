# eda_analysis.py
# Exploratory Data Analysis on APSRTC dataset
# Bus Rush Predictor Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- LOAD DATA ----
df = pd.read_csv("apsrtc_bus_data.csv")

print("=" * 50)
print("APSRTC BUS RUSH DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# ---- BASIC INFO ----
print(f"\nShape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nNull values:")
print(df.isnull().sum())

# ---- RUSH LEVEL DISTRIBUTION ----
print(f"\nRush level counts:")
print(df["rush_level"].value_counts())

print(f"\nRush level percentages:")
print(
    df["rush_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .astype(str)
    + "%"
)

# ---- ROUTE ANALYSIS ----
print(f"\nAverage passengers per route:")
route_avg = df.groupby("route_id")["passenger_count"].mean().round(1)
print(route_avg)

print(f"\nRush distribution by route:")
route_rush = df.groupby("route_id")["rush_level"].value_counts()
print(route_rush)

# ---- TIME ANALYSIS ----
print(f"\nBusiest hours (avg passengers):")
hour_avg = df.groupby("hour")["passenger_count"].mean().round(1)
print(hour_avg.sort_values(ascending=False).head(5))

# ---- WEATHER IMPACT ----
print(f"\nAverage passengers by weather:")
print(df.groupby("weather")["passenger_count"].mean().round(1))

# ---- DAY OF WEEK ----
print(f"\nAverage passengers by day:")
print(
    df.groupby("day_of_week")["passenger_count"]
    .mean()
    .round(1)
    .sort_values(ascending=False)
)

# ---- DISTRICT ANALYSIS ----
print(f"\nAverage passengers by district:")
print(
    df.groupby("district")["passenger_count"]
    .mean()
    .round(1)
    .sort_values(ascending=False)
)

# ==================================
# CHARTS
# ==================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# ---- CHART 1 — Rush Level Distribution ----
plt.figure(figsize=(8, 6))

colors = {
    "Low": "#2ECC71",
    "Medium": "#F39C12",
    "High": "#E74C3C"
}

rush_counts = df["rush_level"].value_counts()

plt.bar(
    rush_counts.index,
    rush_counts.values,
    color=[colors[x] for x in rush_counts.index],
    edgecolor="black",
    linewidth=0.5
)

plt.title(
    "Bus Rush Level Distribution\nAPSRTC Guntur Routes",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Rush Level")
plt.ylabel("Number of Trips")

for i, (label, val) in enumerate(rush_counts.items()):
    plt.text(
        i,
        val + 20,
        str(val),
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("chart1_rush_distribution.png", dpi=150)
plt.show()

print("Chart 1 saved: chart1_rush_distribution.png")

# ---- CHART 2 — Average Passengers by Hour ----
plt.figure(figsize=(12, 6))

hour_avg = df.groupby("hour")["passenger_count"].mean()

plt.plot(
    hour_avg.index,
    hour_avg.values,
    color="#3498DB",
    linewidth=2.5,
    marker="o",
    markersize=6
)

# Highlight peak hours
plt.axvspan(
    7,
    10,
    alpha=0.15,
    color="red",
    label="Morning peak"
)

plt.axvspan(
    17,
    20,
    alpha=0.15,
    color="orange",
    label="Evening peak"
)

plt.title(
    "Average Passengers by Hour of Day\nMorning & Evening Peak Hours Highlighted",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Passengers")
plt.xticks(range(5, 24))
plt.legend()

plt.tight_layout()
plt.savefig("chart2_passengers_by_hour.png", dpi=150)
plt.show()

print("Chart 2 saved: chart2_passengers_by_hour.png")

# ---- CHART 3 — Rush Level by Route ----
plt.figure(figsize=(12, 6))

route_rush_pct = (
    df.groupby("route_id")["rush_level"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack(fill_value=0)
)

# Reorder columns
for col in ["Low", "Medium", "High"]:
    if col not in route_rush_pct.columns:
        route_rush_pct[col] = 0

route_rush_pct = route_rush_pct[
    ["Low", "Medium", "High"]
]

route_rush_pct.plot(
    kind="bar",
    stacked=True,
    color=["#2ECC71", "#F39C12", "#E74C3C"],
    edgecolor="black",
    linewidth=0.5
)

plt.title(
    "Rush Level Distribution by Route (%)\nAPSRTC Guntur Region",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Route ID")
plt.ylabel("Percentage of Trips (%)")
plt.xticks(rotation=0)
plt.legend(title="Rush Level")

plt.tight_layout()
plt.savefig("chart3_rush_by_route.png", dpi=150)
plt.show()

print("Chart 3 saved: chart3_rush_by_route.png")

# ---- CHART 4 — Weather Impact ----
plt.figure(figsize=(8, 6))

weather_avg = (
    df.groupby("weather")["passenger_count"]
    .mean()
    .sort_values()
)

colors_w = ["#85C1E9", "#AED6F1", "#F8C471", "#F39C12"]

plt.barh(
    weather_avg.index,
    weather_avg.values,
    color=colors_w,
    edgecolor="black",
    linewidth=0.5
)

plt.title(
    "Average Passengers by Weather Condition",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Average Passengers")
plt.ylabel("Weather")

for i, val in enumerate(weather_avg.values):
    plt.text(
        val + 0.3,
        i,
        f"{val:.1f}",
        va="center",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("chart4_weather_impact.png", dpi=150)
plt.show()

print("Chart 4 saved: chart4_weather_impact.png")

print("\nAll 4 charts saved successfully!")
print("EDA complete.")