# predictor.py
# ML Prediction module — imported by Flask app
# Includes full passenger flow between stops
# APSRTC Bus Rush Predictor

import pandas as pd
import joblib
import random

# ── LOAD ONCE AT STARTUP ──────────────────────────────────
model           = joblib.load("rush_model.pkl")
encoders        = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")
ROUTES          = joblib.load("routes_config.pkl")

# Available dropdown options
AVAILABLE_ROUTES  = sorted(list(encoders["route_id"].classes_))
AVAILABLE_DAYS    = list(encoders["day_of_week"].classes_)
AVAILABLE_WEATHER = list(encoders["weather"].classes_)
OPERATING_HOURS   = list(range(5, 24))

RUSH_CONFIG = {
    "Low":    {"color":"#27AE60","bg":"#EAFAF1","emoji":"🟢",
               "text":"Comfortable journey — seats available"},
    "Medium": {"color":"#F39C12","bg":"#FEF9EC","emoji":"🟡",
               "text":"Moderately crowded — arrive early"},
    "High":   {"color":"#E74C3C","bg":"#FDEDEC","emoji":"🔴",
               "text":"Very crowded — consider different time"},
}

def is_peak(hour):
    return (7 <= int(hour) <= 10) or (17 <= int(hour) <= 20)

def format_hour(hour):
    hour = int(hour)
    if hour == 0:    return "12:00 AM"
    elif hour < 12:  return f"{hour}:00 AM"
    elif hour == 12: return "12:00 PM"
    else:            return f"{hour-12}:00 PM"

def predict_rush(route, day, hour, weather,
                 is_holiday, is_weekend):
    capacity = ROUTES.get(route, {}).get("capacity", 50)
    sample   = pd.DataFrame([{
        "route_id":    encoders["route_id"].transform([route])[0],
        "day_of_week": encoders["day_of_week"].transform([day])[0],
        "hour":        int(hour),
        "weather":     encoders["weather"].transform([weather])[0],
        "is_holiday":  int(is_holiday),
        "is_weekend":  int(is_weekend),
        "capacity":    int(capacity)
    }])
    pred  = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    rush  = encoders["rush_level"].inverse_transform([pred])[0]
    conf  = round(max(proba) * 100, 1)
    all_proba = {
        label: round(p * 100, 1)
        for label, p in zip(encoders["rush_level"].classes_, proba)
    }
    cfg = RUSH_CONFIG.get(rush, {})
    return {
        "rush_level":        rush,
        "confidence":        conf,
        "all_probabilities": all_proba,
        "color":             cfg.get("color","#333"),
        "bg_color":          cfg.get("bg","#fff"),
        "emoji":             cfg.get("emoji","⚪"),
        "text":              cfg.get("text",""),
    }

def find_best_time(route, day, weather,
                   is_holiday, is_weekend):
    best_hour     = 5
    best_low_prob = -1
    best_rush     = "Low"
    for hour in OPERATING_HOURS:
        result   = predict_rush(route, day, hour, weather,
                                is_holiday, is_weekend)
        low_prob = result["all_probabilities"].get("Low", 0)
        if low_prob > best_low_prob:
            best_low_prob = low_prob
            best_hour     = hour
            best_rush     = result["rush_level"]
    return {
        "best_hour":       best_hour,
        "rush_level":      best_rush,
        "low_probability": round(best_low_prob, 1),
        "formatted":       format_hour(best_hour),
    }

def get_passenger_flow(route, day, hour, weather,
                       is_holiday, is_weekend):
    """
    Calculate how many passengers travel between
    every pair of stops on the route.
    Shows: From Stop → To Stop → Passenger Count → % of Capacity
    """
    stops    = ROUTES.get(route, {}).get("stops", [])
    capacity = ROUTES.get(route, {}).get("capacity", 50)

    if len(stops) < 2:
        return None

    result     = predict_rush(route, day, hour, weather,
                               is_holiday, is_weekend)
    rush_level = result["rush_level"]

    # Base load depends on rush level
    load_map  = {"Low": 0.30, "Medium": 0.58, "High": 0.90}
    base_load = capacity * load_map.get(rush_level, 0.5)

    if is_peak(hour):         base_load *= 1.20
    if weather == "Rainy":    base_load *= 1.18
    if is_holiday:            base_load *= 1.12

    # Use fixed seed based on inputs for reproducibility
    random.seed(hash(f"{route}{day}{hour}{weather}") % (2**31))

    flow_data = []
    for i in range(len(stops) - 1):
        for j in range(i + 1, len(stops)):
            distance = j - i
            # Longer distance = more passengers during peak
            dist_factor   = (0.55 + distance * 0.18) if is_peak(hour) \
                            else (0.38 + distance * 0.13)
            # Earlier stops have more boardings
            origin_factor = max(1.0 - (i * 0.14), 0.40)

            passengers = int(
                base_load * dist_factor * origin_factor
                * random.uniform(0.82, 1.18)
            )
            passengers = max(1, min(passengers, capacity))
            pct        = round(passengers / capacity * 100, 1)

            if pct >= 75:   rush = "High"
            elif pct >= 40: rush = "Medium"
            else:           rush = "Low"

            flow_data.append({
                "from_stop":   stops[i],
                "to_stop":     stops[j],
                "passengers":  passengers,
                "percentage":  pct,
                "rush":        rush,
                "is_direct":   distance == 1,
                "color": "#E74C3C" if rush=="High"
                         else ("#F39C12" if rush=="Medium"
                         else "#27AE60"),
            })

    # Sort by passenger count — most crowded first
    flow_data.sort(key=lambda x: x["passengers"], reverse=True)

    return {
        "flows":          flow_data,
        "most_crowded":   flow_data[0]  if flow_data else None,
        "least_crowded":  flow_data[-1] if flow_data else None,
        "route_stops":    stops,
        "total_stops":    len(stops),
        "capacity":       capacity,
        "rush_level":     rush_level,
        "direct_only":    [f for f in flow_data if f["is_direct"]],
    }

def get_hourly_forecast(route, day, weather,
                        is_holiday, is_weekend):
    forecast = []
    for hour in OPERATING_HOURS:
        result = predict_rush(route, day, hour, weather,
                               is_holiday, is_weekend)
        forecast.append({
            "hour":       hour,
            "hour_label": format_hour(hour),
            "rush":       result["rush_level"],
            "confidence": result["confidence"],
            "low_prob":   result["all_probabilities"].get("Low", 0),
            "color":      result["color"],
            "is_peak":    is_peak(hour),
        })
    return forecast

def get_route_stops(route):
    """Return list of stops for a route."""
    return ROUTES.get(route, {}).get("stops", [])

def get_capacity(route):
    """Return capacity for a route."""
    return ROUTES.get(route, {}).get("capacity", 50)

if __name__ == "__main__":
    print("Testing predictor.py...")
    r = AVAILABLE_ROUTES[0]
    result = predict_rush(r, "Monday", 8, "Rainy",
                          False, False)
    print(f"\nTest: Route={r} Monday 8AM Rainy")
    print(f"  Rush: {result['emoji']} {result['rush_level']}")
    print(f"  Confidence: {result['confidence']}%")

    flow = get_passenger_flow(r, "Monday", 8, "Rainy",
                              False, False)
    if flow:
        print(f"\nPassenger Flow ({len(flow['flows'])} segments):")
        for f in flow["flows"][:4]:
            print(f"  {f['from_stop']:20} → "
                  f"{f['to_stop']:20}: "
                  f"{f['passengers']:3} pax ({f['percentage']}%)")
        print(f"\n  Most crowded : {flow['most_crowded']['from_stop']}"
              f" → {flow['most_crowded']['to_stop']}"
              f" ({flow['most_crowded']['passengers']} pax)")
        print(f"  Least crowded: {flow['least_crowded']['from_stop']}"
              f" → {flow['least_crowded']['to_stop']}"
              f" ({flow['least_crowded']['passengers']} pax)")
    print("\npredictor.py ready ✅")
def get_smart_boarding_tip(passenger_flow):
    """
    Analyse passenger flow and return a smart boarding tip
    telling the user which stop to board at for
    the most comfortable journey.
    """
    if not passenger_flow or not passenger_flow["flows"]:
        return None

    direct_flows  = passenger_flow["direct_only"]
    least_crowded = passenger_flow["least_crowded"]
    most_crowded  = passenger_flow["most_crowded"]

    # Find the best direct segment
    if direct_flows:
        best_direct = min(direct_flows,
                          key=lambda x: x["passengers"])
        tip = {
            "board_at":        best_direct["from_stop"],
            "alight_at":       best_direct["to_stop"],
            "passengers":      best_direct["passengers"],
            "percentage":      best_direct["percentage"],
            "saving":          most_crowded["passengers"]
                               - best_direct["passengers"],
            "rush":            best_direct["rush"],
        }
    else:
        tip = {
            "board_at":        least_crowded["from_stop"],
            "alight_at":       least_crowded["to_stop"],
            "passengers":      least_crowded["passengers"],
            "percentage":      least_crowded["percentage"],
            "saving":          most_crowded["passengers"]
                               - least_crowded["passengers"],
            "rush":            least_crowded["rush"],
        }
    return tip    