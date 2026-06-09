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
# ── LIVE STOP RUSH CHECKER ─────────────────────────────────
# Simulates real-time passenger tracking between stops
# In production: replace simulate_current_bus_data()
# with actual APSRTC ETM API call

import datetime

def simulate_current_bus_data(route, hour, day,
                               weather, is_holiday):
    """
    Simulates what an APSRTC ETM API would return.
    Returns live-like passenger count at each stop
    as if the bus is currently mid-journey.

    In production replace this entire function with:
        response = requests.get(APSRTC_API_URL, params={...})
        return response.json()
    """
    stops    = ROUTES[route]["stops"]
    capacity = ROUTES[route]["capacity"]

    # Use ML model to get overall predicted load
    result     = predict_rush(route, day, hour, weather,
                               is_holiday,
                               day in ["Saturday","Sunday"])
    rush_level = result["rush_level"]

    load_map   = {"Low": 0.28, "Medium": 0.55, "High": 0.88}
    base_load  = capacity * load_map.get(rush_level, 0.5)

    if is_peak(hour):       base_load *= 1.20
    if weather == "Rainy":  base_load *= 1.18

    # Simulate how many passengers are at each stop
    # as the bus travels through the route
    random.seed(hash(f"{route}{day}{hour}{weather}late") % (2**31))

    stop_data = []
    running_passengers = 0

    for i, stop in enumerate(stops):
        if i == 0:
            # Origin stop — most passengers board here
            boarded  = int(base_load * 0.55
                           * random.uniform(0.88, 1.12))
            boarded  = min(boarded, capacity)
            alighted = 0
        else:
            # Subsequent stops — some board, some alight
            board_rate  = max(0.08 - (i * 0.01), 0.03)
            alight_rate = max(0.12 + (i * 0.02), 0.05)

            boarded  = int(capacity * board_rate
                           * random.uniform(0.80, 1.20))
            alighted = int(running_passengers * alight_rate
                           * random.uniform(0.80, 1.20))
            boarded  = min(boarded,
                           capacity - running_passengers)

        running_passengers = (running_passengers
                              + boarded - alighted)
        running_passengers = max(0, min(running_passengers,
                                        capacity))
        pct = round(running_passengers / capacity * 100, 1)

        stop_data.append({
            "stop_index":    i,
            "stop_name":     stop,
            "passengers_boarded_here":  boarded,
            "passengers_alighted_here": alighted,
            "current_on_bus":  running_passengers,
            "capacity":        capacity,
            "percentage":      pct,
            "rush":  "High"   if pct >= 75
                     else ("Medium" if pct >= 40
                     else "Low"),
            "color": "#E74C3C" if pct >= 75
                     else ("#F39C12" if pct >= 40
                     else "#27AE60"),
        })

    return stop_data


def get_avg_time_per_stop(route):
    """Average travel time in minutes between consecutive stops."""
    time_map = {
        "Guntur-Amaravati":    12,
        "Guntur-Brodipet":      8,
        "Guntur-Nallapadu":    10,
        "Guntur-Mangalagiri":  11,
        "Guntur-Tenali":       14,
        "Guntur-Ponnur":       15,
        "Guntur-Sattenapalli": 16,
        "Guntur-Narasaraopet": 18,
        "Tenali-Repalle":      14,
        "Tenali-Bapatla":      13,
        "Tenali-Chirala":      15,
        "Tenali-Guntur":       14,
        "Tenali-Vijayawada":   15,
        "Tenali-Ongole":       20,
    }
    return time_map.get(route, 12)


def get_realtime_stop_rush(route, day, hour,
                            weather, is_holiday,
                            user_stop_name,
                            bus_current_stop_name):
    """
    MAIN FUNCTION for the real-time feature.

    A person is standing at user_stop_name.
    The bus is currently at bus_current_stop_name.

    Returns:
    - How many passengers are on the bus RIGHT NOW
    - How crowded it is at each previous stop
    - How many minutes until it reaches user's stop
    - Predicted rush when it arrives at user's stop
    """
    stops    = ROUTES[route]["stops"]
    capacity = ROUTES[route]["capacity"]

    # Validate stops exist on this route
    if user_stop_name not in stops:
        return {"error": f"'{user_stop_name}' not on {route}"}
    if bus_current_stop_name not in stops:
        return {"error": f"'{bus_current_stop_name}' not on {route}"}

    user_stop_idx = stops.index(user_stop_name)
    bus_stop_idx  = stops.index(bus_current_stop_name)

    # Bus must be BEFORE user's stop
    if bus_stop_idx >= user_stop_idx:
        return {
            "error": "Bus has already passed your stop or "
                     "is at the same stop."
        }

    # Get simulated live data for all stops
    all_stop_data = simulate_current_bus_data(
        route, hour, day, weather, is_holiday
    )

    # Current bus status (at its current stop)
    current_bus = all_stop_data[bus_stop_idx]

    # Stops the bus has already visited
    visited_stops = all_stop_data[:bus_stop_idx + 1]

    # Stops between current bus position and user's stop
    stops_to_travel = user_stop_idx - bus_stop_idx

    # Time until bus reaches user's stop
    mins_per_stop  = get_avg_time_per_stop(route)
    minutes_away   = stops_to_travel * mins_per_stop

    # Predict what load will be when it reaches user's stop
    # Account for boarding and alighting between now and then
    current_passengers = current_bus["current_on_bus"]

    for i in range(bus_stop_idx + 1, user_stop_idx + 1):
        stop_sim     = all_stop_data[i]
        current_passengers = stop_sim["current_on_bus"]

    predicted_at_user_stop    = current_passengers
    predicted_pct             = round(
        predicted_at_user_stop / capacity * 100, 1
    )

    predicted_rush = ("High"   if predicted_pct >= 75
                      else ("Medium" if predicted_pct >= 40
                      else "Low"))
    predicted_color = ("#E74C3C" if predicted_pct >= 75
                       else ("#F39C12" if predicted_pct >= 40
                       else "#27AE60"))

    # Arrival time estimate
    now           = datetime.datetime.now()
    arrival_time  = now + datetime.timedelta(minutes=minutes_away)
    arrival_str   = arrival_time.strftime("%I:%M %p")

    return {
        # Current bus status
        "bus_current_stop":      bus_current_stop_name,
        "bus_current_stop_idx":  bus_stop_idx,
        "current_passengers":    current_bus["current_on_bus"],
        "current_capacity":      capacity,
        "current_percentage":    current_bus["percentage"],
        "current_rush":          current_bus["rush"],
        "current_color":         current_bus["color"],

        # User's stop info
        "user_stop":             user_stop_name,
        "user_stop_idx":         user_stop_idx,

        # Journey info
        "stops_away":            stops_to_travel,
        "minutes_away":          minutes_away,
        "estimated_arrival":     arrival_str,

        # Prediction at user's stop
        "predicted_passengers":  predicted_at_user_stop,
        "predicted_percentage":  predicted_pct,
        "predicted_rush":        predicted_rush,
        "predicted_color":       predicted_color,

        # All stop history for display
        "visited_stops":         visited_stops,
        "all_stops":             all_stop_data,
        "total_stops":           len(stops),
        "route_stops":           stops,
        "capacity":              capacity,
    } 