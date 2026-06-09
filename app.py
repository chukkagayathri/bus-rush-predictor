# app.py
# Flask web application — APSRTC Bus Rush Predictor
# With passenger count between stops feature

from flask import Flask, render_template, request
import predictor as p

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html",
        routes=p.AVAILABLE_ROUTES,
        days=p.AVAILABLE_DAYS,
        weather_options=p.AVAILABLE_WEATHER,
        hours=p.OPERATING_HOURS,
        format_hour=p.format_hour
    )

@app.route("/predict", methods=["POST"])
def predict():
    route      = request.form.get("route")
    day        = request.form.get("day")
    hour       = int(request.form.get("hour"))
    weather    = request.form.get("weather")
    is_holiday = request.form.get("is_holiday") == "on"
    is_weekend = day in ["Saturday","Sunday"]
    capacity   = p.get_capacity(route)

    result          = p.predict_rush(route, day, hour, weather,
                                     is_holiday, is_weekend)
    best_time       = p.find_best_time(route, day, weather,
                                       is_holiday, is_weekend)
    passenger_flow  = p.get_passenger_flow(route, day, hour,
                                           weather, is_holiday,
                                           is_weekend)
    hourly_forecast = p.get_hourly_forecast(route, day, weather,
                                            is_holiday, is_weekend)
    route_stops     = p.get_route_stops(route)
    smart_tip = p.get_smart_boarding_tip(passenger_flow)

    return render_template(
    "result.html",
    route=route,
    day=day,
    hour=hour,
    weather=weather,
    capacity=capacity,
    is_holiday=is_holiday,
    result=result,
    best_time=best_time,
    passenger_flow=passenger_flow,
    hourly_forecast=hourly_forecast,
    route_stops=route_stops,
    smart_tip=smart_tip,
    format_hour=p.format_hour
)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/routes")
def routes_page():
    """Page showing all routes and their stops."""
    return render_template(
        "routes.html",
        routes=p.ROUTES,
        format_hour=p.format_hour
    )

if __name__ == "__main__":
    print("Starting APSRTC Bus Rush Predictor...")
    print("Open: http://localhost:5000")
    app.run(debug=True)