class BusRoute:
    company = "APSRTC"
    peak_hours = [(7, 10), (17, 20)]  # morning and evening peak
    all_routes = []

    def __init__(self, route_id, start, end, capacity):
        self.route_id = route_id
        self.start = start
        self.end = end
        self.capacity = capacity
        self.passengers = 0
        BusRoute.all_routes.append(route_id)

    # ---- REGULAR METHODS ----
    def board_passengers(self, count):
        self.passengers = min(count, self.capacity)

    def get_rush_level(self):
        pct = BusRoute.calculate_percentage(
            self.passengers, self.capacity
        )
        return BusRoute.label_from_percentage(pct)

    def show_info(self):
        print(f"Company  : {BusRoute.company}")
        print(f"Route    : {self.route_id}")
        print(f"From     : {self.start} → {self.end}")
        print(f"Passengers: {self.passengers}/{self.capacity}")
        print(f"Rush     : {self.get_rush_level()}")

    # ---- CLASS METHODS ----
    @classmethod
    def from_string(cls, route_string):
        # Format: "R05|Tenali|Guntur|50"
        parts = route_string.split("|")
        return cls(parts[0], parts[1], parts[2], int(parts[3]))

    @classmethod
    def from_dict(cls, d):
        return cls(d["route_id"], d["start"], d["end"], d["capacity"])

    @classmethod
    def get_all_routes(cls):
        return cls.all_routes

    @classmethod
    def get_company(cls):
        return cls.company

    # ---- STATIC METHODS ----
    @staticmethod
    def calculate_percentage(passengers, capacity):
        return round((passengers / capacity) * 100, 1)

    @staticmethod
    def label_from_percentage(pct):
        if pct < 40:
            return "Low"
        elif pct < 75:
            return "Medium"
        else:
            return "High"

    @staticmethod
    def is_peak_hour(hour):
        return (7 <= hour <= 10) or (17 <= hour <= 20)

    @staticmethod
    def is_valid_capacity(cap):
        return isinstance(cap, int) and 1 <= cap <= 200


# ---- Test everything ----

# Regular constructor
r1 = BusRoute("TNL05", "Tenali", "Guntur", 50)

# Alternative constructors (classmethods)
r2 = BusRoute.from_string("GNT10|Guntur|Tenali|45")

r3 = BusRoute.from_dict({
    "route_id": "TNL07",
    "start": "Tenali",
    "end": "Guntur",
    "capacity": 55
})

# Board passengers
r1.board_passengers(47)
r2.board_passengers(18)
r3.board_passengers(35)

# Show info
r1.show_info()
print()

r2.show_info()
print()

r3.show_info()

# Classmethod calls
print(f"\nAll routes: {BusRoute.get_all_routes()}")
print(f"Company: {BusRoute.get_company()}")

# Staticmethod calls — no object needed
print(f"\nPeak hour check:")
print(f"8 AM: {BusRoute.is_peak_hour(8)}")
print(f"2 PM: {BusRoute.is_peak_hour(14)}")
print(f"6 PM: {BusRoute.is_peak_hour(18)}")

print(f"\nCapacity valid:")
print(BusRoute.is_valid_capacity(50))
print(BusRoute.is_valid_capacity(500))

print(f"\nDirect percentage:")
print(BusRoute.calculate_percentage(47, 50))
print(BusRoute.label_from_percentage(94.0))