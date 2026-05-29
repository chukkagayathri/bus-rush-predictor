class BusRoute:
    # Class variables
    operator = "APSRTC"
    total_routes = 0
    all_routes = []   # list of all route names

    def __init__(self, route_id, start, end, capacity):
        # Instance variables
        self.route_id = route_id
        self.start = start
        self.end = end
        self.capacity = capacity
        self.current_passengers = 0

        # Update class variables whenever new route is created
        BusRoute.total_routes += 1
        BusRoute.all_routes.append(route_id)

    def get_rush_level(self):
        percentage = (self.current_passengers / self.capacity) * 100
        if percentage < 40:
            return "Low"
        elif percentage < 75:
            return "Medium"
        else:
            return "High"

    def update_passengers(self, count):
        self.current_passengers = count

    def show_info(self):
        print(f"Operator  : {BusRoute.operator}")
        print(f"Route ID  : {self.route_id}")
        print(f"From      : {self.start}")
        print(f"To        : {self.end}")
        print(f"Capacity  : {self.capacity}")
        print(f"Passengers: {self.current_passengers}")
        print(f"Rush Level: {self.get_rush_level()}")
        print("-" * 30)


# Create routes
r1 = BusRoute("R05", "tnl", "gnt", 50)
r2 = BusRoute("R10", "gnt", "nrt", 45)
r3 = BusRoute("R07", "nrt","pdg", 55)

# Update passengers
r1.update_passengers(47)   # nearly full
r2.update_passengers(18)   # mostly empty
r3.update_passengers(35)   # medium

# Show info
r1.show_info()
r2.show_info()
r3.show_info()

# Class variable tracking
print(f"Total routes created : {BusRoute.total_routes}")  # 3
print(f"All route IDs        : {BusRoute.all_routes}")
# ['R05', 'R10', 'R07']

# All share the same operator
print(r1.operator)   
print(r2.operator)  