class BusRoute:
    # Class variable — same for all routes
    operator = "APRTC "
    
    def __init__(self, route_id, start, end, capacity):
        # Instance variables — unique to each route
        self.route_id = route_id
        self.start = start
        self.end = end
        self.capacity = capacity
        self.current_passengers = 0
        self.rush_history = []   # list of past rush levels
    
    def update_passengers(self, count):
        self.current_passengers = count
    
    def get_rush_level(self):
        percentage = (self.current_passengers / self.capacity) * 100
        if percentage < 40:
            return "Low"
        elif percentage < 75:
            return "Medium"
        else:
            return "High"
    
    def record_rush(self):
        level = self.get_rush_level()
        self.rush_history.append(level)
        print(f"Route {self.route_id}: {level} rush recorded")
    
    def show_status(self):
        print(f"--- Route {self.route_id} ---")
        print(f"From: {self.start} → To: {self.end}")
        print(f"Passengers: {self.current_passengers}/{self.capacity}")
        print(f"Rush Level: {self.get_rush_level()}")


# Create actual route objects
route5  = BusRoute("R05", "tnl", "gnt", 50)
route10 = BusRoute("R10", "gnt", "nrt", 45)

# Update passengers for morning peak
route5.update_passengers(47)
route10.update_passengers(21)

# Check rush
route5.show_status()
# --- Route R05 ---
# Passengers: 47/50
# Rush Level: High

route10.show_status()
# --- Route R10 ---

# Passengers: 21/45
# Rush Level: Low

# Record in history
route5.record_rush()   # Route R05: High rush recorded
route10.record_rush()  # Route R10: Low rush recorded

# All routes share same operator
print(route5.operator)   
print(route10.operator)  