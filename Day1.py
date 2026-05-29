# Day 1 - Bus Rush Predictor Project
# Created by: Gayatri Chukka
# CSE AI/ML - B.Tech 2nd Year

print("Hello! Bus Rush Predictor project started.")
print("Today: Day 1 - Setup complete")

# Testing Python basics
name = "Gayatri"
project = "Bus Rush Predictor"
print(f"Developer: {name}")
print(f"Project: {project}")

# Simple class practice from today's video
class Bus:
    def __init__(self, route, capacity):
        self.route = route
        self.capacity = capacity
    
    def info(self):
        print(f"Bus Route: {self.route}, Capacity: {self.capacity}")

# Create bus objects
bus1 = Bus("Route 5 - Hyderabad", 50)
bus2 = Bus("Route 10 - Secunderabad", 40)

bus1.info()
bus2.info()