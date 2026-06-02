# Day 5 — Dunder Methods Practice
# Bus Rush Predictor Project

class Bus:
    company = "TSRTC"

    def __init__(self, route_id, capacity, passengers=0):
        self.route_id = route_id
        self.capacity = capacity
        self.passengers = passengers

    def __repr__(self):
        return (f"Bus('{self.route_id}', "
                f"{self.capacity}, {self.passengers})")

    def __str__(self):
        pct = round(self.passengers/self.capacity*100, 1)
        return (f"[{Bus.company}] Route {self.route_id} | "
                f"{self.passengers}/{self.capacity} | {pct}%")

    def __len__(self):
        return self.capacity - self.passengers  # free seats

    def __eq__(self, other):
        return self.passengers == other.passengers

    def __lt__(self, other):
        return self.passengers < other.passengers

    def __gt__(self, other):
        return self.passengers > other.passengers

    def __bool__(self):
        return self.passengers > 0


# Test all dunder methods
b1 = Bus("R05", 50, 47)
b2 = Bus("R10", 40, 18)
b3 = Bus("M01", 20, 0)

print(b1)             # __str__
print(repr(b2))       # __repr__
print(len(b1))        # free seats = 3
print(b1 > b2)        # True
print(b1 == Bus("R07", 55, 47))  # True

buses = [b1, b2, Bus("R07", 55, 35)]
buses.sort()           # uses __lt__
for b in buses:
    print(b)

if b3:
    print("Active")
else:
    print(f"Route {b3.route_id} is empty")