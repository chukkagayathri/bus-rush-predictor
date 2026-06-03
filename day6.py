# Day 6 — Property Decorators Practice
# Bus Rush Predictor Project

class Bus:
    def __init__(self, route_id, capacity, passengers=0):
        self.route_id = route_id
        self.capacity = capacity
        self.passengers = passengers

    @property
    def route_id(self):
        return self._route_id

    @route_id.setter
    def route_id(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Route ID must be non-empty string")
        self._route_id = value.strip().upper()

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        if not isinstance(value, int) or not 1 <= value <= 200:
            raise ValueError("Capacity must be integer between 1-200")
        self._capacity = value

    @property
    def passengers(self):
        return self._passengers

    @passengers.setter
    def passengers(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Passengers must be non-negative integer")
        if value > self._capacity:
            raise ValueError(f"Cannot exceed capacity {self._capacity}")
        self._passengers = value

    @property
    def rush_percentage(self):
        return round(self._passengers / self._capacity * 100, 1)

    @property
    def rush_level(self):
        pct = self.rush_percentage
        if pct < 40: return "Low"
        elif pct < 75: return "Medium"
        else: return "High"

    @property
    def available_seats(self):
        return self._capacity - self._passengers

    @property
    def is_full(self):
        return self._passengers >= self._capacity

    def __str__(self):
        return (f"Bus {self.route_id} | "
                f"{self.passengers}/{self.capacity} | "
                f"{self.rush_level} ({self.rush_percentage}%)")


# Test
b = Bus("r05", 50, 47)
print(b)
print(f"Available seats: {b.available_seats}")
print(f"Is full: {b.is_full}")

# Test validation
try:
    b.passengers = 999
except ValueError as e:
    print(f"Caught: {e}")

try:
    b.capacity = -10
except ValueError as e:
    print(f"Caught: {e}")