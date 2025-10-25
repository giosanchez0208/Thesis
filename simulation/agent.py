import random

class Passenger:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.state = "WALK"

class Jeepney:
    def __init__(self, route_nodes):
        self.route_nodes = route_nodes
        self.position_index = 0

    def move(self):
        self.position_index = (self.position_index + 1) % len(self.route_nodes)
        return self.route_nodes[self.position_index]
