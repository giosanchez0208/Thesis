"""
Jeep simulation class for the passenger simulation system.

Simulates a jeepney moving along a predefined route with constant velocity.
"""

from enum import Enum
from typing import Optional, List, Tuple
import math


class JeepState(Enum):
    """Finite State Machine states for jeepney journey."""
    IDLE = "idle"                    # Not started
    MOVING = "moving"                # In motion along route
    AT_STATION = "at_station"        # Stopped at a station
    COMPLETED = "completed"          # Completed route loop


class Jeep:
    """
    Represents a jeepney moving along a fixed route.
    
    Parameters
    ----------
    jeep_id : str
        Unique identifier for the jeep
    route_nodes : list[tuple[float, float]]
        List of (latitude, longitude) coordinates defining the route
    v_jeep : float
        Velocity in meters per second
    """
    
    def __init__(
        self,
        jeep_id: str,
        route_nodes: List[Tuple[float, float]],
        v_jeep: float = 10.0
    ):
        """Initialize jeepney with route and velocity."""
        if len(route_nodes) < 2:
            raise ValueError("Route must have at least 2 nodes")
        
        self.jeep_id = str(jeep_id)
        self.route_nodes = list(route_nodes)  # [(lat, lon), ...]
        self.v_jeep = float(v_jeep)  # meters per second
        
        # Current position (starts at first node)
        self.curr_lat = float(route_nodes[0][0])
        self.curr_lon = float(route_nodes[0][1])
        
        # Journey tracking
        self.state = JeepState.IDLE
        self.total_time = 0.0  # seconds
        self.total_distance = 0.0  # meters
        self.current_segment_idx = 0  # index of current segment
        self.distance_along_segment = 0.0  # meters from start of current segment
        
    def get_curr_lat_lon(self) -> Tuple[float, float]:
        """Get current latitude and longitude."""
        return float(self.curr_lat), float(self.curr_lon)
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Returns
        -------
        float
            Distance in meters
        """
        R_earth = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R_earth * c
    
    @staticmethod
    def _linear_interpolation(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        fraction: float
    ) -> Tuple[float, float]:
        """
        Linearly interpolate between two points.
        
        Parameters
        ----------
        lat1, lon1 : float
            Starting point
        lat2, lon2 : float
            Ending point
        fraction : float
            Fraction of distance traveled (0 to 1)
        
        Returns
        -------
        tuple
            Interpolated (lat, lon)
        """
        new_lat = lat1 + (lat2 - lat1) * fraction
        new_lon = lon1 + (lon2 - lon1) * fraction
        return new_lat, new_lon
    
    def update(self, dt: float) -> None:
        """
        Update jeep position based on time step.
        
        Parameters
        ----------
        dt : float
            Time increment in seconds
        """
        if self.state == JeepState.IDLE:
            # Start moving on first update
            self.state = JeepState.MOVING
        
        if self.state == JeepState.COMPLETED:
            return  # No more movement
        
        # Distance traveled in this time step
        distance_in_dt = self.v_jeep * dt
        self.distance_along_segment += distance_in_dt
        self.total_distance += distance_in_dt
        self.total_time += dt
        
        # Move along route
        while self.current_segment_idx < len(self.route_nodes) - 1:
            # Current segment: route_nodes[idx] -> route_nodes[idx+1]
            lat1, lon1 = self.route_nodes[self.current_segment_idx]
            lat2, lon2 = self.route_nodes[self.current_segment_idx + 1]
            
            # Distance for the complete segment
            segment_dist = self._haversine_distance(lat1, lon1, lat2, lon2)
            
            if self.distance_along_segment >= segment_dist:
                # Completed this segment, move to next
                self.distance_along_segment -= segment_dist
                self.current_segment_idx += 1
                
                if self.current_segment_idx == len(self.route_nodes) - 1:
                    # Reached last node
                    self.curr_lat = float(self.route_nodes[-1][0])
                    self.curr_lon = float(self.route_nodes[-1][1])
                    self.state = JeepState.COMPLETED
                    return
            else:
                # Still on this segment
                fraction = self.distance_along_segment / segment_dist
                self.curr_lat, self.curr_lon = self._linear_interpolation(
                    lat1, lon1, lat2, lon2, fraction
                )
                return
    
    def restart_route(self) -> None:
        """Restart the route from the beginning."""
        self.curr_lat = float(self.route_nodes[0][0])
        self.curr_lon = float(self.route_nodes[0][1])
        self.state = JeepState.IDLE
        self.total_time = 0.0
        self.total_distance = 0.0
        self.current_segment_idx = 0
        self.distance_along_segment = 0.0
    
    def __repr__(self) -> str:
        return (
            f"Jeep({self.jeep_id}, state={self.state.value}, "
            f"distance={self.total_distance:.1f}m, time={self.total_time:.1f}s)"
        )
