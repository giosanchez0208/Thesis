from .passenger_map import PassengerMap
from enum import Enum

class PassengerState(Enum):
    """Finite State Machine states for passengerpassenger journey."""
    WAITING_TO_WALK = "waiting_to_walk"      # At start position, before walking to boarding point
    WALKING_TO_BOARD = "walking_to_board"    # Walking from start to boarding point
    WAITING_AT_STATION = "waiting_at_station"  # At boarding point, waiting for jeep
    RIDING = "riding"                        # On the jeep
    ALIGHTING = "alighting"                  # Getting off the jeep
    WALKING_FROM_ALIGHT = "walking_from_alight"  # Walking from drop-off to destination
    AT_DESTINATION = "at_destination"        # Reached destination
    COMPLETED = "completed"                  # Journey complete

class Passenger:
    """
    Represents a passenger with a start/end position and journey state machine.
    
    The passenger follows a path calculated by the TravelGraphManager and tracks
    their position as they move through the travel graph.
    """
    def __init__(self, start_node_id: str = None, end_node_id: str = None):
        """
        Parameters
        ----------
        start_node_id : str, optional
            Starting node ID. If None, randomly sampled.
        end_node_id : str, optional
            Destination node ID. If None, randomly sampled (different from start).
        """
        try:
            self.passenger_map = PassengerMap()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Passenger map data not found. Expected file at: {e}\n"
                "Make sure the data files are in the correct location."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PassengerMap: {e}") from e
        
        # Sample or use provided start/end positions
        if start_node_id is None:
            start_sample = self.passenger_map.generate_nodes(n_points=1)
            self.start_node_id = start_sample['base_osmid'].iloc[0]
            self.start_lat = start_sample['lat'].iloc[0]
            self.start_lon = start_sample['lon'].iloc[0]
        else:
            self.start_node_id = start_node_id
            # Lookup lat/lon from passenger_map
            row = self.passenger_map.df[
                self.passenger_map.df['base_osmid'] == start_node_id
            ]
            if len(row) > 0:
                self.start_lat = row['lat'].iloc[0]
                self.start_lon = row['lon'].iloc[0]
            else:
                raise ValueError(f"Start node {start_node_id} not found in passenger map")
        
        # Sample end_pos, ensure it's different from start_pos
        if end_node_id is None:
            while True:
                end_sample = self.passenger_map.generate_nodes(n_points=1)
                self.end_node_id = end_sample['base_osmid'].iloc[0]
                if self.start_node_id != self.end_node_id:
                    break
            self.end_lat = end_sample['lat'].iloc[0]
            self.end_lon = end_sample['lon'].iloc[0]
        else:
            self.end_node_id = end_node_id
            row = self.passenger_map.df[
                self.passenger_map.df['base_osmid'] == end_node_id
            ]
            if len(row) > 0:
                self.end_lat = row['lat'].iloc[0]
                self.end_lon = row['lon'].iloc[0]
            else:
                raise ValueError(f"End node {end_node_id} not found in passenger map")
        
        # Current position as node ID (updated during journey)
        self.curr_node_id = self.start_node_id
        self.curr_lat = self.start_lat
        self.curr_lon = self.start_lon
        
        # Journey tracking
        self.total_time = 0.0  # Total elapsed time in seconds
        self.state = PassengerState.WAITING_TO_WALK
        self.journey_path = []  # List of node IDs traversed
        
    def get_start_lat_lon(self):
        """Get starting latitude and longitude."""
        return float(self.start_lat), float(self.start_lon)
    
    def get_end_lat_lon(self):
        """Get destination latitude and longitude."""
        return float(self.end_lat), float(self.end_lon)
    
    def get_curr_lat_lon(self):
        """Get current latitude and longitude."""
        return float(self.curr_lat), float(self.curr_lon)
    
    def update(self, new_node_id: str, new_lat: float, new_lon: float, 
               state: PassengerState = None, dt: float = 0.0) -> None:
        """
        Update passenger position and state during simulation.
        
        Parameters
        ----------
        new_node_id : str
            New current node ID
        new_lat : float
            New latitude
        new_lon : float
            New longitude
        state : PassengerState, optional
            New state (if None, state unchanged)
        dt : float
            Time increment in seconds
        """
        self.curr_node_id = new_node_id
        self.curr_lat = new_lat
        self.curr_lon = new_lon
        self.total_time += dt
        
        if state is not None:
            self.state = state
        
        if new_node_id not in self.journey_path:
            self.journey_path.append(new_node_id)
    
    def __repr__(self) -> str:
        return (
            f"Passenger(start={self.start_node_id}, end={self.end_node_id}, "
            f"curr={self.curr_node_id}, state={self.state.value}, "
            f"time={self.total_time:.1f}s)"
        )

if __name__ == "__main__":
    passenger = Passenger()
    print(f"Passenger start position: {passenger.start_node_id}")
    print(f"Passenger end position: {passenger.end_node_id}")
    print(f"Passenger current position: {passenger.curr_node_id}")
    print(f"Current state: {passenger.state.value}")
    print(f"Total time: {passenger.total_time}")
    
    print("="*50)
    print(f"Start lat/lon: {passenger.get_start_lat_lon()}")
    print(f"End lat/lon: {passenger.get_end_lat_lon()}")
    print(f"Current lat/lon: {passenger.get_curr_lat_lon()}")
    
    print("="*50)
    # Test update method
    passenger.update(
        new_node_id="test_node",
        new_lat=8.0,
        new_lon=124.0,
        state=PassengerState.WALKING_TO_BOARD,
        dt=5.0
    )
    print(f"After update: {passenger}")