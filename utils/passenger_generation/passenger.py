from .passenger_map import PassengerMap
from enum import Enum
from typing import List, Optional

class PassengerState(Enum):
    """Finite State Machine states for passenger journey."""
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
    def __init__(
        self,
        start_node_id: str = None,
        end_node_id: str = None,
        passenger_map: PassengerMap = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        start_node_id : str, optional
            Starting node ID. If None, randomly sampled.
        end_node_id : str, optional
            Destination node ID. If None, randomly sampled (different from start).
        """
        self.verbose = bool(verbose)
        if passenger_map is None:
            try:
                self.passenger_map = PassengerMap()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Passenger map data not found. Expected file at: {e}\n"
                    "Make sure the data files are in the correct location."
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to initialize PassengerMap: {e}") from e
        else:
            self.passenger_map = passenger_map
        
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
        
        # Pathfinding and routing
        self.travel_graph_mgr = None  # Will be set by simulation
        self.shortest_path_edges = []  # List of edge IDs from start to end
        self.shortest_path_nodes = []  # List of node IDs from start to end
        self.current_path_index = 0  # Current position in shortest path
        self.current_edge_progress_m = 0.0
        self.boarded_jeep_id = None
        self.boarded_route_id = None
        
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
 
        self.curr_node_id = new_node_id
        self.curr_lat = new_lat
        self.curr_lon = new_lon
        self.total_time += dt
        
        if state is not None:
            self.state = state
        
        if new_node_id not in self.journey_path:
            self.journey_path.append(new_node_id)
    
    def set_travel_graph(self, travel_graph_mgr) -> None:
        """
        Set the travel graph manager for pathfinding.
        
        Parameters
        ----------
        travel_graph_mgr : TravelGraphManager
            The manager instance with routing capabilities
        """
        self.travel_graph_mgr = travel_graph_mgr
    
    def calculate_shortest_path(self) -> bool:
        """
        Calculate shortest path from start to end using travel graph.
        Uses the travel graph's Dijkstra shortest path algorithm.
        Maps passenger coordinates to nearest travel graph nodes.
        
        Returns
        -------
        bool
            True if path found successfully, False otherwise
        """
        if self.travel_graph_mgr is None:
            if self.verbose:
                print("Warning: travel graph not set for pathfinding")
            return False
        
        try:
            # Find nearest travel graph nodes to passenger start/end coordinates
            # The travel graph nodes are geo-located and we match by proximity
            start_graph_node_id = self.travel_graph_mgr.find_nearest_node(
                self.start_lat, self.start_lon, layer="start_walk"
            )
            end_graph_node_id = self.travel_graph_mgr.find_nearest_node(
                self.end_lat, self.end_lon, layer="end_walk"
            )
            
            if not start_graph_node_id or not end_graph_node_id:
                if self.verbose:
                    print("Warning: could not find start or end nodes in travel graph")
                return False
            
            # Get shortest path edges using travel graph node IDs
            self.shortest_path_edges = self.travel_graph_mgr.calculate_shortest_path(
                start_graph_node_id, 
                end_graph_node_id
            )
            
            # Reconstruct nodes from edges
            self.shortest_path_nodes = [start_graph_node_id]
            for edge_id in self.shortest_path_edges:
                edge = self.travel_graph_mgr.get_edge(edge_id)
                if edge:
                    self.shortest_path_nodes.append(edge.v)
            
            self.current_path_index = 0
            if self.verbose:
                print(
                    f"Shortest path calculated: {len(self.shortest_path_edges)} edges, "
                    f"{len(self.shortest_path_nodes)} nodes"
                )
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating shortest path: {e}")
            return False
    
    def get_next_path_node(self):
        if not self.shortest_path_nodes or self.current_path_index >= len(self.shortest_path_nodes):
            return None, None, None
        
        node_id = self.shortest_path_nodes[self.current_path_index]
        self.current_path_index += 1
        
        # Look up coordinates
        node_row = self.passenger_map.df[
            self.passenger_map.df['base_osmid'] == int(node_id.split('_')[-1]) if '_' in node_id else False
        ]
        
        if len(node_row) > 0:
            lat = node_row['lat'].iloc[0]
            lon = node_row['lon'].iloc[0]
            return node_id, lat, lon
        
        return node_id, None, None
    
    def is_path_complete(self) -> bool:
        """Check if the passenger has traversed the entire shortest path."""
        return self.current_path_index >= len(self.shortest_path_nodes)
    
    def reset_path_index(self) -> None:
        """Reset path traversal index (e.g., for restart)."""
        self.current_path_index = 0
        self.current_edge_progress_m = 0.0
        self.boarded_jeep_id = None
        self.boarded_route_id = None

    def current_path_edge_id(self):
        if 0 <= self.current_path_index < len(self.shortest_path_edges):
            return self.shortest_path_edges[self.current_path_index]
        return None
    
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
