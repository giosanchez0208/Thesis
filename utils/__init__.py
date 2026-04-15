"""
Utilities package for thesis project.

Provides:
- Travel graph analysis and routing via TravelGraphManager
- JeepneyRoute for defining jeepney services
- Passenger generation, jeep movement, and tandem simulation tools
"""

from .travel_graph import (
    TravelGraphManager,
    JeepneyRoute,
    # Helper functions for graph construction
    make_coord_key,
    resolve_study_area_boundary,
    load_graphs_for_study_area,
    node_table_from_graph,
    extract_uncategorized_nodes,
    graph_edges_to_bidirectional_base,
    duplicate_walk_nodes_to_layers,
    build_direct_edges,
    build_interlayer_edges,
    assign_edge_ids,
    edge_frame_to_csv,
    node_frame_to_csv,
    make_edges_gdf,
    make_nodes_gdf,
    add_nodes_to_digraph,
    add_edges_to_digraph,
    compute_v_to_outgoing,
    attach_accessible_nodes,
)

from .passenger_generation import (
    PassengerMap,
    Passenger,
    PassengerState,
    Jeep,
    JeepState,
    SimulationConfig,
    Simulation,
)

__all__ = [
    # Main classes
    "TravelGraphManager",
    "JeepneyRoute",
    # Graph construction helpers
    "make_coord_key",
    "resolve_study_area_boundary",
    "load_graphs_for_study_area",
    "node_table_from_graph",
    "extract_uncategorized_nodes",
    "graph_edges_to_bidirectional_base",
    "duplicate_walk_nodes_to_layers",
    "build_direct_edges",
    "build_interlayer_edges",
    "assign_edge_ids",
    "edge_frame_to_csv",
    "node_frame_to_csv",
    "make_edges_gdf",
    "make_nodes_gdf",
    "add_nodes_to_digraph",
    "add_edges_to_digraph",
    "compute_v_to_outgoing",
    "attach_accessible_nodes",
    "PassengerMap",
    "Passenger",
    "PassengerState",
    "Jeep",
    "JeepState",
    "SimulationConfig",
    "Simulation",
]
