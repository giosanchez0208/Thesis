"""Travel graph construction and management module."""

from .travel_graph_visualizer import TravelGraphVisualizer, visualize_travel_graph
from .jeepney_system import JeepneySystem, JeepneyRouteScrubber

from .travel_graph import (
    TravelGraphManager,
    JeepneyRoute,
    # Helper functions for graph construction
    make_coord_key,
    resolve_study_area_boundary,
    load_graphs_for_study_area,
    prune_dead_end_nodes,
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

__all__ = [
    # Main classes
    "TravelGraphManager",
    "JeepneyRoute",
    "JeepneySystem",
    "JeepneyRouteScrubber",
    # Graph construction helpers
    "make_coord_key",
    "resolve_study_area_boundary",
    "load_graphs_for_study_area",
    "prune_dead_end_nodes",
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
    # Visualizer
    "TravelGraphVisualizer",
    "visualize_travel_graph",
]
