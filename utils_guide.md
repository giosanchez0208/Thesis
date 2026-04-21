# Utils package guide

## Overview

The `utils` folder is split into two main areas:

- `utils/travel_graph` handles graph construction, routing, and HTML visualization.
- `utils/passenger_generation` handles passenger origin and destination sampling, jeep motion, and multipassenger simulation.

The pieces are connected in this order:

1. `travel_graph.py` builds and manages the stitched travel graph.
2. `travel_graph_visualizer.py` reads a `TravelGraphManager` and exports interactive HTML.
3. `passenger_map.py` samples passenger start and end locations from node data.
4. `passenger.py` uses `PassengerMap` and can query `TravelGraphManager` for routes.
5. `jeep.py` simulates a jeepney moving along a fixed route.
6. `simulation.py` prepares multipassenger simulation state with shared graph caching and flexible fleet sizes.

## Package exports

### `utils/__init__.py`

Re-exports the main travel graph and passenger utilities. It exposes:

- `TravelGraphManager`
- `JeepneyRoute`
- `JeepneySystem`
- `JeepneyRouteScrubber`
- `TravelGraphVisualizer`
- `visualize_travel_graph`
- passenger map, passenger, jeep, and tandem simulation classes
- travel graph construction helpers
- `BaselineRouteGenerator`
- `BaselineRoute`

### `utils/travel_graph/__init__.py`

Re-exports the travel graph core plus the visualizer:

- `TravelGraphManager`
- `JeepneyRoute`
- `JeepneySystem`
- `JeepneyRouteScrubber`
- `TravelGraphVisualizer`
- `visualize_travel_graph`
- graph construction helpers

## `utils/travel_graph/travel_graph.py`

This is the core module. It does two jobs: build the graph data and provide routing on top of it.

### Helper functions

| Function | What it does |
| --- | --- |
| `make_coord_key` | Rounds lon/lat values and combines them into a string key for matching nodes across layers. |
| `resolve_study_area_boundary` | Tries several place queries until it gets a valid study-area boundary from OSMnx. |
| `load_graphs_for_study_area` | Loads walk and drive graphs for the study area and projects them. |
| `node_table_from_graph` | Merges raw and projected graph nodes into one node table. |
| `extract_uncategorized_nodes` | Unions walk and drive nodes and marks graph membership. |
| `graph_edges_to_bidirectional_base` | Converts OSMnx edges into a bidirectional edge table. |
| `duplicate_walk_nodes_to_layers` | Clones walk nodes into `start_walk` and `end_walk` layers. |
| `build_direct_edges` | Connects same-location start and end walk nodes with zero-cost direct edges. |
| `_prepare_match_df` | Internal normalization helper for layer matching. |
| `build_interlayer_edges` | Links layers by exact coordinate match first, then by nearest snap if allowed. |
| `assign_edge_ids` | Adds sequential edge IDs with a prefix. |
| `edge_frame_to_csv` | Saves the edge table with the columns needed by the manager. |
| `node_frame_to_csv` | Saves selected node columns to CSV. |
| `make_edges_gdf` | Builds a GeoDataFrame from edge rows. |
| `make_nodes_gdf` | Builds a GeoDataFrame from node rows. |
| `add_nodes_to_digraph` | Loads node rows into a NetworkX directed graph. |
| `add_edges_to_digraph` | Loads edge rows into a NetworkX directed graph. |
| `compute_v_to_outgoing` | Builds a node to outgoing-edge map from the full stitched edge table. |
| `attach_accessible_nodes` | Adds the `accessible_nodes` column to each edge row. |

### `JeepneyRoute`

A simple wrapper around an ordered circular list of ride-layer node IDs.

- Validates that the route has at least two nodes.
- Validates that every node starts with `ride_`.
- `edge_pairs` returns all directed ride edges in the loop, including the wrap-around edge.
- `node_set` returns the unique ride nodes in the route.

### `TravelGraphManager`

This is the main runtime API.

What it loads:

- `edges_csv`: stitched travel graph edges
- `nodes_csv`: optional node table for map coordinates
- optional `routes`: route restrictions for the ride layer

What it builds:

- a weighted `networkx.DiGraph`
- an edge lookup by `edge_id`
- an accessibility lookup for edge-to-edge traversal
- node coordinate lookup for visualization

Weight rules:

- `start_walk` and `end_walk` use distance times `walk_wt`
- `ride` uses distance times `ride_wt`
- `wait` uses a flat `wait_wt`
- `transfer` uses a flat `transfer_wt`
- `alight` and `direct` are zero cost

Key methods:

- `get_edge`, `get_edges_from_node`, `get_edges_to_node`
- `get_accessible_edges`
- `generate_random_ride_loop`
- `find_nearest_node`
- `calculate_shortest_path`
- `visualize_path`

How it fits together:

- If routes are supplied, the manager filters the graph so only usable ride, wait, alight, and transfer edges remain.
- `calculate_shortest_path` runs Dijkstra on the weighted directed graph and returns edge IDs.
- `visualize_path` uses `nodes_csv` coordinates plus the chosen path edges to write a Leaflet map.

## `utils/travel_graph/travel_graph_visualizer.py`

This module builds a richer HTML explorer than `TravelGraphManager.visualize_path`.
Use this layer when the browser is only rendering playback or inspection, not running the simulation itself.

### `TravelGraphVisualizer`

Inputs:

- a ready `TravelGraphManager`
- an optional title

What it exports:

- toggleable node layers
- toggleable edge layers
- optional route overlays
- edge-type stats
- an in-browser shortest-path tool

Private helpers:

- `_extract_nodes`
- `_extract_edges`
- `_extract_routes`
- `_stats`
- `_build_html`

### `visualize_travel_graph`

A convenience wrapper that:

- creates a `TravelGraphManager`
- creates a `TravelGraphVisualizer`
- writes the HTML file

This is the easiest entry point if you already have the edges and nodes CSV files.

## `utils/passenger_generation/passenger_map.py`

This module provides the sampling base for passengers.

### `PassengerMap`

What it loads:

- `data/iligan_node_with_traffic_data.csv` by default

What it computes:

- `v_ped`, a pedestrian volume score derived from traffic index, building density, and betweenness centrality

What it does:

- computes sampling weights from `v_ped`
- samples nodes probabilistically with `generate_nodes`

This class does not depend on the travel graph. It only supplies plausible passenger origin and destination points.
`generate_nodes()` also accepts an optional `random_state` for reproducible traffic-biased sampling.

## `utils/baseline_route_generator.py`

This module creates the new geometric baseline routes on the physical drivable street network only.

### `BaselineRouteGenerator`

What it does:

- loads the drive-only street graph with the existing graph-loading helpers
- samples exactly four traffic-biased anchors using `PassengerMap`
- rejects anchor sets whose projected quadrilateral area is below the configured minimum
- stitches shortest paths between the four anchors to form a closed loop
- exports either a combined multi-route explorer or per-route HTML maps for inspection

### `BaselineRoute`

A compact route record containing:

- the selected anchor nodes
- ordered anchor geometry
- the full physical path node sequence
- projected area and route length

This module is the baseline route-generation entry point for B4 and later RL work.

## `utils/systemic_fitness_evaluator.py`

This module performs the multi-system statistical fitness test for route synergy.

### `SystemicFitnessEvaluator`

What it does:

- samples the number of evaluation tests from a configurable mean/spread
- samples the number of background noise routes per system from a configurable mean/spread
- generates multiple background route systems with `BaselineRouteGenerator`
- injects the candidate route into each 3-layer behavioral graph instance
- reuses the shared in-memory physical graph context across evaluations
- returns the average and standard deviation of Generalized Travel Cost across runs
- returns passenger-level GTC spread for future robustness analysis

Use this when you want to compare routes under stochastic sensitivity analysis rather than a single isolated background system.

## `utils/jeepney_route_env.py`

This module provides the Gymnasium-style RL environment for geometric route construction.

### `JeepneyRouteEnv`

What it does:

- navigates the primal physical street network step by step
- exposes coordinate-invariant geometric observations
- applies a continuous turn penalty during route construction
- supports route closure as an episode-ending event
- calls `calculate_route_fitness` by default on loop closure, or a provided `SystemicFitnessEvaluator` during PPO training, to score the final physical route on the 3-layer evaluation graph

State design:

- relative turning angles for available next edges
- sinuosity index for zig-zag detection
- Euclidean distance and relative bearing back to the origin node
- no absolute node IDs in the observation

The 3-layer travel graph remains evaluation-only for generalized travel cost scoring.

## `utils/rl_training.py`

This module contains the PPO training helpers used by B4.

### `train_route_agent`

What it does:

- builds a PPO-ready `JeepneyRouteEnv` with a systemic reward evaluator
- runs `stable_baselines3.PPO` on the coordinate-invariant observation space
- tracks the best and worst completed routes during training
- exports separate HTML maps and coordinate JSON for those routes
- writes `training_history.csv` and `training_snapshots.csv` into the chosen results folder so route quality can be plotted across episodes

Use this when you want the notebook to stay thin and keep the PPO plumbing in one utility layer.

### `export_training_results_csvs`

Use this helper when you want the training trail in tabular form without rerunning PPO.

- `training_history.csv` stores each finished episode with return, reward, GTC, and route-length fields
- `training_snapshots.csv` stores the best and worst closed-loop routes with their saved artifact paths

## `utils/route_spectrum_analysis.py`

This module powers the pre/post notebook analysis for B4A and B4B.

### What it does

- builds a numeric feature table for each route using shape, history, topology, demand, and global metrics
- computes Pearson and Spearman-style correlation matrices from that feature table
- extracts the strongest feature pairs for quick inspection
- writes correlation heatmaps and delta heatmaps to disk
- formats short route notes plus an overall summary for the shared route explorer HTML
- the shared route explorer can show that overall summary above the per-route note box

Use this when you want route comparisons to move beyond simple averages and into relationship analysis.

## `utils/passenger_generation/passenger.py`

This module models one passenger journey.

### `PassengerState`

Journey states:

- `WAITING_TO_WALK`
- `WALKING_TO_BOARD`
- `WAITING_AT_STATION`
- `RIDING`
- `ALIGHTING`
- `WALKING_FROM_ALIGHT`
- `AT_DESTINATION`
- `COMPLETED`

### `Passenger`

What it does:

- creates a passenger with start and end nodes
- looks up coordinates through a shared `PassengerMap` by default
- tracks current position, time, state, and traversed nodes
- can attach a `TravelGraphManager` with `set_travel_graph`
- can calculate a shortest path through the travel graph
- can step through the path node by node

Connection to the graph:

- `calculate_shortest_path` finds the nearest travel-graph nodes for the passenger start and end points
- it then calls `TravelGraphManager.calculate_shortest_path`
- path edges are converted back into path nodes for traversal
- it can reuse a shared `PassengerMap` when created inside `Simulation`
- it can reuse a shared `PassengerMap` when created inside `Simulation`

## `utils/passenger_generation/jeep.py`

This module simulates a single jeepney moving along a route.

### `JeepState`

States:

- `IDLE`
- `MOVING`
- `AT_STATION`
- `COMPLETED`

### `Jeep`

What it does:

- stores a route as a list of latitude and longitude pairs
- advances position using constant speed
- uses haversine distance to measure segment length
- uses linear interpolation to place the jeep between route nodes
- resets with `restart_route`

This class is independent of the travel graph and passenger classes. It only needs route coordinates.

## `utils/passenger_generation/simulation.py`

This module prepares the multipassenger experiment layer.

### `SimulationConfig`

Loads numeric values from `configs/travel_graph_config.yaml` so weights, velocities, and fleet defaults stay configurable.
It also pulls passenger generation interval, mean, and standard deviation for batch experiments.

### `Simulation`

What it does:

- keeps one shared `TravelGraphManager`
- keeps one shared `PassengerMap`
- caches nearest-node lookups and shortest paths
- prepares many passengers against the same graph
- builds per-route jeep fleets with configurable counts
- tracks total jeep count for later optimization work
- samples passenger batch sizes from config-driven mean and standard deviation

## Practical dependency map

| Component | Depends on |
| --- | --- |
| `PassengerMap` | CSV data in `data/iligan_node_with_traffic_data.csv` |
| `Passenger` | `PassengerMap`, optional `TravelGraphManager` |
| `Jeep` | Route coordinate list only |
| `Simulation` | Shared `TravelGraphManager`, passengers, routes, and config |
| `SimulationConfig` | `configs/travel_graph_config.yaml` |
| `JeepneyRoute` | Ordered ride node IDs |
| `TravelGraphManager` | Stitched edges CSV, optional nodes CSV, optional routes |
| `TravelGraphVisualizer` | `TravelGraphManager` |
| `visualize_travel_graph` | `TravelGraphManager`, `TravelGraphVisualizer` |

## Typical usage flow

1. Build or load the stitched travel graph.
2. Instantiate `TravelGraphManager`.
3. Optionally define `JeepneyRoute` objects.
4. Use the manager to find paths or generate route loops.
5. Use `visualize_path` for a single journey or `TravelGraphVisualizer` for the full explorer.
6. Generate passengers with `PassengerMap` and attach them to the manager in `Passenger`.
7. Advance a `Jeep` independently if you need vehicle motion over a route.
