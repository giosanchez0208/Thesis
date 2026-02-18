from __future__ import annotations

from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from src.part_a.routes import RouteDef


LAYER_START = 0
LAYER_RIDE = 1
LAYER_END = 2


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    # Fast enough + no extra deps
    R = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _drive_shortest_path_length_m(
    drive_G: nx.MultiDiGraph,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> Optional[float]:
    try:
        u = int(ox.distance.nearest_nodes(drive_G, X=lon1, Y=lat1))
        v = int(ox.distance.nearest_nodes(drive_G, X=lon2, Y=lat2))
        # Use edge length attribute
        length = nx.shortest_path_length(drive_G, u, v, weight="length")
        return float(length)
    except Exception:
        return None


def build_travel_graph_arrays(
    walk_G: nx.MultiDiGraph,
    drive_G: nx.MultiDiGraph,
    routes: List[RouteDef],
    weights: Dict[str, float],
    include_direct_edges: bool = True,
    ride_distance_method: str = "drive_shortest_path",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Output:
      nodes_df: idx, base_osmid, layer, lat, lon, node_type
      edges_df: u, v, weight, length_m, edge_type, route_id
      index_maps: dict with start_idx/end_idx/ride_idx mappings
    """
    beta_walk = float(weights["beta_walk"])
    beta_ride = float(weights["beta_ride"])
    beta_wait = float(weights["beta_wait"])
    beta_transfer = float(weights["beta_transfer"])
    beta_alight = float(weights.get("beta_alight", 0.0))
    beta_direct = float(weights.get("beta_direct", 0.0))

    # --------- Build node indices (arrays-friendly) ----------
    base_nodes = list(walk_G.nodes(data=True))

    # Ride nodes are only those referenced by routes
    ride_base_ids: Set[int] = set()
    stop_base_ids: Set[int] = set()

    for r in routes:
        for p in r.points:
            ride_base_ids.add(int(p.base_osmid))
            if p.point_type in {"stop", "terminal"}:
                stop_base_ids.add(int(p.base_osmid))

    start_idx: Dict[int, int] = {}
    end_idx: Dict[int, int] = {}
    ride_idx: Dict[int, int] = {}

    nodes_rows: List[Dict[str, Any]] = []
    idx_counter = 0

    # Start + End layers include ALL walk nodes (for general OD routing)
    for osmid, data in base_nodes:
        lat = float(data["y"])
        lon = float(data["x"])

        start_idx[int(osmid)] = idx_counter
        nodes_rows.append(
            dict(idx=idx_counter, base_osmid=int(osmid), layer=LAYER_START, lat=lat, lon=lon, node_type="intersection")
        )
        idx_counter += 1

        end_idx[int(osmid)] = idx_counter
        nodes_rows.append(
            dict(idx=idx_counter, base_osmid=int(osmid), layer=LAYER_END, lat=lat, lon=lon, node_type="intersection")
        )
        idx_counter += 1

    # Ride layer includes only route-relevant nodes
    for base_id in sorted(ride_base_ids):
        if base_id not in walk_G.nodes:
            # should not happen due to snapping, but keep safe
            continue
        lat = float(walk_G.nodes[base_id]["y"])
        lon = float(walk_G.nodes[base_id]["x"])
        ride_idx[base_id] = idx_counter
        node_type = "stop" if base_id in stop_base_ids else "waypoint"
        nodes_rows.append(
            dict(idx=idx_counter, base_osmid=int(base_id), layer=LAYER_RIDE, lat=lat, lon=lon, node_type=node_type)
        )
        idx_counter += 1

    nodes_df = pd.DataFrame(nodes_rows).sort_values("idx").reset_index(drop=True)

    # --------- Build edges ----------
    edges_rows: List[Dict[str, Any]] = []

    # A) Walk edges in START and END layers
    # OSMnx graphs can have multiple edges; we iterate over all and keep them directed.
    for u, v, k, data in walk_G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        if length_m <= 0:
            continue

        # START layer
        edges_rows.append(
            dict(
                u=start_idx[int(u)],
                v=start_idx[int(v)],
                length_m=length_m,
                weight=length_m * beta_walk,
                edge_type="walk",
                route_id="",
            )
        )
        # END layer
        edges_rows.append(
            dict(
                u=end_idx[int(u)],
                v=end_idx[int(v)],
                length_m=length_m,
                weight=length_m * beta_walk,
                edge_type="walk",
                route_id="",
            )
        )

    # B) Direct edges: START node -> END node (lets “walk-only” be valid)
    if include_direct_edges:
        for osmid, _ in base_nodes:
            base_id = int(osmid)
            edges_rows.append(
                dict(
                    u=start_idx[base_id],
                    v=end_idx[base_id],
                    length_m=0.0,
                    weight=beta_direct,
                    edge_type="direct",
                    route_id="",
                )
            )

    # C) Inter-layer edges for stops/terminals only
    for base_id in stop_base_ids:
        if base_id not in ride_idx:
            continue

        # Wait: START -> RIDE
        edges_rows.append(
            dict(
                u=start_idx[base_id],
                v=ride_idx[base_id],
                length_m=0.0,
                weight=beta_wait,
                edge_type="wait",
                route_id="",
            )
        )
        # Alight: RIDE -> END
        edges_rows.append(
            dict(
                u=ride_idx[base_id],
                v=end_idx[base_id],
                length_m=0.0,
                weight=beta_alight,
                edge_type="alight",
                route_id="",
            )
        )
        # Transfer: END -> RIDE
        edges_rows.append(
            dict(
                u=end_idx[base_id],
                v=ride_idx[base_id],
                length_m=0.0,
                weight=beta_transfer,
                edge_type="transfer",
                route_id="",
            )
        )

    # D) Ride edges from each route definition (directed, usually cycle)
    for r in routes:
        pts = r.points[:]
        if r.directed_cycle and len(pts) >= 2:
            # close the cycle if not closed already
            if pts[0].base_osmid != pts[-1].base_osmid:
                pts = pts + [pts[0]]

        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            if int(a.base_osmid) not in ride_idx or int(b.base_osmid) not in ride_idx:
                continue

            # distance for ride edge
            length_m: Optional[float] = None
            if ride_distance_method == "drive_shortest_path":
                length_m = _drive_shortest_path_length_m(drive_G, a.lat, a.lon, b.lat, b.lon)

            if length_m is None:
                length_m = _haversine_m(a.lat, a.lon, b.lat, b.lon)

            edges_rows.append(
                dict(
                    u=ride_idx[int(a.base_osmid)],
                    v=ride_idx[int(b.base_osmid)],
                    length_m=float(length_m),
                    weight=float(length_m) * beta_ride,
                    edge_type="ride",
                    route_id=r.route_id,
                )
            )

    edges_df = pd.DataFrame(edges_rows)

    # Basic sanity checks (keeps Part B easy)
    if edges_df.empty:
        raise ValueError("Edges array is empty. Check that the walk graph has edges and routes are valid.")

    # Ensure integer u/v
    edges_df["u"] = edges_df["u"].astype(int)
    edges_df["v"] = edges_df["v"].astype(int)

    index_maps = {
        "start_idx": {str(k): v for k, v in start_idx.items()},
        "end_idx": {str(k): v for k, v in end_idx.items()},
        "ride_idx": {str(k): v for k, v in ride_idx.items()},
        "layers": {"START": LAYER_START, "RIDE": LAYER_RIDE, "END": LAYER_END},
    }

    return nodes_df, edges_df, index_maps