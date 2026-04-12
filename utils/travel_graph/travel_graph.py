"""
Travel graph construction and management module.

This module provides:
- Helper functions for graph construction from OSM data
- JeepneyRoute: A wrapper for a jeepney route as an ordered circular sequence
- TravelGraphManager: Main interface for routing and visualization
"""

from __future__ import annotations

from pathlib import Path as _Path
import json as _json
import random
import warnings

from typing import Optional, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import folium
except ImportError:
    folium = None


# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def make_coord_key(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat", decimals: int = 7) -> pd.Series:
    """Create a coordinate key by rounding lon/lat and combining them."""
    return df[lon_col].round(decimals).astype(str) + "|" + df[lat_col].round(decimals).astype(str)


def resolve_study_area_boundary(place_queries: list):
    """
    Resolve the study area boundary from a list of place queries.
    
    Tries each query in order until one yields a valid boundary.
    Returns the GeoDataFrame and the method used to resolve it.
    """
    last_error = None
    for query in place_queries:
        try:
            gdf = ox.geocode_to_gdf(query)
            if gdf.empty:
                continue

            chosen = gdf.copy()
            if {"class", "type"}.issubset(chosen.columns):
                mask = (
                    chosen["class"].astype(str).str.lower().eq("boundary")
                    & chosen["type"].astype(str).str.lower().eq("administrative")
                )
                if mask.any():
                    chosen = chosen.loc[mask].copy()

            chosen = chosen.iloc[[0]].copy()
            geom = chosen.geometry.iloc[0]
            if geom is None or geom.is_empty:
                continue

            return chosen, f"geocode_to_gdf({query!r})"
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Unable to resolve the study area boundary: {last_error}")
    raise RuntimeError("Unable to resolve the study area boundary from the configured place queries.")


def load_graphs_for_study_area(
    place_queries: list,
    point_query: str | None = None,
    point_dist: float = 30000,
    simplify: bool = True,
    retain_all: bool = True,
):
    """
    Load walk and drive graphs for the study area.
    
    Parameters
    ----------
    place_queries : list
        List of place query strings to try
    point_query : str, optional
        Fallback point query if polygon download fails
    point_dist : float
        Distance in meters for point-based query
    simplify : bool
        Whether to simplify the graphs
    retain_all : bool
        Whether to retain all graph components
    
    Returns
    -------
    tuple
        (study_area_gdf, boundary_source, graph_source, G_walk_raw, G_walk_proj, G_drive_raw, G_drive_proj)
    """
    study_area_gdf, boundary_source = resolve_study_area_boundary(place_queries)
    polygon = study_area_gdf.geometry.union_all() if hasattr(study_area_gdf.geometry, "union_all") else study_area_gdf.unary_union

    try:
        G_walk_raw = ox.graph_from_polygon(
            polygon,
            network_type="walk",
            simplify=simplify,
            retain_all=retain_all,
        )
        G_drive_raw = ox.graph_from_polygon(
            polygon,
            network_type="drive",
            simplify=simplify,
            retain_all=retain_all,
        )
        graph_source = "graph_from_polygon(study_area_boundary)"
    except Exception as exc:
        if point_query is None:
            raise
        point = ox.geocode(point_query)
        G_walk_raw = ox.graph_from_point(
            point,
            dist=point_dist,
            network_type="walk",
            simplify=simplify,
            retain_all=retain_all,
        )
        G_drive_raw = ox.graph_from_point(
            point,
            dist=point_dist,
            network_type="drive",
            simplify=simplify,
            retain_all=retain_all,
        )
        graph_source = f"graph_from_point({point_query!r}, dist={point_dist}) because polygon download failed: {type(exc).__name__}: {exc}"

    G_walk_proj = ox.project_graph(G_walk_raw)
    G_drive_proj = ox.project_graph(G_drive_raw)
    return study_area_gdf, boundary_source, graph_source, G_walk_raw, G_walk_proj, G_drive_raw, G_drive_proj


def node_table_from_graph(G_raw: nx.MultiDiGraph, G_proj: nx.MultiDiGraph) -> pd.DataFrame:
    """Extract node tables from raw and projected graphs."""
    raw_nodes = pd.DataFrame.from_dict(dict(G_raw.nodes(data=True)), orient="index").reset_index()
    raw_nodes = raw_nodes.rename(columns={"index": "base_node_id", "x": "lon", "y": "lat"})

    proj_nodes = pd.DataFrame.from_dict(dict(G_proj.nodes(data=True)), orient="index").reset_index()
    proj_nodes = proj_nodes.rename(columns={"index": "base_node_id", "x": "x", "y": "y"})

    merged = proj_nodes[["base_node_id", "x", "y"]].merge(
        raw_nodes[["base_node_id", "lon", "lat"]],
        on="base_node_id",
        how="left",
    )
    merged["coord_key"] = make_coord_key(merged, "lon", "lat", 7)
    merged["node_id"] = merged["base_node_id"].astype(str)
    return merged[["node_id", "base_node_id", "x", "y", "lon", "lat", "coord_key"]].sort_values("base_node_id").reset_index(drop=True)


def extract_uncategorized_nodes(walk_nodes: pd.DataFrame, drive_nodes: pd.DataFrame) -> pd.DataFrame:
    """Extract all unique nodes from both walk and drive graphs."""
    walk = walk_nodes.copy()
    walk["in_walk_graph"] = True
    walk["in_drive_graph"] = False

    drive = drive_nodes.copy()
    drive["in_walk_graph"] = False
    drive["in_drive_graph"] = True

    union = pd.concat([walk, drive], ignore_index=True, sort=False)
    union = union.groupby("base_node_id", as_index=False).agg(
        {
            "node_id": "first",
            "x": "first",
            "y": "first",
            "lon": "first",
            "lat": "first",
            "coord_key": "first",
            "in_walk_graph": "max",
            "in_drive_graph": "max",
        }
    )
    union["node_id"] = union["base_node_id"].astype(str)
    return union[
        ["node_id", "base_node_id", "x", "y", "lon", "lat", "in_walk_graph", "in_drive_graph", "coord_key"]
    ].sort_values("base_node_id").reset_index(drop=True)


def graph_edges_to_bidirectional_base(G_proj: nx.MultiDiGraph, prefix: str, edge_type: str) -> pd.DataFrame:
    """Convert graph edges to bidirectional edge dataframe."""
    edges_gdf = ox.graph_to_gdfs(G_proj, nodes=False, edges=True).reset_index()

    if "length" not in edges_gdf.columns:
        edges_gdf["length"] = edges_gdf.geometry.length

    edges_gdf["pair_key"] = edges_gdf.apply(lambda row: tuple(sorted((row["u"], row["v"]))), axis=1)
    edges_gdf = (
        edges_gdf.sort_values(["pair_key", "length"])
        .drop_duplicates(subset=["pair_key"], keep="first")
        .copy()
    )

    rows = []
    for row in edges_gdf.itertuples(index=False):
        pair_u, pair_v = row.pair_key
        dist = float(row.length)
        geom = row.geometry

        rows.append(
            {
                "u": f"{prefix}_{pair_u}",
                "v": f"{prefix}_{pair_v}",
                "dist": dist,
                "edge_type": edge_type,
                "geometry": geom,
            }
        )
        rev_geom = LineString(list(geom.coords)[::-1]) if geom is not None else None
        rows.append(
            {
                "u": f"{prefix}_{pair_v}",
                "v": f"{prefix}_{pair_u}",
                "dist": dist,
                "edge_type": edge_type,
                "geometry": rev_geom,
            }
        )
    return pd.DataFrame(rows)


def duplicate_walk_nodes_to_layers(nodes_walk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate walk nodes to start-walk and end-walk layers."""
    start_nodes = nodes_walk.copy()
    start_nodes["node_id"] = "sw_" + start_nodes["base_node_id"].astype(str)

    end_nodes = nodes_walk.copy()
    end_nodes["node_id"] = "ew_" + end_nodes["base_node_id"].astype(str)

    return start_nodes, end_nodes


def build_direct_edges(start_nodes: pd.DataFrame, end_nodes: pd.DataFrame) -> pd.DataFrame:
    """Build direct edges connecting start-walk nodes to end-walk nodes."""
    merged = start_nodes[["base_node_id", "node_id", "x", "y"]].merge(
        end_nodes[["base_node_id", "node_id", "x", "y"]],
        on="base_node_id",
        how="inner",
        suffixes=("_start", "_end"),
    )

    rows = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "u": row.node_id_start,
                "v": row.node_id_end,
                "dist": 0.0,
                "edge_type": "direct",
                "geometry": LineString([(row.x_start, row.y_start), (row.x_end, row.y_end)]),
            }
        )
    return pd.DataFrame(rows)


def _prepare_match_df(df: pd.DataFrame, node_col: str) -> pd.DataFrame:
    """Prepare a dataframe for node matching."""
    return df[[node_col, "base_node_id", "x", "y", "lon", "lat", "coord_key"]].drop_duplicates(node_col).reset_index(drop=True)


def build_interlayer_edges(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_node_col: str,
    right_node_col: str,
    edge_type: str,
    max_snap_dist_m: float | None = None,
    anchor_side: str = "right",
) -> pd.DataFrame:
    """
    Build edges between two node layers using exact (coordinate-key) matching
    with optional snapping to nearest neighbors.
    """
    if left_df.empty or right_df.empty:
        return pd.DataFrame(columns=["u", "v", "dist", "edge_type", "geometry"])

    if anchor_side not in {"left", "right"}:
        raise ValueError("anchor_side must be 'left' or 'right'.")

    source_df = left_df if anchor_side == "left" else right_df
    target_df = right_df if anchor_side == "left" else left_df
    source_node_col = left_node_col if anchor_side == "left" else right_node_col
    target_node_col = right_node_col if anchor_side == "left" else left_node_col

    source = _prepare_match_df(source_df, source_node_col)
    target = _prepare_match_df(target_df, target_node_col)

    records = []

    # Exact coordinate match
    exact = source.merge(target, on="coord_key", how="inner", suffixes=("_source", "_target"))
    source_id_key = source_node_col + "_source" if source_node_col + "_source" in exact.columns else source_node_col
    target_id_key = target_node_col + "_target" if target_node_col + "_target" in exact.columns else target_node_col

    exact_source_ids = set()
    for rec in exact.to_dict("records"):
        exact_source_ids.add(rec[source_id_key])
        records.append(
            {
                "source_node": rec[source_id_key],
                "target_node": rec[target_id_key],
                "x_source": rec["x_source"],
                "y_source": rec["y_source"],
                "x_target": rec["x_target"],
                "y_target": rec["y_target"],
            }
        )

    # Snap remaining sources to nearest target
    source_remaining = source[~source[source_node_col].isin(exact_source_ids)].copy()
    if not source_remaining.empty and not target.empty:
        target_xy = target[["x", "y"]].to_numpy(dtype=float)
        for rec in source_remaining.to_dict("records"):
            sx = float(rec["x"])
            sy = float(rec["y"])
            dists = np.sqrt((target_xy[:, 0] - sx) ** 2 + (target_xy[:, 1] - sy) ** 2)
            nearest_idx = int(dists.argmin())
            snap_dist = float(dists[nearest_idx])

            if max_snap_dist_m is not None and snap_dist > float(max_snap_dist_m):
                continue

            target_row = target.iloc[nearest_idx].to_dict()
            records.append(
                {
                    "source_node": rec[source_node_col],
                    "target_node": target_row[target_node_col],
                    "x_source": rec["x"],
                    "y_source": rec["y"],
                    "x_target": target_row["x"],
                    "y_target": target_row["y"],
                }
            )

    rows = []
    for rec in records:
        if anchor_side == "left":
            u = rec["source_node"]
            v = rec["target_node"]
            x_u, y_u = rec["x_source"], rec["y_source"]
            x_v, y_v = rec["x_target"], rec["y_target"]
        else:
            u = rec["target_node"]
            v = rec["source_node"]
            x_u, y_u = rec["x_target"], rec["y_target"]
            x_v, y_v = rec["x_source"], rec["y_source"]

        rows.append(
            {
                "u": u,
                "v": v,
                "dist": 0.0,
                "edge_type": edge_type,
                "geometry": LineString([(x_u, y_u), (x_v, y_v)]),
            }
        )
    return pd.DataFrame(rows)


def assign_edge_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Assign sequential edge IDs with a given prefix."""
    out = df.copy().reset_index(drop=True)
    out.insert(0, "edge_id", [f"{prefix}_{i + 1:06d}" for i in range(len(out))])
    return out


def edge_frame_to_csv(df: pd.DataFrame, csv_path: _Path) -> None:
    """Save edges to CSV with selected columns."""
    out = df.copy()
    save_cols = [col for col in ["edge_id", "u", "v", "dist", "edge_type", "accessible_nodes"] if col in out.columns]
    out = out[save_cols]
    out.to_csv(csv_path, index=False)


def node_frame_to_csv(df: pd.DataFrame, csv_path: _Path, keep_cols: list[str]) -> None:
    """Save nodes to CSV with selected columns."""
    out = df.copy()
    out = out[keep_cols]
    out.to_csv(csv_path, index=False)


def make_edges_gdf(df: pd.DataFrame, crs):
    """Create a GeoDataFrame from edges."""
    out = df.copy()
    if "geometry" not in out.columns:
        out["geometry"] = pd.Series(dtype="object")
    if gpd is None:
        raise ImportError("geopandas is required for this function")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def make_nodes_gdf(df: pd.DataFrame, crs):
    """Create a GeoDataFrame from nodes."""
    if gpd is None:
        raise ImportError("geopandas is required for this function")
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=crs,
    )
    return gdf


def add_nodes_to_digraph(G: nx.DiGraph, nodes_df: pd.DataFrame) -> None:
    """Add nodes from a DataFrame to a NetworkX DiGraph."""
    for row in nodes_df.itertuples(index=False):
        attrs = row._asdict()
        node_id = attrs.pop("node_id")
        attrs.pop("coord_key", None)
        G.add_node(node_id, **attrs)


def add_edges_to_digraph(G: nx.DiGraph, edges_df: pd.DataFrame) -> None:
    """Add edges from a DataFrame to a NetworkX DiGraph."""
    for row in edges_df.itertuples(index=False):
        attrs = row._asdict()
        edge_id = attrs.pop("edge_id")
        geom = attrs.pop("geometry", None)
        u = attrs.pop("u")
        v = attrs.pop("v")
        G.add_edge(u, v, edge_id=edge_id, geometry=geom, **attrs)


def compute_v_to_outgoing(all_edges: pd.DataFrame) -> dict:
    """
    Build {node_id -> [edge_id, ...]} mapping for all edges departing FROM that node.

    Must be called on the FULLY STITCHED travel_graph_edges DataFrame so that
    cross-layer connections (wait, alight, transfer, direct) are included.
    """
    return (
        all_edges.groupby("u", sort=False)["edge_id"]
        .apply(list)
        .to_dict()
    )


def attach_accessible_nodes(edges_df: pd.DataFrame, v_to_outgoing: dict) -> pd.DataFrame:
    """
    Add the accessible_nodes column to an edge DataFrame.

    For each row, accessible_nodes is a semicolon-delimited string of all
    edge_ids reachable from that edge's v in the full graph.
    Terminal nodes (no onward edges) receive an empty string.
    """
    out = edges_df.copy()
    out["accessible_nodes"] = out["v"].apply(
        lambda v_node: ";".join(v_to_outgoing.get(v_node, []))
    )
    return out


# ────────────────────────────────────────────────────────────────────────────
# JEEPNEY ROUTE CLASS
# ────────────────────────────────────────────────────────────────────────────

class JeepneyRoute:
    """
    A single jeepney route: an ordered circular sequence of ride-layer node IDs.

    Parameters
    ----------
    route_id : str
        Unique identifier (e.g. "R1", "Iligan-Hinaplanon").
    nodes : list[str]
        Ordered ride_ node IDs. The route is circular — the last node connects
        back to the first. Do NOT repeat the first node at the end.

    Example
    -------
    >>> r = JeepneyRoute("R1", ["ride_A", "ride_B", "ride_C"])
    # circular edges: ride_A→ride_B, ride_B→ride_C, ride_C→ride_A
    """

    def __init__(self, route_id: str, nodes: list) -> None:
        if len(nodes) < 2:
            raise ValueError(f"Route {route_id!r} needs at least 2 nodes.")
        bad = [n for n in nodes if not str(n).startswith("ride_")]
        if bad:
            raise ValueError(
                f"Route {route_id!r} has non-ride nodes: {bad[:5]}. "
                "All nodes must carry the 'ride_' prefix."
            )
        self._route_id = str(route_id)
        self._nodes    = list(nodes)

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def route_id(self) -> str:
        """Unique route identifier."""
        return self._route_id

    @property
    def nodes(self) -> list:
        """Ordered ride node IDs (first node not repeated at end)."""
        return list(self._nodes)

    @property
    def edge_pairs(self) -> set:
        """
        Set of directed (u, v) ride-edge pairs that belong to this route,
        including the circular wrap-around from the last node to the first.
        """
        n = self._nodes
        pairs = set(zip(n, n[1:]))   # consecutive pairs
        pairs.add((n[-1], n[0]))      # close the loop
        return pairs

    @property
    def node_set(self) -> set:
        """Set of all ride node IDs on this route."""
        return set(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"JeepneyRoute(id={self._route_id!r}, stops={len(self._nodes)})"


# ────────────────────────────────────────────────────────────────────────────
# TRAVEL GRAPH MANAGER CLASS
# ────────────────────────────────────────────────────────────────────────────

# Edge-type constants
_WALK_TYPES    = {"start_walk", "end_walk"}
_RIDE_TYPE     = "ride"
_WAIT_TYPE     = "wait"
_ALIGHT_TYPE   = "alight"
_DIRECT_TYPE   = "direct"
_TRANSFER_TYPE = "transfer"
_ALL_KNOWN_TYPES = _WALK_TYPES | {_RIDE_TYPE, _WAIT_TYPE, _ALIGHT_TYPE,
                                   _DIRECT_TYPE, _TRANSFER_TYPE}
_REQUIRED_COLUMNS = {"edge_id", "u", "v", "dist", "edge_type"}

# Colour palette for route visualisation (Leaflet hex strings)
_ROUTE_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]


class TravelGraphManager:
    """
    Loads the stitched travel graph and exposes routing + visualisation.

    Parameters
    ----------
    edges_csv : str or Path
        Path to the stitched edges CSV (iligan_travel_graph.csv).
    nodes_csv : str or Path, optional
        Path to the stitched nodes CSV (travel_graph_nodes.csv).
        Required for visualise_path(); contains lat/lon per node.
    routes : list[JeepneyRoute], optional
        If provided the ride layer is restricted to these routes only.
        If None the full ride graph is used (useful for inspection).
    walk_wt : float   β_walk  — multiplied by dist_m for walk edges.
    ride_wt : float   β_ride  — multiplied by dist_m for ride edges.
    wait_wt : float   β_wait  — flat boarding disutility.
    transfer_wt : float  β_transfer — flat vehicle-change disutility.
    """

    def __init__(
        self,
        edges_csv,
        nodes_csv=None,
        routes=None,
        *,
        walk_wt: float,
        ride_wt: float,
        wait_wt: float,
        transfer_wt: float,
    ) -> None:
        self._walk_wt     = float(walk_wt)
        self._ride_wt     = float(ride_wt)
        self._wait_wt     = float(wait_wt)
        self._transfer_wt = float(transfer_wt)
        self._routes      = list(routes) if routes is not None else None

        self._edges_df  = self._load_edges(_Path(edges_csv))
        self._nodes_df  = self._load_nodes(_Path(nodes_csv)) if nodes_csv else None

        # Apply route filtering if routes were supplied
        self._active_edges = (
            self._filter_by_routes(self._edges_df, self._routes)
            if self._routes is not None
            else self._edges_df
        )

        self._graph      = self._build_weighted_graph(self._active_edges)
        self._edge_by_id = {
            row.edge_id: row
            for row in self._active_edges.itertuples(index=False)
        }
        self._accessible  = self._build_accessible_index()
        self._node_coords = self._build_node_coords()

        self._sanity_check()

    # ── Private: loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load_edges(path: _Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Edges CSV not found: {path}")
        df = pd.read_csv(
            path,
            dtype={"edge_id": str, "u": str, "v": str, "edge_type": str},
        )
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Edges CSV missing required columns: {missing}")
        return df

    @staticmethod
    def _load_nodes(path: _Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Nodes CSV not found: {path}")
        return pd.read_csv(
            path,
            dtype={"node_id": str},
        )

    # ── Private: route filtering ─────────────────────────────────────────────

    @staticmethod
    def _filter_by_routes(df: pd.DataFrame, routes: list) -> pd.DataFrame:
        """
        Keep only the edges a passenger can actually use given the supplied routes.

        Rules per edge type
        -------------------
        start_walk / end_walk / direct : always kept (passenger walks freely).
        ride       : kept only if (u, v) is an edge pair of at least one route.
        wait       : kept only if v (ride node) is in at least one route.
        alight     : kept only if u (ride node) is in at least one route.
        transfer   : kept only if v (ride node) is in at least one route.
        """
        valid_pairs = set()
        valid_nodes = set()
        for route in routes:
            valid_pairs |= route.edge_pairs
            valid_nodes |= route.node_set

        def _keep(row) -> bool:
            et = row.edge_type
            if et in _WALK_TYPES or et == _DIRECT_TYPE:
                return True
            if et == _RIDE_TYPE:
                return (row.u, row.v) in valid_pairs
            if et == _WAIT_TYPE:
                return row.v in valid_nodes      # sw → ride: is the stop on a route?
            if et == _ALIGHT_TYPE:
                return row.u in valid_nodes      # ride → ew: is the stop on a route?
            if et == _TRANSFER_TYPE:
                return row.v in valid_nodes      # ew → ride: is the stop on a route?
            return True

        mask = df.apply(_keep, axis=1)
        return df[mask].reset_index(drop=True)

    # ── Private: graph + index construction ─────────────────────────────────

    def _edge_weight(self, edge_type: str, dist: float) -> float:
        if edge_type in _WALK_TYPES:            return dist * self._walk_wt
        if edge_type == _RIDE_TYPE:             return dist * self._ride_wt
        if edge_type == _WAIT_TYPE:             return self._wait_wt
        if edge_type == _TRANSFER_TYPE:         return self._transfer_wt
        if edge_type in (_ALIGHT_TYPE, _DIRECT_TYPE): return 0.0
        warnings.warn(f"Unknown edge_type {edge_type!r}; using raw dist.", stacklevel=2)
        return dist

    def _build_weighted_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        for row in df.itertuples(index=False):
            G.add_edge(
                row.u, row.v,
                edge_id=row.edge_id,
                weight=self._edge_weight(row.edge_type, float(row.dist)),
                edge_type=row.edge_type,
                dist=float(row.dist),
            )
        return G

    def _build_accessible_index(self) -> dict:
        """
        Build edge_id → [reachable edge_ids] from the ACTIVE (possibly filtered) graph.
        Reads the accessible_nodes CSV column if present; otherwise derives from graph.
        """
        if "accessible_nodes" in self._active_edges.columns:
            active_ids = set(self._active_edges["edge_id"])
            idx = {}
            for row in self._active_edges.itertuples(index=False):
                raw = getattr(row, "accessible_nodes", "") or ""
                # Only include neighbours that are still in the active (filtered) graph
                idx[row.edge_id] = [
                    e for e in str(raw).split(";")
                    if e and e in active_ids
                ]
            return idx
        warnings.warn(
            "accessible_nodes column not found — computing from filtered graph.",
            stacklevel=2,
        )
        v_to_out: dict = {}
        for u, v, data in self._graph.edges(data=True):
            v_to_out.setdefault(u, []).append(data["edge_id"])
        return {
            row.edge_id: v_to_out.get(row.v, [])
            for row in self._active_edges.itertuples(index=False)
        }

    def _build_node_coords(self) -> dict:
        """Build node_id → (lat, lon) from the nodes DataFrame (if loaded)."""
        if self._nodes_df is None:
            return {}
        required = {"node_id", "lat", "lon"}
        if not required.issubset(self._nodes_df.columns):
            warnings.warn(
                f"nodes CSV missing one of {required}; visualisation disabled.",
                stacklevel=2,
            )
            return {}
        return {
            row.node_id: (float(row.lat), float(row.lon))
            for row in self._nodes_df.itertuples(index=False)
        }

    def _sanity_check(self) -> None:
        assert self._graph.number_of_nodes() > 0, "Graph has no nodes."
        assert self._graph.number_of_edges() > 0, "Graph has no edges."
        actual_types = set(self._active_edges["edge_type"].unique())
        unknown = actual_types - _ALL_KNOWN_TYPES
        if unknown:
            warnings.warn(f"Unexpected edge_type(s): {unknown}")
        route_str = (
            f"{len(self._routes)} routes"
            if self._routes is not None else "full ride graph"
        )
        print(
            f"[TravelGraphManager] {route_str} | "
            f"{self._graph.number_of_nodes():,} nodes | "
            f"{self._graph.number_of_edges():,} edges"
        )
        print(f"  Edge types : {sorted(actual_types)}")
        print(
            f"  Weights    — walk: {self._walk_wt}, ride: {self._ride_wt}, "
            f"wait: {self._wait_wt}, transfer: {self._transfer_wt}"
        )

    # ── Getters ──────────────────────────────────────────────────────────────

    @property
    def edges(self) -> pd.DataFrame:
        """Active (possibly route-filtered) edges DataFrame (copy)."""
        return self._active_edges.copy()

    @property
    def graph(self) -> nx.DiGraph:
        """Underlying weighted NetworkX DiGraph (active edges only)."""
        return self._graph

    @property
    def routes(self):
        """List of JeepneyRoute objects, or None if using full ride graph."""
        return list(self._routes) if self._routes is not None else None

    @property
    def walk_wt(self) -> float:     return self._walk_wt
    @property
    def ride_wt(self) -> float:     return self._ride_wt
    @property
    def wait_wt(self) -> float:     return self._wait_wt
    @property
    def transfer_wt(self) -> float: return self._transfer_wt

    def get_edge(self, edge_id: str):
        return self._edge_by_id.get(edge_id)

    def get_edges_from_node(self, node_id: str) -> pd.DataFrame:
        return self._active_edges[self._active_edges["u"] == node_id].copy()

    def get_edges_to_node(self, node_id: str) -> pd.DataFrame:
        return self._active_edges[self._active_edges["v"] == node_id].copy()

    def get_accessible_edges(self, edge_id: str) -> list:
        """
        Edge_ids reachable after traversing edge_id. Reflects the active
        (route-filtered) graph — O(1) lookup.
        """
        return list(self._accessible.get(edge_id, []))

    # ── Method 1: generate_random_ride_loop ─────────────────────────────────

    def generate_random_ride_loop(
        self,
        min_length: int,
        max_length: int,
        seed=None,
    ) -> list:
        """
        Generate a random cycle within the *active* ride layer.

        Useful for creating synthetic JeepneyRoute objects for testing:
            nodes = mgr.generate_random_ride_loop(5, 20)
            route = JeepneyRoute("test", nodes[:-1])  # drop repeated last node

        Returns
        -------
        list[str]
            Ride node IDs forming a closed cycle (first == last element).
        """
        if min_length < 2:
            raise ValueError("min_length must be >= 2.")
        if min_length > max_length:
            raise ValueError("min_length must be <= max_length.")

        rng = random.Random(seed)
        ride_df = self._active_edges[self._active_edges["edge_type"] == _RIDE_TYPE]
        if ride_df.empty:
            raise ValueError(
                "No ride edges in active graph. "
                "If routes were provided, ensure they contain valid ride edge pairs."
            )

        ride_G = nx.DiGraph()
        for row in ride_df.itertuples(index=False):
            ride_G.add_edge(row.u, row.v)

        ride_nodes = list(ride_G.nodes())
        if not ride_nodes:
            raise ValueError("No ride nodes found in the active ride sub-graph.")

        for _ in range(300):
            start   = rng.choice(ride_nodes)
            path    = [start]
            current = start
            visited = {start}
            target  = rng.randint(min_length, max_length)

            while len(path) < target:
                unvisited = [n for n in ride_G.successors(current) if n not in visited]
                nbrs = unvisited or list(ride_G.successors(current))
                if not nbrs:
                    break
                current = rng.choice(nbrs)
                path.append(current)
                visited.add(current)

            if len(path) < min_length:
                continue

            if ride_G.has_edge(current, start):
                return path + [start]

            try:
                ret = nx.shortest_path(ride_G, current, start)
                combined = path + ret[1:]
                if len(combined) - 1 >= min_length:
                    return combined
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        raise ValueError(
            f"Could not generate a ride loop of length [{min_length}, {max_length}] "
            "after 300 attempts."
        )

    # ── Method 1b: find_nearest_node ───────────────────────────────────────

    def find_nearest_node(self, lat: float, lon: float, layer: str = None) -> str:
        if self._nodes_df is None:
            raise ValueError("Nodes dataframe not loaded. Provide nodes_csv to TravelGraphManager.")
        
        # Filter by layer if specified
        if layer is not None:
            nodes = self._nodes_df[self._nodes_df['layer'] == layer].copy()
        else:
            nodes = self._nodes_df.copy()
        
        if len(nodes) == 0:
            return None
        
        # Calculate distances using Euclidean distance
        # (OK for small areas, more precise with haversine if needed)
        distances = ((nodes['lat'] - lat)**2 + (nodes['lon'] - lon)**2)**0.5
        nearest_idx = distances.idxmin()
        nearest_node = nodes.loc[nearest_idx]
        
        return nearest_node['node_id']

    # ── Method 2: calculate_shortest_path ───────────────────────────────────

    def calculate_shortest_path(self, u: str, v: str) -> list:
        """
        Minimum-disutility path from node u to node v using Dijkstra.

        Only active (route-filtered) edges are considered.  A passenger
        restricted to routes R1 and R2 cannot ride edges belonging only to R3.

        Returns
        -------
        list[str]  Ordered edge_ids traversed. Empty list if u == v.

        Raises
        ------
        nx.NodeNotFound    If u or v is absent from the active graph.
        nx.NetworkXNoPath  If no path exists (e.g. destination unreachable
                           with the given routes).
        """
        if u not in self._graph:
            raise nx.NodeNotFound(f"Origin not in active graph: {u!r}")
        if v not in self._graph:
            raise nx.NodeNotFound(f"Destination not in active graph: {v!r}")
        if u == v:
            return []
        node_path = nx.dijkstra_path(self._graph, u, v, weight="weight")
        return [self._graph[a][b]["edge_id"] for a, b in zip(node_path[:-1], node_path[1:])]

    # ── Method 3: visualize_path ─────────────────────────────────────────────

    def visualize_path(
        self,
        path_edges: list,
        output_html,
        title: str = "Jeepney Journey",
    ) -> _Path:
        """
        Generate a self-contained Leaflet HTML map of a journey.

        Shows:
        - All active jeepney routes as coloured polylines (toggleable).
        - The journey path highlighted by segment type (walk / ride / transfer).
        - Origin (green) and destination (red) markers with popups.
        - A legend and layer control.

        Parameters
        ----------
        path_edges : list[str]
            Output of calculate_shortest_path().
        output_html : str or Path
            Where to write the HTML file.
        title : str
            Map title shown in the top-left panel.

        Returns
        -------
        Path  Absolute path to the generated HTML file.

        Raises
        ------
        RuntimeError  If nodes_csv was not provided at initialisation.
        """
        if not self._node_coords:
            raise RuntimeError(
                "Node coordinates are unavailable. "
                "Pass nodes_csv= when creating TravelGraphManager."
            )
        if not path_edges:
            raise ValueError("path_edges is empty — nothing to visualise.")

        out_path = _Path(output_html).resolve()

        # ── Collect journey segments ──────────────────────────────────────
        SEGMENT_COLOURS = {
            "start_walk": "#1f77b4",   # blue
            "end_walk":   "#1f77b4",
            "direct":     "#7f7f7f",   # grey
            "ride":       "#d62728",   # red  (overridden by route colour below)
            "wait":       "#2ca02c",   # green
            "alight":     "#9467bd",   # purple
            "transfer":   "#ff7f0e",   # orange
        }

        # Map ride edge_id → route colour
        ride_edge_to_colour = {}
        if self._routes:
            for idx, route in enumerate(self._routes):
                colour = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
                for u_node, v_node in route.edge_pairs:
                    # Find the matching edge_id
                    matches = self._active_edges[
                        (self._active_edges["u"] == u_node) &
                        (self._active_edges["v"] == v_node) &
                        (self._active_edges["edge_type"] == _RIDE_TYPE)
                    ]["edge_id"]
                    for eid in matches:
                        ride_edge_to_colour[eid] = colour

        segments = []
        for eid in path_edges:
            row = self.get_edge(eid)
            if row is None:
                continue
            u_coord = self._node_coords.get(row.u)
            v_coord = self._node_coords.get(row.v)
            if u_coord is None or v_coord is None:
                continue
            et = row.edge_type
            colour = (
                ride_edge_to_colour.get(eid, SEGMENT_COLOURS["ride"])
                if et == _RIDE_TYPE
                else SEGMENT_COLOURS.get(et, "#333333")
            )
            segments.append({
                "coords": [list(u_coord), list(v_coord)],
                "edge_id": eid,
                "edge_type": et,
                "dist": round(float(row.dist), 1),
                "colour": colour,
            })

        if not segments:
            raise RuntimeError(
                "No segments could be drawn — node coordinates may be missing for "
                "the nodes in this path."
            )

        origin = segments[0]["coords"][0]
        dest   = segments[-1]["coords"][1]

        # ── Build route polyline data ─────────────────────────────────────
        route_layers = []
        if self._routes:
            for idx, route in enumerate(self._routes):
                colour = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
                coords = []
                for nid in route.nodes:
                    c = self._node_coords.get(nid)
                    if c:
                        coords.append(list(c))
                # close the loop visually
                if coords:
                    coords.append(coords[0])
                route_layers.append({
                    "id":     route.route_id,
                    "colour": colour,
                    "coords": coords,
                })

        # ── Summary stats ─────────────────────────────────────────────────
        type_counts = {}
        for s in segments:
            type_counts[s["edge_type"]] = type_counts.get(s["edge_type"], 0) + 1
        walk_dist = sum(
            s["dist"] for s in segments
            if s["edge_type"] in _WALK_TYPES or s["edge_type"] == _DIRECT_TYPE
        )
        ride_dist = sum(s["dist"] for s in segments if s["edge_type"] == _RIDE_TYPE)
        n_transfers = type_counts.get("transfer", 0)

        summary_html = (
            f"<b>Walk</b> {walk_dist:.0f} m &nbsp;|&nbsp; "
            f"<b>Ride</b> {ride_dist:.0f} m &nbsp;|&nbsp; "
            f"<b>Transfers</b> {n_transfers}"
        )

        # ── Build legend HTML ─────────────────────────────────────────────
        legend_items = ""
        seen = set()
        for s in segments:
            et = s["edge_type"]
            c  = s["colour"]
            if et not in seen:
                seen.add(et)
                legend_items += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
                    f'<span style="display:inline-block;width:24px;height:4px;' +
                    f'background:{c};border-radius:2px"></span>' +
                    f'<span style="font-size:12px">{et.replace("_"," ")}</span></div>'
                )
        if self._routes:
            for rl in route_layers:
                legend_items += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">' +
                    f'<span style="display:inline-block;width:24px;height:4px;opacity:0.5;' +
                    f'background:{rl["colour"]};border-radius:2px"></span>' +
                    f'<span style="font-size:12px">Route {rl["id"]}</span></div>'
                )

        # ── Embed data + write HTML ───────────────────────────────────────
        segments_json    = _json.dumps(segments)
        routes_json      = _json.dumps(route_layers)
        origin_json      = _json.dumps(origin)
        dest_json        = _json.dumps(dest)

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body {{margin:0;padding:0;height:100%;font-family:sans-serif}}
  #map {{height:100vh;width:100%}}
  #panel {{
    position:fixed;top:12px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(255,255,255,0.96);
    border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.25);
    padding:10px 16px;max-width:500px;text-align:center;
  }}
  #panel h3 {{margin:0 0 4px;font-size:15px}}
  #panel p  {{margin:0;font-size:12px;color:#555}}
  #legend {{
    position:fixed;bottom:28px;right:12px;z-index:9999;
    background:rgba(255,255,255,0.95);border-radius:8px;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);padding:10px 14px;min-width:140px;
  }}
  #legend h4 {{margin:0 0 6px;font-size:13px}}
</style>
</head>
<body>
<div id="panel">
  <h3>{title}</h3>
  <p>{summary_html}</p>
</div>
<div id="legend">
  <h4>Legend</h4>
  {legend_items}
</div>
<div id="map"></div>
<script>
const map = L.map("map");
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",{{
  attribution: "&copy; OpenStreetMap contributors",maxZoom:19
}}).addTo(map);

const routes   = {routes_json};
const segments = {segments_json};
const origin   = {origin_json};
const dest     = {dest_json};

// ── Draw jeepney routes (background, semi-transparent) ──────────────────
const routeGroup = L.layerGroup().addTo(map);
routes.forEach(r => {{
  if (r.coords.length < 2) return;
  L.polyline(r.coords, {{
    color: r.colour, weight: 4, opacity: 0.35, dashArray: "8 5"
  }}).bindPopup(`<b>Route ${{r.id}}</b>`).addTo(routeGroup);
}});

// ── Draw journey segments ────────────────────────────────────────────────
const journeyGroup = L.layerGroup().addTo(map);
segments.forEach(s => {{
  const weight = (s.edge_type === "ride") ? 6 : 4;
  const popup  = `<b>${{s.edge_type.replace(/_/g," ")}}</b><br/>` +
                 `ID: ${{s.edge_id}}<br/>dist: ${{s.dist}} m`;
  L.polyline(s.coords, {{
    color: s.colour, weight: weight, opacity: 0.9
  }}).bindPopup(popup).addTo(journeyGroup);
}});

// ── Origin / destination markers ────────────────────────────────────────
const greenIcon = L.divIcon({{className:"",html:
  '<div style="width:14px;height:14px;border-radius:50%;background:#2ca02c;' +
  'border:2px solid #fff;box-shadow:0 0 4px rgba(0,0,0,0.5)"></div>'
}});
const redIcon = L.divIcon({{className:"",html:
  '<div style="width:14px;height:14px;border-radius:50%;background:#d62728;' +
  'border:2px solid #fff;box-shadow:0 0 4px rgba(0,0,0,0.5)"></div>'
}});
L.marker(origin, {{icon:greenIcon}}).bindPopup("<b>Origin</b>").addTo(map);
L.marker(dest,   {{icon:redIcon  }}).bindPopup("<b>Destination</b>").addTo(map);

// ── Layer control ────────────────────────────────────────────────────────
L.control.layers(null, {{
  "Jeepney routes": routeGroup,
  "Journey path":   journeyGroup,
}}, {{collapsed:false, position:"topright"}}).addTo(map);

// ── Fit bounds to journey ────────────────────────────────────────────────
const allCoords = segments.flatMap(s => s.coords);
if (allCoords.length) map.fitBounds(allCoords, {{padding:[40,40]}});
</script>
</body>
</html>"""

        out_path.write_text(html, encoding="utf-8")
        print(f"Saved journey map → {out_path}")
        return out_path

    def __repr__(self) -> str:
        route_str = (
            f"{len(self._routes)} routes"
            if self._routes is not None else "full ride graph"
        )
        return (
            f"TravelGraphManager("
            f"{route_str}, "
            f"nodes={self._graph.number_of_nodes():,}, "
            f"edges={self._graph.number_of_edges():,})"
        )
