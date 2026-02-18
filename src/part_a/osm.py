from __future__ import annotations

from pathlib import Path
from typing import Tuple

import osmnx as ox
import networkx as nx
import geopandas as gpd


def _load_or_create_boundary(place_query: str, boundary_path: Path) -> gpd.GeoDataFrame:
    boundary_path.parent.mkdir(parents=True, exist_ok=True)

    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path)
        if gdf.empty:
            raise ValueError(f"Boundary file exists but is empty: {boundary_path}")
        return gdf

    gdf = ox.geocode_to_gdf(place_query)
    gdf.to_file(boundary_path, driver="GeoJSON")
    return gdf


def _largest_weakly_connected_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if len(G) == 0:
        return G
    UG = ox.convert.to_undirected(G)
    largest = max(nx.connected_components(UG), key=len)
    return G.subgraph(largest).copy()


def clean_graph(
    G: nx.MultiDiGraph,
    simplify: bool,
    consolidate_intersections: bool,
    tolerance_m: float,
    keep_largest_component: bool,
) -> nx.MultiDiGraph:
    if simplify:
        G = ox.simplify_graph(G)

    if consolidate_intersections:
        G_proj = ox.project_graph(G)
        G_proj = ox.consolidate_intersections(
            G_proj,
            tolerance=tolerance_m,
            rebuild_graph=True,
            dead_ends=False,
        )
        G = ox.project_graph(G_proj, to_crs="EPSG:4326")

    if keep_largest_component:
        G = _largest_weakly_connected_component(G)

    G = ox.distance.add_edge_lengths(G)
    return G


def build_iligan_graphs(cfg: dict) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph, gpd.GeoDataFrame]:
    # ✅ Cache config (set here, not at import time)
    cache_dir = Path(cfg.get("paths", {}).get("osmnx_cache_dir", "data/raw/osm_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_dir)

    ox.settings.use_cache = bool(cfg["osmnx"].get("use_cache", True))
    ox.settings.log_console = bool(cfg["osmnx"].get("log_console", True))

    place_query = str(cfg["place_query"])
    boundary_path = Path(cfg["paths"]["boundary_geojson"])

    boundary_gdf = _load_or_create_boundary(place_query, boundary_path)
    polygon = boundary_gdf.geometry.iloc[0]

    walk_type = str(cfg["network"]["walk_type"])
    drive_type = str(cfg["network"]["drive_type"])

    walk_G = ox.graph_from_polygon(polygon, network_type=walk_type, simplify=False)
    drive_G = ox.graph_from_polygon(polygon, network_type=drive_type, simplify=False)

    c = cfg["cleaning"]
    walk_G = clean_graph(
        walk_G,
        simplify=bool(c.get("simplify", True)),
        consolidate_intersections=bool(c.get("consolidate_intersections", True)),
        tolerance_m=float(c.get("tolerance_m", 8.0)),
        keep_largest_component=bool(c.get("keep_largest_component", True)),
    )
    drive_G = clean_graph(
        drive_G,
        simplify=bool(c.get("simplify", True)),
        consolidate_intersections=bool(c.get("consolidate_intersections", True)),
        tolerance_m=float(c.get("tolerance_m", 8.0)),
        keep_largest_component=bool(c.get("keep_largest_component", True)),
    )

    return walk_G, drive_G, boundary_gdf