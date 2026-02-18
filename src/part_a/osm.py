from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from pyproj import CRS


def _load_or_create_boundary(place_query: str, boundary_path: Path) -> gpd.GeoDataFrame:
    boundary_path.parent.mkdir(parents=True, exist_ok=True)

    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path)
        if gdf.empty:
            raise ValueError(f"Boundary file exists but is empty: {boundary_path}")
        return gdf

    gdf = ox.geocode_to_gdf(place_query)
    if gdf.empty:
        raise ValueError(f"Nominatim returned empty boundary for: {place_query}")

    gdf.to_file(boundary_path, driver="GeoJSON")
    return gdf


def _fix_polygon(boundary_gdf: gpd.GeoDataFrame) -> Polygon:
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

    geom = boundary_gdf.geometry.unary_union

    if isinstance(geom, MultiPolygon):
        geom = max(list(geom.geoms), key=lambda g: g.area)

    geom = geom.buffer(0)  # fix many invalid geometries

    if geom.is_empty or not isinstance(geom, Polygon):
        raise ValueError(f"Invalid boundary geometry: {geom.geom_type}")

    return geom


def _utm_crs_from_latlon(lat: float, lon: float) -> CRS:
    # UTM zone: 1..60
    zone = int((lon + 180) // 6) + 1
    # EPSG: 326## for Northern hemisphere, 327## for Southern
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _area_km2(polygon: Polygon) -> float:
    lat = float(polygon.centroid.y)
    lon = float(polygon.centroid.x)
    utm_crs = _utm_crs_from_latlon(lat, lon)

    g = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(utm_crs)
    return float(g.area.iloc[0]) / 1e6


def _buffer_polygon_m(polygon: Polygon, meters: float) -> Polygon:
    lat = float(polygon.centroid.y)
    lon = float(polygon.centroid.x)
    utm_crs = _utm_crs_from_latlon(lat, lon)

    g = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(utm_crs)
    g = g.buffer(meters)
    return g.to_crs("EPSG:4326").iloc[0]


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


def _graph_from_polygon_safe(polygon: Polygon, network_type: str) -> nx.MultiDiGraph:
    return ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=False,
        truncate_by_edge=True,
        retain_all=True,
    )


def _graph_from_point_then_truncate(polygon: Polygon, network_type: str, dist_m: float) -> nx.MultiDiGraph:
    center = (polygon.centroid.y, polygon.centroid.x)  # (lat, lon)
    G = ox.graph_from_point(center, dist=dist_m, network_type=network_type, simplify=False)
    G = ox.truncate.truncate_graph_polygon(G, polygon, truncate_by_edge=True)
    return G


def build_iligan_graphs(cfg: dict) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph, gpd.GeoDataFrame]:
    # Cache settings
    cache_dir = Path(cfg.get("paths", {}).get("osmnx_cache_dir", "data/raw/osm_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_dir)

    ox.settings.use_cache = bool(cfg.get("osmnx", {}).get("use_cache", True))
    ox.settings.log_console = bool(cfg.get("osmnx", {}).get("log_console", True))

    place_query = str(cfg["place_query"])
    boundary_path = Path(cfg["paths"]["boundary_geojson"])

    boundary_gdf = _load_or_create_boundary(place_query, boundary_path)
    polygon = _fix_polygon(boundary_gdf)

    # Boundary robustness knobs (from config)
    min_area_km2 = float(cfg.get("boundary", {}).get("min_area_km2", 20.0))
    buffer_m = float(cfg.get("boundary", {}).get("buffer_m", 50.0))
    fallback_dist = float(cfg.get("fallback", {}).get("dist_m", 25000))

    # If the boundary is suspiciously small, expand it (common Nominatim issue)
    if _area_km2(polygon) < min_area_km2:
        polygon = _buffer_polygon_m(polygon, meters=fallback_dist)

    # Small tolerance buffer helps avoid edge truncation problems
    polygon = _buffer_polygon_m(polygon, meters=buffer_m)

    walk_type = str(cfg["network"]["walk_type"])
    drive_type = str(cfg["network"]["drive_type"])

    # Build graphs with fallback
    try:
        walk_G = _graph_from_polygon_safe(polygon, network_type=walk_type)
    except ValueError:
        walk_G = _graph_from_point_then_truncate(polygon, network_type=walk_type, dist_m=fallback_dist)

    try:
        drive_G = _graph_from_polygon_safe(polygon, network_type=drive_type)
    except ValueError:
        drive_G = _graph_from_point_then_truncate(polygon, network_type=drive_type, dist_m=fallback_dist)

    # Clean graphs
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