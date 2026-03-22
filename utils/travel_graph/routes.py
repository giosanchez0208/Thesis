from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import networkx as nx
import osmnx as ox


@dataclass(frozen=True)
class RoutePoint:
    base_osmid: int
    lat: float
    lon: float
    point_type: str  # stop | terminal | waypoint


@dataclass
class RouteDef:
    route_id: str
    name: str
    directed_cycle: bool
    points: List[RoutePoint]


def _snap_latlon_to_walk_node(walk_G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    # OSMnx nearest node by x=lon, y=lat
    return int(ox.distance.nearest_nodes(walk_G, X=lon, Y=lat))


def _get_node_latlon(G: nx.MultiDiGraph, osmid: int) -> Tuple[float, float]:
    data = G.nodes[osmid]
    return float(data["y"]), float(data["x"])  # lat, lon


def load_routes_and_snap(
    routes_json_path: Path,
    walk_G: nx.MultiDiGraph,
    snap_to_walk: bool = True,
) -> List[RouteDef]:
    if not routes_json_path.exists():
        raise FileNotFoundError(
            f"Routes file not found: {routes_json_path}\n"
            "Create it using the provided template: data/raw/routes/jeepney_routes.json"
        )

    raw = json.loads(routes_json_path.read_text(encoding="utf-8"))
    routes_raw = raw.get("routes", [])
    if not routes_raw:
        raise ValueError("Routes JSON has no 'routes' entries.")

    routes: List[RouteDef] = []

    for r in routes_raw:
        route_id = str(r["route_id"])
        name = str(r.get("name", route_id))
        directed_cycle = bool(r.get("directed_cycle", True))
        points_raw = r.get("points", [])
        if len(points_raw) < 2:
            raise ValueError(f"Route {route_id} must have at least 2 points.")

        points: List[RoutePoint] = []
        for p in points_raw:
            ptype = str(p.get("type", "stop")).lower()

            osmid: Optional[int] = None
            lat: Optional[float] = None
            lon: Optional[float] = None

            if "osmid" in p:
                osmid = int(p["osmid"])
                if osmid not in walk_G.nodes:
                    # If the osmid isn't in the simplified graph, snap by coords if given
                    if "lat" in p and "lon" in p and snap_to_walk:
                        osmid = _snap_latlon_to_walk_node(walk_G, float(p["lat"]), float(p["lon"]))
                    else:
                        raise ValueError(
                            f"Point osmid {p['osmid']} not found in walk graph for route {route_id}. "
                            "Provide lat/lon or ensure it exists in the extracted network."
                        )
                lat, lon = _get_node_latlon(walk_G, osmid)

            else:
                if "lat" not in p or "lon" not in p:
                    raise ValueError(f"Point in route {route_id} must include either osmid or lat/lon.")
                lat = float(p["lat"])
                lon = float(p["lon"])
                osmid = _snap_latlon_to_walk_node(walk_G, lat, lon) if snap_to_walk else None
                if osmid is None:
                    raise ValueError("snap_to_walk is False but point has no osmid.")
                lat, lon = _get_node_latlon(walk_G, osmid)

            points.append(RoutePoint(base_osmid=osmid, lat=lat, lon=lon, point_type=ptype))

        routes.append(RouteDef(route_id=route_id, name=name, directed_cycle=directed_cycle, points=points))

    return routes