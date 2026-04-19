"""Traffic-biased baseline route generation on the physical street network.

The minimum-area quadrilateral filter follows the route morphology idea cited at
https://arxiv.org/html/2603.28385v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .passenger_generation import PassengerMap
from .travel_graph import load_graphs_for_study_area, make_coord_key, node_table_from_graph

try:
    import folium
except ImportError:  # pragma: no cover - optional visualisation dependency
    folium = None

_ROUTE_COLOURS = [
    "#ef4444",
    "#3b82f6",
    "#22c55e",
    "#a855f7",
    "#f97316",
    "#06b6d4",
    "#f59e0b",
    "#ec4899",
]


@dataclass(slots=True)
class BaselineRoute:
    """A single closed baseline route on the drivable street network."""

    route_id: str
    anchor_node_ids: tuple[int, int, int, int]
    anchor_latlon: list[tuple[float, float]]
    anchor_xy: list[tuple[float, float]]
    ordered_anchor_node_ids: tuple[int, int, int, int]
    path_node_ids: list[int]
    path_latlon: list[tuple[float, float]]
    polygon_area_m2: float
    path_length_m: float
    attempt_index: int

    @property
    def closed_anchor_latlon(self) -> list[tuple[float, float]]:
        return self.anchor_latlon + [self.anchor_latlon[0]]

    @property
    def closed_path_latlon(self) -> list[tuple[float, float]]:
        if not self.path_latlon:
            return []
        if self.path_latlon[0] == self.path_latlon[-1]:
            return list(self.path_latlon)
        return self.path_latlon + [self.path_latlon[0]]

    @property
    def nodes(self) -> list[int]:
        """Route node sequence compatible with the shared jeepney visualizer."""
        if not self.path_node_ids:
            return []
        if len(self.path_node_ids) > 1 and self.path_node_ids[0] == self.path_node_ids[-1]:
            return list(self.path_node_ids[:-1])
        return list(self.path_node_ids)


class BaselineRouteGenerator:
    """
    Generate traffic-biased closed baseline routes on the physical drive graph.

    The anchor sampler is driven by PassengerMap traffic weights (v_ped), while
    the geometry is validated on the projected drive network before shortest
    paths are stitched on the physical drivable street graph.
    """

    def __init__(
        self,
        passenger_map: PassengerMap | None = None,
        *,
        place_queries: Sequence[str] | None = None,
        point_query: str | None = None,
        point_dist: float = 30_000.0,
        min_area_m2: float = 2_000_000.0,
        anchor_pool_size: int = 64,
        max_attempts: int = 500,
        seed: int | None = None,
        drive_graph_raw: nx.MultiDiGraph | None = None,
        drive_graph_proj: nx.MultiDiGraph | None = None,
    ) -> None:
        self.passenger_map = passenger_map or PassengerMap()
        self.place_queries = list(place_queries) if place_queries is not None else [
            "Iligan City, Philippines"
        ]
        self.point_query = point_query
        self.point_dist = float(point_dist)
        self.min_area_m2 = float(min_area_m2)
        self.anchor_pool_size = max(int(anchor_pool_size), 4)
        self.max_attempts = max(int(max_attempts), 1)
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        if drive_graph_raw is None or drive_graph_proj is None:
            (
                self.study_area_gdf,
                self.boundary_source,
                self.graph_source,
                _walk_raw,
                _walk_proj,
                self.drive_graph_raw,
                self.drive_graph_proj,
            ) = load_graphs_for_study_area(
                self.place_queries,
                point_query=self.point_query,
                point_dist=self.point_dist,
            )
        else:
            self.study_area_gdf = None
            self.boundary_source = None
            self.graph_source = "preloaded drive graph"
            self.drive_graph_raw = drive_graph_raw
            self.drive_graph_proj = drive_graph_proj

        self.node_table = node_table_from_graph(self.drive_graph_raw, self.drive_graph_proj)
        self.node_table["base_node_id"] = self.node_table["base_node_id"].astype(int)
        self.node_table["coord_key"] = make_coord_key(self.node_table, "lon", "lat")
        self._node_table_by_base = self.node_table.set_index("base_node_id", drop=False)
        self._drive_node_ids = self.node_table["base_node_id"].to_numpy(dtype=int)
        self._drive_node_lats = self.node_table["lat"].to_numpy(dtype=float)
        self._drive_node_lons = self.node_table["lon"].to_numpy(dtype=float)
        self._coord_by_node_id = {
            int(row.base_node_id): (float(row.lat), float(row.lon), float(row.x), float(row.y))
            for row in self.node_table.itertuples(index=False)
        }
        self._lat_by_node_id = {node_id: coords[0] for node_id, coords in self._coord_by_node_id.items()}
        self._lon_by_node_id = {node_id: coords[1] for node_id, coords in self._coord_by_node_id.items()}
        self._x_by_node_id = {node_id: coords[2] for node_id, coords in self._coord_by_node_id.items()}
        self._y_by_node_id = {node_id: coords[3] for node_id, coords in self._coord_by_node_id.items()}

    def _random_state(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, np.iinfo(np.int32).max))

    def _snap_passenger_samples(self, sampled: pd.DataFrame) -> pd.DataFrame:
        if sampled.empty:
            raise ValueError("PassengerMap returned no nodes to sample from.")

        sample_lats = sampled["lat"].to_numpy(dtype=float)[:, None]
        sample_lons = sampled["lon"].to_numpy(dtype=float)[:, None]
        distances = (sample_lats - self._drive_node_lats[None, :]) ** 2 + (
            sample_lons - self._drive_node_lons[None, :]
        ) ** 2
        nearest_idx = np.argmin(distances, axis=1)

        snapped = sampled.copy()
        snapped["anchor_node_id"] = self._drive_node_ids[nearest_idx]
        snapped["anchor_lat"] = snapped["anchor_node_id"].map(self._lat_by_node_id)
        snapped["anchor_lon"] = snapped["anchor_node_id"].map(self._lon_by_node_id)
        snapped["anchor_x"] = snapped["anchor_node_id"].map(self._x_by_node_id)
        snapped["anchor_y"] = snapped["anchor_node_id"].map(self._y_by_node_id)

        grouped = (
            snapped.groupby("anchor_node_id", as_index=False)
            .agg(
                v_ped=("v_ped", "sum"),
                sample_count=("anchor_node_id", "size"),
                traffic_base_osmid=("base_osmid", "first"),
                passenger_lat=("lat", "mean"),
                passenger_lon=("lon", "mean"),
                anchor_lat=("anchor_lat", "first"),
                anchor_lon=("anchor_lon", "first"),
                anchor_x=("anchor_x", "first"),
                anchor_y=("anchor_y", "first"),
            )
            .sort_values("v_ped", ascending=False)
            .reset_index(drop=True)
        )
        grouped["coord_key"] = make_coord_key(grouped, "anchor_lon", "anchor_lat")
        return grouped

    def _sample_anchor_candidates(self, rng: np.random.Generator) -> pd.DataFrame:
        sampled = self.passenger_map.generate_nodes(
            n_points=self.anchor_pool_size,
            random_state=self._random_state(rng),
        )
        sampled["base_osmid"] = sampled["base_osmid"].astype(int)
        sampled["lat"] = sampled["lat"].astype(float)
        sampled["lon"] = sampled["lon"].astype(float)
        candidates = self._snap_passenger_samples(sampled)
        if len(candidates) >= 4:
            return candidates

        fallback = self.passenger_map.df.copy()
        fallback["base_osmid"] = fallback["base_osmid"].astype(int)
        fallback["lat"] = fallback["lat"].astype(float)
        fallback["lon"] = fallback["lon"].astype(float)
        candidates = self._snap_passenger_samples(fallback)
        return candidates

    def _pick_anchor_nodes(self, rng: np.random.Generator) -> pd.DataFrame:
        candidates = self._sample_anchor_candidates(rng)
        if len(candidates) == 4:
            selected = candidates.copy()
        else:
            weights = candidates["v_ped"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            chosen_idx = rng.choice(candidates.index.to_numpy(), size=4, replace=False, p=weights)
            selected = candidates.loc[chosen_idx].copy()
        return selected.reset_index(drop=True)

    def _anchor_metadata(self, anchors: pd.DataFrame) -> list[dict]:
        rows: list[dict] = []
        for row in anchors.itertuples(index=False):
            node_id = int(row.anchor_node_id)
            lat, lon, x, y = self._coord_by_node_id[node_id]
            rows.append(
                {
                    "node_id": node_id,
                    "lat": lat,
                    "lon": lon,
                    "x": x,
                    "y": y,
                    "coord_key": str(row.coord_key),
                    "v_ped": float(getattr(row, "v_ped", np.nan)),
                    "traffic_base_osmid": int(row.traffic_base_osmid),
                    "sample_count": int(row.sample_count),
                    "passenger_lat": float(row.passenger_lat),
                    "passenger_lon": float(row.passenger_lon),
                }
            )
        return rows

    @staticmethod
    def _polygon_simple_and_area(points_xy: list[tuple[float, float]]) -> tuple[bool, float]:
        polygon = Polygon(points_xy)
        return bool(polygon.is_valid and not polygon.is_empty and polygon.area > 0.0), float(polygon.area)

    def _order_anchors(self, anchors: list[dict]) -> tuple[list[dict], Polygon]:
        best_polygon: Polygon | None = None
        best_order: list[dict] | None = None

        for perm in permutations(anchors):
            points_xy = [(item["x"], item["y"]) for item in perm]
            polygon = Polygon(points_xy)
            if not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0:
                continue
            if best_polygon is None or polygon.area > best_polygon.area:
                best_polygon = polygon
                best_order = list(perm)

        if best_polygon is None or best_order is None:
            raise ValueError("The sampled anchors did not form a valid simple quadrilateral.")
        return best_order, best_polygon

    def _stitch_path(self, ordered_anchors: list[dict]) -> tuple[list[int], float]:
        path_nodes: list[int] = [int(ordered_anchors[0]["node_id"])]
        total_length = 0.0

        for current_anchor, next_anchor in zip(ordered_anchors, ordered_anchors[1:] + ordered_anchors[:1]):
            start = int(current_anchor["node_id"])
            end = int(next_anchor["node_id"])
            segment_length, segment_nodes = nx.bidirectional_dijkstra(self.drive_graph_raw, start, end, weight="length")
            if path_nodes and segment_nodes and path_nodes[-1] == segment_nodes[0]:
                segment_nodes = segment_nodes[1:]
            path_nodes.extend(int(node_id) for node_id in segment_nodes)
            total_length += segment_length

        return path_nodes, total_length

    def _path_latlon(self, path_nodes: Sequence[int]) -> list[tuple[float, float]]:
        return [(self._lat_by_node_id[int(node_id)], self._lon_by_node_id[int(node_id)]) for node_id in path_nodes]

    def generate_route(self, *, route_id: str = "B01", seed: int | None = None) -> BaselineRoute:
        rng = np.random.default_rng(self._seed if seed is None else seed)

        for attempt_index in range(1, self.max_attempts + 1):
            anchors_df = self._pick_anchor_nodes(rng)
            anchors = self._anchor_metadata(anchors_df)
            ordered_anchors, polygon = self._order_anchors(anchors)
            if polygon.area < self.min_area_m2:
                continue

            try:
                path_nodes, path_length = self._stitch_path(ordered_anchors)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path_nodes) < 4:
                continue

            anchor_node_ids = tuple(item["node_id"] for item in anchors)
            ordered_anchor_node_ids = tuple(item["node_id"] for item in ordered_anchors)
            anchor_latlon = [(item["lat"], item["lon"]) for item in ordered_anchors]
            anchor_xy = [(item["x"], item["y"]) for item in ordered_anchors]
            path_latlon = self._path_latlon(path_nodes)

            return BaselineRoute(
                route_id=route_id,
                anchor_node_ids=anchor_node_ids,
                anchor_latlon=anchor_latlon,
                anchor_xy=anchor_xy,
                ordered_anchor_node_ids=ordered_anchor_node_ids,
                path_node_ids=path_nodes,
                path_latlon=path_latlon,
                polygon_area_m2=float(polygon.area),
                path_length_m=float(path_length),
                attempt_index=attempt_index,
            )

        raise ValueError(
            f"Could not generate a quadrilateral route with area >= {self.min_area_m2:.0f} m² "
            f"after {self.max_attempts} attempts."
        )

    def generate_routes(
        self,
        count: int,
        *,
        route_prefix: str = "B",
        seed: int | None = None,
    ) -> list[BaselineRoute]:
        if count < 1:
            raise ValueError("count must be at least 1.")

        routes: list[BaselineRoute] = []
        seen_signatures: set[tuple[str, ...]] = set()
        base_rng = np.random.default_rng(self._seed if seed is None else seed)

        for index in range(count):
            retry = 0
            while True:
                route_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
                route = self.generate_route(route_id=f"{route_prefix}{index + 1:02d}", seed=route_seed)
                signature = tuple(sorted(int(node_id) for node_id in route.anchor_node_ids))
                if signature not in seen_signatures:
                    break
                retry += 1
                if retry >= self.max_attempts:
                    raise ValueError("Could not generate a unique baseline route after repeated attempts.")
            seen_signatures.add(signature)
            routes.append(route)
        return routes

    def export_route_html(
        self,
        route: BaselineRoute,
        output_html: str | Path,
        *,
        title: str | None = None,
    ) -> Path:
        if folium is None:  # pragma: no cover - optional visualisation dependency
            raise ImportError("folium is required to export baseline route HTML maps.")

        out = Path(output_html).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        if route.path_latlon:
            centre = route.path_latlon[len(route.path_latlon) // 2]
        else:
            centre = route.anchor_latlon[0]

        fmap = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap")
        folium.PolyLine(
            route.path_latlon,
            color="#2563eb",
            weight=5,
            opacity=0.9,
            tooltip=f"{route.route_id} route path",
        ).add_to(fmap)
        folium.PolyLine(
            route.closed_anchor_latlon,
            color="#dc2626",
            weight=2,
            opacity=0.75,
            dash_array="6,6",
            tooltip="Anchor quadrilateral",
        ).add_to(fmap)

        for idx, (lat, lon) in enumerate(route.anchor_latlon, start=1):
            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="#b91c1c",
                fill=True,
                fill_color="#fca5a5",
                fill_opacity=0.95,
                popup=folium.Popup(
                    f"Anchor {idx}<br>Node: {route.ordered_anchor_node_ids[idx - 1]}<br>"
                    f"Area: {route.polygon_area_m2:,.0f} m²<br>Path length: {route.path_length_m:,.0f} m",
                    max_width=260,
                ),
            ).add_to(fmap)

        lat_values = [lat for lat, _ in route.path_latlon] + [lat for lat, _ in route.anchor_latlon]
        lon_values = [lon for _, lon in route.path_latlon] + [lon for _, lon in route.anchor_latlon]
        fmap.fit_bounds([[min(lat_values), min(lon_values)], [max(lat_values), max(lon_values)]])

        header = title or f"Baseline Route {route.route_id}"
        root = folium.Element(
            f"""
            <div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
                        background: white; padding: 10px 12px; border: 1px solid #cbd5e1;
                        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.12);
                        font-family: Arial, sans-serif; font-size: 13px;">
              <div style="font-weight: 700; margin-bottom: 4px;">{header}</div>
              <div>Anchors: {len(route.anchor_node_ids)} | Area: {route.polygon_area_m2:,.0f} m²</div>
              <div>Length: {route.path_length_m:,.0f} m | Attempts: {route.attempt_index}</div>
            </div>
            """
        )
        fmap.get_root().html.add_child(root)
        fmap.save(str(out))
        return out

    def export_routes_html(
        self,
        routes: Sequence[BaselineRoute],
        output_html: str | Path,
        *,
        title: str = "Baseline Route Explorer",
    ) -> Path:
        if folium is None:  # pragma: no cover - optional visualisation dependency
            raise ImportError("folium is required to export baseline route HTML maps.")

        routes = list(routes)
        if not routes:
            raise ValueError("routes must contain at least one BaselineRoute.")

        all_lat = [lat for route in routes for lat, _ in route.path_latlon]
        all_lon = [lon for route in routes for _, lon in route.path_latlon]
        all_lat.extend(lat for route in routes for lat, _ in route.anchor_latlon)
        all_lon.extend(lon for route in routes for _, lon in route.anchor_latlon)

        centre = (
            float(sum(all_lat) / len(all_lat)),
            float(sum(all_lon) / len(all_lon)),
        )
        fmap = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap")

        for idx, route in enumerate(routes):
            colour = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
            group = folium.FeatureGroup(name=route.route_id, show=True)
            folium.PolyLine(
                route.closed_path_latlon,
                color=colour,
                weight=5,
                opacity=0.9,
                tooltip=f"{route.route_id} route",
            ).add_to(group)
            folium.PolyLine(
                route.closed_anchor_latlon,
                color=colour,
                weight=2,
                opacity=0.7,
                dash_array="6,6",
                tooltip=f"{route.route_id} anchors",
            ).add_to(group)

            for anchor_idx, (lat, lon) in enumerate(route.anchor_latlon, start=1):
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color=colour,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.9,
                    popup=folium.Popup(
                        f"{route.route_id} · Anchor {anchor_idx}<br>"
                        f"Node: {route.ordered_anchor_node_ids[anchor_idx - 1]}<br>"
                        f"Area: {route.polygon_area_m2:,.0f} m²<br>"
                        f"Length: {route.path_length_m:,.0f} m",
                        max_width=260,
                    ),
                ).add_to(group)

            group.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.fit_bounds([[min(all_lat), min(all_lon)], [max(all_lat), max(all_lon)]])

        route_rows = "".join(
            f"<tr><td>{route.route_id}</td><td>{route.polygon_area_m2:,.0f}</td>"
            f"<td>{route.path_length_m:,.0f}</td><td>{route.attempt_index}</td></tr>"
            for route in routes
        )
        header = folium.Element(
            f"""
            <div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
                        background: white; padding: 10px 12px; border: 1px solid #cbd5e1;
                        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.12);
                        font-family: Arial, sans-serif; font-size: 13px; max-width: 320px;">
              <div style="font-weight: 700; margin-bottom: 4px;">{title}</div>
              <div>Routes: {len(routes)} | Minimum area: {self.min_area_m2:,.0f} m²</div>
              <div style="margin-top: 6px;">
                <table style="border-collapse: collapse; width: 100%; font-size: 12px;">
                  <thead>
                    <tr><th align="left">Route</th><th align="left">Area</th><th align="left">Length</th><th align="left">Try</th></tr>
                  </thead>
                  <tbody>{route_rows}</tbody>
                </table>
              </div>
            </div>
            """
        )
        fmap.get_root().html.add_child(header)

        out = Path(output_html).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out))
        return out
