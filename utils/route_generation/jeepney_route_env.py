"""Gymnasium-style RL environment for geometric jeepney route construction.

The environment operates only on the primal physical street network. The
three-layer travel graph remains reserved for downstream evaluation of the
final route under generalized travel cost.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import cos, hypot, pi, sin
from pathlib import Path as _Path
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Polygon

from ..passenger_generation import Passenger, PassengerMap, Simulation, SimulationConfig
from ..travel_graph import JeepneyRoute, TravelGraphManager, load_graphs_for_study_area, make_coord_key, node_table_from_graph

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - local compatibility shim
    from types import SimpleNamespace

    class _BaseSpace:
        def sample(self):  # noqa: D401 - simple fallback helper
            raise NotImplementedError

    class _Discrete(_BaseSpace):
        def __init__(self, n: int):
            self.n = int(n)

        def sample(self):
            return int(np.random.default_rng().integers(self.n))

    class _Box(_BaseSpace):
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            rng = np.random.default_rng()
            return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    class _Dict(_BaseSpace):
        def __init__(self, spaces_dict):
            self.spaces = dict(spaces_dict)

        def sample(self):
            return {key: space.sample() for key, space in self.spaces.items()}

    class _Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None

    gym = SimpleNamespace(Env=_Env)
    spaces = SimpleNamespace(Discrete=_Discrete, Box=_Box, Dict=_Dict)


_REPO_ROOT = _Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "travel_graph_config.yaml"
_DEFAULT_EDGES_CSV = _REPO_ROOT / "data" / "iligan_travel_graph.csv"
_DEFAULT_NODES_CSV = _REPO_ROOT / "data" / "travel_graph_nodes.csv"
_DEFAULT_WEIGHT_PROFILE = "full_ride_manager"
_DEFAULT_UNSERVED_PENALTY_BETA = 2.0

__all__ = ["RouteFitnessResult", "calculate_route_fitness", "JeepneyRouteEnv"]

_PHYSICAL_TO_RIDE_NODE_MAP_CACHE: dict[tuple[int, int, str], dict[int, str]] = {}


@dataclass(slots=True)
class RouteFitnessResult:
    """Summary of passenger-based route fitness."""

    reward: float
    average_gtc: float
    passenger_gtc_std: float
    total_gtc: float
    passenger_count: int
    served_passenger_count: int
    unserved_passenger_count: int
    unserved_penalty_beta: float
    route_node_count: int
    route_edge_count: int
    batch_size: int
    seed: int | None

    def __float__(self) -> float:
        return float(self.reward)


def _coerce_node_id(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return int(token)
        if "_" in token:
            tail = token.rsplit("_", 1)[-1]
            if tail.isdigit() or (tail.startswith("-") and tail[1:].isdigit()):
                return int(tail)
    return None


def _extract_physical_path_nodes(generated_path: Any) -> list[int]:
    if generated_path is None:
        raise ValueError("generated_path cannot be None.")

    if hasattr(generated_path, "path_node_ids"):
        raw_nodes = list(getattr(generated_path, "path_node_ids"))
    elif hasattr(generated_path, "nodes") and not isinstance(generated_path, (str, bytes)):
        raw_nodes = list(getattr(generated_path, "nodes"))
    elif isinstance(generated_path, Sequence) and not isinstance(generated_path, (str, bytes)):
        raw_nodes = list(generated_path)
    else:
        raise TypeError(
            "generated_path must be a route object or a sequence of physical node IDs."
        )

    nodes: list[int] = []
    for item in raw_nodes:
        node_id = _coerce_node_id(item)
        if node_id is None:
            raise TypeError(f"Unsupported node identifier in generated_path: {item!r}")
        nodes.append(node_id)

    if len(nodes) < 2:
        raise ValueError("generated_path needs at least two nodes.")
    return nodes


def _stitch_physical_loop(
    node_ids: Sequence[int],
    drive_graph_raw: nx.MultiDiGraph,
) -> list[int]:
    loop_nodes = list(node_ids)
    if loop_nodes[0] != loop_nodes[-1]:
        loop_nodes.append(loop_nodes[0])

    stitched: list[int] = [int(loop_nodes[0])]
    for start, end in zip(loop_nodes[:-1], loop_nodes[1:]):
        start = int(start)
        end = int(end)
        if start == end:
            continue
        if drive_graph_raw.has_edge(start, end):
            segment_nodes = [start, end]
        else:
            _, segment_nodes = nx.bidirectional_dijkstra(drive_graph_raw, start, end, weight="length")
        segment_nodes = [int(node_id) for node_id in segment_nodes]
        if stitched[-1] == segment_nodes[0]:
            stitched.extend(segment_nodes[1:])
        else:
            stitched.extend(segment_nodes)
    return stitched


def _physical_path_to_route_nodes(
    generated_path: Any,
    drive_graph_raw: nx.MultiDiGraph,
) -> list[int]:
    physical_nodes = _extract_physical_path_nodes(generated_path)
    if all(drive_graph_raw.has_edge(u, v) for u, v in zip(physical_nodes[:-1], physical_nodes[1:])):
        return physical_nodes
    return _stitch_physical_loop(physical_nodes, drive_graph_raw)


def _repair_minimal_closed_loop(
    node_ids: Sequence[int],
    drive_graph_raw: nx.MultiDiGraph,
) -> list[int]:
    anchor_candidates = [int(node_id) for node_id in node_ids if node_id is not None]
    if not anchor_candidates:
        raise ValueError("generated_path needs at least one physical node.")

    anchor = int(anchor_candidates[0])
    outgoing = list(drive_graph_raw.out_edges(anchor, data=True))
    if not outgoing:
        raise ValueError(f"Route anchor {anchor!r} has no outgoing edges to build a closed loop.")

    next_node = min(
        outgoing,
        key=lambda edge: float(edge[2].get("length", 0.0)),
    )[1]
    return [anchor, int(next_node), anchor]


def _build_physical_to_ride_node_map(
    drive_graph_raw: nx.MultiDiGraph,
    drive_graph_proj: nx.MultiDiGraph,
    travel_nodes_df: pd.DataFrame,
) -> dict[int, str]:
    physical_nodes = node_table_from_graph(drive_graph_raw, drive_graph_proj)
    physical_nodes["base_node_id"] = physical_nodes["base_node_id"].astype(int)
    physical_nodes["coord_key"] = make_coord_key(physical_nodes, "lon", "lat")

    ride_nodes = travel_nodes_df.copy()
    if "layer" in ride_nodes.columns:
        ride_nodes = ride_nodes[ride_nodes["layer"].astype(str) == "ride"].copy()
    if ride_nodes.empty:
        raise ValueError("The travel graph has no ride-layer nodes to map against.")

    ride_nodes["base_node_id"] = ride_nodes["base_node_id"].astype(int) if "base_node_id" in ride_nodes.columns else ride_nodes["node_id"].astype(str)
    if "node_id" not in ride_nodes.columns:
        ride_nodes["node_id"] = ride_nodes["base_node_id"].astype(str)
    ride_nodes["coord_key"] = make_coord_key(ride_nodes, "lon", "lat")

    merged = physical_nodes[["base_node_id", "coord_key"]].merge(
        ride_nodes[["node_id", "coord_key"]],
        on="coord_key",
        how="inner",
    )
    node_map = {
        int(row.base_node_id): str(row.node_id)
        for row in merged.itertuples(index=False)
    }

    missing = physical_nodes.loc[~physical_nodes["base_node_id"].isin(node_map.keys())].copy()
    if not missing.empty:
        ride_xy = ride_nodes[["lon", "lat"]].to_numpy(dtype=float)
        ride_ids = ride_nodes["node_id"].astype(str).to_numpy()
        for row in missing.itertuples(index=False):
            source_xy = np.asarray([float(row.lon), float(row.lat)], dtype=float)
            deltas = ride_xy - source_xy
            nearest_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
            node_map[int(row.base_node_id)] = str(ride_ids[nearest_idx])

    if not node_map:
        raise ValueError("Could not map the generated physical route onto the ride layer.")

    return node_map


def _physical_nodes_to_ride_route(
    route_nodes: Sequence[int],
    route_id: str,
    physical_to_ride_node_map: dict[int, str],
) -> JeepneyRoute:
    ride_nodes: list[str] = []
    missing_nodes: list[int] = []
    for node_id in route_nodes:
        ride_node_id = physical_to_ride_node_map.get(int(node_id))
        if ride_node_id is None:
            missing_nodes.append(int(node_id))
            continue
        ride_nodes.append(ride_node_id)

    if missing_nodes:
        raise ValueError(
            "Could not map some physical route nodes onto the ride layer: "
            f"{missing_nodes[:5]}"
        )

    if len(ride_nodes) > 1 and ride_nodes[0] == ride_nodes[-1]:
        ride_nodes = ride_nodes[:-1]
    return JeepneyRoute(route_id, ride_nodes)


def _coerce_route_like(
    route_like: Any,
    route_id: str,
    drive_graph_raw: nx.MultiDiGraph,
    drive_graph_proj: nx.MultiDiGraph,
    physical_to_ride_node_map: dict[int, str],
) -> JeepneyRoute:
    if isinstance(route_like, JeepneyRoute):
        return route_like

    if hasattr(route_like, "path_node_ids"):
        return _physical_nodes_to_ride_route(
            _physical_path_to_route_nodes(route_like, drive_graph_raw),
            route_id=str(getattr(route_like, "route_id", route_id)),
            physical_to_ride_node_map=physical_to_ride_node_map,
        )

    if hasattr(route_like, "nodes"):
        nodes = list(getattr(route_like, "nodes"))
        if nodes and all(str(node).startswith("ride_") for node in nodes):
            return JeepneyRoute(str(getattr(route_like, "route_id", route_id)), nodes)
        if nodes:
            return _physical_nodes_to_ride_route(
                _physical_path_to_route_nodes(nodes, drive_graph_raw),
                route_id=str(getattr(route_like, "route_id", route_id)),
                physical_to_ride_node_map=physical_to_ride_node_map,
            )

    if isinstance(route_like, Sequence) and not isinstance(route_like, (str, bytes)):
        items = list(route_like)
        if items and all(str(item).startswith("ride_") for item in items):
            return JeepneyRoute(route_id, [str(item) for item in items])
        return _physical_nodes_to_ride_route(
            _physical_path_to_route_nodes(items, drive_graph_raw),
            route_id=route_id,
            physical_to_ride_node_map=physical_to_ride_node_map,
        )

    raise TypeError(f"Unsupported route object: {type(route_like).__name__}")


def _normalize_background_routes(
    background_routes: Sequence[Any] | None,
    drive_graph_raw: nx.MultiDiGraph,
    drive_graph_proj: nx.MultiDiGraph,
    physical_to_ride_node_map: dict[int, str],
) -> list[JeepneyRoute]:
    routes: list[JeepneyRoute] = []
    if not background_routes:
        return routes
    for index, route_like in enumerate(background_routes, start=1):
        routes.append(
            _coerce_route_like(
                route_like,
                f"BG{index:02d}",
                drive_graph_raw,
                drive_graph_proj,
                physical_to_ride_node_map,
            )
        )
    return routes


@lru_cache(maxsize=8)
def _default_simulation_config(
    config_path: str | _Path | None = None,
    weight_profile: str = _DEFAULT_WEIGHT_PROFILE,
) -> SimulationConfig:
    return SimulationConfig.from_yaml(
        _Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH,
        weight_profile=weight_profile,
    )


@lru_cache(maxsize=8)
def _default_route_constraints(config_path: str | _Path | None = None) -> tuple[int, int | None]:
    config = _default_simulation_config(config_path)
    raw = getattr(config, "raw", {}) or {}
    route_cfg = raw.get("route_cfg")
    if not isinstance(route_cfg, dict):
        route_cfg = raw.get("route_generation") or {}
    min_nodes = max(int(route_cfg.get("min_nodes", 6)), 2)
    max_value = route_cfg.get("max_nodes", None)
    if max_value is None:
        max_nodes = None
    else:
        max_nodes = max(int(max_value), min_nodes)
    return min_nodes, max_nodes


def _graph_path_key(path_value: str | _Path | None, default_path: _Path) -> str:
    return str(_Path(path_value)) if path_value is not None else str(default_path)


@lru_cache(maxsize=8)
def _load_fitness_graph_frames(
    edges_csv: str | _Path | None = None,
    nodes_csv: str | _Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges_path = _Path(edges_csv) if edges_csv is not None else _DEFAULT_EDGES_CSV
    nodes_path = _Path(nodes_csv) if nodes_csv is not None else _DEFAULT_NODES_CSV
    edges_df = pd.read_csv(
        edges_path,
        dtype={"edge_id": str, "u": str, "v": str, "edge_type": str},
    )
    nodes_df = pd.read_csv(
        nodes_path,
        dtype={"node_id": str},
    )
    return edges_df, nodes_df


def _edge_weight_from_manager(manager: TravelGraphManager, edge_id: str) -> float:
    edge_weight = getattr(manager, "get_edge_weight", None)
    if callable(edge_weight):
        value = edge_weight(edge_id)
        if value is not None:
            return float(value)
    edge = manager.get_edge(edge_id)
    if edge is None:
        raise KeyError(f"Edge not found in travel graph manager: {edge_id}")
    return float(manager._edge_weight(edge.edge_type, float(edge.dist)))


def _baseline_manager_factory(
    *,
    edges_csv: str | _Path | None,
    nodes_csv: str | _Path | None,
    background_jeep_routes: Sequence[Any],
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    config: SimulationConfig,
) -> Callable[[], TravelGraphManager]:
    def _factory() -> TravelGraphManager:
        return TravelGraphManager(
            edges_csv or _DEFAULT_EDGES_CSV,
            nodes_csv or _DEFAULT_NODES_CSV,
            routes=background_jeep_routes or None,
            edges_df=edges_df,
            nodes_df=nodes_df,
            quiet=True,
            walk_wt=config.walk_wt,
            ride_wt=config.ride_wt,
            wait_wt=config.wait_wt,
            transfer_wt=config.transfer_wt,
        )

    return _factory


def _evaluate_passenger_batch(
    simulation: Simulation,
    passengers: Sequence[Passenger],
    *,
    baseline_manager: TravelGraphManager | None,
    baseline_manager_factory: Callable[[], TravelGraphManager] | None,
    unserved_penalty_beta: float,
) -> RouteFitnessResult:
    served_count = 0
    unserved_count = 0
    passenger_costs: list[float] = []

    manager = simulation.travel_graph_mgr

    for passenger in passengers:
        try:
            payload = simulation.prepare_passenger(passenger)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            payload = {"found": False}
        if not payload.get("found", True):
            unserved_count += 1
            start_graph_node_id = payload.get("start_graph_node") or simulation.find_nearest_node(
                passenger.start_lat, passenger.start_lon, layer="start_walk"
            )
            end_graph_node_id = payload.get("end_graph_node") or simulation.find_nearest_node(
                passenger.end_lat, passenger.end_lon, layer="end_walk"
            )
            elapsed_time = float(passenger.total_time)
            remaining_travel_time = 0.0
            if start_graph_node_id is not None and end_graph_node_id is not None:
                try:
                    if baseline_manager is None:
                        if baseline_manager_factory is None:
                            raise RuntimeError("baseline_manager_factory was not provided.")
                        baseline_manager = baseline_manager_factory()
                    fallback_edges = baseline_manager.calculate_shortest_path(
                        start_graph_node_id,
                        end_graph_node_id,
                    )
                    remaining_travel_time = sum(
                        _edge_weight_from_manager(baseline_manager, edge_id)
                        for edge_id in fallback_edges
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    remaining_travel_time = float(
                        np.hypot(
                            float(passenger.start_lat) - float(passenger.end_lat),
                            float(passenger.start_lon) - float(passenger.end_lon),
                        )
                        * 111_000.0
                    )
            passenger_costs.append(elapsed_time + (remaining_travel_time * float(unserved_penalty_beta)))
            continue

        path_edges = list(passenger.shortest_path_edges)
        path_cost = sum(_edge_weight_from_manager(manager, edge_id) for edge_id in path_edges)
        served_count += 1
        passenger_costs.append(path_cost)

    passenger_count = len(passengers)
    total_gtc = float(sum(passenger_costs))
    average_gtc = total_gtc / passenger_count if passenger_count else 0.0
    passenger_gtc_std = float(np.std(passenger_costs, ddof=0)) if len(passenger_costs) > 1 else 0.0
    return RouteFitnessResult(
        reward=-float(average_gtc),
        average_gtc=float(average_gtc),
        passenger_gtc_std=passenger_gtc_std,
        total_gtc=float(total_gtc),
        passenger_count=passenger_count,
        served_passenger_count=served_count,
        unserved_passenger_count=unserved_count,
        unserved_penalty_beta=float(unserved_penalty_beta),
        route_node_count=0,
        route_edge_count=0,
        batch_size=passenger_count,
        seed=None,
    )


def calculate_route_fitness(
    generated_path: Any,
    background_routes: Sequence[Any] | None = None,
    *,
    passenger_map: PassengerMap | None = None,
    drive_graph_raw: nx.MultiDiGraph | None = None,
    drive_graph_proj: nx.MultiDiGraph | None = None,
    config_path: str | _Path | None = None,
    edges_csv: str | _Path | None = None,
    nodes_csv: str | _Path | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    route_id: str = "FIT_ROUTE",
    weight_profile: str = _DEFAULT_WEIGHT_PROFILE,
    unserved_penalty_beta: float = _DEFAULT_UNSERVED_PENALTY_BETA,
) -> RouteFitnessResult:
    """
    Score a generated physical route against passenger generalized travel cost.

    The supplied physical path is inserted into the existing three-layer travel
    graph as a ride-layer route, then a stochastic passenger batch is sampled
    from the heatmap utilities and evaluated with Dijkstra shortest paths.
    """

    config = _default_simulation_config(config_path, weight_profile)
    edges_cache_key = _graph_path_key(edges_csv, _DEFAULT_EDGES_CSV)
    nodes_cache_key = _graph_path_key(nodes_csv, _DEFAULT_NODES_CSV)
    edges_df, nodes_df = _load_fitness_graph_frames(edges_cache_key, nodes_cache_key)
    if drive_graph_raw is None or drive_graph_proj is None:
        (
            _study_area_gdf,
            _boundary_source,
            _graph_source,
            drive_graph_raw,
            drive_graph_proj,
            _drive_graph_raw,
            _drive_graph_proj,
        ) = load_graphs_for_study_area(
            ["Iligan City, Philippines"],
            point_query="Iligan City, Philippines",
            point_dist=30_000.0,
        )

    map_cache_key = (id(drive_graph_raw), id(drive_graph_proj), nodes_cache_key)
    physical_to_ride_node_map = _PHYSICAL_TO_RIDE_NODE_MAP_CACHE.get(map_cache_key)
    if physical_to_ride_node_map is None:
        physical_to_ride_node_map = _build_physical_to_ride_node_map(drive_graph_raw, drive_graph_proj, nodes_df)
        _PHYSICAL_TO_RIDE_NODE_MAP_CACHE[map_cache_key] = physical_to_ride_node_map
    route_nodes = _physical_path_to_route_nodes(generated_path, drive_graph_raw)
    if len(route_nodes) < 2 or len(set(route_nodes)) < 2:
        route_nodes = _repair_minimal_closed_loop(route_nodes, drive_graph_raw)
    route = _physical_nodes_to_ride_route(
        route_nodes,
        route_id=route_id,
        physical_to_ride_node_map=physical_to_ride_node_map,
    )
    background_jeep_routes = _normalize_background_routes(
        background_routes,
        drive_graph_raw,
        drive_graph_proj,
        physical_to_ride_node_map,
    )
    passenger_map = passenger_map or PassengerMap()
    manager = TravelGraphManager(
        edges_csv or _DEFAULT_EDGES_CSV,
        nodes_csv or _DEFAULT_NODES_CSV,
        routes=background_jeep_routes + [route] if background_jeep_routes else [route],
        edges_df=edges_df,
        nodes_df=nodes_df,
        quiet=True,
        walk_wt=config.walk_wt,
        ride_wt=config.ride_wt,
        wait_wt=config.wait_wt,
        transfer_wt=config.transfer_wt,
    )
    baseline_manager = None
    baseline_manager_factory = _baseline_manager_factory(
        edges_csv=edges_csv,
        nodes_csv=nodes_csv,
        background_jeep_routes=background_jeep_routes,
        edges_df=edges_df,
        nodes_df=nodes_df,
        config=config,
    )
    simulation = Simulation(
        manager,
        routes=[route],
        config=config,
        passenger_map=passenger_map,
    )
    if batch_size is None:
        batch_size = simulation.sample_passenger_batch_size(seed=seed)
    batch_size = max(0, int(batch_size))
    passengers = simulation.generate_passenger_batch(batch_size=batch_size, seed=seed)

    result = _evaluate_passenger_batch(
        simulation,
        passengers,
        baseline_manager=baseline_manager,
        baseline_manager_factory=baseline_manager_factory,
        unserved_penalty_beta=unserved_penalty_beta,
    )
    result.route_node_count = len(route.nodes)
    result.route_edge_count = len(route.edge_pairs)
    result.batch_size = batch_size
    result.seed = seed
    return result


@dataclass(slots=True)
class _CandidateEdge:
    next_node_id: int
    length_m: float
    vector_xy: tuple[float, float]


class JeepneyRouteEnv(gym.Env):
    """Coordinate-invariant route-construction environment on the physical graph."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        drive_graph_raw: nx.MultiDiGraph | None = None,
        drive_graph_proj: nx.MultiDiGraph | None = None,
        *,
        passenger_map: PassengerMap | None = None,
        place_queries: list[str] | None = None,
        point_query: str | None = None,
        point_dist: float = 30_000.0,
        systemic_evaluator: Any | None = None,
        systemic_std_penalty_weight: float = 1.0,
        seed: int | None = None,
        max_steps: int = 128,
        min_route_nodes: int | None = None,
        max_route_nodes: int | None = None,
        max_candidates: int | None = None,
        turn_penalty_weight: float = 1.0,
        repeat_uturn_penalty: float = 1.5,
        length_penalty_weight: float = 0.01,
        revisit_penalty_weight: float = 0.5,
        survival_bonus: float = 0.25,
        dead_end_penalty: float = 10_000.0,
        invalid_action_penalty: float = 5_000.0,
        closure_bonus: float = 5.0,
        termination_penalty: float = 10_000.0,
    ) -> None:
        super().__init__()
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

        if drive_graph_raw is None or drive_graph_proj is None:
            (
                _study_area,
                _boundary_source,
                self.graph_source,
                _walk_raw,
                _walk_proj,
                self.drive_graph_raw,
                self.drive_graph_proj,
            ) = load_graphs_for_study_area(
                place_queries or ["Iligan City, Philippines"],
                point_query=point_query,
                point_dist=point_dist,
            )
        else:
            self.graph_source = "preloaded drive graph"
            self.drive_graph_raw = drive_graph_raw
            self.drive_graph_proj = drive_graph_proj

        self.passenger_map = passenger_map or PassengerMap()
        self.systemic_evaluator = systemic_evaluator
        self.systemic_std_penalty_weight = float(systemic_std_penalty_weight)
        self.node_table = node_table_from_graph(self.drive_graph_raw, self.drive_graph_proj)
        self.node_table["base_node_id"] = self.node_table["base_node_id"].astype(int)
        self.node_table["node_key"] = self.node_table["base_node_id"].astype(int)
        self._x_by_node = {
            int(row.base_node_id): float(row.x) for row in self.node_table.itertuples(index=False)
        }
        self._y_by_node = {
            int(row.base_node_id): float(row.y) for row in self.node_table.itertuples(index=False)
        }
        self._lat_by_node = {
            int(row.base_node_id): float(row.lat) for row in self.node_table.itertuples(index=False)
        }
        self._lon_by_node = {
            int(row.base_node_id): float(row.lon) for row in self.node_table.itertuples(index=False)
        }
        self._node_ids = np.array(self.node_table["base_node_id"].to_list(), dtype=int)
        self._out_degree_by_node = {int(node): int(self.drive_graph_raw.out_degree(node)) for node in self._node_ids}
        self._in_degree_by_node = {int(node): int(self.drive_graph_raw.in_degree(node)) for node in self._node_ids}
        self._max_out_degree = max(self._out_degree_by_node.values(), default=1)
        self._max_in_degree = max(self._in_degree_by_node.values(), default=1)

        demand_series = self.passenger_map.df.groupby("base_osmid")["v_ped"].mean()
        self._demand_by_node = {int(node_id): float(value) for node_id, value in demand_series.items()}
        self._demand_values = np.array(list(self._demand_by_node.values()), dtype=np.float64)
        self._demand_scale = max(float(np.percentile(self._demand_values, 95)) if self._demand_values.size else 1.0, 1.0)

        self._successors = self._build_successor_cache()
        self._start_nodes = [node for node, succ in self._successors.items() if len(succ) > 0]
        self._origin_candidates = [node for node, succ in self._successors.items() if len(succ) >= 2]
        if not self._origin_candidates:
            self._origin_candidates = list(self._start_nodes)

        if not self._start_nodes:
            raise ValueError("The drive graph has no navigable outgoing edges.")

        route_min_nodes, route_max_nodes = _default_route_constraints()
        if min_route_nodes is None:
            self.min_route_nodes = route_min_nodes
        else:
            self.min_route_nodes = max(int(min_route_nodes), route_min_nodes)
        self.min_route_nodes = max(self.min_route_nodes, 2)

        if max_route_nodes is None:
            self.max_route_nodes = route_max_nodes
        else:
            self.max_route_nodes = max(int(max_route_nodes), self.min_route_nodes)
        if self.max_route_nodes is not None:
            self.max_route_nodes = max(self.max_route_nodes, self.min_route_nodes)

        self.max_steps = max(int(max_steps), 1)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.repeat_uturn_penalty = float(repeat_uturn_penalty)
        self.length_penalty_weight = float(length_penalty_weight)
        self.revisit_penalty_weight = float(revisit_penalty_weight)
        self.survival_bonus = float(survival_bonus)
        self.dead_end_penalty = float(dead_end_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.closure_bonus = float(closure_bonus)
        self.termination_penalty = float(termination_penalty)
        self.u_turn_threshold = 0.85 * pi
        self.sharp_turn_threshold = pi / 3

        self.max_candidates = int(max_candidates or max((len(v) for v in self._successors.values()), default=1))
        self.max_candidates = max(self.max_candidates, 1)

        self._x_span = float(self.node_table["x"].max() - self.node_table["x"].min()) if not self.node_table.empty else 1.0
        self._y_span = float(self.node_table["y"].max() - self.node_table["y"].min()) if not self.node_table.empty else 1.0
        self._distance_scale = max(hypot(self._x_span, self._y_span), 1.0)
        self._area_scale = max(self._distance_scale**2, 1.0)
        self._edge_scale = max(self._max_edge_length(), 1.0)

        self.action_space = spaces.Discrete(self.max_candidates + 1)
        self.observation_space = spaces.Dict(
            {
                "shape": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "history": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                "topology": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                "demand": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "global": spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32),
                "candidates": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.max_candidates, 10), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.max_candidates + 1,), dtype=np.float32
                ),
            }
        )

        self.origin_node_id: int | None = None
        self.current_node_id: int | None = None
        self.previous_node_id: int | None = None
        self.path_node_ids: list[int] = []
        self.visited_node_ids: set[int] = set()
        self.current_candidates: list[_CandidateEdge] = []
        self.cumulative_length_m = 0.0
        self.cumulative_turn_penalty = 0.0
        self.consecutive_uturns = 0
        self.steps_taken = 0
        self._last_turn_angle = 0.0
        self._current_reference_vector: tuple[float, float] = (1.0, 0.0)
        self._turn_history = deque(maxlen=6)
        self._recent_step_lengths = deque(maxlen=6)
        self._sharp_turns_since_reset = 0
        self._steps_since_sharp_turn = 0

    def _build_successor_cache(self) -> dict[int, list[_CandidateEdge]]:
        cache: dict[int, list[_CandidateEdge]] = {}
        for node_id in self.node_table["base_node_id"].astype(int):
            best_by_successor: dict[int, tuple[float, tuple[float, float]]] = {}
            x0 = self._x_by_node[int(node_id)]
            y0 = self._y_by_node[int(node_id)]
            for _, successor, data in self.drive_graph_raw.out_edges(int(node_id), data=True):
                successor = int(successor)
                x1 = self._x_by_node.get(successor)
                y1 = self._y_by_node.get(successor)
                if x1 is None or y1 is None:
                    continue
                length_m = float(data.get("length", hypot(x1 - x0, y1 - y0)))
                vector_xy = (x1 - x0, y1 - y0)
                existing = best_by_successor.get(successor)
                if existing is None or length_m < existing[0]:
                    best_by_successor[successor] = (length_m, vector_xy)
            cache[int(node_id)] = [
                _CandidateEdge(next_node_id=succ, length_m=length, vector_xy=vector_xy)
                for succ, (length, vector_xy) in best_by_successor.items()
            ]
        return cache

    def _max_edge_length(self) -> float:
        max_length = 0.0
        for _, _, data in self.drive_graph_raw.edges(data=True):
            max_length = max(max_length, float(data.get("length", 0.0)))
        return max_length

    def _current_points_xy(self) -> list[tuple[float, float]]:
        return [(self._x_by_node[node_id], self._y_by_node[node_id]) for node_id in self.path_node_ids]

    def _current_bbox_diagonal_m(self) -> float:
        points = self._current_points_xy()
        if not points:
            return 0.0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return float(hypot(max(xs) - min(xs), max(ys) - min(ys)))

    def _current_hull_area_m2(self) -> float:
        points = self._current_points_xy()
        if len(points) < 2:
            return 0.0
        return float(MultiPoint(points).convex_hull.area)

    def _node_demand(self, node_id: int) -> float:
        return float(self._demand_by_node.get(int(node_id), 0.0))

    def _normalized_demand(self, node_id: int) -> float:
        return float(np.clip(self._node_demand(node_id) / self._demand_scale, 0.0, 1.0))

    def _candidate_demand_values(self) -> list[float]:
        return [self._node_demand(cand.next_node_id) for cand in self._legal_candidates()]

    def _legal_candidates_for_state(
        self,
        *,
        current_node_id: int,
        previous_node_id: int | None,
        route_length: int,
    ) -> list[_CandidateEdge]:
        candidates = list(self._successors.get(int(current_node_id), []))
        legal_candidates: list[_CandidateEdge] = []
        for cand in candidates:
            if previous_node_id is not None and cand.next_node_id == previous_node_id:
                continue
            if cand.next_node_id == self.origin_node_id and route_length < self.min_route_nodes:
                continue
            legal_candidates.append(cand)
        return legal_candidates

    def _candidate_has_future(self, candidate: _CandidateEdge) -> bool:
        next_route_length = len(self.path_node_ids) + 1
        next_candidates = self._legal_candidates_for_state(
            current_node_id=candidate.next_node_id,
            previous_node_id=self.current_node_id,
            route_length=next_route_length,
        )
        if next_candidates:
            return True
        return next_route_length >= self.min_route_nodes

    def _legal_candidates(self) -> list[_CandidateEdge]:
        if self.current_node_id is None:
            return []
        candidates = list(self.current_candidates[: self.max_candidates])
        if not candidates:
            return []
        return [cand for cand in candidates if self._candidate_has_future(cand)]

    def _shape_features(self) -> np.ndarray:
        distance, bearing = self._distance_and_bearing_to_origin()
        path_length = float(self.cumulative_length_m)
        bbox_diag = self._current_bbox_diagonal_m()
        hull_area = self._current_hull_area_m2()
        compactness = hull_area / max(hull_area + path_length**2, 1.0)
        return np.asarray(
            [
                float(np.clip(distance / self._distance_scale, 0.0, 1.0)),
                float(sin(bearing)),
                float(cos(bearing)),
                float(np.clip(path_length / self._distance_scale, 0.0, 1.0)),
                float(np.clip(bbox_diag / self._distance_scale, 0.0, 1.0)),
                float(np.clip(hull_area / self._area_scale, 0.0, 1.0)),
                float(np.clip(compactness, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _history_features(self) -> np.ndarray:
        turns = list(self._turn_history)
        recent_lengths = list(self._recent_step_lengths)
        if not turns and not recent_lengths:
            return np.zeros((8,), dtype=np.float32)
        abs_turns = np.abs(np.asarray(turns, dtype=np.float64)) if turns else np.asarray([], dtype=np.float64)
        last_turn = float(turns[-1]) if turns else 0.0
        recent_length_mean = float(np.mean(recent_lengths)) if recent_lengths else 0.0
        recent_length_std = float(np.std(recent_lengths, ddof=0)) if len(recent_lengths) > 1 else 0.0
        mean_abs_turn = float(np.mean(abs_turns) / pi) if abs_turns.size else 0.0
        max_abs_turn = float(np.max(abs_turns) / pi) if abs_turns.size else 0.0
        mean_signed_turn = float(np.mean(turns) / pi) if turns else 0.0
        return np.asarray(
            [
                float(np.clip(mean_abs_turn, 0.0, 1.0)),
                float(np.clip(max_abs_turn, 0.0, 1.0)),
                float(np.clip(mean_signed_turn, -1.0, 1.0)),
                float(sin(last_turn)),
                float(cos(last_turn)),
                float(np.clip(recent_length_mean / self._edge_scale, 0.0, 1.0)),
                float(np.clip(recent_length_std / self._edge_scale, 0.0, 1.0)),
                float(np.clip(self._steps_since_sharp_turn / max(self.max_steps, 1), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _topology_features(self) -> np.ndarray:
        if self.current_node_id is None:
            return np.zeros((5,), dtype=np.float32)
        current = int(self.current_node_id)
        current_out = self._out_degree_by_node.get(current, 0)
        current_in = self._in_degree_by_node.get(current, 0)
        legal_candidates = self._legal_candidates()
        candidate_count = len(legal_candidates)
        next_outs = [self._out_degree_by_node.get(cand.next_node_id, 0) for cand in legal_candidates]
        mean_candidate_out = float(np.mean(next_outs)) if next_outs else 0.0
        dead_end_flag = 1.0 if candidate_count == 0 else 0.0
        return np.asarray(
            [
                float(np.clip(current_out / max(self._max_out_degree, 1), 0.0, 1.0)),
                float(np.clip(current_in / max(self._max_in_degree, 1), 0.0, 1.0)),
                float(dead_end_flag),
                float(np.clip(candidate_count / max(self.max_candidates, 1), 0.0, 1.0)),
                float(np.clip(mean_candidate_out / max(self._max_out_degree, 1), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _demand_features(self) -> np.ndarray:
        if self.current_node_id is None:
            return np.zeros((4,), dtype=np.float32)
        current_demand = self._normalized_demand(self.current_node_id)
        candidate_demands = np.asarray(self._candidate_demand_values(), dtype=np.float64)
        if candidate_demands.size:
            candidate_mean = float(np.clip(candidate_demands.mean() / self._demand_scale, 0.0, 1.0))
            candidate_max = float(np.clip(candidate_demands.max() / self._demand_scale, 0.0, 1.0))
            demand_gap = float(np.clip((candidate_demands.max() - self._node_demand(self.current_node_id)) / self._demand_scale, -1.0, 1.0))
        else:
            candidate_mean = 0.0
            candidate_max = 0.0
            demand_gap = 0.0
        return np.asarray([current_demand, candidate_mean, candidate_max, demand_gap], dtype=np.float32)

    @staticmethod
    def _normalize_vector(vector_xy: tuple[float, float]) -> np.ndarray:
        vec = np.asarray(vector_xy, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vec / norm

    @staticmethod
    def _signed_angle(reference_xy: tuple[float, float], vector_xy: tuple[float, float]) -> float:
        ref = JeepneyRouteEnv._normalize_vector(reference_xy)
        vec = JeepneyRouteEnv._normalize_vector(vector_xy)
        cross = ref[0] * vec[1] - ref[1] * vec[0]
        dot = float(np.clip(ref[0] * vec[0] + ref[1] * vec[1], -1.0, 1.0))
        return float(np.arctan2(cross, dot))

    @staticmethod
    def _continuous_turn_penalty(theta_rad: float) -> float:
        # Continuous, bounded turn penalty; sharp turns approach 1.0 while straight
        # motion approaches 0.0. See Sensors 2024: https://www.mdpi.com/1424-8220/24/10/3149
        return 0.5 * (1.0 - cos(float(theta_rad)))

    def _route_sinuosity(self) -> float:
        # Sinuosity / tortuosity ratio for zig-zag detection:
        # https://www.researchgate.net/publication/8501613_How_to_reliably_estimate_the_tortuosity_of_an_animal's_path_Straightness_sinuosity_or_fractal_dimension
        if len(self.path_node_ids) < 2:
            return 1.0
        start = self.path_node_ids[0]
        current = self.path_node_ids[-1]
        straight = float(
            hypot(
                self._x_by_node[current] - self._x_by_node[start],
                self._y_by_node[current] - self._y_by_node[start],
            )
        )
        if straight <= 1e-9:
            return 1.0
        return max(self.cumulative_length_m / straight, 1.0)

    def _reference_vector(self) -> tuple[float, float]:
        return self._current_reference_vector

    @staticmethod
    def _canonical_reference_from_vectors(vectors: list[_CandidateEdge]) -> tuple[float, float]:
        # Coordinate-invariant representations should rely on local relational geometry,
        # not absolute coordinates; this mirrors coordinate-invariant GNN design ideas:
        # https://openreview.net/pdf?id=HlBZ9I3qBW
        if not vectors:
            return (1.0, 0.0)
        raw = np.array([cand.vector_xy for cand in vectors], dtype=np.float64)
        if raw.shape[0] == 1:
            ref = raw[0]
        else:
            cov = raw.T @ raw
            eigvals, eigvecs = np.linalg.eigh(cov)
            ref = eigvecs[:, int(np.argmax(eigvals))]
            longest = raw[int(np.argmax(np.linalg.norm(raw, axis=1)))]
            if float(np.dot(ref, longest)) < 0.0:
                ref = -ref
        return float(ref[0]), float(ref[1])

    def _distance_and_bearing_to_origin(self) -> tuple[float, float]:
        if self.origin_node_id is None or self.current_node_id is None:
            return 0.0, 0.0
        origin = (
            self._x_by_node[self.origin_node_id],
            self._y_by_node[self.origin_node_id],
        )
        current = (
            self._x_by_node[self.current_node_id],
            self._y_by_node[self.current_node_id],
        )
        vector_to_origin = (origin[0] - current[0], origin[1] - current[1])
        distance = float(hypot(*vector_to_origin))
        ref = self._reference_vector()
        bearing = self._signed_angle(ref, vector_to_origin) if distance > 1e-9 else 0.0
        return distance, bearing

    def _candidate_features(self) -> tuple[np.ndarray, np.ndarray]:
        ref = self._reference_vector()
        origin = self.origin_node_id
        current = self.current_node_id
        if current is None:
            return np.zeros((self.max_candidates, 10), dtype=np.float32), np.zeros(
                (self.max_candidates + 1,), dtype=np.float32
            )

        legal_candidates = self._legal_candidates()
        rows: list[list[float]] = []
        current_demand = self._node_demand(current)
        for cand in legal_candidates:
            angle = self._signed_angle(ref, cand.vector_xy)
            cand_dir = self._normalize_vector(cand.vector_xy)
            if origin is not None:
                vector_to_origin = (
                    self._x_by_node[origin] - self._x_by_node[current],
                    self._y_by_node[origin] - self._y_by_node[current],
                )
                origin_dir = self._normalize_vector(vector_to_origin) if np.linalg.norm(vector_to_origin) > 1e-12 else np.array([0.0, 0.0])
                alignment = float(np.clip(np.dot(cand_dir, origin_dir), -1.0, 1.0)) if np.linalg.norm(origin_dir) > 0 else 0.0
            else:
                alignment = 0.0
            backtrack = 1.0 if cand.next_node_id == self.previous_node_id else 0.0
            uturn = 1.0 if abs(angle) >= self.u_turn_threshold else 0.0
            next_demand = self._node_demand(cand.next_node_id)
            demand_delta = float(np.clip((next_demand - current_demand) / self._demand_scale, -1.0, 1.0))
            rows.append(
                [
                    float(sin(angle)),
                    float(cos(angle)),
                    float(np.clip(cand.length_m / self._edge_scale, 0.0, 1.0)),
                    float(np.clip(self._out_degree_by_node.get(cand.next_node_id, 0) / max(self._max_out_degree, 1), 0.0, 1.0)),
                    float(1.0 if self._out_degree_by_node.get(cand.next_node_id, 0) <= 1 else 0.0),
                    float(np.clip(next_demand / self._demand_scale, 0.0, 1.0)),
                    demand_delta,
                    backtrack,
                    uturn,
                    alignment,
                ]
            )

        pad_rows = self.max_candidates - len(rows)
        if pad_rows > 0:
            rows.extend([[0.0] * 10 for _ in range(pad_rows)])

        mask = np.zeros((self.max_candidates + 1,), dtype=np.float32)
        mask[: len(legal_candidates)] = 1.0
        mask[self.max_candidates] = 1.0 if len(self.path_node_ids) >= self.min_route_nodes else 0.0
        return np.asarray(rows, dtype=np.float32), mask

    def _global_features(self) -> np.ndarray:
        return np.concatenate(
            [
                self._shape_features(),
                self._history_features(),
                self._topology_features(),
                self._demand_features(),
            ]
        ).astype(np.float32, copy=False)

    def _observation(self) -> dict[str, np.ndarray]:
        candidates, mask = self._candidate_features()
        shape = self._shape_features()
        history = self._history_features()
        topology = self._topology_features()
        demand = self._demand_features()
        return {
            "shape": shape,
            "history": history,
            "topology": topology,
            "demand": demand,
            "global": np.concatenate([shape, history, topology, demand]).astype(np.float32, copy=False),
            "candidates": candidates,
            "action_mask": mask,
        }

    def _flat_state(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                observation["shape"].ravel(),
                observation["history"].ravel(),
                observation["topology"].ravel(),
                observation["demand"].ravel(),
                observation["candidates"].ravel(),
                observation["action_mask"].ravel(),
            ]
        ).astype(np.float32, copy=False)

    def _current_polygon_area_m2(self) -> float:
        if len(self.path_node_ids) < 3:
            return 0.0
        points = self._current_points_xy()
        polygon = Polygon(points)
        if polygon.is_valid and not polygon.is_empty:
            return float(polygon.area)
        return self._current_hull_area_m2()

    def _set_candidates(self, node_id: int) -> None:
        candidates = list(self._successors.get(int(node_id), []))
        if self.previous_node_id is not None and self.current_node_id is not None:
            ref = (
                self._x_by_node[self.current_node_id] - self._x_by_node[self.previous_node_id],
                self._y_by_node[self.current_node_id] - self._y_by_node[self.previous_node_id],
            )
        else:
            ref = self._canonical_reference_from_vectors(candidates)
        self._current_reference_vector = ref
        candidates.sort(key=lambda cand: self._signed_angle(ref, cand.vector_xy))
        self.current_candidates = candidates

    def _minimal_closed_route_path(self) -> list[int]:
        anchor_node_id = self.origin_node_id if self.origin_node_id is not None else self.current_node_id
        if anchor_node_id is None:
            return list(self.path_node_ids)

        candidates = self.current_candidates or list(self._successors.get(int(anchor_node_id), []))
        if not candidates:
            return [int(anchor_node_id)]

        best = min(candidates, key=lambda cand: cand.length_m)
        return [int(anchor_node_id), int(best.next_node_id), int(anchor_node_id)]

    def _closed_route_path_nodes(self) -> list[int]:
        if len(self.path_node_ids) < 2:
            return self._minimal_closed_route_path()
        closed_nodes = _stitch_physical_loop(self.path_node_ids, self.drive_graph_raw)
        if len(closed_nodes) < 2 or len(set(closed_nodes)) < 2:
            return self._minimal_closed_route_path()
        return closed_nodes

    def _evaluate_closed_route(self, route_node_ids: Sequence[int] | None = None) -> RouteFitnessResult:
        closed_nodes = list(self.path_node_ids if route_node_ids is None else route_node_ids)
        if self.systemic_evaluator is None:
            return calculate_route_fitness(
                closed_nodes,
                passenger_map=self.passenger_map,
                drive_graph_raw=self.drive_graph_raw,
                drive_graph_proj=self.drive_graph_proj,
                seed=int(self.np_random.integers(0, 2**32 - 1)),
            )
        return self.systemic_evaluator.evaluate(
            closed_nodes,
            seed=int(self.np_random.integers(0, 2**32 - 1)),
        )

    def _finalize_closed_route(
        self,
        *,
        natural: bool,
        penalty: float = 0.0,
    ) -> tuple[float, RouteFitnessResult, list[int]]:
        closed_nodes = self._closed_route_path_nodes()
        fitness = self._evaluate_closed_route(closed_nodes)
        reward = float(fitness.reward)
        if natural:
            reward += self.closure_bonus
        else:
            reward -= float(penalty)
        return reward, fitness, closed_nodes

    def _step_info(
        self,
        *,
        terminated_reason: str,
        closure_mode: str | None = None,
        base_info: dict[str, Any] | None = None,
        route_fitness: RouteFitnessResult | None = None,
        route_path_node_ids: Sequence[int] | None = None,
        fitness_reward: float | None = None,
        fitness_average_gtc: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation = self._observation()
        info: dict[str, Any] = dict(base_info or {})
        info.update(
            {
                "route_length_m": self.cumulative_length_m,
                "sinuosity_index": self._route_sinuosity(),
                "distance_to_origin_m": self._distance_and_bearing_to_origin()[0],
                "bearing_to_origin_rad": self._distance_and_bearing_to_origin()[1],
                "route_area_m2": self._current_polygon_area_m2(),
                "state_vector": self._flat_state(observation),
                "terminated_reason": terminated_reason,
            }
        )
        if closure_mode is not None:
            info["closure_mode"] = closure_mode
        if route_fitness is not None:
            info["route_fitness"] = route_fitness
            info["fitness_reward"] = float(route_fitness.reward)
            info["fitness_average_gtc"] = float(route_fitness.average_gtc)
            info["fitness_passenger_gtc_std"] = float(
                getattr(route_fitness, "passenger_gtc_std", getattr(route_fitness, "std_gtc", np.nan))
            )
        elif fitness_reward is not None:
            info["fitness_reward"] = float(fitness_reward)
        if fitness_average_gtc is not None:
            info["fitness_average_gtc"] = float(fitness_average_gtc)
        if route_path_node_ids is None:
            route_path_node_ids = list(self.path_node_ids)
        info["route_path_node_ids"] = list(route_path_node_ids)
        return observation, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng(self._seed)

        options = dict(options or {})
        origin_node_id = options.get("origin_node_id")
        if origin_node_id is None:
            pool = self._origin_candidates or self._start_nodes
            origin_node_id = int(self.np_random.choice(pool))
        origin_node_id = int(origin_node_id)

        self.origin_node_id = origin_node_id
        self.current_node_id = origin_node_id
        self.previous_node_id = None
        self.path_node_ids = [origin_node_id]
        self.visited_node_ids = {origin_node_id}
        self.cumulative_length_m = 0.0
        self.cumulative_turn_penalty = 0.0
        self.consecutive_uturns = 0
        self.steps_taken = 0
        self._last_turn_angle = 0.0
        self._turn_history.clear()
        self._recent_step_lengths.clear()
        self._sharp_turns_since_reset = 0
        self._steps_since_sharp_turn = 0
        self._set_candidates(origin_node_id)
        observation = self._observation()
        info = {
            "state_vector": self._flat_state(observation),
            "origin_node_id": origin_node_id,
            "current_node_id": origin_node_id,
        }
        return observation, info

    def _apply_move(self, next_node_id: int) -> tuple[float, bool, dict[str, Any]]:
        assert self.current_node_id is not None
        current = self.current_node_id
        current_xy = (self._x_by_node[current], self._y_by_node[current])
        next_xy = (self._x_by_node[next_node_id], self._y_by_node[next_node_id])
        vector_xy = (next_xy[0] - current_xy[0], next_xy[1] - current_xy[1])
        length_m = float(hypot(*vector_xy))
        turn_angle = 0.0
        turn_penalty = 0.0
        if self.previous_node_id is not None:
            incoming = (
                current_xy[0] - self._x_by_node[self.previous_node_id],
                current_xy[1] - self._y_by_node[self.previous_node_id],
            )
            signed_turn = self._signed_angle(incoming, vector_xy)
            turn_angle = float(signed_turn)
            abs_turn = abs(signed_turn)
            turn_penalty = self._continuous_turn_penalty(abs_turn) * self.turn_penalty_weight
            self._turn_history.append(turn_angle)
            if abs_turn >= self.sharp_turn_threshold:
                self._steps_since_sharp_turn = 0
            else:
                self._steps_since_sharp_turn += 1
            if abs_turn >= self.u_turn_threshold:
                self.consecutive_uturns += 1
                turn_penalty += self.repeat_uturn_penalty * max(self.consecutive_uturns - 1, 0)
            else:
                self.consecutive_uturns = 0
        else:
            self.consecutive_uturns = 0
            self._steps_since_sharp_turn = 0

        reward = self.survival_bonus
        step_reward = reward

        self.previous_node_id = current
        self.current_node_id = next_node_id
        self.path_node_ids.append(next_node_id)
        self.visited_node_ids.add(next_node_id)
        self.cumulative_length_m += length_m
        self.cumulative_turn_penalty += turn_penalty
        self._last_turn_angle = turn_angle
        self._current_reference_vector = vector_xy
        self._recent_step_lengths.append(length_m)
        self.steps_taken += 1
        self._set_candidates(next_node_id)

        terminated = False
        if next_node_id == self.origin_node_id:
            terminated = True
            natural = len(self.path_node_ids) >= self.min_route_nodes
            reward, fitness, closed_nodes = self._finalize_closed_route(
                natural=natural,
                penalty=self.termination_penalty,
            )
            reward += step_reward
        else:
            reward = step_reward

        info = {
            "turn_angle_rad": turn_angle,
            "turn_penalty": turn_penalty,
            "step_survival_bonus": step_reward,
            "route_length_m": self.cumulative_length_m,
            "sinuosity_index": self._route_sinuosity(),
            "distance_to_origin_m": self._distance_and_bearing_to_origin()[0],
            "bearing_to_origin_rad": self._distance_and_bearing_to_origin()[1],
            "route_area_m2": self._current_polygon_area_m2(),
            "state_vector": self._flat_state(self._observation()),
        }
        if terminated:
            info["route_fitness"] = fitness
            info["fitness_reward"] = float(fitness.reward)
            info["fitness_average_gtc"] = float(fitness.average_gtc)
            info["route_path_node_ids"] = list(closed_nodes)
            info["closure_mode"] = "natural" if len(self.path_node_ids) >= self.min_route_nodes else "forced"
            info["terminated_reason"] = "closed_loop"
        return reward, terminated, info

    def step(self, action: int):
        if self.current_node_id is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        valid_moves = self._legal_candidates()
        terminate_action = self.max_candidates
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        terminate_allowed = len(self.path_node_ids) >= self.min_route_nodes

        if not valid_moves and not terminate_allowed:
            self.steps_taken += 1
            reward = -self.dead_end_penalty
            observation, info = self._step_info(
                terminated_reason="dead_end",
                closure_mode="blocked",
                base_info=info if info else None,
                route_path_node_ids=list(self.path_node_ids),
            )
            info["dead_end"] = True
            return observation, float(reward), True, False, info

        if action == terminate_action:
            if not terminate_allowed:
                self.steps_taken += 1
                reward = -self.invalid_action_penalty
                if self.steps_taken >= self.max_steps:
                    reward -= self.termination_penalty
                    terminated = True
                    observation, info = self._step_info(
                        terminated_reason="max_steps",
                        closure_mode="forced",
                        base_info=info if info else None,
                        route_path_node_ids=list(self.path_node_ids),
                    )
                    info["invalid_action"] = True
                    info["max_steps_reached"] = True
                    return observation, float(reward), terminated, False, info
                observation, info = self._step_info(
                    terminated_reason="invalid_action",
                    closure_mode="blocked",
                    base_info=info if info else None,
                    route_path_node_ids=list(self.path_node_ids),
                )
                info["invalid_action"] = True
                return observation, float(reward), False, False, info

            closed = self.current_node_id == self.origin_node_id and len(self.path_node_ids) >= self.min_route_nodes
            terminated = True
            reward, fitness, closed_nodes = self._finalize_closed_route(
                natural=closed,
                penalty=self.termination_penalty,
            )
            observation, info = self._step_info(
                terminated_reason="closed_loop",
                closure_mode="natural" if closed else "forced",
                base_info=info,
                route_fitness=fitness,
                route_path_node_ids=closed_nodes,
            )
            return observation, float(reward), terminated, truncated, info

        if action < 0 or action >= len(valid_moves):
            reward = -self.invalid_action_penalty
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                reward -= self.termination_penalty
                terminated = True
                observation, info = self._step_info(
                    terminated_reason="max_steps",
                    closure_mode="forced",
                    base_info=info if info else None,
                    route_path_node_ids=list(self.path_node_ids),
                )
                info["invalid_action"] = True
                info["max_steps_reached"] = True
                return observation, float(reward), terminated, False, info
            observation, info = self._step_info(
                terminated_reason="invalid_action",
                closure_mode="blocked",
                base_info=info if info else None,
                route_path_node_ids=list(self.path_node_ids),
            )
            info["invalid_action"] = True
            return observation, float(reward), False, False, info

        chosen = valid_moves[action]
        reward, terminated, info = self._apply_move(chosen.next_node_id)

        if terminated:
            info["terminated_reason"] = "closed_loop"
        else:
            legal_after_move = self._legal_candidates()
            if not legal_after_move and len(self.path_node_ids) < self.min_route_nodes:
                terminated = True
                reward -= self.dead_end_penalty
                observation, info = self._step_info(
                    terminated_reason="dead_end",
                    closure_mode="blocked",
                    base_info=info,
                    route_path_node_ids=list(self.path_node_ids),
                )
                info["dead_end"] = True

            elif self.max_route_nodes is not None and len(self.path_node_ids) >= self.max_route_nodes and not terminated:
                terminated = True
                reward -= self.termination_penalty
                observation, info = self._step_info(
                    terminated_reason="max_nodes",
                    closure_mode="forced",
                    base_info=info,
                    route_path_node_ids=list(self.path_node_ids),
                )
                info["max_nodes_reached"] = True
                info["route_path_node_ids"] = list(self.path_node_ids)
                return observation, float(reward), terminated, False, info

        if not terminated and self.steps_taken >= self.max_steps:
            terminated = True
            reward -= self.termination_penalty
            observation, info = self._step_info(
                terminated_reason="max_steps",
                closure_mode="forced",
                base_info=info,
                route_path_node_ids=list(self.path_node_ids),
            )
            info["max_steps_reached"] = True
            return observation, float(reward), terminated, False, info

        if not terminated:
            observation, info = self._step_info(
                terminated_reason=info.get("terminated_reason", "in_progress"),
                closure_mode=info.get("closure_mode"),
                base_info=info,
                route_fitness=info.get("route_fitness"),
                route_path_node_ids=info.get("route_path_node_ids", list(self.path_node_ids)),
                fitness_reward=info.get("fitness_reward"),
                fitness_average_gtc=info.get("fitness_average_gtc"),
            )
        else:
            observation, info = self._step_info(
                terminated_reason=info.get("terminated_reason", "closed_loop"),
                closure_mode=info.get("closure_mode"),
                base_info=info,
                route_fitness=info.get("route_fitness"),
                route_path_node_ids=info.get("route_path_node_ids", list(self.path_node_ids)),
                fitness_reward=info.get("fitness_reward"),
                fitness_average_gtc=info.get("fitness_average_gtc"),
            )
        return observation, float(reward), terminated, False, info

    def close(self) -> None:  # pragma: no cover - trivial lifecycle hook
        return None

