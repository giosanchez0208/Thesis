"""Passenger and jeep simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
import numpy as np
import yaml

from .jeep import Jeep, JeepState
from .passenger import Passenger, PassengerState
from .passenger_map import PassengerMap


@dataclass(slots=True)
class SimulationConfig:
    """Numerical configuration used by the simulation."""

    walk_wt: float
    ride_wt: float
    wait_wt: float
    transfer_wt: float
    v_jeep: float
    v_passenger: float
    dt: float = 1.0
    headway_s: float = 60.0
    default_num_jeeps: int = 3
    max_timesteps: int = 10000
    passenger_generation_interval_dt: float = 2.0
    passenger_generation_mean: float = 120.0
    passenger_generation_std: float = 30.0
    coord_rounding: int = 7
    max_interlayer_snap_distance_m: float = 80.0
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        weight_profile: str = "full_ride_manager",
    ) -> "SimulationConfig":
        path = Path(config_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        weights = data["weights"][weight_profile]
        velocities = data["velocities"]
        network = data.get("network", {})
        route_cfg = data.get("route_cfg", data.get("route_generation", {}))
        route_generation = data.get("route_generation", {})
        simulation = data.get("simulation", {})
        passenger_generation = data.get("passenger_generation", {})

        return cls(
            walk_wt=float(weights["walk_wt"]),
            ride_wt=float(weights["ride_wt"]),
            wait_wt=float(weights["wait_wt"]),
            transfer_wt=float(weights["transfer_wt"]),
            v_jeep=float(velocities["v_jeep"]),
            v_passenger=float(velocities["v_passenger"]),
            dt=1.0,
            headway_s=float(simulation.get("headway_s", 60.0)),
            default_num_jeeps=int(simulation.get("default_num_jeeps", 3)),
            max_timesteps=int(simulation.get("max_timesteps", 10000)),
            passenger_generation_interval_dt=float(passenger_generation.get("interval_dt", 2.0)),
            passenger_generation_mean=float(passenger_generation.get("mean_per_interval", 120.0)),
            passenger_generation_std=float(passenger_generation.get("std_per_interval", 30.0)),
            coord_rounding=int(network.get("coord_rounding", 7)),
            max_interlayer_snap_distance_m=float(network.get("max_interlayer_snap_distance_m", 80.0)),
            raw={
                "network": network,
                "route_cfg": route_cfg,
                "route_generation": route_generation,
                "simulation": simulation,
                "passenger_generation": passenger_generation,
                "weights": data.get("weights", {}),
                "velocities": velocities,
            },
        )


class Simulation:
    """Discrete multipassenger simulation with per-timestep playback recording."""

    def __init__(
        self,
        travel_graph_mgr,
        passengers: Optional[Iterable[Passenger]] = None,
        routes: Optional[Iterable] = None,
        *,
        config: Optional[SimulationConfig] = None,
        config_path: str | Path | None = None,
        passenger_map: Optional[PassengerMap] = None,
        fleet_sizes: Optional[Mapping[str, int]] = None,
        default_num_jeeps: Optional[int] = None,
    ) -> None:
        if config is None:
            if config_path is None:
                raise ValueError("Provide either config or config_path.")
            config = SimulationConfig.from_yaml(config_path)

        self.config = config
        self.travel_graph_mgr = travel_graph_mgr
        self.passengers: list[Passenger] = list(passengers or [])
        self.routes = list(routes or [])
        self.default_num_jeeps = int(default_num_jeeps or config.default_num_jeeps)
        self.fleet_sizes = dict(fleet_sizes or {})
        self.passenger_map = passenger_map or PassengerMap()

        self._nearest_node_cache: dict[tuple[str, float, float], Optional[str]] = {}
        self._path_cache: dict[tuple[str, str], list[str]] = {}
        self._route_fleets: dict[str, list[Jeep]] = {}
        self._passenger_path_cache: dict[tuple[str, str], dict] = {}
        self._playback_rows: list[dict] = []

        self._build_route_fleets()

    @property
    def total_jeeps(self) -> int:
        return sum(len(fleet) for fleet in self._route_fleets.values())

    def set_passengers(self, passengers: Iterable[Passenger]) -> None:
        self.passengers = list(passengers)

    def set_routes(self, routes: Iterable) -> None:
        self.routes = list(routes)
        self._build_route_fleets()

    def set_fleet_size(self, route_id: str, num_jeeps: int) -> None:
        self.fleet_sizes[str(route_id)] = int(num_jeeps)
        self._build_route_fleets()

    def set_fleet_sizes(self, fleet_sizes: Mapping[str, int]) -> None:
        self.fleet_sizes = dict(fleet_sizes)
        self._build_route_fleets()

    def _node_coords(self) -> dict:
        return getattr(self.travel_graph_mgr, "_node_coords", {}) or {}

    def _all_jeeps(self) -> list[Jeep]:
        return [jeep for fleet in self._route_fleets.values() for jeep in fleet]

    def _coords_close(self, a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> bool:
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    def _route_ids_for_edge(self, edge_id: str) -> list[str]:
        if hasattr(self.travel_graph_mgr, "get_route_ids_for_edge"):
            return list(self.travel_graph_mgr.get_route_ids_for_edge(edge_id))
        return []

    def _route_id_for_edge(self, edge_id: str) -> Optional[str]:
        route_ids = self._route_ids_for_edge(edge_id)
        return route_ids[0] if route_ids else None

    def _build_route_fleets(self) -> None:
        self._route_fleets = {}

        if not self.routes:
            return

        node_coords = self._node_coords()
        for route in self.routes:
            route_id = str(route.route_id)
            route_coords: list[tuple[float, float]] = []
            for node_id in route.nodes:
                coord = node_coords.get(node_id)
                if coord is not None:
                    route_coords.append((float(coord[0]), float(coord[1])))

            fleet: list[Jeep] = []
            if route_coords:
                fleet_size = self._fleet_size_for_route(route_id)
                headway = self.config.headway_s / max(fleet_size, 1)
                for idx in range(fleet_size):
                    jeep = Jeep(
                        jeep_id=f"{route_id}_{idx:03d}",
                        route_nodes=route_coords,
                        route_id=route_id,
                        route_node_ids=route.nodes,
                        v_jeep=self.config.v_jeep,
                    )
                    jeep.start_time = idx * headway
                    fleet.append(jeep)
            self._route_fleets[route_id] = fleet

    def _fleet_size_for_route(self, route_id: str) -> int:
        return int(self.fleet_sizes.get(route_id, self.default_num_jeeps))

    def sample_passenger_batch_size(self, seed: Optional[int] = None) -> int:
        rng = np.random.default_rng(seed)
        size = int(round(rng.normal(self.config.passenger_generation_mean, self.config.passenger_generation_std)))
        return max(0, size)

    def generate_passenger_batch(
        self,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[Passenger]:
        if batch_size is None:
            batch_size = self.sample_passenger_batch_size(seed=seed)
        rng = np.random.default_rng(seed)
        passengers = [
            Passenger(
                passenger_map=self.passenger_map,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            for _ in range(int(batch_size))
        ]
        self.passengers.extend(passengers)
        return passengers

    def generate_passenger_batches(
        self,
        num_batches: int,
        seed: Optional[int] = None,
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        counts: list[int] = []
        for _ in range(int(num_batches)):
            counts.append(
                self.sample_passenger_batch_size(seed=int(rng.integers(0, 2**32 - 1)))
            )
        return counts

    def _nearest_node_cache_key(self, lat: float, lon: float, layer: Optional[str]) -> tuple[str, float, float]:
        return (
            layer or "any",
            round(float(lat), self.config.coord_rounding),
            round(float(lon), self.config.coord_rounding),
        )

    def find_nearest_node(self, lat: float, lon: float, layer: Optional[str] = None) -> Optional[str]:
        key = self._nearest_node_cache_key(lat, lon, layer)
        if key not in self._nearest_node_cache:
            self._nearest_node_cache[key] = self.travel_graph_mgr.find_nearest_node(lat, lon, layer=layer)
        return self._nearest_node_cache[key]

    def calculate_shortest_path(self, passenger: Passenger) -> list[str]:
        start_graph_node_id = self.find_nearest_node(passenger.start_lat, passenger.start_lon, layer="start_walk")
        end_graph_node_id = self.find_nearest_node(passenger.end_lat, passenger.end_lon, layer="end_walk")
        if not start_graph_node_id or not end_graph_node_id:
            return []

        cache_key = (start_graph_node_id, end_graph_node_id)
        if cache_key not in self._path_cache:
            self._path_cache[cache_key] = self.travel_graph_mgr.calculate_shortest_path(
                start_graph_node_id,
                end_graph_node_id,
            )
        return list(self._path_cache[cache_key])

    def prepare_passenger(self, passenger: Passenger) -> dict:
        passenger.set_travel_graph(self.travel_graph_mgr)
        path_edges = self.calculate_shortest_path(passenger)
        passenger.current_path_edge_index = 0
        passenger.current_edge_progress_m = 0.0
        passenger.boarded_jeep_id = None
        passenger.boarded_route_id = None

        if not path_edges:
            passenger.shortest_path_edges = []
            passenger.shortest_path_nodes = []
            passenger.state = PassengerState.COMPLETED
            cache_key = (str(passenger.start_node_id), str(passenger.end_node_id))
            payload = {"found": False, "path_edges": [], "path_nodes": []}
            self._passenger_path_cache[cache_key] = payload
            return payload

        start_graph_node_id = self.find_nearest_node(passenger.start_lat, passenger.start_lon, layer="start_walk")
        nodes = [start_graph_node_id]
        for edge_id in path_edges:
            edge = self.travel_graph_mgr.get_edge(edge_id)
            if edge is not None:
                nodes.append(edge.v)

        passenger.shortest_path_edges = list(path_edges)
        passenger.shortest_path_nodes = nodes
        passenger.current_path_index = 0

        payload = {
            "found": True,
            "start_graph_node": start_graph_node_id,
            "end_graph_node": self.find_nearest_node(passenger.end_lat, passenger.end_lon, layer="end_walk"),
            "path_edges": list(path_edges),
            "path_nodes": nodes,
        }
        cache_key = (str(passenger.start_node_id), str(passenger.end_node_id))
        self._passenger_path_cache[cache_key] = payload
        return payload

    def prepare_passengers(self, passengers: Optional[Iterable[Passenger]] = None) -> list[dict]:
        items = list(passengers or self.passengers)
        return [self.prepare_passenger(passenger) for passenger in items]

    def _passenger_edge(self, passenger: Passenger):
        edge_id = passenger.current_path_edge_id()
        if edge_id is None:
            return None
        return self.travel_graph_mgr.get_edge(edge_id)

    def _edge_coords(self, edge) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        coords = self._node_coords()
        if edge is None:
            return None
        u = coords.get(edge.u)
        v = coords.get(edge.v)
        if u is None or v is None:
            return None
        return (float(u[0]), float(u[1])), (float(v[0]), float(v[1]))

    def _passenger_position(self, passenger: Passenger) -> tuple[float, float]:
        coords = self._node_coords()
        edge = self._passenger_edge(passenger)
        if edge is None:
            return float(passenger.curr_lat), float(passenger.curr_lon)
        edge_coords = self._edge_coords(edge)
        if edge_coords is None:
            return float(passenger.curr_lat), float(passenger.curr_lon)
        start, end = edge_coords
        if edge.dist <= 0:
            return end
        frac = min(max(passenger.current_edge_progress_m / float(edge.dist), 0.0), 1.0)
        lat = start[0] + (end[0] - start[0]) * frac
        lon = start[1] + (end[1] - start[1]) * frac
        return lat, lon

    def _matching_jeep(self, route_ids: list[str], stop_node_id: str) -> Optional[Jeep]:
        stop_coords = self._node_coords().get(stop_node_id)
        if stop_coords is None:
            return None
        for route_id in route_ids:
            for jeep in self._route_fleets.get(route_id, []):
                if jeep.state == JeepState.COMPLETED:
                    continue
                if jeep.total_time < getattr(jeep, "start_time", 0.0):
                    continue
                if self._coords_close(jeep.get_curr_lat_lon(), (float(stop_coords[0]), float(stop_coords[1]))):
                    return jeep
        return None

    def _start_waiting(self, passenger: Passenger) -> None:
        passenger.state = PassengerState.WAITING_AT_STATION
        passenger.curr_lat, passenger.curr_lon = self._passenger_position(passenger)

    def _advance_passenger_walk(self, passenger: Passenger, dt: float) -> None:
        edge = self._passenger_edge(passenger)
        if edge is None:
            passenger.state = PassengerState.COMPLETED
            return
        edge_coords = self._edge_coords(edge)
        if edge_coords is None:
            passenger.state = PassengerState.COMPLETED
            return

        start, end = edge_coords
        if edge.dist <= 0:
            passenger.current_path_index += 1
            passenger.current_edge_progress_m = 0.0
            passenger.curr_lat, passenger.curr_lon = end
            return

        passenger.current_edge_progress_m += self.config.v_passenger * dt
        if passenger.current_edge_progress_m >= float(edge.dist):
            passenger.curr_lat, passenger.curr_lon = end
            passenger.curr_node_id = edge.v
            passenger.current_path_index += 1
            passenger.current_edge_progress_m = 0.0
        else:
            passenger.curr_lat, passenger.curr_lon = self._passenger_position(passenger)

    def _advance_passenger_pre_jeep(self, passenger: Passenger, dt: float) -> None:
        edge = self._passenger_edge(passenger)
        if edge is None:
            passenger.state = PassengerState.COMPLETED
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, PassengerState.COMPLETED, dt=dt)
            return

        edge_type = str(edge.edge_type)
        if passenger.state == PassengerState.WAITING_TO_WALK:
            passenger.state = PassengerState.WALKING_TO_BOARD

        if edge_type in {"start_walk", "end_walk"}:
            passenger.state = PassengerState.WALKING_TO_BOARD if passenger.current_path_index == 0 else PassengerState.WALKING_FROM_ALIGHT
            self._advance_passenger_walk(passenger, dt)
            if passenger.current_path_index < len(passenger.shortest_path_edges):
                next_edge = self.travel_graph_mgr.get_edge(passenger.shortest_path_edges[passenger.current_path_index])
                if next_edge is not None and str(next_edge.edge_type) == "wait":
                    passenger.curr_node_id = next_edge.u
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)
            return

        if edge_type in {"direct", "alight"}:
            passenger.curr_node_id = edge.v
            passenger.current_path_index += 1
            passenger.current_edge_progress_m = 0.0
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)
            return

        if edge_type == "transfer":
            passenger.state = PassengerState.WAITING_AT_STATION
            passenger.curr_node_id = edge.v
            passenger.current_path_index += 1
            passenger.current_edge_progress_m = 0.0
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)
            return

        if edge_type == "wait":
            passenger.state = PassengerState.WAITING_AT_STATION
            route_ids = self._route_ids_for_edge(passenger.shortest_path_edges[passenger.current_path_index + 1]) if passenger.current_path_index + 1 < len(passenger.shortest_path_edges) else self._route_ids_for_edge(edge.edge_id)
            if not route_ids:
                route_ids = self._route_ids_for_edge(edge.edge_id)
            stop_node_id = edge.v
            match = self._matching_jeep(route_ids, stop_node_id)
            if match is not None:
                passenger.boarded_jeep_id = match.jeep_id
                passenger.boarded_route_id = match.route_id
                passenger.state = PassengerState.RIDING
                passenger.curr_node_id = edge.u
                passenger.current_path_index += 1
                passenger.current_edge_progress_m = 0.0
            passenger.curr_lat, passenger.curr_lon = self._node_coords().get(stop_node_id, (passenger.curr_lat, passenger.curr_lon))
            passenger.curr_node_id = stop_node_id
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)
            return

        if edge_type == "ride":
            if passenger.boarded_jeep_id is None:
                route_ids = self._route_ids_for_edge(edge.edge_id)
                match = self._matching_jeep(route_ids, edge.u)
                if match is not None:
                    passenger.boarded_jeep_id = match.jeep_id
                    passenger.boarded_route_id = match.route_id
                    passenger.state = PassengerState.RIDING
                    passenger.curr_node_id = edge.u
            passenger.curr_lat, passenger.curr_lon = self._passenger_position(passenger)
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)
            return

        passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, passenger.state, dt=dt)

    def _advance_jeeps(self, dt: float, sim_time: float) -> None:
        for fleet in self._route_fleets.values():
            for jeep in fleet:
                if sim_time < getattr(jeep, "start_time", 0.0):
                    continue
                jeep.update(dt)

    def _sync_riding_passengers_post_jeep(self, dt: float) -> None:
        coords = self._node_coords()
        jeep_by_id = {jeep.jeep_id: jeep for jeep in self._all_jeeps()}
        for passenger in self.passengers:
            if passenger.state != PassengerState.RIDING or not passenger.boarded_jeep_id:
                continue
            jeep = jeep_by_id.get(passenger.boarded_jeep_id)
            if jeep is None:
                continue
            passenger.curr_lat, passenger.curr_lon = jeep.get_curr_lat_lon()
            edge = self._passenger_edge(passenger)
            if edge is not None and not passenger.curr_node_id:
                passenger.curr_node_id = edge.u
            passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, PassengerState.RIDING, dt=dt)

            if edge is None or str(edge.edge_type) != "ride":
                continue
            end_coords = coords.get(edge.v)
            if end_coords is not None and self._coords_close(jeep.get_curr_lat_lon(), (float(end_coords[0]), float(end_coords[1]))):
                passenger.curr_node_id = edge.v
                passenger.current_path_index += 1
                passenger.current_edge_progress_m = 0.0
                next_edge = self._passenger_edge(passenger)
                if next_edge is None:
                    passenger.boarded_jeep_id = None
                    passenger.boarded_route_id = None
                    passenger.state = PassengerState.AT_DESTINATION
                    passenger.update(passenger.curr_node_id, passenger.curr_lat, passenger.curr_lon, PassengerState.COMPLETED, dt=0.0)
                elif str(next_edge.edge_type) != "ride":
                    passenger.boarded_jeep_id = None
                    passenger.boarded_route_id = None
                    if str(next_edge.edge_type) in {"alight", "direct"}:
                        passenger.state = PassengerState.WALKING_FROM_ALIGHT if str(next_edge.edge_type) == "alight" else PassengerState.COMPLETED
                    elif str(next_edge.edge_type) == "transfer":
                        passenger.state = PassengerState.WAITING_AT_STATION

    def advance_fleets(self, sim_time: float, dt: Optional[float] = None) -> None:
        step = float(self.config.dt if dt is None else dt)
        for passenger in self.passengers:
            if passenger.shortest_path_edges == [] and passenger.state not in {PassengerState.COMPLETED, PassengerState.AT_DESTINATION}:
                self.prepare_passenger(passenger)
            if passenger.state not in {PassengerState.COMPLETED, PassengerState.AT_DESTINATION}:
                self._advance_passenger_pre_jeep(passenger, step)
        self._advance_jeeps(step, sim_time)
        self._sync_riding_passengers_post_jeep(step)

    def _snapshot_rows(self, timestep: int, sim_time: float) -> list[dict]:
        rows: list[dict] = []
        for passenger in self.passengers:
            edge_id = passenger.current_path_edge_id()
            route_id = passenger.boarded_route_id or (self._route_id_for_edge(edge_id) if edge_id else None)
            rows.append(
                {
                    "time_step": timestep,
                    "sim_time": float(sim_time),
                    "item_type": "passenger",
                    "id": str(passenger.curr_node_id),
                    "lat": float(passenger.curr_lat),
                    "lon": float(passenger.curr_lon),
                    "state": passenger.state.value if hasattr(passenger.state, "value") else str(passenger.state),
                    "route_id": route_id,
                    "parent_id": passenger.boarded_jeep_id,
                    "path_index": passenger.current_path_index,
                    "edge_id": edge_id,
                }
            )
        for jeep in self._all_jeeps():
            rows.append(
                {
                    "time_step": timestep,
                    "sim_time": float(sim_time),
                    "item_type": "jeep",
                    "id": str(jeep.jeep_id),
                    "lat": float(jeep.curr_lat),
                    "lon": float(jeep.curr_lon),
                    "state": jeep.state.value if hasattr(jeep.state, "value") else str(jeep.state),
                    "route_id": jeep.route_id,
                    "parent_id": jeep.route_id,
                    "path_index": jeep.current_segment_idx,
                    "edge_id": None,
                }
            )
        return rows

    def record_snapshot(self, timestep: int, sim_time: float) -> None:
        self._playback_rows.extend(self._snapshot_rows(timestep, sim_time))

    def export_playback_csv(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._playback_rows)
        df.to_csv(out, index=False)
        return out

    def all_passengers_completed(self) -> bool:
        return all(p.state in {PassengerState.COMPLETED, PassengerState.AT_DESTINATION} for p in self.passengers)

    def run(
        self,
        *,
        max_timesteps: Optional[int] = None,
        dt: Optional[float] = None,
        output_csv_path: str | Path | None = None,
    ) -> pd.DataFrame:
        step_dt = float(self.config.dt if dt is None else dt)
        cap = int(self.config.max_timesteps if max_timesteps is None else max_timesteps)
        cap = min(cap, int(self.config.max_timesteps))

        self._playback_rows = []
        for passenger in self.passengers:
            if not passenger.shortest_path_edges:
                self.prepare_passenger(passenger)
        sim_time = 0.0
        self.record_snapshot(0, sim_time)

        for timestep in range(1, cap + 1):
            if self.all_passengers_completed():
                break
            self.advance_fleets(sim_time=sim_time, dt=step_dt)
            sim_time += step_dt
            self.record_snapshot(timestep, sim_time)

        frame = pd.DataFrame(self._playback_rows)
        if output_csv_path is not None:
            self.export_playback_csv(output_csv_path)
        return frame

    def fleet_summary(self) -> dict[str, int]:
        return {route_id: len(fleet) for route_id, fleet in self._route_fleets.items()}

    def summary(self) -> dict:
        return {
            "passengers": len(self.passengers),
            "routes": len(self.routes),
            "total_jeeps": self.total_jeeps,
            "fleet_sizes": self.fleet_summary(),
            "passenger_generation_interval_dt": self.config.passenger_generation_interval_dt,
            "passenger_generation_mean": self.config.passenger_generation_mean,
            "passenger_generation_std": self.config.passenger_generation_std,
            "max_timesteps": self.config.max_timesteps,
            "cached_paths": len(self._path_cache),
            "cached_nodes": len(self._nearest_node_cache),
        }


TandemPassengerJeepSimulation = Simulation
