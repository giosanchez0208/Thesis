"""Passenger and jeep simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import numpy as np
import yaml

from .jeep import Jeep
from .passenger import Passenger
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
            passenger_generation_interval_dt=float(passenger_generation.get("interval_dt", 2.0)),
            passenger_generation_mean=float(passenger_generation.get("mean_per_interval", 120.0)),
            passenger_generation_std=float(passenger_generation.get("std_per_interval", 30.0)),
            coord_rounding=int(network.get("coord_rounding", 7)),
            max_interlayer_snap_distance_m=float(network.get("max_interlayer_snap_distance_m", 80.0)),
            raw={
                "network": network,
                "route_generation": route_generation,
                "simulation": simulation,
                "passenger_generation": passenger_generation,
                "weights": data.get("weights", {}),
                "velocities": velocities,
            },
        )


class Simulation:
    """
    Experimental multipassenger simulation scaffold.

    The class is designed around a shared TravelGraphManager, cached shortest
    paths, and a flexible per-route fleet size model.
    """

    def __init__(
        self,
        travel_graph_mgr,
        passengers: Optional[Iterable[Passenger]] = None,
        routes: Optional[Iterable] = None,
        *,
        config: Optional[SimulationConfig] = None,
        config_path: str | Path | None = None,
        fleet_sizes: Optional[Mapping[str, int]] = None,
        default_num_jeeps: Optional[int] = None,
    ) -> None:
        if config is None:
            if config_path is None:
                raise ValueError("Provide either config or config_path.")
            config = SimulationConfig.from_yaml(config_path)

        self.config = config
        self.travel_graph_mgr = travel_graph_mgr
        self.passengers: List[Passenger] = list(passengers or [])
        self.routes = list(routes or [])
        self.default_num_jeeps = int(default_num_jeeps or config.default_num_jeeps)
        self.fleet_sizes = dict(fleet_sizes or {})
        self.passenger_map = PassengerMap()

        self._nearest_node_cache: dict[tuple[str, float, float], Optional[str]] = {}
        self._path_cache: dict[tuple[str, str], list[str]] = {}
        self._route_fleets: dict[str, list[Jeep]] = {}
        self._passenger_path_cache: dict[tuple[str, str], dict] = {}

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

    def _fleet_size_for_route(self, route_id: str) -> int:
        return int(self.fleet_sizes.get(route_id, self.default_num_jeeps))

    def _node_coords(self) -> dict:
        return getattr(self.travel_graph_mgr, "_node_coords", {}) or {}

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
        passengers = [Passenger(passenger_map=self.passenger_map) for _ in range(int(batch_size))]
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
                        v_jeep=self.config.v_jeep,
                    )
                    jeep.route_id = route_id
                    jeep.start_time = idx * headway
                    fleet.append(jeep)
            self._route_fleets[route_id] = fleet

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
            try:
                self._path_cache[cache_key] = self.travel_graph_mgr.calculate_shortest_path(
                    start_graph_node_id,
                    end_graph_node_id,
                )
            except Exception:
                self._path_cache[cache_key] = []
        return list(self._path_cache[cache_key])

    def prepare_passenger(self, passenger: Passenger) -> dict:
        passenger.set_travel_graph(self.travel_graph_mgr)
        path_edges = self.calculate_shortest_path(passenger)
        if not path_edges:
            passenger.shortest_path_edges = []
            passenger.shortest_path_nodes = []
            passenger.current_path_index = 0
            cache_key = (str(passenger.start_node_id), str(passenger.end_node_id))
            self._passenger_path_cache[cache_key] = {
                "found": False,
                "path_edges": [],
                "path_nodes": [],
            }
            return self._passenger_path_cache[cache_key]

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

    def advance_fleets(self, sim_time: float, dt: Optional[float] = None) -> None:
        step = float(self.config.dt if dt is None else dt)
        for fleet in self._route_fleets.values():
            for jeep in fleet:
                if sim_time < getattr(jeep, "start_time", 0.0):
                    continue
                jeep.update(step)

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
            "cached_paths": len(self._path_cache),
            "cached_nodes": len(self._nearest_node_cache),
        }


TandemPassengerJeepSimulation = Simulation
