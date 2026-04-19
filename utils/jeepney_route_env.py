"""Gymnasium-style RL environment for geometric jeepney route construction.

The environment operates only on the primal physical street network. The
three-layer travel graph remains reserved for downstream evaluation of the
final route under generalized travel cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from .travel_graph import load_graphs_for_study_area, node_table_from_graph

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
        place_queries: list[str] | None = None,
        point_query: str | None = None,
        point_dist: float = 30_000.0,
        seed: int | None = None,
        max_steps: int = 128,
        min_route_nodes: int = 4,
        max_candidates: int | None = None,
        turn_penalty_weight: float = 1.0,
        repeat_uturn_penalty: float = 1.5,
        length_penalty_weight: float = 0.01,
        revisit_penalty_weight: float = 0.5,
        dead_end_penalty: float = 2.0,
        invalid_action_penalty: float = 1.0,
        closure_bonus: float = 5.0,
        termination_penalty: float = 4.0,
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

        self.node_table = node_table_from_graph(self.drive_graph_raw, self.drive_graph_proj)
        self.node_table["base_node_id"] = self.node_table["base_node_id"].astype(int)
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

        self._successors = self._build_successor_cache()
        self._start_nodes = [node for node, succ in self._successors.items() if len(succ) > 0]
        self._origin_candidates = [node for node, succ in self._successors.items() if len(succ) >= 2]
        if not self._origin_candidates:
            self._origin_candidates = list(self._start_nodes)

        if not self._start_nodes:
            raise ValueError("The drive graph has no navigable outgoing edges.")

        self.max_steps = max(int(max_steps), 1)
        self.min_route_nodes = max(int(min_route_nodes), 2)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.repeat_uturn_penalty = float(repeat_uturn_penalty)
        self.length_penalty_weight = float(length_penalty_weight)
        self.revisit_penalty_weight = float(revisit_penalty_weight)
        self.dead_end_penalty = float(dead_end_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.closure_bonus = float(closure_bonus)
        self.termination_penalty = float(termination_penalty)
        self.u_turn_threshold = 0.85 * pi

        self.max_candidates = int(max_candidates or max((len(v) for v in self._successors.values()), default=1))
        self.max_candidates = max(self.max_candidates, 1)

        self._x_span = float(self.node_table["x"].max() - self.node_table["x"].min()) if not self.node_table.empty else 1.0
        self._y_span = float(self.node_table["y"].max() - self.node_table["y"].min()) if not self.node_table.empty else 1.0
        self._distance_scale = max(hypot(self._x_span, self._y_span), 1.0)
        self._edge_scale = max(self._max_edge_length(), 1.0)

        self.action_space = spaces.Discrete(self.max_candidates + 1)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
                "candidates": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.max_candidates, 6), dtype=np.float32
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
            return np.zeros((self.max_candidates, 6), dtype=np.float32), np.zeros(
                (self.max_candidates + 1,), dtype=np.float32
            )

        rows: list[list[float]] = []
        for cand in self.current_candidates[: self.max_candidates]:
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
            rows.append(
                [
                    float(sin(angle)),
                    float(cos(angle)),
                    float(np.clip(cand.length_m / self._edge_scale, 0.0, 1.0)),
                    alignment,
                    backtrack,
                    uturn,
                ]
            )

        pad_rows = self.max_candidates - len(rows)
        if pad_rows > 0:
            rows.extend([[0.0] * 6 for _ in range(pad_rows)])

        mask = np.zeros((self.max_candidates + 1,), dtype=np.float32)
        mask[: len(self.current_candidates[: self.max_candidates])] = 1.0
        mask[self.max_candidates] = 1.0
        return np.asarray(rows, dtype=np.float32), mask

    def _global_features(self) -> np.ndarray:
        if self.current_node_id is None:
            return np.zeros((7,), dtype=np.float32)
        distance, bearing = self._distance_and_bearing_to_origin()
        sinuosity = self._route_sinuosity()
        return np.asarray(
            [
                float(np.clip(distance / self._distance_scale, 0.0, 1.0)),
                float(sin(bearing)),
                float(cos(bearing)),
                float(np.clip(sinuosity / 10.0, 0.0, 1.0)),
                float(np.clip(self.cumulative_length_m / self._distance_scale, 0.0, 1.0)),
                float(np.clip(self.cumulative_turn_penalty / 10.0, 0.0, 1.0)),
                float(np.clip(self.consecutive_uturns / 5.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        candidates, mask = self._candidate_features()
        return {
            "global": self._global_features(),
            "candidates": candidates,
            "action_mask": mask,
        }

    def _flat_state(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                observation["global"].ravel(),
                observation["candidates"].ravel(),
                observation["action_mask"].ravel(),
            ]
        ).astype(np.float32, copy=False)

    def _current_polygon_area_m2(self) -> float:
        if len(self.path_node_ids) < 4:
            return 0.0
        points = [
            (self._x_by_node[node_id], self._y_by_node[node_id])
            for node_id in self.path_node_ids
        ]
        polygon = Polygon(points)
        if polygon.is_empty:
            return 0.0
        return float(polygon.area)

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
            turn_angle = abs(self._signed_angle(incoming, vector_xy))
            turn_penalty = self._continuous_turn_penalty(turn_angle) * self.turn_penalty_weight
            if turn_angle >= self.u_turn_threshold:
                self.consecutive_uturns += 1
                turn_penalty += self.repeat_uturn_penalty * max(self.consecutive_uturns - 1, 0)
            else:
                self.consecutive_uturns = 0
        else:
            self.consecutive_uturns = 0

        revisit_penalty = self.revisit_penalty_weight if next_node_id in self.visited_node_ids else 0.0
        length_penalty = self.length_penalty_weight * (length_m / self._distance_scale)
        reward = -(turn_penalty + revisit_penalty + length_penalty)

        self.previous_node_id = current
        self.current_node_id = next_node_id
        self.path_node_ids.append(next_node_id)
        self.visited_node_ids.add(next_node_id)
        self.cumulative_length_m += length_m
        self.cumulative_turn_penalty += turn_penalty
        self._last_turn_angle = turn_angle
        self._current_reference_vector = vector_xy
        self.steps_taken += 1
        self._set_candidates(next_node_id)

        terminated = False
        if next_node_id == self.origin_node_id and len(self.path_node_ids) >= self.min_route_nodes:
            terminated = True
            reward += self.closure_bonus + min(self._current_polygon_area_m2() / (self._distance_scale**2), 1.0)

        info = {
            "turn_angle_rad": turn_angle,
            "turn_penalty": turn_penalty,
            "route_length_m": self.cumulative_length_m,
            "sinuosity_index": self._route_sinuosity(),
            "distance_to_origin_m": self._distance_and_bearing_to_origin()[0],
            "bearing_to_origin_rad": self._distance_and_bearing_to_origin()[1],
            "route_area_m2": self._current_polygon_area_m2(),
            "state_vector": self._flat_state(self._observation()),
        }
        return reward, terminated, info

    def step(self, action: int):
        if self.current_node_id is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        valid_moves = self.current_candidates[: self.max_candidates]
        terminate_action = self.max_candidates
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if action == terminate_action:
            closed = self.current_node_id == self.origin_node_id and len(self.path_node_ids) >= self.min_route_nodes
            terminated = True
            if closed:
                reward = self.closure_bonus + min(self._current_polygon_area_m2() / (self._distance_scale**2), 1.0)
            else:
                reward = -self.termination_penalty
            info = {
                "turn_angle_rad": 0.0,
                "turn_penalty": 0.0,
                "route_length_m": self.cumulative_length_m,
                "sinuosity_index": self._route_sinuosity(),
                "distance_to_origin_m": self._distance_and_bearing_to_origin()[0],
                "bearing_to_origin_rad": self._distance_and_bearing_to_origin()[1],
                "route_area_m2": self._current_polygon_area_m2(),
                "state_vector": self._flat_state(self._observation()),
                "terminated_reason": "closed_loop" if closed else "agent_terminated",
            }
            observation = self._observation()
            return observation, float(reward), terminated, truncated, info

        if action < 0 or action >= len(valid_moves):
            reward = -self.invalid_action_penalty
            self.steps_taken += 1
            truncated = self.steps_taken >= self.max_steps
            observation = self._observation()
            info = {
                "turn_angle_rad": 0.0,
                "turn_penalty": 0.0,
                "route_length_m": self.cumulative_length_m,
                "sinuosity_index": self._route_sinuosity(),
                "distance_to_origin_m": self._distance_and_bearing_to_origin()[0],
                "bearing_to_origin_rad": self._distance_and_bearing_to_origin()[1],
                "route_area_m2": self._current_polygon_area_m2(),
                "state_vector": self._flat_state(observation),
                "terminated_reason": "invalid_action",
            }
            return observation, float(reward), terminated, truncated, info

        chosen = valid_moves[action]
        reward, terminated, info = self._apply_move(chosen.next_node_id)

        if len(self.current_candidates) == 0 and not terminated:
            terminated = True
            reward -= self.dead_end_penalty
            info["terminated_reason"] = "dead_end"
        elif terminated:
            info["terminated_reason"] = "closed_loop"

        truncated = self.steps_taken >= self.max_steps and not terminated
        if truncated:
            info["terminated_reason"] = "max_steps"

        observation = self._observation()
        info["state_vector"] = self._flat_state(observation)
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:  # pragma: no cover - trivial lifecycle hook
        return None

