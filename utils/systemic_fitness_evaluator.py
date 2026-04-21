"""Stochastic sensitivity analysis for candidate route fitness.

The evaluator compares a candidate physical route against multiple background
route systems so route learning favors network synergy over isolated "super"
routes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baseline_route_generator import BaselineRouteGenerator
from .jeepney_route_env import RouteFitnessResult, calculate_route_fitness
from .passenger_generation import PassengerMap


@dataclass(slots=True)
class SystemicFitnessResult:
    """Summary statistics for stochastic multi-system route evaluation."""

    average_gtc: float
    std_gtc: float
    average_passenger_gtc_std: float
    std_passenger_gtc_std: float
    average_reward: float
    std_reward: float
    n_tests: int
    per_test_gtc: list[float]
    per_test_passenger_gtc_std: list[float]
    per_test_reward: list[float]
    per_test_background_route_counts: list[int]
    per_test_route_results: list[RouteFitnessResult]


class SystemicFitnessEvaluator:
    """Evaluate a candidate route across multiple stochastic background systems.

    Stochastic sensitivity analysis follows the transit-network resilience idea in:
    https://www.researchgate.net/publication/330809142_Evaluating_transit_network_resilience_through_graph_theory_and_demand-elastic_measures_Case_study_of_the_Toronto_transit_system
    """

    def __init__(
        self,
        *,
        passenger_map: PassengerMap | None = None,
        drive_graph_raw=None,
        drive_graph_proj=None,
        place_queries: Sequence[str] | None = None,
        point_query: str | None = None,
        point_dist: float = 30_000.0,
        edges_csv: str | Path | None = None,
        nodes_csv: str | Path | None = None,
        config_path: str | Path | None = None,
        weight_profile: str = "full_ride_manager",
        unserved_penalty_beta: float = 2.0,
        evaluation_test_mean: float = 10.0,
        evaluation_test_std: float = 5.0,
        min_evaluation_tests: int = 1,
        max_evaluation_tests: int | None = None,
        background_route_mean: float = 2.0,
        background_route_std: float = 1.0,
        min_noise_routes: int = 1,
        max_noise_routes: int = 3,
        batch_size: int | None = None,
        max_workers: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.passenger_map = passenger_map or PassengerMap()
        self.edges_csv = Path(edges_csv) if edges_csv is not None else None
        self.nodes_csv = Path(nodes_csv) if nodes_csv is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self.weight_profile = str(weight_profile)
        self.unserved_penalty_beta = float(unserved_penalty_beta)
        self.evaluation_test_mean = float(evaluation_test_mean)
        self.evaluation_test_std = max(float(evaluation_test_std), 0.0)
        self.min_evaluation_tests = max(int(min_evaluation_tests), 1)
        self.max_evaluation_tests = (
            max(int(max_evaluation_tests), self.min_evaluation_tests)
            if max_evaluation_tests is not None
            else None
        )
        self.background_route_mean = float(background_route_mean)
        self.background_route_std = max(float(background_route_std), 0.0)
        self.min_noise_routes = max(int(min_noise_routes), 0)
        self.max_noise_routes = max(int(max_noise_routes), self.min_noise_routes)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.max_workers = max(int(max_workers), 1) if max_workers is not None else None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._background_generator = BaselineRouteGenerator(
            passenger_map=self.passenger_map,
            place_queries=list(place_queries) if place_queries is not None else None,
            point_query=point_query,
            point_dist=point_dist,
            drive_graph_raw=drive_graph_raw,
            drive_graph_proj=drive_graph_proj,
            seed=seed,
        )
        self.drive_graph_raw = self._background_generator.drive_graph_raw
        self.drive_graph_proj = self._background_generator.drive_graph_proj
        self._evaluation_cache: dict[tuple, SystemicFitnessResult] = {}

    @staticmethod
    def _route_signature(route_like: Any) -> tuple[Any, ...]:
        if hasattr(route_like, "path_node_ids"):
            return tuple(int(node_id) for node_id in getattr(route_like, "path_node_ids"))
        if hasattr(route_like, "nodes"):
            return tuple(str(node_id) for node_id in getattr(route_like, "nodes"))
        if isinstance(route_like, Sequence) and not isinstance(route_like, (str, bytes)):
            items = list(route_like)
            if items and all(str(item).startswith("ride_") for item in items):
                return tuple(str(item) for item in items)
            return tuple(int(item) for item in items)
        return (str(route_like),)

    def _sample_positive_int(
        self,
        mean: float,
        std: float,
        rng: np.random.Generator,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        if std <= 0.0:
            sample = int(round(mean))
        else:
            sample = int(round(rng.normal(mean, std)))
        sample = max(sample, minimum)
        if maximum is not None:
            sample = min(sample, maximum)
        return sample

    def _sample_evaluation_count(self, rng: np.random.Generator) -> int:
        return self._sample_positive_int(
            self.evaluation_test_mean,
            self.evaluation_test_std,
            rng,
            minimum=self.min_evaluation_tests,
            maximum=self.max_evaluation_tests,
        )

    def _sample_noise_route_count(self, rng: np.random.Generator) -> int:
        return self._sample_positive_int(
            self.background_route_mean,
            self.background_route_std,
            rng,
            minimum=self.min_noise_routes,
            maximum=self.max_noise_routes,
        )

    def _worker_count(self, task_count: int) -> int:
        if task_count < 2:
            return 1
        if self.max_workers is not None:
            return min(self.max_workers, task_count)
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, task_count)

    def _evaluate_single_system(
        self,
        *,
        candidate_route: Any,
        route_id: str,
        noise_route_prefix: str,
        resolved_batch_size: int,
        test_index: int,
        noise_count: int,
        background_seed: int,
        fitness_seed: int,
    ) -> tuple[float, float, float, int, RouteFitnessResult]:
        background_routes = self._background_generator.generate_routes(
            noise_count,
            route_prefix=f"{noise_route_prefix}{test_index + 1:02d}",
            seed=background_seed,
        )
        route_result = calculate_route_fitness(
            candidate_route,
            background_routes=background_routes,
            passenger_map=self.passenger_map,
            drive_graph_raw=self.drive_graph_raw,
            drive_graph_proj=self.drive_graph_proj,
            config_path=self.config_path,
            edges_csv=self.edges_csv,
            nodes_csv=self.nodes_csv,
            batch_size=resolved_batch_size,
            seed=fitness_seed,
            route_id=route_id,
            weight_profile=self.weight_profile,
            unserved_penalty_beta=self.unserved_penalty_beta,
        )
        return (
            float(route_result.average_gtc),
            float(getattr(route_result, "passenger_gtc_std", 0.0)),
            float(route_result.reward),
            noise_count,
            route_result,
        )

    def evaluate(
        self,
        candidate_route: Any,
        *,
        n_tests: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        route_id: str = "CANDIDATE",
        noise_route_prefix: str = "SYS",
    ) -> SystemicFitnessResult:
        rng = np.random.default_rng(self._seed if seed is None else seed)
        resolved_n_tests = self._sample_evaluation_count(rng) if n_tests is None else int(n_tests)
        if resolved_n_tests < 1:
            raise ValueError("n_tests must be at least 1.")

        resolved_batch_size = self.batch_size if batch_size is None else int(batch_size)
        cache_key = (
            self._route_signature(candidate_route),
            resolved_n_tests,
            int(self._seed if seed is None else seed),
            resolved_batch_size,
            str(route_id),
            str(noise_route_prefix),
        )
        cached = self._evaluation_cache.get(cache_key)
        if cached is not None:
            return cached

        gtc_values: list[float] = []
        passenger_gtc_std_values: list[float] = []
        reward_values: list[float] = []
        noise_counts: list[int] = []
        route_results: list[RouteFitnessResult] = []

        test_specs: list[tuple[int, int, int, int]] = []
        for test_index in range(resolved_n_tests):
            # Stochastic sensitivity analysis for transit network design:
            # the candidate is scored against multiple background route systems
            # to discourage routes that only work well in a vacuum.
            noise_count = self._sample_noise_route_count(rng)
            test_specs.append(
                (
                    test_index,
                    noise_count,
                    int(rng.integers(0, np.iinfo(np.int32).max)),
                    int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            )

        worker_count = self._worker_count(resolved_n_tests)
        if worker_count == 1:
            results = [
                self._evaluate_single_system(
                    candidate_route=candidate_route,
                    route_id=route_id,
                    noise_route_prefix=noise_route_prefix,
                    resolved_batch_size=resolved_batch_size,
                    test_index=test_index,
                    noise_count=noise_count,
                    background_seed=background_seed,
                    fitness_seed=fitness_seed,
                )
                for test_index, noise_count, background_seed, fitness_seed in test_specs
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    executor.map(
                        lambda spec: self._evaluate_single_system(
                            candidate_route=candidate_route,
                            route_id=route_id,
                            noise_route_prefix=noise_route_prefix,
                            resolved_batch_size=resolved_batch_size,
                            test_index=spec[0],
                            noise_count=spec[1],
                            background_seed=spec[2],
                            fitness_seed=spec[3],
                        ),
                        test_specs,
                    )
                )

        for (test_index, noise_count, _background_seed, _fitness_seed), (avg_gtc, passenger_std, reward, _noise_count, route_result) in zip(
            test_specs,
            results,
        ):
            noise_counts.append(noise_count)
            route_results.append(route_result)
            gtc_values.append(avg_gtc)
            passenger_gtc_std_values.append(passenger_std)
            reward_values.append(reward)

        average_gtc = float(np.mean(gtc_values)) if gtc_values else 0.0
        std_gtc = float(np.std(gtc_values, ddof=0)) if len(gtc_values) > 1 else 0.0
        average_passenger_gtc_std = float(np.mean(passenger_gtc_std_values)) if passenger_gtc_std_values else 0.0
        std_passenger_gtc_std = (
            float(np.std(passenger_gtc_std_values, ddof=0)) if len(passenger_gtc_std_values) > 1 else 0.0
        )
        average_reward = float(np.mean(reward_values)) if reward_values else 0.0
        std_reward = float(np.std(reward_values, ddof=0)) if len(reward_values) > 1 else 0.0

        result = SystemicFitnessResult(
            average_gtc=average_gtc,
            std_gtc=std_gtc,
            average_passenger_gtc_std=average_passenger_gtc_std,
            std_passenger_gtc_std=std_passenger_gtc_std,
            average_reward=average_reward,
            std_reward=std_reward,
            n_tests=int(resolved_n_tests),
            per_test_gtc=gtc_values,
            per_test_passenger_gtc_std=passenger_gtc_std_values,
            per_test_reward=reward_values,
            per_test_background_route_counts=noise_counts,
            per_test_route_results=route_results,
        )
        self._evaluation_cache[cache_key] = result
        return result
