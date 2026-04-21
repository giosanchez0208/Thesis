"""PPO training helpers for jeepney route exploration."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .jeepney_route_env import JeepneyRouteEnv
from .systemic_fitness_evaluator import SystemicFitnessEvaluator


@dataclass(slots=True)
class RouteTrainingSnapshot:
    """Captured route data for a standout training episode."""

    episode_index: int
    episode_return: float
    fitness_reward: float
    average_gtc: float
    std_gtc: float
    route_path_node_ids: list[int]
    route_latlon: list[tuple[float, float]]
    output_html: Path
    output_json: Path


@dataclass(slots=True)
class RouteTrainingArtifacts:
    """Outputs from a short PPO training run."""

    model: PPO
    best_snapshot: RouteTrainingSnapshot | None = None
    worst_snapshot: RouteTrainingSnapshot | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    history_csv: Path | None = None
    snapshot_csv: Path | None = None


def _serialise_route_nodes(route_node_ids: Iterable[int]) -> str:
    return json.dumps([int(node_id) for node_id in route_node_ids])


def _snapshot_to_row(label: str, snapshot: RouteTrainingSnapshot) -> dict[str, Any]:
    return {
        "snapshot_label": label,
        "episode_index": snapshot.episode_index,
        "episode_return": snapshot.episode_return,
        "fitness_reward": snapshot.fitness_reward,
        "average_gtc": snapshot.average_gtc,
        "std_gtc": snapshot.std_gtc,
        "route_node_count": len(snapshot.route_path_node_ids),
        "route_path_node_ids": _serialise_route_nodes(snapshot.route_path_node_ids),
        "output_html": str(snapshot.output_html),
        "output_json": str(snapshot.output_json),
    }


def export_training_results_csvs(
    *,
    output_dir: str | Path,
    history: list[dict[str, Any]],
    best_snapshot: RouteTrainingSnapshot | None,
    worst_snapshot: RouteTrainingSnapshot | None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    history_rows: list[dict[str, Any]] = []
    for record in history:
        row = dict(record)
        route_node_ids = row.get("route_path_node_ids")
        if isinstance(route_node_ids, Iterable) and not isinstance(route_node_ids, (str, bytes)):
            route_node_ids_list = [int(node_id) for node_id in route_node_ids]
            row["route_node_count"] = len(route_node_ids_list)
            row["route_path_node_ids"] = _serialise_route_nodes(route_node_ids_list)
        else:
            row.setdefault("route_node_count", 0)
        history_rows.append(row)

    history_columns = [
        "episode_index",
        "episode_return",
        "terminated_reason",
        "closure_mode",
        "closed_loop",
        "fitness_reward",
        "average_gtc",
        "std_gtc",
        "route_node_count",
        "route_path_node_ids",
    ]
    history_csv = out_dir / "training_history.csv"
    pd.DataFrame(history_rows, columns=history_columns).to_csv(history_csv, index=False)

    snapshot_rows: list[dict[str, Any]] = []
    if best_snapshot is not None:
        snapshot_rows.append(_snapshot_to_row("best", best_snapshot))
    if worst_snapshot is not None:
        snapshot_rows.append(_snapshot_to_row("worst", worst_snapshot))

    snapshot_csv = out_dir / "training_snapshots.csv"
    snapshot_columns = [
        "snapshot_label",
        "episode_index",
        "episode_return",
        "fitness_reward",
        "average_gtc",
        "std_gtc",
        "route_node_count",
        "route_path_node_ids",
        "output_html",
        "output_json",
    ]
    pd.DataFrame(snapshot_rows, columns=snapshot_columns).to_csv(snapshot_csv, index=False)

    return history_csv, snapshot_csv


def route_nodes_to_latlon(route_node_ids: Iterable[int], drive_graph_raw) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for node_id in route_node_ids:
        try:
            node = drive_graph_raw.nodes[int(node_id)]
        except KeyError:
            continue
        lon = float(node.get("x", 0.0))
        lat = float(node.get("y", 0.0))
        coords.append((lat, lon))
    return coords


def export_physical_route_html(
    route_node_ids: Iterable[int],
    drive_graph_raw,
    output_html: str | Path,
    *,
    title: str,
    subtitle: str | None = None,
) -> Path:
    route_latlon = route_nodes_to_latlon(route_node_ids, drive_graph_raw)
    if not route_latlon:
        raise ValueError("route_node_ids did not resolve to any coordinates.")

    out = Path(output_html).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    center = route_latlon[len(route_latlon) // 2]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    folium.PolyLine(route_latlon, color="#2563eb", weight=5, opacity=0.9).add_to(fmap)
    folium.Marker(route_latlon[0], popup="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(route_latlon[-1], popup="End", icon=folium.Icon(color="red")).add_to(fmap)

    header = title if subtitle is None else f"{title}<br><small>{subtitle}</small>"
    fmap.get_root().html.add_child(
        folium.Element(
            f"""
            <div style="position: fixed; top: 10px; left: 10px; z-index: 9999;
                        background: white; padding: 10px 12px; border: 1px solid #cbd5e1;
                        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.12);
                        font-family: Arial, sans-serif; font-size: 13px;">
              <div style="font-weight: 700; margin-bottom: 4px;">{header}</div>
              <div>Nodes: {len(route_node_ids)}</div>
            </div>
            """
        )
    )
    fmap.save(str(out))
    return out


class BestWorstRouteCallback(BaseCallback):
    """Track and export the best and worst routes seen during PPO training."""

    def __init__(
        self,
        *,
        drive_graph_raw,
        output_dir: str | Path,
        heartbeat_seconds: float = 60.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.drive_graph_raw = drive_graph_raw
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_seconds = max(float(heartbeat_seconds), 0.0)
        self.best_snapshot: RouteTrainingSnapshot | None = None
        self.worst_snapshot: RouteTrainingSnapshot | None = None
        self.history: list[dict[str, Any]] = []
        self._episode_returns: list[float] = []
        self._episode_index = 0
        self._closed_loop_count = 0
        self._forced_loop_count = 0
        self._last_heartbeat = 0.0

    def _init_callback(self) -> None:
        self._episode_returns = [0.0 for _ in range(self.training_env.num_envs)]
        self._closed_loop_count = 0
        self._forced_loop_count = 0
        self._last_heartbeat = time.monotonic()

    def _emit_heartbeat(self) -> None:
        if self.heartbeat_seconds <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_heartbeat < self.heartbeat_seconds:
            return
        self._last_heartbeat = now
        best_return = self.best_snapshot.episode_return if self.best_snapshot is not None else float("nan")
        print(
            f"[training] episodes={self._episode_index} "
            f"closed_loops={self._closed_loop_count} "
            f"forced_loops={self._forced_loop_count} "
            f"best_return={best_return:.3f}",
            flush=True,
        )

    def _capture_snapshot(
        self,
        *,
        episode_index: int,
        episode_return: float,
        info: dict[str, Any],
    ) -> RouteTrainingSnapshot | None:
        route_node_ids = [int(node_id) for node_id in info.get("route_path_node_ids", [])]
        if not route_node_ids:
            return None

        fitness = info.get("route_fitness")
        average_gtc = float(getattr(fitness, "average_gtc", info.get("fitness_average_gtc", 0.0)))
        std_gtc = float(getattr(fitness, "std_gtc", 0.0))
        fitness_reward = float(getattr(fitness, "reward", info.get("fitness_reward", episode_return)))
        route_latlon = route_nodes_to_latlon(route_node_ids, self.drive_graph_raw)
        html_path = self.output_dir / "route_snapshot.html"
        json_path = self.output_dir / "route_snapshot.json"
        payload = {
            "episode_index": episode_index,
            "episode_return": episode_return,
            "fitness_reward": fitness_reward,
            "average_gtc": average_gtc,
            "std_gtc": std_gtc,
            "route_path_node_ids": route_node_ids,
            "route_latlon": route_latlon,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return RouteTrainingSnapshot(
            episode_index=episode_index,
            episode_return=episode_return,
            fitness_reward=fitness_reward,
            average_gtc=average_gtc,
            std_gtc=std_gtc,
            route_path_node_ids=route_node_ids,
            route_latlon=route_latlon,
            output_html=html_path,
            output_json=json_path,
        )

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for index, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
            self._episode_returns[index] += float(reward)
            if not done:
                continue

            episode_return = self._episode_returns[index]
            self._episode_returns[index] = 0.0
            self._episode_index += 1
            closed_loop = info.get("terminated_reason") == "closed_loop"
            closure_mode = info.get("closure_mode")
            fitness = info.get("route_fitness")
            route_node_ids = [int(node_id) for node_id in info.get("route_path_node_ids", [])]
            history_record: dict[str, Any] = {
                "episode_index": self._episode_index,
                "episode_return": episode_return,
                "terminated_reason": info.get("terminated_reason"),
                "closure_mode": closure_mode,
                "closed_loop": closed_loop,
                "fitness_reward": float(getattr(fitness, "reward", info.get("fitness_reward", episode_return))),
                "average_gtc": float(getattr(fitness, "average_gtc", info.get("fitness_average_gtc", np.nan))),
                "std_gtc": float(getattr(fitness, "passenger_gtc_std", info.get("fitness_passenger_gtc_std", np.nan))),
                "route_node_count": len(route_node_ids),
            }
            if route_node_ids:
                history_record["route_path_node_ids"] = route_node_ids
            self.history.append(history_record)
            if closed_loop:
                self._closed_loop_count += 1
            if closure_mode == "forced":
                self._forced_loop_count += 1

            if not closed_loop:
                self._emit_heartbeat()
                continue

            snapshot = self._capture_snapshot(
                episode_index=self._episode_index,
                episode_return=episode_return,
                info=info,
            )
            if snapshot is None:
                continue

            if self.best_snapshot is None or snapshot.episode_return > self.best_snapshot.episode_return:
                self.best_snapshot = snapshot
                self.best_snapshot.output_html = export_physical_route_html(
                    snapshot.route_path_node_ids,
                    self.drive_graph_raw,
                    self.output_dir / "best_route.html",
                    title="Best Route",
                    subtitle=f"episode {snapshot.episode_index} | return {snapshot.episode_return:.3f}",
                )
                self.best_snapshot.output_json = self.output_dir / "best_route.json"
                self.best_snapshot.output_json.write_text(
                    json.dumps(
                        {
                            "episode_index": snapshot.episode_index,
                            "episode_return": snapshot.episode_return,
                            "fitness_reward": snapshot.fitness_reward,
                            "average_gtc": snapshot.average_gtc,
                            "std_gtc": snapshot.std_gtc,
                            "route_path_node_ids": snapshot.route_path_node_ids,
                            "route_latlon": snapshot.route_latlon,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if self.worst_snapshot is None or snapshot.episode_return < self.worst_snapshot.episode_return:
                self.worst_snapshot = snapshot
                self.worst_snapshot.output_html = export_physical_route_html(
                    snapshot.route_path_node_ids,
                    self.drive_graph_raw,
                    self.output_dir / "worst_route.html",
                    title="Worst Route",
                    subtitle=f"episode {snapshot.episode_index} | return {snapshot.episode_return:.3f}",
                )
                self.worst_snapshot.output_json = self.output_dir / "worst_route.json"
                self.worst_snapshot.output_json.write_text(
                    json.dumps(
                        {
                            "episode_index": snapshot.episode_index,
                            "episode_return": snapshot.episode_return,
                            "fitness_reward": snapshot.fitness_reward,
                            "average_gtc": snapshot.average_gtc,
                            "std_gtc": snapshot.std_gtc,
                            "route_path_node_ids": snapshot.route_path_node_ids,
                            "route_latlon": snapshot.route_latlon,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            self._emit_heartbeat()

        return True


def build_training_env(
    *,
    passenger_map,
    drive_graph_raw,
    drive_graph_proj,
    systemic_test_mean: float = 2.0,
    systemic_test_std: float = 0.0,
    background_route_mean: float = 1.0,
    background_route_std: float = 0.0,
    systemic_batch_size: int = 8,
    systemic_std_penalty_weight: float = 1.0,
    max_workers: int | None = None,
    seed: int | None = None,
    **env_kwargs: Any,
) -> JeepneyRouteEnv:
    evaluator = SystemicFitnessEvaluator(
        passenger_map=passenger_map,
        drive_graph_raw=drive_graph_raw,
        drive_graph_proj=drive_graph_proj,
        evaluation_test_mean=systemic_test_mean,
        evaluation_test_std=systemic_test_std,
        background_route_mean=background_route_mean,
        background_route_std=background_route_std,
        batch_size=systemic_batch_size,
        std_penalty_weight=systemic_std_penalty_weight,
        max_workers=max_workers,
        seed=seed,
    )
    return JeepneyRouteEnv(
        drive_graph_raw=drive_graph_raw,
        drive_graph_proj=drive_graph_proj,
        passenger_map=passenger_map,
        systemic_evaluator=evaluator,
        systemic_std_penalty_weight=systemic_std_penalty_weight,
        seed=seed,
        **env_kwargs,
    )


def train_route_agent(
    *,
    passenger_map,
    drive_graph_raw,
    drive_graph_proj,
    output_dir: str | Path,
    seed: int | None = None,
    total_timesteps: int = 4_000,
    systemic_test_mean: float = 2.0,
    systemic_test_std: float = 0.0,
    background_route_mean: float = 1.0,
    background_route_std: float = 0.0,
    systemic_batch_size: int = 8,
    systemic_std_penalty_weight: float = 1.0,
    systemic_max_workers: int | None = None,
    ppo_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    heartbeat_seconds: float = 60.0,
) -> RouteTrainingArtifacts:
    env_kwargs = dict(env_kwargs or {})
    ppo_kwargs = dict(ppo_kwargs or {})
    ppo_verbose = int(ppo_kwargs.pop("verbose", 1))
    training_env = DummyVecEnv(
        [
            lambda: Monitor(
                build_training_env(
                    passenger_map=passenger_map,
                    drive_graph_raw=drive_graph_raw,
                    drive_graph_proj=drive_graph_proj,
                    systemic_test_mean=systemic_test_mean,
                    systemic_test_std=systemic_test_std,
                    background_route_mean=background_route_mean,
                    background_route_std=background_route_std,
                    systemic_batch_size=systemic_batch_size,
                    systemic_std_penalty_weight=systemic_std_penalty_weight,
                    max_workers=systemic_max_workers,
                    seed=seed,
                    **env_kwargs,
                )
            )
        ]
    )

    model = PPO(
        "MultiInputPolicy",
        training_env,
        seed=seed,
        verbose=ppo_verbose,
        **ppo_kwargs,
    )
    callback = BestWorstRouteCallback(
        drive_graph_raw=drive_graph_raw,
        output_dir=output_dir,
        heartbeat_seconds=heartbeat_seconds,
    )
    model.learn(total_timesteps=int(total_timesteps), callback=callback)
    history_csv, snapshot_csv = export_training_results_csvs(
        output_dir=output_dir,
        history=callback.history,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
    )
    return RouteTrainingArtifacts(
        model=model,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
        history=callback.history,
        history_csv=history_csv,
        snapshot_csv=snapshot_csv,
    )
