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

try:  # pragma: no cover - optional dashboard dependency
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - degrade gracefully without live UI
    Console = None
    Live = None
    Panel = None
    Table = None


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


def _deserialize_route_nodes(route_node_ids: Any) -> list[int]:
    if route_node_ids is None or (isinstance(route_node_ids, float) and np.isnan(route_node_ids)):
        return []
    if isinstance(route_node_ids, list):
        return [int(node_id) for node_id in route_node_ids]
    if isinstance(route_node_ids, tuple):
        return [int(node_id) for node_id in route_node_ids]
    if isinstance(route_node_ids, str):
        text = route_node_ids.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [int(token) for token in text.split(",") if token.strip()]
        if isinstance(parsed, list):
            return [int(node_id) for node_id in parsed]
    return []


def _load_training_history(output_dir: str | Path) -> list[dict[str, Any]]:
    out_dir = Path(output_dir).resolve()
    history_csv = out_dir / "training_history.csv"
    if not history_csv.exists():
        return []

    frame = pd.read_csv(history_csv)
    history: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record = dict(row)
        for key in ("episode_index", "route_node_count"):
            if key in record and pd.notna(record[key]):
                record[key] = int(record[key])
        if "closed_loop" in record and pd.notna(record["closed_loop"]):
            record["closed_loop"] = bool(record["closed_loop"])
        for key in ("terminated_reason", "closure_mode"):
            if key in record and pd.isna(record[key]):
                record[key] = None
        route_node_ids = _deserialize_route_nodes(record.get("route_path_node_ids"))
        if route_node_ids:
            record["route_path_node_ids"] = route_node_ids
        else:
            record.pop("route_path_node_ids", None)
        history.append(record)
    return history


def _load_snapshot_from_json(snapshot_json: Path) -> RouteTrainingSnapshot | None:
    if not snapshot_json.exists():
        return None
    payload = json.loads(snapshot_json.read_text(encoding="utf-8"))
    route_node_ids = [int(node_id) for node_id in payload.get("route_path_node_ids", [])]
    route_latlon = [tuple(float(coord) for coord in pair) for pair in payload.get("route_latlon", [])]
    return RouteTrainingSnapshot(
        episode_index=int(payload.get("episode_index", 0)),
        episode_return=float(payload.get("episode_return", 0.0)),
        fitness_reward=float(payload.get("fitness_reward", payload.get("episode_return", 0.0))),
        average_gtc=float(payload.get("average_gtc", 0.0)),
        std_gtc=float(payload.get("std_gtc", 0.0)),
        route_path_node_ids=route_node_ids,
        route_latlon=route_latlon,
        output_html=snapshot_json.with_suffix(".html"),
        output_json=snapshot_json,
    )


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
        heartbeat_steps: int = 1000,
        checkpoint_seconds: float = 300.0,
        checkpoint_steps: int = 2000,
        checkpoint_path: str | Path | None = None,
        initial_history: list[dict[str, Any]] | None = None,
        initial_best_snapshot: RouteTrainingSnapshot | None = None,
        initial_worst_snapshot: RouteTrainingSnapshot | None = None,
        enable_rich_dashboard: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.drive_graph_raw = drive_graph_raw
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_seconds = max(float(heartbeat_seconds), 0.0)
        self.heartbeat_steps = max(int(heartbeat_steps), 0)
        self.checkpoint_seconds = max(float(checkpoint_seconds), 0.0)
        self.checkpoint_steps = max(int(checkpoint_steps), 0)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path is not None else self.output_dir / "ppo_latest_model.zip"
        self.state_path = self.output_dir / "training_state.json"
        self.best_snapshot = initial_best_snapshot
        self.worst_snapshot = initial_worst_snapshot
        self.history = [dict(record) for record in (initial_history or [])]
        self._episode_returns: list[float] = []
        self._episode_index = max((int(record.get("episode_index", 0)) for record in self.history), default=0)
        self._closed_loop_count = sum(1 for record in self.history if bool(record.get("closed_loop")))
        self._forced_loop_count = sum(1 for record in self.history if record.get("closure_mode") == "forced")
        self._last_heartbeat = 0.0
        self._last_heartbeat_step = 0
        self._last_checkpoint = 0.0
        self._last_checkpoint_step = 0
        self._latest_episode_return = 0.0
        self._latest_closure_mode = "unknown"
        self._latest_terminated_reason = "unknown"
        self._latest_route_node_count = 0
        self._dashboard_enabled = bool(
            enable_rich_dashboard and Console is not None and Live is not None and Panel is not None and Table is not None
        )
        self._console = Console() if self._dashboard_enabled and Console is not None else None
        self._live = None
        self.history_csv = self.output_dir / "training_history.csv"
        self.snapshot_csv = self.output_dir / "training_snapshots.csv"

    def _init_callback(self) -> None:
        self._episode_returns = [0.0 for _ in range(self.training_env.num_envs)]
        self._last_heartbeat = time.monotonic()
        self._last_heartbeat_step = self.num_timesteps
        self._last_checkpoint = self._last_heartbeat
        self._last_checkpoint_step = self.num_timesteps
        self._start_live_dashboard()
        self._maybe_checkpoint(force=True)

    def _dashboard_panel(self):
        if not self._dashboard_enabled or Panel is None or Table is None:
            return None

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="bold cyan")
        table.add_column(style="white")
        best_return = self.best_snapshot.episode_return if self.best_snapshot is not None else float("nan")
        table.add_row("timesteps", f"{self.num_timesteps}")
        table.add_row("episodes", f"{self._episode_index}")
        table.add_row("closed loops", f"{self._closed_loop_count}")
        table.add_row("forced loops", f"{self._forced_loop_count}")
        table.add_row("best return", f"{best_return:.3f}")
        table.add_row("latest return", f"{self._latest_episode_return:.3f}")
        table.add_row("closure mode", str(self._latest_closure_mode))
        table.add_row("termination", str(self._latest_terminated_reason))
        table.add_row("route nodes", f"{self._latest_route_node_count}")
        table.add_row("checkpoint", str(self.checkpoint_path))
        return Panel(table, title="B4B PPO training", border_style="cyan")

    def _start_live_dashboard(self) -> None:
        if not self._dashboard_enabled or self._live is not None or self._console is None or Live is None:
            return
        self._live = Live(self._dashboard_panel(), console=self._console, refresh_per_second=4, transient=False)
        self._live.__enter__()

    def _stop_live_dashboard(self) -> None:
        if self._live is None:
            return
        self._live.update(self._dashboard_panel())
        self._live.__exit__(None, None, None)
        self._live = None

    def _persist_state(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.checkpoint_path))
        self.history_csv, self.snapshot_csv = export_training_results_csvs(
            output_dir=self.output_dir,
            history=self.history,
            best_snapshot=self.best_snapshot,
            worst_snapshot=self.worst_snapshot,
        )
        self.state_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str(self.checkpoint_path),
                    "num_timesteps": int(self.num_timesteps),
                    "episode_index": int(self._episode_index),
                    "closed_loop_count": int(self._closed_loop_count),
                    "forced_loop_count": int(self._forced_loop_count),
                    "latest_episode_return": float(self._latest_episode_return),
                    "latest_closure_mode": self._latest_closure_mode,
                    "latest_terminated_reason": self._latest_terminated_reason,
                    "latest_route_node_count": int(self._latest_route_node_count),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _maybe_checkpoint(self, *, force: bool = False) -> None:
        now = time.monotonic()
        time_due = self.checkpoint_seconds <= 0.0 or now - self._last_checkpoint >= self.checkpoint_seconds
        step_due = self.checkpoint_steps > 0 and (self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_steps)
        if not force and not time_due and not step_due:
            return
        self._last_checkpoint = now
        self._last_checkpoint_step = self.num_timesteps
        self._persist_state()

    def _emit_heartbeat(self) -> None:
        now = time.monotonic()
        time_due = self.heartbeat_seconds <= 0.0 or now - self._last_heartbeat >= self.heartbeat_seconds
        step_due = self.heartbeat_steps > 0 and (self.num_timesteps - self._last_heartbeat_step >= self.heartbeat_steps)
        if not time_due and not step_due:
            return
        self._last_heartbeat = now
        self._last_heartbeat_step = self.num_timesteps
        self._maybe_checkpoint()
        if self._live is not None:
            self._live.update(self._dashboard_panel())
        best_return = self.best_snapshot.episode_return if self.best_snapshot is not None else float("nan")
        print(
            f"[training] timesteps={self.num_timesteps} "
            f"episodes={self._episode_index} "
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
            self._emit_heartbeat()
            if not done:
                continue

            episode_return = self._episode_returns[index]
            self._episode_returns[index] = 0.0
            self._episode_index += 1
            closed_loop = info.get("terminated_reason") == "closed_loop"
            closure_mode = info.get("closure_mode")
            fitness = info.get("route_fitness")
            route_node_ids = [int(node_id) for node_id in info.get("route_path_node_ids", [])]
            self._latest_episode_return = float(episode_return)
            self._latest_closure_mode = str(closure_mode) if closure_mode is not None else "unknown"
            self._latest_terminated_reason = str(info.get("terminated_reason") or "unknown")
            self._latest_route_node_count = len(route_node_ids)
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
            print(
                f"[training] episode={self._episode_index} "
                f"return={episode_return:.3f} "
                f"closed_loop={closed_loop} "
                f"closure_mode={closure_mode} "
                f"nodes={len(route_node_ids)}",
                flush=True,
            )
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
                self._emit_heartbeat()
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
                print(
                    f"[training] best snapshot updated: episode={snapshot.episode_index} "
                    f"return={snapshot.episode_return:.3f}",
                    flush=True,
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
                print(
                    f"[training] worst snapshot updated: episode={snapshot.episode_index} "
                    f"return={snapshot.episode_return:.3f}",
                    flush=True,
                )
            self._emit_heartbeat()

        return True

    def _on_training_end(self) -> None:
        self._persist_state()
        self._stop_live_dashboard()


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
    heartbeat_steps: int = 1000,
    checkpoint_seconds: float = 300.0,
    checkpoint_steps: int = 2000,
    resume_from_checkpoint: bool = True,
    use_rich_dashboard: bool = True,
) -> RouteTrainingArtifacts:
    env_kwargs = dict(env_kwargs or {})
    ppo_kwargs = dict(ppo_kwargs or {})
    ppo_verbose = int(ppo_kwargs.pop("verbose", 1))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    initial_history = _load_training_history(output_dir) if resume_from_checkpoint else []
    initial_best_snapshot = _load_snapshot_from_json(output_dir / "best_route.json") if resume_from_checkpoint else None
    initial_worst_snapshot = _load_snapshot_from_json(output_dir / "worst_route.json") if resume_from_checkpoint else None
    latest_checkpoint = output_dir / "ppo_latest_model.zip"
    final_model_path = output_dir / "ppo_final_model.zip"

    model: PPO
    if resume_from_checkpoint and latest_checkpoint.exists():
        model = PPO.load(str(latest_checkpoint), env=training_env, device="auto")
        print(f"[training] Resuming from checkpoint: {latest_checkpoint}", flush=True)
    elif resume_from_checkpoint and final_model_path.exists():
        model = PPO.load(str(final_model_path), env=training_env, device="auto")
        print(f"[training] Resuming from final model: {final_model_path}", flush=True)
    else:
        model = PPO(
            "MultiInputPolicy",
            training_env,
            seed=seed,
            verbose=ppo_verbose,
            **ppo_kwargs,
        )
        print("[training] Starting a fresh PPO model.", flush=True)

    current_timesteps = int(getattr(model, "num_timesteps", 0))
    remaining_timesteps = max(int(total_timesteps) - current_timesteps, 0)
    if current_timesteps > 0:
        print(
            f"[training] Loaded timesteps={current_timesteps}; target={int(total_timesteps)}; remaining={remaining_timesteps}.",
            flush=True,
        )
    else:
        print(f"[training] Target timesteps={int(total_timesteps)}.", flush=True)

    callback = BestWorstRouteCallback(
        drive_graph_raw=drive_graph_raw,
        output_dir=output_dir,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_steps=heartbeat_steps,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_steps=checkpoint_steps,
        checkpoint_path=latest_checkpoint,
        initial_history=initial_history,
        initial_best_snapshot=initial_best_snapshot,
        initial_worst_snapshot=initial_worst_snapshot,
        enable_rich_dashboard=use_rich_dashboard,
    )
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            reset_num_timesteps=current_timesteps == 0,
        )
    else:
        print("[training] Target timesteps already reached; skipping additional PPO updates.", flush=True)
    history_csv, snapshot_csv = export_training_results_csvs(
        output_dir=output_dir,
        history=callback.history,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
    )
    model.save(str(final_model_path))
    return RouteTrainingArtifacts(
        model=model,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
        history=callback.history,
        history_csv=history_csv,
        snapshot_csv=snapshot_csv,
    )
