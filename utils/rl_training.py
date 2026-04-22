"""PPO training helpers for jeepney route exploration."""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
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

try:  # pragma: no cover - notebook detection
    from IPython import get_ipython
except ImportError:  # pragma: no cover - optional dependency
    get_ipython = None

try:  # pragma: no cover - optional dashboard dependency
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - degrade gracefully without live UI
    Console = None
    Group = None
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
    telemetry_csv: Path | None = None
    snapshot_csv: Path | None = None
    snapshot_dir: Path | None = None


class TelemetryPPO(PPO):
    """PPO variant that retains the latest training telemetry after each update."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latest_ppo_metrics: dict[str, float] = {}
        self.latest_ppo_update_step = 0

    def train(self) -> None:
        super().train()
        logger_values = getattr(getattr(self, "logger", None), "name_to_value", None)
        if not isinstance(logger_values, dict):
            return
        telemetry: dict[str, float] = {}
        for key in (
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
            "train/loss",
            "train/learning_rate",
            "time/fps",
        ):
            value = _safe_float(logger_values.get(key))
            if value is not None:
                telemetry[key] = value
        if telemetry:
            self.latest_ppo_metrics = telemetry
            self.latest_ppo_update_step = int(self.num_timesteps)


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


def _load_training_telemetry(output_dir: str | Path) -> list[dict[str, Any]]:
    out_dir = Path(output_dir).resolve()
    telemetry_csv = out_dir / "training_telemetry.csv"
    if not telemetry_csv.exists():
        return []

    frame = pd.read_csv(telemetry_csv)
    telemetry: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record = dict(row)
        for key in (
            "sequence_index",
            "episode_index",
            "timesteps",
            "route_node_count",
            "closed_loop_count",
            "forced_loop_count",
        ):
            if key in record and pd.notna(record[key]):
                record[key] = int(record[key])
        for key in (
            "episode_return",
            "fitness_reward",
            "average_gtc",
            "std_gtc",
            "latest_episode_return",
            "latest_fitness_reward",
            "latest_average_gtc",
            "latest_std_gtc",
            "best_return",
            "closed_loop_rate",
            "forced_loop_rate",
            "elapsed_s",
            "episode_duration_s",
            "steps_per_second",
        ):
            if key in record and pd.notna(record[key]):
                record[key] = float(record[key])
        for key in (
            "closed_loop",
            "is_terminal",
        ):
            if key in record and pd.notna(record[key]):
                record[key] = bool(record[key])
        for key in ("terminated_reason", "closure_mode", "status_message", "event_type", "event_message"):
            if key in record and pd.isna(record[key]):
                record[key] = None
        telemetry.append(record)
    return telemetry


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _rolling_mean(values: Iterable[float], window: int = 10) -> float | None:
    samples = [float(value) for value in values]
    if not samples:
        return None
    if window > 0 and len(samples) > window:
        samples = samples[-window:]
    return float(np.mean(samples))


def _format_signed_change(delta: float | None, *, higher_is_better: bool) -> str:
    if delta is None:
        return "n/a"
    if abs(delta) < 1e-9:
        return "→ flat"
    improving = delta > 0 if higher_is_better else delta < 0
    arrow = "↑" if delta > 0 else "↓"
    label = "better" if improving else "worse"
    return f"{arrow} {abs(delta):.3g} ({label})"


def _series_trend(values: Iterable[float], *, higher_is_better: bool, window: int = 10) -> tuple[float | None, float | None, str]:
    samples = [float(value) for value in values]
    if not samples:
        return None, None, "n/a"
    recent = samples[-window:] if window > 0 and len(samples) > window else samples
    prior = samples[-2 * window : -window] if window > 0 and len(samples) > window else []
    recent_mean = float(np.mean(recent))
    prior_mean = float(np.mean(prior)) if prior else None
    delta = None if prior_mean is None else recent_mean - prior_mean
    return recent_mean, prior_mean, _format_signed_change(delta, higher_is_better=higher_is_better)


def export_training_results_csvs(
    *,
    output_dir: str | Path,
    history: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
    best_snapshot: RouteTrainingSnapshot | None,
    worst_snapshot: RouteTrainingSnapshot | None,
) -> tuple[Path, Path, Path]:
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

    telemetry_rows: list[dict[str, Any]] = []
    for record in telemetry:
        row = dict(record)
        telemetry_rows.append(row)

    telemetry_columns = [
        "sequence_index",
        "event_type",
        "timesteps",
        "wall_time_s",
        "episode_index",
        "episode_return",
        "closed_loop",
        "closure_mode",
        "terminated_reason",
        "route_node_count",
        "fitness_reward",
        "average_gtc",
        "std_gtc",
        "latest_episode_return",
        "latest_fitness_reward",
        "latest_average_gtc",
        "latest_std_gtc",
        "best_return",
        "closed_loop_count",
        "forced_loop_count",
        "closed_loop_rate",
        "forced_loop_rate",
        "episode_duration_s",
        "steps_per_second",
        "status_message",
        "event_message",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/loss",
        "train/learning_rate",
        "time/fps",
    ]
    telemetry_csv = out_dir / "training_telemetry.csv"
    pd.DataFrame(telemetry_rows, columns=telemetry_columns).to_csv(telemetry_csv, index=False)

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

    return history_csv, telemetry_csv, snapshot_csv


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
        initial_telemetry: list[dict[str, Any]] | None = None,
        initial_best_snapshot: RouteTrainingSnapshot | None = None,
        initial_worst_snapshot: RouteTrainingSnapshot | None = None,
        target_timesteps: int | None = None,
        status_message: str = "Training",
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
        self.telemetry = [dict(record) for record in (initial_telemetry or [])]
        self.target_timesteps = int(target_timesteps) if target_timesteps is not None else None
        self.status_message = str(status_message)
        self._episode_returns: list[float] = []
        self._episode_start_time = 0.0
        self._run_start_time = 0.0
        self._episode_durations: deque[float] = deque(maxlen=20)
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
        self._latest_fitness_reward = float("nan")
        self._latest_average_gtc = float("nan")
        self._latest_std_gtc = float("nan")
        self._latest_event_message = "waiting for the first closed loop"
        self._tracked_ppo_keys = (
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/explained_variance",
            "train/loss",
            "train/learning_rate",
            "time/fps",
        )
        self._ppo_metric_history: dict[str, deque[float]] = {key: deque(maxlen=20) for key in self._tracked_ppo_keys}
        self._latest_ppo_metrics: dict[str, float] = {}
        self._dashboard_enabled = bool(
            enable_rich_dashboard and Console is not None and Live is not None and Panel is not None and Table is not None
        )
        self._in_notebook = False
        if get_ipython is not None:
            shell = get_ipython()
            self._in_notebook = shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        self._console = Console(force_jupyter=self._in_notebook) if self._dashboard_enabled and Console is not None else None
        self._live = None
        self.history_csv = self.output_dir / "training_history.csv"
        self.telemetry_csv = self.output_dir / "training_telemetry.csv"
        self.snapshot_csv = self.output_dir / "training_snapshots.csv"
        self.snapshot_dir = self.output_dir / "route_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._restore_telemetry_state()

    def _init_callback(self) -> None:
        self._episode_returns = [0.0 for _ in range(self.training_env.num_envs)]
        self._run_start_time = time.monotonic()
        self._episode_start_time = self._run_start_time
        self._last_heartbeat = time.monotonic()
        self._last_heartbeat_step = self.num_timesteps
        self._last_checkpoint = self._last_heartbeat
        self._last_checkpoint_step = self.num_timesteps
        self._start_live_dashboard()
        self._maybe_checkpoint(force=True)

    def _refresh_ppo_metrics(self) -> None:
        latest_metrics = getattr(self.model, "latest_ppo_metrics", None)
        if isinstance(latest_metrics, dict) and latest_metrics:
            source = latest_metrics
        else:
            logger = getattr(self.model, "logger", None)
            logger_values = getattr(logger, "name_to_value", None) if logger is not None else None
            source = logger_values if isinstance(logger_values, dict) else {}

        for key in self._tracked_ppo_keys:
            value = _safe_float(source.get(key))
            if value is None:
                continue
            self._latest_ppo_metrics[key] = value
            self._ppo_metric_history.setdefault(key, deque(maxlen=20)).append(value)

        if "rollout/ep_rew_mean" not in self._latest_ppo_metrics:
            episode_returns = self._history_series("episode_return")
            fallback = _rolling_mean(episode_returns, window=10)
            if fallback is not None:
                self._latest_ppo_metrics["rollout/ep_rew_mean"] = fallback
                self._ppo_metric_history.setdefault("rollout/ep_rew_mean", deque(maxlen=20)).append(fallback)

        if "rollout/ep_len_mean" not in self._latest_ppo_metrics:
            episode_lengths = [max(int(record.get("route_node_count", 0)) - 1, 0) for record in self.history]
            fallback = _rolling_mean(episode_lengths, window=10)
            if fallback is not None:
                self._latest_ppo_metrics["rollout/ep_len_mean"] = fallback
                self._ppo_metric_history.setdefault("rollout/ep_len_mean", deque(maxlen=20)).append(fallback)

        if "time/fps" not in self._latest_ppo_metrics and self._run_start_time:
            elapsed = max(time.monotonic() - self._run_start_time, 1e-9)
            fps = float(self.num_timesteps / elapsed)
            self._latest_ppo_metrics["time/fps"] = fps
            self._ppo_metric_history.setdefault("time/fps", deque(maxlen=20)).append(fps)

    def _restore_telemetry_state(self) -> None:
        if not self.telemetry:
            return
        for record in self.telemetry:
            for key in self._tracked_ppo_keys:
                value = _safe_float(record.get(key))
                if value is None:
                    continue
                self._latest_ppo_metrics[key] = value
                self._ppo_metric_history.setdefault(key, deque(maxlen=20)).append(value)
        last_record = self.telemetry[-1]
        status_message = last_record.get("status_message")
        if isinstance(status_message, str) and status_message.strip():
            self.status_message = status_message.strip()
        event_message = last_record.get("event_message")
        if isinstance(event_message, str) and event_message.strip():
            self._latest_event_message = event_message.strip()
        latest_episode_return = _safe_float(last_record.get("latest_episode_return"))
        if latest_episode_return is not None:
            self._latest_episode_return = latest_episode_return
        latest_fitness_reward = _safe_float(last_record.get("latest_fitness_reward"))
        if latest_fitness_reward is not None:
            self._latest_fitness_reward = latest_fitness_reward
        latest_average_gtc = _safe_float(last_record.get("latest_average_gtc"))
        if latest_average_gtc is not None:
            self._latest_average_gtc = latest_average_gtc
        latest_std_gtc = _safe_float(last_record.get("latest_std_gtc"))
        if latest_std_gtc is not None:
            self._latest_std_gtc = latest_std_gtc

    def _history_series(self, key: str) -> list[float]:
        values: list[float] = []
        for record in self.history:
            value = _safe_float(record.get(key))
            if value is not None:
                values.append(value)
        return values

    def _avg_route_node_count(self, window: int = 10) -> float | None:
        values = self._history_series("route_node_count")
        return _rolling_mean(values, window=window)

    def _episode_duration_mean(self) -> float | None:
        if not self._episode_durations:
            return None
        return float(np.mean(list(self._episode_durations)))

    def _dashboard_tables(self):
        if not self._dashboard_enabled or Table is None:
            return None

        episode_returns = self._history_series("episode_return")
        avg_gtc_values = self._history_series("average_gtc")
        std_gtc_values = self._history_series("std_gtc")
        route_counts = self._history_series("route_node_count")
        closed_history = [1.0 if record.get("closed_loop") else 0.0 for record in self.history]
        forced_history = [1.0 if record.get("closure_mode") == "forced" else 0.0 for record in self.history]

        latest_return = self._latest_episode_return
        latest_avg_gtc = self._latest_average_gtc
        latest_std_gtc = self._latest_std_gtc
        return_mean_10, return_prev_10, return_trend = _series_trend(episode_returns, higher_is_better=True, window=10)
        gtc_mean_10, gtc_prev_10, gtc_trend = _series_trend(avg_gtc_values, higher_is_better=False, window=10)
        std_mean_10, std_prev_10, std_trend = _series_trend(std_gtc_values, higher_is_better=False, window=10)

        summary = Table(show_header=False, box=None, pad_edge=False)
        summary.add_column("metric", style="bold cyan")
        summary.add_column("value")
        summary.add_row("status", self.status_message)
        summary.add_row("event", self._latest_event_message)
        summary.add_row("timesteps", f"{self.num_timesteps:,}")
        if self.target_timesteps and self.target_timesteps > 0:
            progress = 100.0 * min(self.num_timesteps / self.target_timesteps, 1.0)
            summary.add_row("progress", f"{progress:.1f}%")
        elapsed = max(time.monotonic() - self._run_start_time, 1e-9) if self._run_start_time else None
        if elapsed is not None:
            summary.add_row("elapsed", f"{elapsed/60.0:.1f} min")
            summary.add_row("speed", f"{self.num_timesteps / elapsed:.1f} steps/s")
            if self.target_timesteps and self.num_timesteps < self.target_timesteps:
                remaining = max(self.target_timesteps - self.num_timesteps, 0)
                rate = self.num_timesteps / elapsed if elapsed > 0 else 0.0
                eta = remaining / rate if rate > 1e-9 else None
                summary.add_row("eta", f"{eta/60.0:.1f} min" if eta is not None else "n/a")
        summary.add_row("episodes", f"{len(self.history):,}")
        summary.add_row("closed loops", f"{self._closed_loop_count:,}")
        summary.add_row("forced loops", f"{self._forced_loop_count:,}")
        summary.add_row("closed loop rate", f"{(self._closed_loop_count / max(len(self.history), 1)):.1%}")
        summary.add_row("latest return", f"{latest_return:.3f}")
        summary.add_row("best return", f"{self.best_snapshot.episode_return:.3f}" if self.best_snapshot is not None else "n/a")
        summary.add_row("latest avg gtc", f"{latest_avg_gtc:.3f}")
        summary.add_row("latest std gtc", f"{latest_std_gtc:.3f}")
        summary.add_row("avg route nodes", f"{_rolling_mean(route_counts, window=10):.1f}" if route_counts else "n/a")
        summary.add_row("checkpoint", self.checkpoint_path.name)

        trends = Table(show_header=True, header_style="bold magenta", box=None, pad_edge=False)
        trends.add_column("signal")
        trends.add_column("rolling avg")
        trends.add_column("trend vs prev 10")
        trends.add_row(
            "episode return",
            f"{return_mean_10:.3f}" if return_mean_10 is not None else "n/a",
            return_trend,
        )
        trends.add_row(
            "average GTC",
            f"{gtc_mean_10:.3f}" if gtc_mean_10 is not None else "n/a",
            gtc_trend,
        )
        trends.add_row(
            "GTC std",
            f"{std_mean_10:.3f}" if std_mean_10 is not None else "n/a",
            std_trend,
        )
        trends.add_row(
            "closed loop rate",
            f"{_rolling_mean(closed_history, window=10):.1%}" if closed_history else "n/a",
            "n/a",
        )
        trends.add_row(
            "forced loop rate",
            f"{_rolling_mean(forced_history, window=10):.1%}" if forced_history else "n/a",
            "n/a",
        )

        ppo = Table(show_header=True, header_style="bold green", box=None, pad_edge=False)
        ppo.add_column("metric")
        ppo.add_column("latest")
        ppo.add_column("rolling avg")
        ppo.add_column("trend")
        ppo_rows = [
            ("rollout/ep_rew_mean", True),
            ("rollout/ep_len_mean", False),
            ("train/policy_gradient_loss", False),
            ("train/value_loss", False),
            ("train/entropy_loss", False),
            ("train/approx_kl", False),
            ("train/clip_fraction", False),
            ("train/explained_variance", True),
            ("train/loss", False),
            ("train/learning_rate", True),
            ("time/fps", True),
        ]
        for key, higher_is_better in ppo_rows:
            history = list(self._ppo_metric_history.get(key, []))
            latest = self._latest_ppo_metrics.get(key)
            recent_mean, _prior_mean, trend = _series_trend(history, higher_is_better=higher_is_better, window=5)
            ppo.add_row(
                key,
                f"{latest:.4g}" if latest is not None else "n/a",
                f"{recent_mean:.4g}" if recent_mean is not None else "n/a",
                trend,
            )

        return Group(
            Panel(summary, title="B4B training summary", border_style="cyan"),
            Panel(trends, title="Episode trends", border_style="magenta"),
            Panel(ppo, title="PPO telemetry", border_style="green"),
        )

    def _dashboard_panel(self):
        if not self._dashboard_enabled or Panel is None or Table is None or Group is None:
            return None
        return self._dashboard_tables()

    def _telemetry_record(self, event_type: str) -> dict[str, Any]:
        now = time.monotonic()
        elapsed_s = float(max(now - self._run_start_time, 0.0)) if self._run_start_time else 0.0
        episode_duration_s = float(max(now - self._episode_start_time, 0.0)) if self._episode_start_time else 0.0
        current_episode_return = float(self._episode_returns[0]) if self._episode_returns else float(self._latest_episode_return)
        closed_loop_rate = float(self._closed_loop_count / max(len(self.history), 1))
        forced_loop_rate = float(self._forced_loop_count / max(len(self.history), 1))
        record: dict[str, Any] = {
            "sequence_index": len(self.telemetry) + 1,
            "event_type": str(event_type),
            "timesteps": int(self.num_timesteps),
            "wall_time_s": elapsed_s,
            "episode_index": int(self._episode_index),
            "episode_return": current_episode_return,
            "closed_loop": bool(self._latest_terminated_reason == "closed_loop"),
            "closure_mode": self._latest_closure_mode,
            "terminated_reason": self._latest_terminated_reason,
            "route_node_count": int(self._latest_route_node_count),
            "fitness_reward": self._latest_fitness_reward,
            "average_gtc": self._latest_average_gtc,
            "std_gtc": self._latest_std_gtc,
            "latest_episode_return": float(self._latest_episode_return),
            "latest_fitness_reward": float(self._latest_fitness_reward),
            "latest_average_gtc": float(self._latest_average_gtc),
            "latest_std_gtc": float(self._latest_std_gtc),
            "best_return": float(self.best_snapshot.episode_return if self.best_snapshot is not None else np.nan),
            "closed_loop_count": int(self._closed_loop_count),
            "forced_loop_count": int(self._forced_loop_count),
            "closed_loop_rate": closed_loop_rate,
            "forced_loop_rate": forced_loop_rate,
            "episode_duration_s": episode_duration_s,
            "steps_per_second": float(self.num_timesteps / elapsed_s) if elapsed_s > 1e-9 else np.nan,
            "status_message": self.status_message,
            "event_message": self._latest_event_message,
        }
        for key in self._tracked_ppo_keys:
            record[key] = self._latest_ppo_metrics.get(key, np.nan)
        return record

    def _append_telemetry(self, event_type: str) -> None:
        self.telemetry.append(self._telemetry_record(event_type))

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
        self.history_csv, self.telemetry_csv, self.snapshot_csv = export_training_results_csvs(
            output_dir=self.output_dir,
            history=self.history,
            telemetry=self.telemetry,
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
        self._refresh_ppo_metrics()
        self._append_telemetry("heartbeat")
        self._maybe_checkpoint()
        if self._live is not None:
            self._live.update(self._dashboard_panel())

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
        html_path = self.snapshot_dir / f"episode_{episode_index:06d}_route_snapshot.html"
        json_path = self.snapshot_dir / f"episode_{episode_index:06d}_route_snapshot.json"
        payload = {
            "episode_index": episode_index,
            "episode_return": episode_return,
            "fitness_reward": fitness_reward,
            "average_gtc": average_gtc,
            "std_gtc": std_gtc,
            "route_path_node_ids": route_node_ids,
            "route_latlon": route_latlon,
            "route_system": {
                "closed_loop": bool(info.get("terminated_reason") == "closed_loop"),
                "closure_mode": info.get("closure_mode"),
                "terminated_reason": info.get("terminated_reason"),
                "route_node_count": len(route_node_ids),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        export_physical_route_html(
            route_node_ids,
            self.drive_graph_raw,
            html_path,
            title="Training Route Snapshot",
            subtitle=f"episode {episode_index} | return {episode_return:.3f}",
        )
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
                self._refresh_ppo_metrics()
                self._emit_heartbeat()
                continue

            self._refresh_ppo_metrics()
            episode_duration = time.monotonic() - self._episode_start_time
            self._episode_durations.append(max(episode_duration, 0.0))
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
            self._latest_fitness_reward = float(getattr(fitness, "reward", info.get("fitness_reward", episode_return)))
            self._latest_average_gtc = float(getattr(fitness, "average_gtc", info.get("fitness_average_gtc", np.nan)))
            self._latest_std_gtc = float(getattr(fitness, "passenger_gtc_std", info.get("fitness_passenger_gtc_std", np.nan)))
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
            self._episode_start_time = time.monotonic()
            self._append_telemetry("episode")

            if not closed_loop:
                self._maybe_checkpoint()
                if self._live is not None:
                    self._live.update(self._dashboard_panel())
                continue

            snapshot = self._capture_snapshot(
                episode_index=self._episode_index,
                episode_return=episode_return,
                info=info,
            )
            if snapshot is None:
                self._maybe_checkpoint()
                if self._live is not None:
                    self._live.update(self._dashboard_panel())
                continue

            if self.best_snapshot is None or snapshot.episode_return > self.best_snapshot.episode_return:
                self.best_snapshot = snapshot
                self._latest_event_message = f"best route updated at episode {snapshot.episode_index}"
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
                export_physical_route_html(
                    snapshot.route_path_node_ids,
                    self.drive_graph_raw,
                    self.snapshot_dir / f"episode_{snapshot.episode_index:06d}_best_route.html",
                    title="Best Route Snapshot",
                    subtitle=f"episode {snapshot.episode_index} | return {snapshot.episode_return:.3f}",
                )
                (self.snapshot_dir / f"episode_{snapshot.episode_index:06d}_best_route.json").write_text(
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
                self._latest_event_message = f"worst route updated at episode {snapshot.episode_index}"
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
                export_physical_route_html(
                    snapshot.route_path_node_ids,
                    self.drive_graph_raw,
                    self.snapshot_dir / f"episode_{snapshot.episode_index:06d}_worst_route.html",
                    title="Worst Route Snapshot",
                    subtitle=f"episode {snapshot.episode_index} | return {snapshot.episode_return:.3f}",
                )
                (self.snapshot_dir / f"episode_{snapshot.episode_index:06d}_worst_route.json").write_text(
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
            self._maybe_checkpoint()
            if self._live is not None:
                self._live.update(self._dashboard_panel())

        return True

    def _on_training_end(self) -> None:
        self.status_message = "training complete"
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
    initial_telemetry = _load_training_telemetry(output_dir) if resume_from_checkpoint else []
    initial_best_snapshot = _load_snapshot_from_json(output_dir / "best_route.json") if resume_from_checkpoint else None
    initial_worst_snapshot = _load_snapshot_from_json(output_dir / "worst_route.json") if resume_from_checkpoint else None
    latest_checkpoint = output_dir / "ppo_latest_model.zip"
    final_model_path = output_dir / "ppo_final_model.zip"

    model: TelemetryPPO
    if resume_from_checkpoint and latest_checkpoint.exists():
        model = TelemetryPPO.load(str(latest_checkpoint), env=training_env, device="auto")
    else:
        model = TelemetryPPO(
            "MultiInputPolicy",
            training_env,
            seed=seed,
            verbose=ppo_verbose,
            **ppo_kwargs,
        )

    current_timesteps = int(getattr(model, "num_timesteps", 0))
    if resume_from_checkpoint and latest_checkpoint.exists() and current_timesteps >= int(total_timesteps):
        initial_history = []
        initial_best_snapshot = None
        initial_worst_snapshot = None
        model = TelemetryPPO(
            "MultiInputPolicy",
            training_env,
            seed=seed,
            verbose=ppo_verbose,
            **ppo_kwargs,
        )
        current_timesteps = 0

    remaining_timesteps = max(int(total_timesteps) - current_timesteps, 0)
    if current_timesteps > 0:
        status_message = f"resuming from {current_timesteps:,} timesteps"
    else:
        status_message = "starting a fresh PPO run"

    callback = BestWorstRouteCallback(
        drive_graph_raw=drive_graph_raw,
        output_dir=output_dir,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_steps=heartbeat_steps,
        checkpoint_seconds=checkpoint_seconds,
        checkpoint_steps=checkpoint_steps,
        checkpoint_path=latest_checkpoint,
        initial_history=initial_history,
        initial_telemetry=initial_telemetry,
        initial_best_snapshot=initial_best_snapshot,
        initial_worst_snapshot=initial_worst_snapshot,
        target_timesteps=int(total_timesteps),
        status_message=status_message,
        enable_rich_dashboard=use_rich_dashboard,
    )
    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            reset_num_timesteps=current_timesteps == 0,
        )
    history_csv, telemetry_csv, snapshot_csv = export_training_results_csvs(
        output_dir=output_dir,
        history=callback.history,
        telemetry=callback.telemetry,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
    )
    if callback.snapshot_dir is not None:
        callback.snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_csv, callback.snapshot_dir / history_csv.name)
        shutil.copy2(telemetry_csv, callback.snapshot_dir / telemetry_csv.name)
        shutil.copy2(snapshot_csv, callback.snapshot_dir / snapshot_csv.name)
    model.save(str(final_model_path))
    return RouteTrainingArtifacts(
        model=model,
        best_snapshot=callback.best_snapshot,
        worst_snapshot=callback.worst_snapshot,
        history=callback.history,
        history_csv=history_csv,
        telemetry_csv=telemetry_csv,
        snapshot_csv=snapshot_csv,
        snapshot_dir=callback.snapshot_dir,
    )
