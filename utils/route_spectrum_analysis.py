"""Route spectrum analysis helpers for B4A and B4B notebooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _route_nodes(route: Any) -> list[int]:
    nodes = getattr(route, "path_node_ids", None)
    if nodes is None:
        nodes = getattr(route, "nodes", None)
    if nodes is None:
        raise TypeError("route must expose path_node_ids or nodes")
    return [int(node_id) for node_id in nodes]


def _route_points(route: Any) -> list[tuple[float, float]]:
    points = getattr(route, "path_latlon", None)
    if points is None:
        raise TypeError("route must expose path_latlon")
    return [(float(lat), float(lon)) for lat, lon in points]


def _pairwise_max_distance(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    max_dist = 0.0
    for i, (lat1, lon1) in enumerate(points[:-1]):
        for lat2, lon2 in points[i + 1 :]:
            dist = math.hypot(lat2 - lat1, lon2 - lon1) * 111_000.0
            if dist > max_dist:
                max_dist = dist
    return max_dist


def _turning_metrics(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 3:
        return 0.0, 0.0
    cycle = list(points)
    if cycle[0] != cycle[-1]:
        cycle = cycle + [cycle[0]]
    cycle = cycle + [cycle[1] if len(cycle) > 1 else cycle[0]]
    total = 0.0
    turn_count = 0.0
    for i in range(1, len(cycle) - 1):
        a = math.atan2(cycle[i][1] - cycle[i - 1][1], cycle[i][0] - cycle[i - 1][0])
        b = math.atan2(cycle[i + 1][1] - cycle[i][1], cycle[i + 1][0] - cycle[i][0])
        delta = abs(b - a)
        while delta > math.pi:
            delta = abs(delta - 2 * math.pi)
        total += delta
        if delta > 1e-9:
            turn_count += 1.0
    return total / (2 * math.pi), turn_count


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 == o2 == o3 == o4 == 0:
        return False
    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _self_intersections(points: Sequence[tuple[float, float]]) -> int:
    if len(points) < 4:
        return 0
    cycle = list(points)
    if cycle[0] != cycle[-1]:
        cycle = cycle + [cycle[0]]
    segments = list(zip(cycle[:-1], cycle[1:]))
    count = 0
    for i in range(len(segments)):
        for j in range(i + 2, len(segments)):
            if i == 0 and j == len(segments) - 1:
                continue
            if _segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                count += 1
    return count


def _traffic_values(route: Any, node_vped_lookup: Mapping[int, float] | None) -> list[float]:
    if not node_vped_lookup:
        return []
    values: list[float] = []
    for node_id in _route_nodes(route):
        value = node_vped_lookup.get(int(node_id))
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        values.append(value)
    return values


def _finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(np.asarray(values, dtype=float)))


def _finite_max(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.max(np.asarray(values, dtype=float)))


def _feature_row(
    route: Any,
    fitness: Any,
    node_vped_lookup: Mapping[int, float] | None,
    *,
    phase: str,
    index: int,
) -> dict[str, Any]:
    path_node_ids = _route_nodes(route)
    path_points = _route_points(route)
    length_m = float(getattr(route, "path_length_m", np.nan))
    area_m2 = float(getattr(route, "polygon_area_m2", np.nan))
    node_count = len(path_node_ids)
    unique_node_count = len(set(path_node_ids))
    unique_node_ratio = unique_node_count / max(node_count, 1)
    revisit_count = max(node_count - unique_node_count, 0)
    approx_diameter_m = _pairwise_max_distance(path_points)
    sinuosity = length_m / max(approx_diameter_m, 1.0)
    turn_ratio, turn_count = _turning_metrics(path_points)
    self_intersections = _self_intersections(path_points)
    values = _traffic_values(route, node_vped_lookup)
    anchor_ids = list(getattr(route, "ordered_anchor_node_ids", path_node_ids[:4]))
    anchor_values = []
    if node_vped_lookup:
        for node_id in anchor_ids:
            value = node_vped_lookup.get(int(node_id))
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            anchor_values.append(value)

    compactness = area_m2 / max(length_m**2, 1.0)
    area_to_length = area_m2 / max(length_m, 1.0)
    reward = float(getattr(fitness, "reward", np.nan))
    average_gtc = float(getattr(fitness, "average_gtc", np.nan))
    passenger_std = float(getattr(fitness, "passenger_gtc_std", np.nan))
    reward_per_km = reward / max(length_m / 1000.0, 1.0)
    avg_gtc_per_km = average_gtc / max(length_m / 1000.0, 1.0)
    demand_mean = _finite_mean(values)
    demand_max = _finite_max(values)
    demand_anchor_mean = _finite_mean(anchor_values)
    demand_anchor_max = _finite_max(anchor_values)

    return {
        "phase": phase,
        "route_index": index,
        "route_id": getattr(route, "route_id", f"R{index + 1:02d}"),
        "shape_length_m": length_m,
        "shape_area_m2": area_m2,
        "shape_compactness": compactness,
        "shape_area_to_length": area_to_length,
        "shape_diameter_m": approx_diameter_m,
        "history_turn_ratio": turn_ratio,
        "history_turn_count": turn_count,
        "history_self_intersections": self_intersections,
        "history_sinuosity": sinuosity,
        "topology_node_count": node_count,
        "topology_unique_node_count": unique_node_count,
        "topology_unique_node_ratio": unique_node_ratio,
        "topology_revisit_count": revisit_count,
        "demand_mean_v_ped": demand_mean,
        "demand_max_v_ped": demand_max,
        "demand_anchor_mean_v_ped": demand_anchor_mean,
        "demand_anchor_max_v_ped": demand_anchor_max,
        "global_reward": reward,
        "global_average_gtc": average_gtc,
        "global_passenger_gtc_std": passenger_std,
        "global_reward_per_km": reward_per_km,
        "global_average_gtc_per_km": avg_gtc_per_km,
    }


def build_route_spectrum_frame(
    routes: Sequence[Any],
    *,
    score_route_fn: Callable[[Any, int], Any],
    node_vped_lookup: Mapping[int, float] | None = None,
    phase: str,
) -> pd.DataFrame:
    rows = [
        _feature_row(route, score_route_fn(route, index), node_vped_lookup, phase=phase, index=index)
        for index, route in enumerate(routes)
    ]
    frame = pd.DataFrame(rows)
    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        frame.loc[:, numeric_cols] = frame.loc[:, numeric_cols].fillna(frame.loc[:, numeric_cols].median())
    return frame


def route_spectrum_correlation(frame: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["route_index"], errors="ignore")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(method=method)


def route_correlation_pairs(
    corr: pd.DataFrame,
    *,
    limit: int = 10,
    min_abs: float = 0.45,
) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if math.isnan(value) or abs(value) < min_abs:
                continue
            rows.append({"feature_a": left, "feature_b": right, "correlation": value, "abs_correlation": abs(value)})
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .head(limit)
        .drop(columns=["abs_correlation"])
        .reset_index(drop=True)
    )


def summarize_route_spectrum(frame: pd.DataFrame, *, label: str) -> str:
    if frame.empty:
        return f"{label}: no routes to summarize."

    lines = [f"{label}: {len(frame)} routes"]
    lines.append(
        "means | reward={:.3f}, avg_gtc={:.3f}, gtc_std={:.3f}, length_m={:.1f}, area_m2={:.1f}".format(
            frame["global_reward"].mean(),
            frame["global_average_gtc"].mean(),
            frame["global_passenger_gtc_std"].mean(),
            frame["shape_length_m"].mean(),
            frame["shape_area_m2"].mean(),
        )
    )
    lines.append(
        "spread | reward_std={:.3f}, avg_gtc_std={:.3f}, sinuosity_mean={:.3f}, demand_mean={:.3f}".format(
            frame["global_reward"].std(ddof=0),
            frame["global_average_gtc"].std(ddof=0),
            frame["history_sinuosity"].mean(),
            frame["demand_mean_v_ped"].mean(),
        )
    )

    corr = route_spectrum_correlation(frame)
    if not corr.empty:
        for target in ["global_reward", "global_average_gtc", "global_passenger_gtc_std"]:
            if target not in corr.columns:
                continue
            series = corr[target].drop(labels=[target], errors="ignore").dropna()
            if series.empty:
                continue
            strongest = series.abs().sort_values(ascending=False).head(3)
            labels = ", ".join(f"{name} ({series[name]:+.2f})" for name in strongest.index)
            lines.append(f"top | {target}: {labels}")

    return "\n".join(lines)


def build_route_notes(
    routes: Sequence[Any],
    frame: pd.DataFrame,
    *,
    phase: str,
) -> dict[str, dict[str, str]]:
    notes: dict[str, dict[str, str]] = {}
    row_by_route = {str(row.route_id): row for row in frame.itertuples(index=False)}
    for route in routes:
        row = row_by_route.get(str(getattr(route, "route_id", "")))
        if row is None:
            continue
        notes[str(route.route_id)] = {
            "encoding": f"route_id: {route.route_id}",
            "interpretation": (
                f"{phase} reward={row.global_reward:.3f} | avg_gtc={row.global_average_gtc:.3f} | "
                f"gtc_std={row.global_passenger_gtc_std:.3f}\n"
                f"length_m={row.shape_length_m:.1f} | area_m2={row.shape_area_m2:.1f} | "
                f"sinuosity={row.history_sinuosity:.3f} | demand_mean={row.demand_mean_v_ped:.3f}"
            ),
        }
    return notes


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    if corr.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        ax.set_axis_off()
    else:
        im = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_correlation_delta(
    pre_corr: pd.DataFrame,
    post_corr: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    common = [col for col in pre_corr.columns if col in post_corr.columns]
    if not common:
        return plot_correlation_heatmap(pd.DataFrame(), output_path, title=title)
    delta = post_corr.loc[common, common] - pre_corr.loc[common, common]
    return plot_correlation_heatmap(delta, output_path, title=title)


def compare_route_spectrum_frames(pre_frame: pd.DataFrame, post_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        "global_reward",
        "global_average_gtc",
        "global_passenger_gtc_std",
        "shape_length_m",
        "shape_area_m2",
        "shape_compactness",
        "history_sinuosity",
        "history_turn_count",
        "history_self_intersections",
        "demand_mean_v_ped",
    ]
    for col in numeric_cols:
        if col not in pre_frame.columns or col not in post_frame.columns:
            continue
        rows.append(
            {
                "feature": col,
                "pre_mean": float(pre_frame[col].mean()),
                "post_mean": float(post_frame[col].mean()),
                "delta_mean": float(post_frame[col].mean() - pre_frame[col].mean()),
                "pre_std": float(pre_frame[col].std(ddof=0)),
                "post_std": float(post_frame[col].std(ddof=0)),
                "delta_std": float(post_frame[col].std(ddof=0) - pre_frame[col].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)

