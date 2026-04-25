"""Pure-Python fallback for route speedups."""

from __future__ import annotations

from itertools import permutations
from math import isclose

import numpy as np


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _shoelace_area(xs: list[float], ys: list[float], order: tuple[int, int, int, int]) -> float:
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        idx_i = order[i]
        idx_j = order[j]
        area += xs[idx_i] * ys[idx_j] - ys[idx_i] * xs[idx_j]
    return abs(area) * 0.5


def _is_simple_quad(xs: list[float], ys: list[float], order: tuple[int, int, int, int]) -> bool:
    a, b, c, d = order
    if _segments_intersect(xs[a], ys[a], xs[b], ys[b], xs[c], ys[c], xs[d], ys[d]):
        return False
    if _segments_intersect(xs[b], ys[b], xs[c], ys[c], xs[d], ys[d], xs[a], ys[a]):
        return False
    return True


def best_anchor_order(xs, ys):
    xs = [float(value) for value in xs]
    ys = [float(value) for value in ys]
    if len(xs) != 4 or len(ys) != 4:
        raise ValueError("best_anchor_order expects exactly four x/y coordinates.")

    best_order = None
    best_area = -1.0
    for order in permutations(range(4), 4):
        if not _is_simple_quad(xs, ys, order):
            continue
        area = _shoelace_area(xs, ys, order)
        if area > best_area:
            best_order = order
            best_area = area

    if best_order is None:
        raise ValueError("The sampled anchors did not form a valid simple quadrilateral.")
    return list(best_order), float(best_area)


def summarize_costs(costs):
    values = np.asarray(costs, dtype=np.float64)
    count = int(values.size)
    if count == 0:
        return 0.0, 0.0, 0.0

    total = float(values.sum())
    average = total / count
    if count > 1:
        variance = float(np.mean((values - average) ** 2))
        std = float(np.sqrt(max(variance, 0.0)))
    else:
        std = 0.0
    return average, std, total
