"""Route container and quick route scrubber visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import math
from html import escape
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .travel_graph import JeepneyRoute


@dataclass(slots=True)
class JeepneySystem:
    """Container for a set of jeepney routes with scrub state."""

    routes: list[JeepneyRoute] = field(default_factory=list)
    route_loops: dict[str, list[str]] = field(default_factory=dict)
    current_index: int = 0

    def __post_init__(self) -> None:
        self.routes = list(self.routes)
        self.route_loops = {str(k): list(v) for k, v in self.route_loops.items()}
        if self.routes:
            self.current_index %= len(self.routes)
        else:
            self.current_index = 0

    def __len__(self) -> int:
        return len(self.routes)

    @property
    def current_route(self) -> JeepneyRoute:
        if not self.routes:
            raise ValueError("JeepneySystem has no routes.")
        return self.routes[self.current_index]

    def set_index(self, index: int) -> None:
        if not self.routes:
            raise ValueError("JeepneySystem has no routes.")
        self.current_index = index % len(self.routes)

    def next_route(self) -> JeepneyRoute:
        self.set_index(self.current_index + 1)
        return self.current_route

    def previous_route(self) -> JeepneyRoute:
        self.set_index(self.current_index - 1)
        return self.current_route

    @classmethod
    def from_routes(cls, routes: Sequence[JeepneyRoute]) -> "JeepneySystem":
        return cls(routes=list(routes))

    @classmethod
    def generate_random_routes(
        cls,
        manager,
        count: int,
        min_nodes: int,
        max_nodes: int,
        *,
        seed_start: int = 0,
        route_prefix: str = "R",
    ) -> "JeepneySystem":
        routes: list[JeepneyRoute] = []
        route_loops: dict[str, list[str]] = {}
        for i in range(int(count)):
            loop = manager.generate_random_ride_loop(min_nodes, max_nodes, seed=seed_start + i)
            route_id = f"{route_prefix}{i + 1:02d}"
            routes.append(JeepneyRoute(route_id, loop[:-1]))
            route_loops[route_id] = list(loop)
        return cls(routes=routes, route_loops=route_loops)

    @staticmethod
    def _orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _on_segment(a, b, c) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    @classmethod
    def _segments_intersect(cls, p1, p2, p3, p4) -> bool:
        o1 = cls._orient(p1, p2, p3)
        o2 = cls._orient(p1, p2, p4)
        o3 = cls._orient(p3, p4, p1)
        o4 = cls._orient(p3, p4, p2)

        if o1 == o2 == o3 == o4 == 0:
            return False
        if o1 == 0 and cls._on_segment(p1, p3, p2):
            return True
        if o2 == 0 and cls._on_segment(p1, p4, p2):
            return True
        if o3 == 0 and cls._on_segment(p3, p1, p4):
            return True
        if o4 == 0 and cls._on_segment(p3, p2, p4):
            return True
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    def _route_points(self, route: JeepneyRoute, manager) -> list[tuple[float, float]]:
        coords = getattr(manager, "_node_coords", None) or {}
        points: list[tuple[float, float]] = []
        for node_id in route.nodes:
            coord = coords.get(node_id)
            if coord is not None:
                points.append((float(coord[1]), float(coord[0])))
        return points

    def _self_intersection_count(self, points: list[tuple[float, float]]) -> int:
        if len(points) < 4:
            return 0
        segments = list(zip(points, points[1:]))
        count = 0
        for i in range(len(segments)):
            for j in range(i + 2, len(segments)):
                if i == 0 and j == len(segments) - 1:
                    continue
                if self._segments_intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    count += 1
        return count

    def _turning_count(self, points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        cycle = points + [points[0], points[1]]
        total = 0.0
        for i in range(1, len(cycle) - 1):
            a = math.atan2(cycle[i][1] - cycle[i - 1][1], cycle[i][0] - cycle[i - 1][0])
            b = math.atan2(cycle[i + 1][1] - cycle[i][1], cycle[i + 1][0] - cycle[i][0])
            delta = abs(b - a)
            while delta > math.pi:
                delta = abs(delta - 2 * math.pi)
            total += delta
        return total / (2 * math.pi)

    def analyze_routes(self, manager) -> list[dict]:
        """Return per-route loop diagnostics."""
        rows: list[dict] = []
        for route in self.routes:
            points = self._route_points(route, manager)
            rows.append(
                {
                    "route_id": route.route_id,
                    "nodes": len(route.nodes),
                    "self_intersections": self._self_intersection_count(points),
                    "turning_count": round(self._turning_count(points), 2),
                    "closed_cycle": True,
                }
            )
        return rows

    def build_route_toggle_html(self, manager, *, title: str = "Jeepney Route Explorer") -> str:
        """Build inline HTML with one-route-at-a-time navigation and display-all mode."""
        coords = getattr(manager, "_node_coords", None) or {}
        if not coords:
            raise ValueError("Node coordinates are required for HTML route display.")

        road_edges = []
        for row in manager.edges.itertuples(index=False):
            if getattr(row, "edge_type", None) != "start_walk":
                continue
            u = coords.get(row.u)
            v = coords.get(row.v)
            if u is None or v is None:
                continue
            road_edges.append(((float(u[1]), float(u[0])), (float(v[1]), float(v[0]))))

        route_points = {route.route_id: self._route_points(route, manager) for route in self.routes}

        all_lons = [p[0] for seg in road_edges for p in seg]
        all_lats = [p[1] for seg in road_edges for p in seg]
        for points in route_points.values():
            all_lons.extend(p[0] for p in points)
            all_lats.extend(p[1] for p in points)

        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        width, height = 1200, 900
        pad = 40
        dx = max(lon_max - lon_min, 1e-9)
        dy = max(lat_max - lat_min, 1e-9)

        def project(point: tuple[float, float]) -> tuple[float, float]:
            lon, lat = point
            x = pad + (lon - lon_min) / dx * (width - 2 * pad)
            y = pad + (lat_max - lat) / dy * (height - 2 * pad)
            return x, y

        road_svg = []
        for (p1, p2) in road_edges:
            x1, y1 = project(p1)
            x2, y2 = project(p2)
            road_svg.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="road-line" />'
            )

        route_data = []
        for route in self.routes:
            points = route_points[route.route_id]
            if len(points) < 2:
                continue
            closed = points + [points[0]]
            route_data.append(
                {
                    "id": route.route_id,
                    "points": [list(project(p)) for p in points],
                    "closed": [list(project(p)) for p in closed],
                    "first": list(project(points[0])),
                    "node_count": len(points),
                }
            )

        route_json = json.dumps(route_data)

        analysis = self.analyze_routes(manager)
        analysis_rows = "".join(
            f"<tr><td>{escape(row['route_id'])}</td><td>{row['nodes']}</td>"
            f"<td>{row['self_intersections']}</td><td>{row['turning_count']}</td></tr>"
            for row in analysis
        )

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {{
    margin: 0;
    background: white;
    color: #111827;
    font-family: Arial, sans-serif;
  }}
  #toolbar {{
    position: sticky;
    top: 0;
    background: white;
    border-bottom: 1px solid #e5e7eb;
    padding: 10px 12px;
    z-index: 10;
  }}
  #title {{
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  #controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }}
  .route-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    background: #f8fafc;
    font-size: 12px;
  }}
  #buttons {{
    margin-left: auto;
    display: flex;
    gap: 8px;
  }}
  button {{
    border: 1px solid #cbd5e1;
    background: #fff;
    border-radius: 8px;
    padding: 6px 10px;
    cursor: pointer;
  }}
  #summary {{
    padding: 10px 12px;
    font-size: 13px;
    color: #334155;
  }}
  #mapwrap {{
    padding: 12px;
  }}
  svg {{
    width: 100%;
    height: auto;
    border: 1px solid #e5e7eb;
    background: white;
  }}
  .road-line {{
    stroke: #d1d5db;
    stroke-width: 1;
    opacity: 0.18;
  }}
  .route-line {{
    stroke: #2563eb;
    stroke-width: 3;
    fill: none;
  }}
  .route-layer {{
    opacity: 1;
    transition: opacity 0.15s ease;
  }}
  .route-layer.is-hidden {{
    opacity: 0;
    pointer-events: none;
  }}
  .route-label {{
    font-size: 13px;
    fill: #111827;
    paint-order: stroke;
    stroke: white;
    stroke-width: 3px;
    stroke-linejoin: round;
  }}
  .route-node {{
    fill: #2563eb;
    stroke: white;
    stroke-width: 2px;
  }}
  .route-dot {{
    fill: #ef4444;
    stroke: white;
    stroke-width: 2px;
    filter: drop-shadow(0 0 4px rgba(239,68,68,0.7));
  }}
  table {{
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 12px;
  }}
  th, td {{
    border: 1px solid #e5e7eb;
    padding: 4px 8px;
    text-align: left;
  }}
</style>
</head>
<body>
  <div id="toolbar">
    <div id="title">{escape(title)}</div>
    <div id="controls">
      <button type="button" onclick="prevRoute()">Previous</button>
      <button type="button" onclick="nextRoute()">Next</button>
      <input id="customRoute" type="text" placeholder="R01 or 1" style="padding:6px 8px;border:1px solid #cbd5e1;border-radius:8px;width:110px" />
      <button type="button" onclick="goCustom()">Go</button>
      <div id="buttons">
        <button type="button" onclick="toggleAll()">Display all</button>
      </div>
    </div>
  </div>
  <div id="summary">
    One route is shown by default. Use Previous/Next or enter a route ID (e.g. R01) or number.
    <table>
      <thead><tr><th>Route</th><th>Nodes</th><th>Self-intersections</th><th>Turning count</th></tr></thead>
      <tbody>{analysis_rows}</tbody>
    </table>
  </div>
  <div id="mapwrap">
    <svg id="routeSvg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
      <g id="roads">
        {''.join(road_svg)}
      </g>
      <g id="routes"></g>
    </svg>
  </div>
  <script>
    const routes = {route_json};
    const routeGroup = document.getElementById("routes");
    const svg = document.getElementById("routeSvg");
    let currentIndex = 0;
    let displayAll = false;
    let timer = null;

    function clearRouteGroup() {{
      while (routeGroup.firstChild) routeGroup.removeChild(routeGroup.firstChild);
    }}

    function makeSvgEl(tag, attrs) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}

    function renderSingleRoute() {{
      clearRouteGroup();
      const route = routes[currentIndex];
      if (!route) return;
      const g = makeSvgEl("g", {{ id: `route-${{route.id}}`, class: "route-layer" }});
      g.appendChild(makeSvgEl("polyline", {{
        points: route.closed.map(p => p.join(",")).join(" "),
        class: "route-line"
      }}));
      route.points.forEach((p, idx) => {{
        g.appendChild(makeSvgEl("circle", {{
          cx: p[0], cy: p[1], r: 3.5, class: "route-node"
        }}));
      }});
      g.appendChild(makeSvgEl("circle", {{
        cx: route.points[0][0],
        cy: route.points[0][1],
        r: 7,
        class: "route-dot",
        id: "movingDot"
      }}));
      g.appendChild(makeSvgEl("text", {{
        x: route.first[0] + 8, y: route.first[1] - 8, class: "route-label"
      }}));
      g.lastChild.textContent = route.id;
      routeGroup.appendChild(g);
      startAnimation(route);
      updateButtons();
    }}

    function renderAllRoutes() {{
      clearRouteGroup();
      routes.forEach(route => {{
        const g = makeSvgEl("g", {{ id: `route-${{route.id}}`, class: "route-layer" }});
        g.appendChild(makeSvgEl("polyline", {{
          points: route.closed.map(p => p.join(",")).join(" "),
          class: "route-line"
        }}));
        route.points.forEach((p, idx) => {{
          g.appendChild(makeSvgEl("circle", {{
            cx: p[0], cy: p[1], r: 3.5, class: "route-node"
          }}));
        }});
        g.appendChild(makeSvgEl("text", {{
          x: route.first[0] + 8, y: route.first[1] - 8, class: "route-label"
        }}));
        g.lastChild.textContent = route.id;
        routeGroup.appendChild(g);
      }});
      stopAnimation();
      updateButtons();
    }}

    function stopAnimation() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
    }}

    function startAnimation(route) {{
      stopAnimation();
      if (!route || route.points.length === 0) return;
      let idx = 0;
      const g = document.getElementById(`route-${{route.id}}`);
      const dot = g ? g.querySelector("#movingDot") : null;
      if (!dot) return;
      const place = () => {{
        const p = route.points[idx];
        dot.setAttribute("cx", p[0]);
        dot.setAttribute("cy", p[1]);
        idx = (idx + 1) % route.points.length;
      }};
      place();
      timer = setInterval(place, 550);
    }}

    function updateButtons() {{
      document.querySelector('#buttons button').textContent = displayAll ? 'Show one' : 'Display all';
    }}

    function nextRoute() {{
      displayAll = false;
      currentIndex = (currentIndex + 1) % routes.length;
      renderSingleRoute();
    }}

    function prevRoute() {{
      displayAll = false;
      currentIndex = (currentIndex - 1 + routes.length) % routes.length;
      renderSingleRoute();
    }}

    function goCustom() {{
      const raw = document.getElementById("customRoute").value.trim();
      if (!raw) return;
      let idx = routes.findIndex(r => r.id.toLowerCase() === raw.toLowerCase());
      if (idx < 0 && /^\\d+$/.test(raw)) {{
        idx = Math.max(0, Math.min(routes.length - 1, parseInt(raw, 10) - 1));
      }}
      if (idx >= 0) {{
        displayAll = false;
        currentIndex = idx;
        renderSingleRoute();
      }}
    }}

    function toggleAll() {{
      displayAll = !displayAll;
      if (displayAll) {{
        renderAllRoutes();
      }} else {{
        renderSingleRoute();
      }}
    }}

    document.addEventListener("keydown", (e) => {{
      if (e.key === "ArrowRight") nextRoute();
      if (e.key === "ArrowLeft") prevRoute();
      if (e.key === "Enter") goCustom();
    }});
    renderSingleRoute();
  </script>
</body>
</html>"""

    def export_route_toggle_html(self, manager, output_html, *, title: str = "Jeepney Route Explorer") -> Path:
        """Write the route explorer HTML to disk and return the output path."""
        out = Path(output_html).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self.build_route_toggle_html(manager, title=title)
        out.write_text(html, encoding="utf-8")
        return out


class JeepneyRouteScrubber:
    """Light-mode matplotlib grid viewer for jeepney routes."""

    def __init__(
        self,
        manager,
        routes: Iterable[JeepneyRoute] | JeepneySystem,
        *,
        title: str = "Jeepney Route Scrubber",
        route_color: str = "#2563eb",
        road_color: str = "#d1d5db",
        road_alpha: float = 0.35,
    ) -> None:
        self.manager = manager
        self.system = routes if isinstance(routes, JeepneySystem) else JeepneySystem(list(routes))
        self.title = title
        self.route_color = route_color
        self.road_color = road_color
        self.road_alpha = road_alpha
        self.fig = None
        self.axes = None

    def _coords(self) -> dict:
        coords = getattr(self.manager, "_node_coords", None)
        if coords is None:
            return {}
        return coords

    def _edge_segments(self):
        coords = self._coords()
        edges = getattr(self.manager, "edges", None)
        if edges is None:
            return []
        segments = []
        for row in edges.itertuples(index=False):
            u = coords.get(row.u)
            v = coords.get(row.v)
            if u is None or v is None:
                continue
            segments.append(((float(u[1]), float(u[0])), (float(v[1]), float(v[0]))))
        return segments

    def _route_segments(self, route: JeepneyRoute):
        coords = self._coords()
        points = []
        for node_id in route.nodes:
            coord = coords.get(node_id)
            if coord is not None:
                points.append((float(coord[1]), float(coord[0])))
        if len(points) >= 2:
            points.append(points[0])
        return points

    def _fit_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        coords = self._coords()
        if not coords:
            return ((0.0, 1.0), (0.0, 1.0))
        lons = [float(v[1]) for v in coords.values()]
        lats = [float(v[0]) for v in coords.values()]
        pad_lon = max((max(lons) - min(lons)) * 0.05, 0.002)
        pad_lat = max((max(lats) - min(lats)) * 0.05, 0.002)
        return (
            (min(lons) - pad_lon, max(lons) + pad_lon),
            (min(lats) - pad_lat, max(lats) + pad_lat),
        )

    def _grid_shape(self) -> tuple[int, int]:
        n = max(len(self.system), 1)
        cols = min(5, max(1, math.ceil(math.sqrt(n))))
        rows = math.ceil(n / cols)
        return rows, cols

    def _draw_subplot(self, ax, route: JeepneyRoute, index: int, bounds) -> None:
        road_segments = self._edge_segments()
        if road_segments:
            ax.add_collection(
                LineCollection(
                    road_segments,
                    colors=self.road_color,
                    linewidths=0.5,
                    alpha=self.road_alpha,
                )
            )

        route_points = self._route_segments(route)
        if route_points:
            xs = [p[0] for p in route_points]
            ys = [p[1] for p in route_points]
            ax.plot(xs, ys, color=self.route_color, linewidth=2.2)
            ax.scatter(xs[:-1], ys[:-1], s=8, color=self.route_color, zorder=3)

        (lon_min, lon_max), (lat_min, lat_max) = bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(route.route_id, fontsize=10, color="#111827", pad=4)

    def _build_figure(self):
        rows, cols = self._grid_shape()
        self.fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2), facecolor="white")
        self.axes = axes
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
        bounds = self._fit_bounds()
        self.fig.suptitle(self.title, fontsize=16, color="#111827")

        for idx, ax in enumerate(axes_list):
            if idx < len(self.system):
                self._draw_subplot(ax, self.system.routes[idx], idx, bounds)
            else:
                ax.axis("off")

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        return self.fig

    def show(self):
        if len(self.system) == 0:
            raise ValueError("No routes to display.")
        self._build_figure()
        plt.show()
        return self.fig

    def save(self, output_path: str | Path) -> Path:
        if self.fig is None:
            self._build_figure()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        return out
