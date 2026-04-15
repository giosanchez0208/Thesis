# pyright: ignore
"""
travel_graph_visualizer.py
==========================
Standalone HTML visualizer for a TravelGraphManager instance.

What it shows
-------------
- Every graph layer as a toggleable overlay on a Leaflet map:
    • Start-walk nodes  (blue dots)
    • End-walk nodes    (teal dots)
    • Ride nodes        (red dots)
    • Start-walk edges  (blue lines)
    • End-walk edges    (teal lines)
    • Ride edges        (red lines, per-route colouring if routes loaded)
    • Wait edges        (green)
    • Alight edges      (purple)
    • Direct edges      (grey)
    • Transfer edges    (orange)
- Layer-control panel to toggle each layer on/off independently.
- Node info popup on click (node_id, layer, lat/lon).
- Edge info popup on click (edge_id, type, dist).
- Stats panel: counts per edge type.
- A "Shortest Path" tool: click two nodes on the map to compute + draw a path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

# Node layer colours
_NODE_COLOURS = {
    "start_walk": "#3b82f6",   # blue
    "end_walk":   "#14b8a6",   # teal
    "ride":       "#ef4444",   # red
}

# Edge type colours
_EDGE_COLOURS = {
    "start_walk": "#3b82f6",
    "end_walk":   "#14b8a6",
    "ride":       "#ef4444",
    "wait":       "#22c55e",
    "alight":     "#a855f7",
    "direct":     "#94a3b8",
    "transfer":   "#f97316",
}

# Per-route colours (cycles when >10 routes)
_ROUTE_COLOURS = [
    "#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316",
    "#06b6d4", "#f59e0b", "#ec4899", "#84cc16", "#6366f1",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main visualizer class
# ─────────────────────────────────────────────────────────────────────────────

class TravelGraphVisualizer:
    """
    Generates a rich self-contained HTML explorer from a TravelGraphManager.

    Parameters
    ----------
    manager : TravelGraphManager
        A fully initialised TravelGraphManager (with nodes_csv loaded).
    title : str
        Title shown in the map header.
    """

    def __init__(self, manager, title: str = "Iligan Travel Graph Explorer"):
        self._mgr   = manager
        self._title = title

    # ── Public API ───────────────────────────────────────────────────────────

    def export(self, output_html) -> Path:
        """
        Generate the HTML explorer and write it to *output_html*.

        Returns the absolute Path of the written file.
        """
        out = Path(output_html).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html()
        out.write_text(html, encoding="utf-8")
        print(f"Travel graph explorer saved to {out}")
        return out

    # ── Private: data extraction ─────────────────────────────────────────────

    def _extract_nodes(self) -> dict:
        """
        Returns { layer_name: [ {node_id, lat, lon}, ... ] }
        Layers: start_walk, end_walk, ride
        """
        mgr = self._mgr
        if mgr._nodes_df is None:
            return {}

        nodes_df = mgr._nodes_df
        layers = {}
        for layer in ["start_walk", "end_walk", "ride"]:
            subset = nodes_df[nodes_df["layer"] == layer] if "layer" in nodes_df.columns else nodes_df
            records = []
            for row in subset.itertuples(index=False):
                node_id = str(row.node_id)
                # Check node is actually in the active graph
                if node_id not in mgr._graph:
                    continue
                records.append({
                    "id":  node_id,
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                })
            if records:
                layers[layer] = records
        return layers

    def _extract_edges(self) -> dict:
        """
        Returns { edge_type: [ {edge_id, u, v, lat_u, lon_u, lat_v, lon_v, dist, colour}, ... ] }
        For ride edges with routes, colour is per-route.
        """
        mgr = self._mgr
        coords = mgr._node_coords   # node_id → (lat, lon)

        # Build ride edge → route colour mapping
        ride_colour_map: dict[str, str] = {}
        if mgr._routes:
            for idx, route in enumerate(mgr._routes):
                colour = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
                for u_node, v_node in route.edge_pairs:
                    matches = mgr._active_edges[
                        (mgr._active_edges["u"] == u_node) &
                        (mgr._active_edges["v"] == v_node) &
                        (mgr._active_edges["edge_type"] == "ride")
                    ]["edge_id"]
                    for eid in matches:
                        ride_colour_map[eid] = colour

        buckets: dict[str, list] = {}
        for row in mgr._active_edges.itertuples(index=False):
            et  = row.edge_type
            eid = row.edge_id
            cu  = coords.get(row.u)
            cv  = coords.get(row.v)
            if cu is None or cv is None:
                continue
            colour = (
                ride_colour_map.get(eid, _EDGE_COLOURS["ride"])
                if et == "ride"
                else _EDGE_COLOURS.get(et, "#888888")
            )
            record = {
                "id":    eid,
                "u":     row.u,
                "v":     row.v,
                "lat_u": cu[0], "lon_u": cu[1],
                "lat_v": cv[0], "lon_v": cv[1],
                "dist":  round(float(row.dist), 1),
                "colour": colour,
            }
            buckets.setdefault(et, []).append(record)
        return buckets

    def _extract_routes(self) -> list:
        """Returns [ {id, colour, coords: [[lat,lon], ...]} ] for each JeepneyRoute."""
        mgr = self._mgr
        if not mgr._routes:
            return []
        coords = mgr._node_coords
        route_list = []
        for idx, route in enumerate(mgr._routes):
            colour = _ROUTE_COLOURS[idx % len(_ROUTE_COLOURS)]
            pts = []
            for nid in route.nodes:
                c = coords.get(nid)
                if c:
                    pts.append([c[0], c[1]])
            if pts:
                pts.append(pts[0])  # close loop
            route_list.append({"id": route.route_id, "colour": colour, "coords": pts})
        return route_list

    def _stats(self) -> dict:
        """Edge-type counts."""
        mgr = self._mgr
        return mgr._active_edges["edge_type"].value_counts().to_dict()

    # ── Private: HTML generation ─────────────────────────────────────────────

    def _build_html(self) -> str:
        nodes_by_layer = self._extract_nodes()
        edges_by_type  = self._extract_edges()
        routes         = self._extract_routes()
        stats          = self._stats()

        nodes_json  = json.dumps(nodes_by_layer)
        edges_json  = json.dumps(edges_by_type)
        routes_json = json.dumps(routes)
        stats_json  = json.dumps(stats)

        node_colours_json = json.dumps(_NODE_COLOURS)
        edge_colours_json = json.dumps(_EDGE_COLOURS)

        title = self._title

        # Count totals for the header
        total_nodes = sum(len(v) for v in nodes_by_layer.values())
        total_edges = sum(len(v) for v in edges_by_type.values())

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --surface2: #222636;
    --border:   #2e3248;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --accent:   #6366f1;
    --accent2:  #22d3ee;
    --radius:   10px;
    --shadow:   0 4px 24px rgba(0,0,0,0.5);
    --font-mono: 'Space Mono', monospace;
    --font-sans: 'DM Sans', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; background: var(--bg); color: var(--text); font-family: var(--font-sans); }}

  /* ── Layout ── */
  #app {{ display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{
    width: 300px; min-width: 260px; max-width: 360px;
    display: flex; flex-direction: column;
    background: var(--surface); border-right: 1px solid var(--border);
    overflow-y: auto; z-index: 100; resize: horizontal;
  }}
  #map-wrap {{ flex: 1; position: relative; }}
  #map {{ width: 100%; height: 100%; }}

  /* ── Sidebar header ── */
  #sidebar-header {{
    padding: 18px 16px 14px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #1e2235 0%, #16192b 100%);
  }}
  #sidebar-header h1 {{
    font-family: var(--font-mono);
    font-size: 13px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent2); margin-bottom: 6px;
  }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .stat-pill {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px;
    font-size: 11px; font-family: var(--font-mono);
    color: var(--muted);
  }}
  .stat-pill span {{ color: var(--text); font-weight: 700; }}

  /* ── Section blocks ── */
  .section {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
  .section-title {{
    font-size: 10px; font-family: var(--font-mono);
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 10px;
  }}

  /* ── Layer toggles ── */
  .layer-list {{ display: flex; flex-direction: column; gap: 5px; }}
  .layer-item {{
    display: flex; align-items: center; gap: 8px;
    cursor: pointer; padding: 6px 8px; border-radius: 6px;
    transition: background 0.15s;
    user-select: none;
  }}
  .layer-item:hover {{ background: var(--surface2); }}
  .layer-item.inactive {{ opacity: 0.4; }}
  .layer-swatch {{
    width: 14px; height: 14px; border-radius: 50%;
    flex-shrink: 0; border: 2px solid rgba(255,255,255,0.2);
  }}
  .layer-swatch.line {{
    border-radius: 3px; height: 4px; width: 20px;
  }}
  .layer-label {{ font-size: 12px; color: var(--text); flex: 1; }}
  .layer-count {{
    font-size: 10px; font-family: var(--font-mono);
    color: var(--muted); background: var(--surface2);
    padding: 1px 6px; border-radius: 4px;
  }}
  .layer-toggle {{
    width: 30px; height: 16px; border-radius: 8px;
    background: var(--border); position: relative;
    transition: background 0.2s; flex-shrink: 0;
  }}
  .layer-toggle::after {{
    content: ''; position: absolute;
    width: 12px; height: 12px; border-radius: 50%;
    background: #fff; top: 2px; left: 2px;
    transition: transform 0.2s;
  }}
  .layer-item:not(.inactive) .layer-toggle {{ background: var(--accent); }}
  .layer-item:not(.inactive) .layer-toggle::after {{ transform: translateX(14px); }}

  /* ── Stats breakdown ── */
  .stats-grid {{ display: flex; flex-direction: column; gap: 4px; }}
  .stats-bar-row {{ display: flex; align-items: center; gap: 8px; }}
  .stats-bar-label {{ font-size: 11px; width: 90px; color: var(--muted); font-family: var(--font-mono); }}
  .stats-bar-track {{
    flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
  }}
  .stats-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.5s; }}
  .stats-bar-val {{ font-size: 10px; font-family: var(--font-mono); color: var(--muted); width: 44px; text-align: right; }}

  /* ── Path tool ── */
  #path-section {{ background: linear-gradient(135deg, #1c1f35, #181b2e); }}
  #path-mode-btn {{
    width: 100%; padding: 8px; border-radius: var(--radius);
    border: 1px solid var(--accent); background: transparent;
    color: var(--accent); font-family: var(--font-mono);
    font-size: 11px; cursor: pointer; letter-spacing: 0.06em;
    transition: all 0.2s; text-transform: uppercase;
  }}
  #path-mode-btn:hover, #path-mode-btn.active {{
    background: var(--accent); color: #fff;
  }}
  #path-status {{ margin-top: 8px; font-size: 11px; color: var(--muted); min-height: 16px; }}
  #path-result {{
    margin-top: 8px; padding: 8px; border-radius: 6px;
    background: var(--surface2); border: 1px solid var(--border);
    font-size: 11px; font-family: var(--font-mono); display: none;
  }}
  #path-result .path-stat {{ color: var(--text); margin: 2px 0; }}
  #path-clear-btn {{
    margin-top: 6px; padding: 4px 10px; border-radius: 6px;
    border: 1px solid var(--border); background: transparent;
    color: var(--muted); font-size: 11px; cursor: pointer;
    font-family: var(--font-mono); display: none;
  }}
  #path-clear-btn:hover {{ color: var(--text); border-color: var(--muted); }}

  /* ── Map controls overlay ── */
  #tile-switcher {{
    position: absolute; bottom: 24px; left: 12px; z-index: 500;
    display: flex; gap: 6px;
  }}
  .tile-btn {{
    padding: 5px 10px; border-radius: 6px;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); font-size: 11px; cursor: pointer;
    font-family: var(--font-mono); transition: all 0.2s;
  }}
  .tile-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .tile-btn:hover {{ color: var(--text); }}

  /* Leaflet popup override */
  .leaflet-popup-content-wrapper {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    color: var(--text) !important;
  }}
  .leaflet-popup-tip {{ background: var(--surface) !important; }}
  .leaflet-popup-content {{ font-family: var(--font-mono); font-size: 11px; line-height: 1.7; }}
  .popup-key {{ color: var(--muted); }}
  .popup-val {{ color: var(--accent2); font-weight: 700; }}

  /* Scrollbar */
  #sidebar::-webkit-scrollbar {{ width: 4px; }}
  #sidebar::-webkit-scrollbar-track {{ background: transparent; }}
  #sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
</style>
</head>
<body>
<div id="app">

  <!-- ── SIDEBAR ── -->
  <div id="sidebar">

    <div id="sidebar-header">
      <h1>⬡ {title}</h1>
      <div class="stat-row">
        <div class="stat-pill">Nodes <span>{total_nodes:,}</span></div>
        <div class="stat-pill">Edges <span>{total_edges:,}</span></div>
      </div>
    </div>

    <!-- Node layers -->
    <div class="section">
      <div class="section-title">Node Layers</div>
      <div class="layer-list" id="node-layer-list"></div>
    </div>

    <!-- Edge layers -->
    <div class="section">
      <div class="section-title">Edge Layers</div>
      <div class="layer-list" id="edge-layer-list"></div>
    </div>

    <!-- Route overlays (only shown if routes exist) -->
    <div class="section" id="route-section" style="display:none">
      <div class="section-title">Jeepney Routes</div>
      <div class="layer-list" id="route-layer-list"></div>
    </div>

    <!-- Stats -->
    <div class="section">
      <div class="section-title">Edge Breakdown</div>
      <div class="stats-grid" id="stats-grid"></div>
    </div>

    <!-- Shortest path tool -->
    <div class="section" id="path-section">
      <div class="section-title">Shortest Path Tool</div>
      <button id="path-mode-btn" onclick="togglePathMode()">
        ⟳ Pick two nodes
      </button>
      <div id="path-status">Click to activate, then click two nodes on the map.</div>
      <div id="path-result"></div>
      <button id="path-clear-btn" onclick="clearPath()">✕ Clear path</button>
    </div>

  </div><!-- /sidebar -->

  <!-- ── MAP ── -->
  <div id="map-wrap">
    <div id="map"></div>
    <div id="tile-switcher">
      <button class="tile-btn active" onclick="setTile('dark')">Dark</button>
      <button class="tile-btn" onclick="setTile('light')">Light</button>
      <button class="tile-btn" onclick="setTile('satellite')">Satellite</button>
    </div>
  </div>

</div><!-- /app -->

<script>
// ── Data injected from Python ────────────────────────────────────────────────
const NODES_BY_LAYER  = {nodes_json};
const EDGES_BY_TYPE   = {edges_json};
const ROUTES          = {routes_json};
const STATS           = {stats_json};
const NODE_COLOURS    = {node_colours_json};
const EDGE_COLOURS    = {edge_colours_json};

// ── Leaflet map init ─────────────────────────────────────────────────────────
const map = L.map("map", {{ zoomControl: true }});

const TILES = {{
  dark:      L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{ attribution:"&copy; OpenStreetMap &copy; CARTO", maxZoom:20 }}),
  light:     L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{ attribution:"&copy; OpenStreetMap &copy; CARTO", maxZoom:20 }}),
  satellite: L.tileLayer("https://{{s}}.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}", {{ subdomains:["mt0","mt1","mt2","mt3"], attribution:"&copy; Google", maxZoom:20 }}),
}};
let currentTile = "dark";
TILES.dark.addTo(map);

function setTile(name) {{
  TILES[currentTile].remove();
  TILES[name].addTo(map);
  currentTile = name;
  document.querySelectorAll(".tile-btn").forEach(b => b.classList.toggle("active", b.textContent.toLowerCase()===name));
}}

// ── Layer groups ─────────────────────────────────────────────────────────────
const leafletLayers = {{}};   // key → L.LayerGroup
const layerVisible  = {{}};   // key → bool

function makeLayerKey(category, name) {{ return category + "__" + name; }}

// ── Build node layers ────────────────────────────────────────────────────────
const nodeLayerOrder = ["start_walk", "end_walk", "ride"];
nodeLayerOrder.forEach(layer => {{
  const nodes = NODES_BY_LAYER[layer];
  if (!nodes || nodes.length === 0) return;
  const key    = makeLayerKey("node", layer);
  const colour = NODE_COLOURS[layer] || "#888";
  const group  = L.layerGroup();

  nodes.forEach(n => {{
    const circle = L.circleMarker([n.lat, n.lon], {{
      radius: 3, color: colour, fillColor: colour,
      fillOpacity: 0.85, weight: 1, opacity: 0.9,
    }});
    circle.bindPopup(
      `<span class="popup-key">node_id</span><br/><span class="popup-val">${{n.id}}</span><br/>` +
      `<span class="popup-key">layer</span>&nbsp;<span class="popup-val">${{layer}}</span><br/>` +
      `<span class="popup-key">lat</span>&nbsp;<span class="popup-val">${{n.lat.toFixed(6)}}</span>&nbsp;` +
      `<span class="popup-key">lon</span>&nbsp;<span class="popup-val">${{n.lon.toFixed(6)}}</span>`,
      {{ maxWidth: 240 }}
    );
    // Path mode: click to set origin/dest
    circle.on("click", e => {{ if (pathMode) {{ L.DomEvent.stopPropagation(e); handleNodeClick(n.id, n.lat, n.lon); }} }});
    circle.addTo(group);
  }});

  leafletLayers[key] = group;
  layerVisible[key]  = true;
  group.addTo(map);
}});

// ── Build edge layers ────────────────────────────────────────────────────────
const edgeLayerOrder = ["start_walk", "end_walk", "ride", "wait", "alight", "direct", "transfer"];
edgeLayerOrder.forEach(et => {{
  const edges = EDGES_BY_TYPE[et];
  if (!edges || edges.length === 0) return;
  const key   = makeLayerKey("edge", et);
  const group = L.layerGroup();
  const defaultHidden = ["start_walk","end_walk","direct","alight"].includes(et);

  edges.forEach(e => {{
    const line = L.polyline(
      [[e.lat_u, e.lon_u], [e.lat_v, e.lon_v]],
      {{ color: e.colour, weight: et === "ride" ? 3 : 1.5, opacity: 0.75 }}
    );
    line.bindPopup(
      `<span class="popup-key">edge_id</span><br/><span class="popup-val">${{e.id}}</span><br/>` +
      `<span class="popup-key">type</span>&nbsp;<span class="popup-val">${{et}}</span><br/>` +
      `<span class="popup-key">u</span>&nbsp;<span class="popup-val">${{e.u}}</span><br/>` +
      `<span class="popup-key">v</span>&nbsp;<span class="popup-val">${{e.v}}</span><br/>` +
      `<span class="popup-key">dist</span>&nbsp;<span class="popup-val">${{e.dist}} m</span>`,
      {{ maxWidth: 280 }}
    );
    line.addTo(group);
  }});

  leafletLayers[key] = group;
  layerVisible[key]  = !defaultHidden;
  if (!defaultHidden) group.addTo(map);
}});

// ── Build route layers ───────────────────────────────────────────────────────
if (ROUTES.length > 0) {{
  document.getElementById("route-section").style.display = "";
  ROUTES.forEach(r => {{
    const key   = makeLayerKey("route", r.id);
    const group = L.layerGroup();
    if (r.coords.length >= 2) {{
      L.polyline(r.coords, {{ color: r.colour, weight: 4, opacity: 0.45, dashArray: "8 5" }})
       .bindPopup(`<span class="popup-key">Route</span><br/><span class="popup-val">${{r.id}}</span>`)
       .addTo(group);
    }}
    leafletLayers[key] = group;
    layerVisible[key]  = true;
    group.addTo(map);
  }});
}}

// ── Fit map to all visible nodes ─────────────────────────────────────────────
(function fitBounds() {{
  const allCoords = [];
  Object.values(NODES_BY_LAYER).forEach(arr => arr.forEach(n => allCoords.push([n.lat, n.lon])));
  if (allCoords.length > 1) map.fitBounds(allCoords, {{ padding: [30, 30] }});
  else if (allCoords.length === 1) map.setView(allCoords[0], 13);
}})();

// ── Sidebar: node layer toggles ──────────────────────────────────────────────
function buildLayerItem(key, label, colour, count, isLine) {{
  const item = document.createElement("div");
  item.className = "layer-item" + (layerVisible[key] ? "" : " inactive");
  item.id = "layer-item-" + key;
  item.innerHTML = `
    <div class="layer-swatch${{isLine ? " line" : ""}}" style="background:${{colour}}"></div>
    <span class="layer-label">${{label}}</span>
    <span class="layer-count">${{count.toLocaleString()}}</span>
    <div class="layer-toggle"></div>
  `;
  item.onclick = () => toggleLayer(key);
  return item;
}}

const nodeList = document.getElementById("node-layer-list");
nodeLayerOrder.forEach(layer => {{
  const key  = makeLayerKey("node", layer);
  if (!leafletLayers[key]) return;
  const n = (NODES_BY_LAYER[layer] || []).length;
  nodeList.appendChild(buildLayerItem(key, layer.replace("_"," "), NODE_COLOURS[layer] || "#888", n, false));
}});

const edgeList = document.getElementById("edge-layer-list");
edgeLayerOrder.forEach(et => {{
  const key = makeLayerKey("edge", et);
  if (!leafletLayers[key]) return;
  const n = (EDGES_BY_TYPE[et] || []).length;
  edgeList.appendChild(buildLayerItem(key, et.replace("_"," "), EDGE_COLOURS[et] || "#888", n, true));
}});

const routeList = document.getElementById("route-layer-list");
ROUTES.forEach(r => {{
  const key = makeLayerKey("route", r.id);
  routeList.appendChild(buildLayerItem(key, "Route " + r.id, r.colour, 1, true));
}});

// ── Toggle a layer ───────────────────────────────────────────────────────────
function toggleLayer(key) {{
  const group = leafletLayers[key];
  if (!group) return;
  const visible = layerVisible[key];
  if (visible) {{ group.remove(); }}
  else         {{ group.addTo(map); }}
  layerVisible[key] = !visible;
  const item = document.getElementById("layer-item-" + key);
  if (item) item.classList.toggle("inactive", visible);
}}

// ── Stats bars ───────────────────────────────────────────────────────────────
(function buildStats() {{
  const grid  = document.getElementById("stats-grid");
  const total = Object.values(STATS).reduce((a, b) => a + b, 0) || 1;
  const order = ["start_walk","end_walk","ride","wait","alight","direct","transfer"];
  order.forEach(et => {{
    if (!STATS[et]) return;
    const pct   = (STATS[et] / total * 100).toFixed(1);
    const colour= EDGE_COLOURS[et] || "#888";
    const row   = document.createElement("div");
    row.className = "stats-bar-row";
    row.innerHTML = `
      <span class="stats-bar-label">${{et.replace("_"," ")}}</span>
      <div class="stats-bar-track">
        <div class="stats-bar-fill" style="width:${{pct}}%;background:${{colour}}"></div>
      </div>
      <span class="stats-bar-val">${{STATS[et].toLocaleString()}}</span>
    `;
    grid.appendChild(row);
  }});
}})();

// ── Shortest path tool ───────────────────────────────────────────────────────
let pathMode    = false;
let pathOrigin  = null;   // {{id, lat, lon}}
let pathDest    = null;
let pathPolylines = [];
let pathMarkers   = [];

// Pre-build adjacency from edge data for client-side Dijkstra
// We use a simplified weight: dist for walk/ride, large flat for wait/transfer
const ADJ = {{}};   // node_id → [{{to, edge_id, et, dist, w}}]

(function buildAdj() {{
  const W = {{ start_walk:0.0142, end_walk:0.0142, ride:0.0071, direct:0, alight:0, wait:8.5, transfer:14.2 }};
  edgeLayerOrder.forEach(et => {{
    (EDGES_BY_TYPE[et] || []).forEach(e => {{
      if (!ADJ[e.u]) ADJ[e.u] = [];
      ADJ[e.u].push({{ to: e.v, id: e.id, et, dist: e.dist, w: (W[et] !== undefined ? (e.dist * W[et] + (et==="wait"?8.5:et==="transfer"?14.2:0)) : e.dist) }});
    }});
  }});
}})();

// Simple Dijkstra on ADJ
function dijkstra(src, dst) {{
  const dist = {{}};
  const prev = {{}};
  const prevEdge = {{}};
  const pq = [[0, src]];
  dist[src] = 0;
  while (pq.length) {{
    pq.sort((a,b) => a[0]-b[0]);
    const [d, u] = pq.shift();
    if (d > (dist[u] ?? Infinity)) continue;
    if (u === dst) break;
    for (const nb of (ADJ[u] || [])) {{
      const nd = d + nb.w;
      if (nd < (dist[nb.to] ?? Infinity)) {{
        dist[nb.to] = nd; prev[nb.to] = u; prevEdge[nb.to] = nb;
        pq.push([nd, nb.to]);
      }}
    }}
  }}
  if (dist[dst] === undefined) return null;
  // Reconstruct
  const path = [];
  let cur = dst;
  while (prev[cur] !== undefined) {{
    path.unshift(prevEdge[cur]);
    cur = prev[cur];
  }}
  return path;
}}

function togglePathMode() {{
  pathMode = !pathMode;
  const btn = document.getElementById("path-mode-btn");
  btn.classList.toggle("active", pathMode);
  if (!pathMode) {{
    clearPath();
  }} else {{
    clearPath();
    document.getElementById("path-status").textContent = "Click a node on the map to set origin.";
    map.getContainer().style.cursor = "crosshair";
  }}
}}

function clearPath() {{
  pathOrigin = null; pathDest = null;
  pathPolylines.forEach(p => p.remove()); pathPolylines = [];
  pathMarkers.forEach(m => m.remove());   pathMarkers   = [];
  document.getElementById("path-result").style.display = "none";
  document.getElementById("path-clear-btn").style.display = "none";
  document.getElementById("path-status").textContent = pathMode ? "Click a node to set origin." : "";
  map.getContainer().style.cursor = "";
}}

// Node click coord lookup
const NODE_COORD_MAP = {{}};
Object.values(NODES_BY_LAYER).forEach(arr => arr.forEach(n => {{ NODE_COORD_MAP[n.id] = [n.lat, n.lon]; }}));

function handleNodeClick(nodeId, lat, lon) {{
  if (!pathOrigin) {{
    pathOrigin = {{id: nodeId, lat, lon}};
    const m = L.circleMarker([lat,lon], {{radius:8, color:"#22c55e", fillColor:"#22c55e", fillOpacity:1, weight:2}})
               .bindPopup("<b>Origin</b><br/>" + nodeId).addTo(map).openPopup();
    pathMarkers.push(m);
    document.getElementById("path-status").textContent = "Origin set. Now click destination node.";
  }} else if (!pathDest) {{
    if (nodeId === pathOrigin.id) return;
    pathDest = {{id: nodeId, lat, lon}};
    const m = L.circleMarker([lat,lon], {{radius:8, color:"#ef4444", fillColor:"#ef4444", fillOpacity:1, weight:2}})
               .bindPopup("<b>Destination</b><br/>" + nodeId).addTo(map).openPopup();
    pathMarkers.push(m);
    document.getElementById("path-status").textContent = "Computing shortest path…";
    setTimeout(() => computeAndDrawPath(), 10);
  }}
}}

function computeAndDrawPath() {{
  const edgePath = dijkstra(pathOrigin.id, pathDest.id);
  const resultEl = document.getElementById("path-result");
  if (!edgePath || edgePath.length === 0) {{
    document.getElementById("path-status").textContent = "No path found between selected nodes.";
    return;
  }}

  // Draw segments
  edgePath.forEach(e => {{
    const coords = [[NODE_COORD_MAP[e.to]?.[0] ?? 0, NODE_COORD_MAP[e.to]?.[1] ?? 0]];
    // We need u coords too — find from ADJ parent implicitly via edgePath order
  }});

  // Rebuild coords from node path
  const nodePath = [pathOrigin.id];
  edgePath.forEach(e => nodePath.push(e.to));

  const ECOL = {{ start_walk:"#3b82f6", end_walk:"#14b8a6", ride:"#ef4444", wait:"#22c55e", alight:"#a855f7", direct:"#94a3b8", transfer:"#f97316" }};
  let walkDist = 0, rideDist = 0, nTransfer = 0;

  for (let i = 0; i < edgePath.length; i++) {{
    const e   = edgePath[i];
    const u   = nodePath[i];
    const v   = nodePath[i+1];
    const cu  = NODE_COORD_MAP[u];
    const cv  = NODE_COORD_MAP[v];
    if (!cu || !cv) continue;
    const col = ECOL[e.et] || "#888";
    const pl  = L.polyline([cu, cv], {{ color: col, weight: 5, opacity: 0.95 }})
                  .bindPopup(`<span class="popup-val">${{e.et}}</span><br/>dist: ${{e.dist.toFixed(0)}} m`)
                  .addTo(map);
    pathPolylines.push(pl);
    if (["start_walk","end_walk","direct"].includes(e.et)) walkDist += e.dist;
    if (e.et === "ride") rideDist += e.dist;
    if (e.et === "transfer") nTransfer++;
  }}

  // Fit bounds
  const allCoords = nodePath.map(n => NODE_COORD_MAP[n]).filter(Boolean);
  if (allCoords.length > 1) map.fitBounds(allCoords, {{ padding:[60,60] }});

  resultEl.innerHTML =
    `<div class="path-stat">🚶 Walk <b>${{walkDist.toFixed(0)}} m</b></div>` +
    `<div class="path-stat">🚌 Ride <b>${{rideDist.toFixed(0)}} m</b></div>` +
    `<div class="path-stat">🔄 Transfers <b>${{nTransfer}}</b></div>` +
    `<div class="path-stat">📍 Edges <b>${{edgePath.length}}</b></div>`;
  resultEl.style.display = "block";
  document.getElementById("path-clear-btn").style.display = "inline-block";
  document.getElementById("path-status").textContent = "Path drawn. Click clear to reset.";
}}

</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def visualize_travel_graph(
    edges_csv: str,
    nodes_csv: str,
    output_html: str,
    routes=None,
    walk_wt: float = 0.0142,
    ride_wt: float = 0.0071,
    wait_wt: float = 8.5,
    transfer_wt: float = 14.2,
    title: str = "Iligan Travel Graph Explorer",
) -> Path:
    """
    One-shot helper: load a TravelGraphManager and export the HTML explorer.

    Parameters
    ----------
    edges_csv    : path to iligan_travel_graph.csv
    nodes_csv    : path to travel_graph_nodes.csv
    output_html  : where to write the HTML file
    routes       : list[JeepneyRoute] or None (full ride graph if None)
    walk_wt / ride_wt / wait_wt / transfer_wt : disutility weights
    title        : page title

    Returns
    -------
    Path  Absolute path to the written HTML file.

    Example
    -------
    >>> from travel_graph_visualizer import visualize_travel_graph
    >>> visualize_travel_graph(
    ...     edges_csv="data/iligan_travel_graph.csv",
    ...     nodes_csv="data/travel_graph_nodes.csv",
    ...     output_html="results/travel_graph/explorer.html",
    ... )
    """
    # Import here so the module doesn't hard-require it at import time
    import sys, importlib
    # Try to import TravelGraphManager from the utils package or travel_graph module
    TravelGraphManager = None
    for module_name in ["utils.travel_graph", "travel_graph", "utils"]:
        try:
            mod = importlib.import_module(module_name)
            TravelGraphManager = getattr(mod, "TravelGraphManager", None)
            if TravelGraphManager:
                break
        except ImportError:
            continue

    if TravelGraphManager is None:
        raise ImportError(
            "Could not import TravelGraphManager. "
            "Make sure utils/travel_graph.py is on the Python path."
        )

    manager = TravelGraphManager(
        edges_csv=edges_csv,
        nodes_csv=nodes_csv,
        routes=routes,
        walk_wt=walk_wt,
        ride_wt=ride_wt,
        wait_wt=wait_wt,
        transfer_wt=transfer_wt,
    )
    viz = TravelGraphVisualizer(manager, title=title)
    return viz.export(output_html)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point (optional)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TravelGraph HTML explorer")
    parser.add_argument("--edges",  required=True, help="Path to iligan_travel_graph.csv")
    parser.add_argument("--nodes",  required=True, help="Path to travel_graph_nodes.csv")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--title",  default="Iligan Travel Graph Explorer")
    args = parser.parse_args()

    visualize_travel_graph(
        edges_csv=args.edges,
        nodes_csv=args.nodes,
        output_html=args.output,
        title=args.title,
    )
