from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import osmnx as ox

from .routes import RouteDef


def plot_road_network(drive_G: nx.MultiDiGraph, routes: List[RouteDef], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = ox.plot_graph(
        drive_G,
        show=False,
        close=False,
        node_size=0,
        edge_linewidth=0.6,
        bgcolor="white",
    )

    # Overlay stops/terminals (from routes)
    lats, lons, colors = [], [], []
    for r in routes:
        for p in r.points:
            lats.append(p.lat)
            lons.append(p.lon)
            colors.append("red" if p.point_type == "terminal" else "blue")

    if lats:
        ax.scatter(lons, lats, s=8, c=colors, alpha=0.8)

    ax.set_title(title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_travel_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    out_path: Path,
    title: str,
    max_edges_to_draw: int = 120000,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw a light visualization: plot edges as straight segments between node coords
    # (This is just for thesis visualization; routing uses arrays.)
    n = nodes_df.set_index("idx")

    e = edges_df
    if len(e) > max_edges_to_draw:
        e = e.sample(n=max_edges_to_draw, random_state=1)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot edges
    for _, row in e.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        x1, y1 = float(n.loc[u, "lon"]), float(n.loc[u, "lat"])
        x2, y2 = float(n.loc[v, "lon"]), float(n.loc[v, "lat"])
        ax.plot([x1, x2], [y1, y2], linewidth=0.2, alpha=0.25)

    # Plot nodes lightly
    ax.scatter(nodes_df["lon"], nodes_df["lat"], s=0.2, alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)