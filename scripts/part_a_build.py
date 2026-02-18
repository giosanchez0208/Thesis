#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import yaml

from src.part_a.osm import build_iligan_graphs
from src.part_a.routes import load_routes_and_snap
from src.part_a.travel_graph import build_travel_graph_arrays
from src.part_a.exporting import export_graphml, export_arrays
from src.part_a.visualize import plot_road_network, plot_travel_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Part A: Iligan OSMnx extraction + Travel Graph build")
    parser.add_argument("--config", type=str, default="configs/part_a.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Ensure dirs
    graphs_dir = Path(cfg["paths"]["graphs_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    graphs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build Iligan graphs (walk + drive)
    walk_G, drive_G, boundary_gdf = build_iligan_graphs(cfg)

    # 2) Load routes and snap points to walk graph nodes (and capture stop nodes)
    routes = load_routes_and_snap(
        routes_json_path=Path(cfg["routes"]["routes_json"]),
        walk_G=walk_G,
        snap_to_walk=bool(cfg["routes"].get("snap_to_walk_graph", True)),
    )

    # 3) Build travel graph arrays (nodes.csv, edges.csv)
    nodes_df, edges_df, index_maps = build_travel_graph_arrays(
        walk_G=walk_G,
        drive_G=drive_G,
        routes=routes,
        weights=cfg["travel_graph"]["weights"],
        include_direct_edges=bool(cfg["travel_graph"].get("include_direct_edges", True)),
        ride_distance_method=str(cfg["routes"].get("ride_distance_method", "drive_shortest_path")),
    )

    # 4) Export graphs + arrays
    export_graphml(walk_G, graphs_dir / "iligan_walk.graphml")
    export_graphml(drive_G, graphs_dir / "iligan_drive.graphml")

    export_arrays(
        nodes_df=nodes_df,
        edges_df=edges_df,
        index_maps=index_maps,
        out_dir=processed_dir,
    )

    # 5) Figures
    plot_road_network(
        drive_G=drive_G,
        routes=routes,
        out_path=figures_dir / "fig_iligan_road_network.png",
        title="Iligan City Road Network (Drive) + Route Stops/Terminals",
    )

    plot_travel_graph(
        nodes_df=nodes_df,
        edges_df=edges_df,
        out_path=figures_dir / "fig_iligan_travel_graph.png",
        title="Iligan Travel Graph (Start-Walk / Ride / End-Walk)",
        max_edges_to_draw=120000,  # keeps plot reasonable
    )

    # Also store the resolved config used for reproducibility
    (processed_dir / "part_a_config_used.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )

    print("✅ Part A complete.")
    print(f"- Graphs:   {graphs_dir}")
    print(f"- Arrays:   {processed_dir / 'nodes.csv'} and {processed_dir / 'edges.csv'}")
    print(f"- Figures:  {figures_dir}")


if __name__ == "__main__":
    main()