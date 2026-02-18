from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import pandas as pd
import networkx as nx
import osmnx as ox


def export_graphml(G: nx.MultiDiGraph, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=str(out_path))


def export_arrays(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, index_maps: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    maps_path = out_dir / "index_maps.json"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    maps_path.write_text(json.dumps(index_maps, indent=2), encoding="utf-8")