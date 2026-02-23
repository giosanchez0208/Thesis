## Thesis Project — Iligan City Multimodal Transport Model

Current Stage: **Data Synthesis & Network Generation** phase. Transforms raw OpenStreetMap (OSM) data and local transit information into a unified mathematical graph for routing optimization.

---

### Core Deliverables

The execution of the model generation script produces the foundational datasets required for the GA-ACO agent-based simulation:

* **Spatial Road Networks:** High-fidelity walking and driving graphs extracted from OSM.
* **Transit Infrastructure:** Jeepney stops and terminal nodes mapped directly onto the walking graph.
* **Multimodal Travel Graph:** A "generalized cost" network exported as `nodes.csv` and `edges.csv`, including weighted edges for walking, waiting, and transferring.
* **Analytical Visualizations:** Spatial plots of the road network and the resulting abstract travel graph.

---

### Requirements & Environment

* **Runtime:** Python 3.11
* **Essential Libraries:** `osmnx`, `networkx`, `geopandas`, `shapely`, `pyyaml`, `scikit-learn`.
* **Performance Boosters:** `pyogrio` for faster GeoPackage writing and `rtree` for spatial indexing.

```bash
# Environment Setup
python3 -m venv Thesis_venv
source Thesis_venv/bin/activate
pip install -r requirements.txt
pip install pyogrio rtree

```

---

### Execution & Configuration

The system relies on a centralized configuration file to define city boundaries, weight parameters for the travel model (e.g., `beta_wait`, `beta_walk`), and file paths.

**Configuration File:** `configs/iligan_transport_config.yaml`

**To generate the model:**

```bash
python3 scripts/build_network_model.py --config configs/iligan_transport_config.yaml

```

---

### Data Output Architecture

| Category | File Path | Description |
| --- | --- | --- |
| **Network Graphs** | `data/processed/graphs/iligan_*.graphml` | XML-based graphs for Walk and Drive networks. |
| **Model Arrays** | `data/processed/nodes.csv` & `edges.csv` | Flat files representing the multimodal travel graph. |
| **Metadata** | `data/processed/config_used.json` | A snapshot of the parameters used for reproducibility. |
| **GIS Assets** | `results/gpkg/iligan_drive.gpkg` | GeoPackage layers (nodes/edges) for spatial validation in QGIS. |
| **Visuals** | `results/figures/fig_*.png` | Mapped visualizations of the road network and travel logic. |

---

### Implementation Note

The **road network** provides the physical constraints of Iligan City, while the **travel graph** integrates the specific behavior of the jeepney system. By separating the physical geometry from the transit logic, you can iterate on route optimizations in the simulation phase without re-downloading the entire city map.