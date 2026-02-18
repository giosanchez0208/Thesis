# Thesis Project — Iligan City Network + Travel Graph (Part A → Part B)

Part A produces the core datasets you will use in Part B:

- **OSM road networks**
  - Walk network graph
  - Drive network graph
- **Route stops/terminals**
  - Loaded from a JSON file (dummy for now, replace with real data later)
- **Travel graph arrays**
  - `nodes.csv`
  - `edges.csv`
  - `index maps` for consistent node indexing
- **Figures**
  - Road network image (OSM drive network + stop markers)
  - Travel graph image

**Important:** The **road network** is the general OSM road network — not the jeepney route geometry. The jeepney-related part is represented by your **stops/terminals** (and later by real route geometry or computed drive paths).

---

## Requirements

### Python
- Python **3.11**

### Python Packages (Minimum)
- osmnx>=1.8.1
- networkx>=3.2
- geopandas>=0.14
- shapely>=2.0
- pyproj>=3.6
- pandas>=2.1
- numpy>=1.26
- pyyaml>=6.0
- matplotlib>=3.8
- scikit-learn>=1.3

### Strongly Recommended Add-ons
These prevent common runtime issues and improve exporting:
- **pyogrio** (faster + more reliable GeoPackage writing)
- **rtree** (spatial index speedups)

Install:

```bash
pip install -r requirements.txt
pip install pyogrio rtree
```

## Setup

```bash
python3 -m venv Thesis_venv
source Thesis_venv/bin/activate
pip install -r requirements.tx
```

## Configuration

configs/part_a.yaml controls:

- City query (Iligan City)
- Output folders
- Routes JSON path
- Travel graph weights and options

## Run Part A

```bash
python3 scripts/part_a_build.py --config configs/part_a.yaml
```