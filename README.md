# Thesis: Iligan City Multimodal Transportation Model

A multimodal transportation network model for Iligan City integrating OpenStreetMap road networks, jeepney transit routes, and passenger distribution data. Produces a weighted travel graph for agent-based simulation and routing optimization.

## Project Workflow

The project is organized into two main parts with supporting analysis:

### Part A: Travel Graph Construction

| Notebook | Purpose | Output |
|----------|---------|--------|
| [A0_travel_graph_construction.ipynb](notebooks/A0_travel_graph_construction.ipynb) | Builds layered multimodal network from OSM walking/driving graphs and jeepney routes | [travel_graph_nodes.csv](data/travel_graph_nodes.csv), [travel_graph.csv](data/iligan_travel_graph.csv) |
| [A1_data_review.ipynb](notebooks/A1_data_review.ipynb) | Validates and compares original vs. TomTom-imputed datasets | Data quality assessment |

**Key Resources:**
- Walk & Drive Graphs: [data/processed/graphs/](data) (GraphML format)
- Configuration: [configs/travel_graph_config.yaml](configs/travel_graph_config.yaml)
- Utilities: [utils/travel_graph/](utils/travel_graph/)

### Part B: Passenger Distribution Model

| Notebook | Purpose | Output |
|----------|---------|--------|
| [B0_build_passenger_heatmap.ipynb](notebooks/B0_build_passenger_heatmap.ipynb) | Gathers traffic data via TomTom API and imputes Average Daily Traffic (ADT) citywide | [iligan_node_with_traffic_data.csv](data/iligan_node_with_traffic_data.csv) |
| [B1_passenger_generation_map.ipynb](notebooks/B1_passenger_generation_map.ipynb) | Estimates pedestrian volumes using traffic intensity and network centrality | Passenger origin/destination spawn points |

**Key Resources:**
- Passenger model: [utils/passenger_generation/](utils/passenger_generation/)
- Traffic heatmap: [results/B0_passenger_heatmap/passenger_generation_heatmap.html](results/B0_passenger_heatmap/passenger_generation_heatmap.html)
- Journey explorer: [results/A0_travel_graph/interactive_passenger_explorer.html](results/A0_travel_graph/interactive_passenger_explorer.html)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Key Data Files

| File | Description |
|------|-------------|
| [data/travel_graph_nodes.csv](data/travel_graph_nodes.csv) | Node indices with layer assignment (START/RIDE/END), coordinates, type |
| [data/iligan_travel_graph.csv](data/iligan_travel_graph.csv) | Edge list with weights, distance, edge type (walk/ride/wait/transfer/alight) |
| [data/iligan_node_with_traffic_data.csv](data/iligan_node_with_traffic_data.csv) | Nodes enriched with traffic intensity and pedestrian volume estimates |

## Visualization Outputs

Interactive HTML maps are available in [results/](results/):
- [A0_travel_graph/](results/A0_travel_graph/) — Network topology and journey mapping
- [B0_passenger_heatmap/](results/B0_passenger_heatmap/) — Traffic and passenger density heatmaps