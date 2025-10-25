import argparse
from simulation.simulation_core import run_simulation
from visualization.map_visualizer import visualize_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "visualize"], required=True)
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulation(
            map_path="data/osm/iligan_roads.graphml",
            route_file="data/routes/route_set_1.json"
        )
    elif args.mode == "visualize":
        visualize_map(
            road_map="data/osm/iligan_roads.graphml",
            building_data="data/osm/iligan_buildings.geojson",
            route_data="data/results/optimized_routes.json"
        )

if __name__ == "__main__":
    main()
