import osmnx as ox
import networkx as nx

def load_environment(map_path):
    print("📦 Loading environment...")
    G = ox.load_graphml(map_path)
    print(f"✅ Loaded {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G
