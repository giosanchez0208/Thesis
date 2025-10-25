import networkx as nx

def shortest_path(G, origin, destination):
    try:
        path = nx.shortest_path(G, origin, destination, weight="length")
        return path
    except nx.NetworkXNoPath:
        return []
