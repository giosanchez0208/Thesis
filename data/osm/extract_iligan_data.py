import osmnx as ox

# ✅ Bounding box coordinates for Iligan City (approx)
north, south, east, west = 8.28, 8.15, 124.30, 124.20

print("⏳ Downloading road network (Iligan City)...")
G = ox.graph.graph_from_bbox(
    bbox=(north, south, east, west),
    network_type="drive"
)
ox.graph.save_graphml(G, "data/osm/iligan_roads.graphml")

print("⏳ Downloading building footprints...")
buildings = ox.features_from_bbox(
    bbox=(north, south, east, west),
    tags={"building": True}
)
buildings.to_file("data/osm/iligan_buildings.geojson", driver="GeoJSON")

print("✅ Done! Iligan data saved locally to /data/osm/")
