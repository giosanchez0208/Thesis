import folium
import geopandas as gpd
import osmnx as ox
import json

def visualize_map(road_map, building_data, route_data):
    print("🗺️ Generating visualization...")
    G = ox.load_graphml(road_map)
    buildings = gpd.read_file(building_data)

    center = ox.graph_to_gdfs(G, nodes=False, edges=False).unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=13)
    folium.TileLayer('cartodb positron').add_to(m)

    # Draw buildings
    for _, row in buildings.iterrows():
        if row.get("building") == "residential":
            color = "green"
        else:
            color = "blue"
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=2, color=color, fill=True, fill_opacity=0.5
        ).add_to(m)

    # Draw jeepney routes
    with open(route_data) as f:
        routes = json.load(f)

    colors = ["red", "orange", "purple", "cyan", "yellow"]
    for i, (name, coords) in enumerate(routes.items()):
        folium.PolyLine(
            locations=coords,
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.8,
            tooltip=name
        ).add_to(m)

    m.save("visualization/iligan_jeepney_map.html")
    print("✅ Saved map → visualization/iligan_jeepney_map.html")
