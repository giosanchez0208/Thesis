import pygame
import random
import csv
from simulation.environment import load_environment
from simulation.agent import Passenger, Jeepney
from simulation.pathfinding import shortest_path

def run_simulation(map_path, route_file):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Jeepney Simulation (Local)")
    clock = pygame.time.Clock()

    # Load map
    G = load_environment(map_path)
    nodes = list(G.nodes)
    
    # Random passengers (sample)
    passengers = [
        Passenger(random.choice(nodes), random.choice(nodes)) for _ in range(10)
    ]

    # Sample jeepney route
    with open(route_file, "r") as f:
        import json
        routes = json.load(f)
    jeepneys = [Jeepney(route) for route in routes["routes"]]

    running = True
    tick = 0
    print("▶️ Simulation running...")

    while running:
        screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move jeepneys
        for jeep in jeepneys:
            node = jeep.move()
            x, y = G.nodes[node]["x"], G.nodes[node]["y"]
            pygame.draw.circle(screen, (0, 100, 255), (int(x % 800), int(y % 600)), 4)

        # Draw passengers (static demo)
        for p in passengers:
            x, y = G.nodes[p.origin]["x"], G.nodes[p.origin]["y"]
            pygame.draw.circle(screen, (255, 0, 0), (int(x % 800), int(y % 600)), 3)

        tick += 1
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("✅ Simulation ended.")

    # Save dummy results
    with open("data/results/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Avg_Commute_Time", "Avg_Wait_Time"])
        writer.writerow([35.8, 12.4])

    optimized_routes = {f"Route_{i}": route for i, route in enumerate(routes["routes"])}
    import json
    with open("data/results/optimized_routes.json", "w") as f:
        json.dump(optimized_routes, f)

    print("💾 Results saved to data/results/")
