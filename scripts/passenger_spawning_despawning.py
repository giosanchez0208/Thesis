import argparse
from pathlib import Path
from time import perf_counter
from datetime import datetime

from src.part_b.spawner import PassengerSpawner

def run_sim(config, timesteps=600):
    sp = PassengerSpawner(config)
    sp.load_nodes_edges()
    sim_time = 0.0
    dt = sp.dt_minutes * 60.0
    t0 = perf_counter()
    for t in range(timesteps):
        spawned = sp.spawn_step(sim_time_seconds=sim_time)
        sp.save_iteration(spawned, timestep=t)
        sp.despawn_step(sim_time_seconds=sim_time)
        sim_time += dt
    duration = perf_counter() - t0
    print(f"Done: {timesteps} timesteps in {duration:.2f}s; outputs -> {sp.out_folder}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--timesteps", type=int, default=600, help="Number of timesteps (e.g. 600 = 10 hours at dt=1min)")
    args = p.parse_args()
    run_sim(args.config, timesteps=args.timesteps)