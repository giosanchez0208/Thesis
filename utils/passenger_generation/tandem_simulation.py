"""Backward-compatible wrapper for the experimental Simulation class."""

from .simulation import Simulation, SimulationConfig, TandemPassengerJeepSimulation

__all__ = [
    "Simulation",
    "SimulationConfig",
    "TandemPassengerJeepSimulation",
]
