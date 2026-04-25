"""Route generation, fitness, and RL utilities."""

from .baseline_route_generator import BaselineRoute, BaselineRouteGenerator
from .jeepney_route_env import JeepneyRouteEnv, RouteFitnessResult, calculate_route_fitness
from .systemic_fitness_evaluator import SystemicFitnessEvaluator, SystemicFitnessResult
from .rl_training import (
    BestWorstRouteCallback,
    RouteTrainingArtifacts,
    RouteTrainingSnapshot,
    build_training_env,
    export_physical_route_html,
    export_training_results_csvs,
    route_nodes_to_latlon,
    train_route_agent,
)
from .route_spectrum_analysis import (
    build_route_nodes,
    build_route_spectrum_frame,
    compare_route_spectrum_frames,
    plot_correlation_delta,
    plot_correlation_heatmap,
    route_correlation_pairs,
    route_spectrum_correlation,
    summarize_route_spectrum,
)

__all__ = [
    "BaselineRoute",
    "BaselineRouteGenerator",
    "JeepneyRouteEnv",
    "RouteFitnessResult",
    "calculate_route_fitness",
    "SystemicFitnessEvaluator",
    "SystemicFitnessResult",
    "BestWorstRouteCallback",
    "RouteTrainingArtifacts",
    "RouteTrainingSnapshot",
    "build_training_env",
    "export_physical_route_html",
    "export_training_results_csvs",
    "route_nodes_to_latlon",
    "train_route_agent",
    "build_route_nodes",
    "build_route_spectrum_frame",
    "compare_route_spectrum_frames",
    "plot_correlation_delta",
    "plot_correlation_heatmap",
    "route_correlation_pairs",
    "route_spectrum_correlation",
    "summarize_route_spectrum",
]
