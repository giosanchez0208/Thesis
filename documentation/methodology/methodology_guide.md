# Methodology Guide

Use this file as the entry point for the methodology snippet set. It is meant to help future agent work by providing a fast map of what each snippet covers and how the pieces fit together.

## How to use these snippets

- Start with the highest-level summaries first.
- Use the most specific snippet for implementation details.
- Keep the thesis assumptions intact unless a task explicitly asks to change them.
- Prefer these snippets as the source of truth for methodology-related coding, drafting, or explanation work.

## Suggested reading order

1. `3.3.9 formal model summary.txt`
2. `3.2.6 graph construction summary.txt`
3. `3.2.2 three-layer graph architecture.txt`
4. `3.2.3 inter-layer connections.txt`
5. `3.2.1 spatial data preparation and network construction.txt`
6. `3.3.1 land-use data extraction and residential-non-residential classification.txt` (includes 3.3.2 data extraction procedure)
7. `3.3.7 passenger demand and trip generation.txt`
8. `3.3.3 passenger spawning and despawning via foot traffic mapping.txt`
9. `3.3.6 jeepney operations and fleet management.txt`
10. `3.3.4 path determination via shortest-path search.txt`
11. `3.2.4 rationale for differentiated wait and transfer weights.txt`
12. `3.2.5 congestion and dynamic weighting.txt`
13. `3.3.5 passenger state machine and travel execution.txt`
14. `3.3.8 synchronization and update sequence.txt`

## Snippet index

| File | Purpose |
| --- | --- |
| `3.3.9 formal model summary.txt` | Defines the simulation as a discrete-event system and names the core entities. |
| `3.2.6 graph construction summary.txt` | Gives the end-to-end recipe for building the travel graph. |
| `3.2.2 three-layer graph architecture.txt` | Explains the start-walk, ride, and end-walk graph layers. |
| `3.2.3 inter-layer connections.txt` | Describes boarding, alighting, transfer, and direct-walk links between layers. |
| `3.2.1 spatial data preparation and network construction.txt` | Covers OSMnx-based street network extraction and simplification. |
| `3.3.1 land-use data extraction and residential-non-residential classification.txt` | Describes how land-use and building data are classified into residential and non-residential zones; also covers 3.3.2 data extraction procedure. |
| `3.3.7 passenger demand and trip generation.txt` | Defines passenger arrivals as a Poisson process. |
| `3.3.3 passenger spawning and despawning via foot traffic mapping.txt` | Explains demand weighting, passenger spawn logic, and stop-based mapping. |
| `3.3.6 jeepney operations and fleet management.txt` | Covers route structure, fleet behavior, capacity, and movement rules. |
| `3.3.4 path determination via shortest-path search.txt` | Shows how shortest-path search produces full passenger itineraries. |
| `3.2.4 rationale for differentiated wait and transfer weights.txt` | Justifies treating waiting and transfer penalties differently. |
| `3.2.5 congestion and dynamic weighting.txt` | Explains congestion effects and time-varying edge weights. |
| `3.3.5 passenger state machine and travel execution.txt` | Describes passenger state transitions during simulation. |
| `3.3.8 synchronization and update sequence.txt` | Defines the per-timestep update order for vehicles and passengers. |
| `references.txt` | Stores supporting citations and source notes for the methodology section. |

## Citation block

Use this as the fast lookup layer before opening `references.txt`.

| Snippet | Citation keys |
| --- | --- |
| `3.2.1 spatial data preparation and network construction.txt` | `sanchez2025`, `boeing2025a`, `boeing2024` |
| `3.2.4 rationale for differentiated wait and transfer weights.txt` | `sanchez2025`, `jara-diaz2022`, `yap2024` |
| `3.3.6 jeepney operations and fleet management.txt` | `mendoza2021`, `inquirer2022`, `ranosa2021`, `changingtransport2021`, `sitchon2023` |
| `3.3.1 land-use data extraction and residential-non-residential classification.txt` | `schiff2024`, `zhangb2024`, `liotta2022`, `gaigne2022`, `boeing2024`, `atwal2024`, `jochem2018` |
| `3.3.7 passenger demand and trip generation.txt` | `zhangj2023`, `christensen2023`, `ross1996stochastic`, `cinlar2013introduction` |
| `3.3.3 passenger spawning and despawning via foot traffic mapping.txt` | `chen2022`, `boeing2021`, `pun2019`, `sobreira2023`, `zhangp2024`, `baddeley2015spatial`, `manser2020`, `gatarin2024`, `singh2023` |
| `3.3.5 passenger state machine and travel execution.txt` | `tozluoglu2024`, `verbas2024`, `bhuiyan2024`, `katsaros2024`, `toprakli2024` |

## Key assumptions to preserve

- The system is modeled around designated stops unless a task explicitly targets informal boarding.
- Path cost is generalized travel time in equivalent in-vehicle minutes.
- The travel graph is the main interface between demand, routing, and simulation.
- Waiting and transfer penalties are not interchangeable.

## Maintenance rule

When a new methodology snippet is added, update this guide with the file name and a one-line purpose so future searches stay fast.
