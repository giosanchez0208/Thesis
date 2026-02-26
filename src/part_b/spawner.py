from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json as _json
import logging
import random
import heapq
import uuid
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.neighbors import KernelDensity
import networkx as nx
import yaml

logger = logging.getLogger("part_b.spawner")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class Passenger:
    pid: str
    origin: int
    destination: int
    spawn_time: float
    ttl_minutes: float
    status: str = "waiting"   # waiting | timed_out | done
    state: str = "WALK"       # WALK -> ... -> DONE
    metadata: dict | None = None

    def to_record(self) -> dict:
        rec = {
            "pid": self.pid,
            "origin": self.origin,
            "destination": self.destination,
            "spawn_time": self.spawn_time,
            "ttl_minutes": self.ttl_minutes,
            "status": self.status,
            "state": self.state,
        }
        if self.metadata:
            rec.update(self.metadata)
        return rec


class PassengerSpawner:
    def __init__(self, config_path: str):
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        self.cfg = cfg

        d = cfg.get("data", {})
        self.nodes_path = Path(d.get("nodes_csv", "data/processed/nodes.csv"))
        self.edges_path = Path(d.get("edges_csv", "data/processed/edges.csv"))
        self.out_folder = Path(d.get("output_iter_folder", "data/route_systems"))
        self.out_folder.mkdir(parents=True, exist_ok=True)

        s = cfg.get("spawner", {}) or {}
        self.base_rate_per_hour = float(s.get("base_demand_rate_per_hour", 100.0))
        self.dt_minutes = float(s.get("dt_minutes", 1.0))
        self.max_wait_minutes = float(s.get("max_wait_minutes", 120.0))
        self.seed = int(s.get("seed", 0))
        random.seed(self.seed); np.random.seed(self.seed)

        self.lambda_per_timestep = s.get("lambda_per_timestep", None)
        if self.lambda_per_timestep is not None:
            self.lambda_per_timestep = float(self.lambda_per_timestep)

        w = cfg.get("weights", {}) or {}
        self.w_kde = float(w.get("w_kde", 0.3))
        self.w_pop = float(w.get("w_pop", 0.5))
        self.w_poi = float(w.get("w_poi", 0.2))
        self.kde_bandwidth_m = float(w.get("kde_bandwidth_m", 500.0))
        self.beta_distance = float(w.get("beta_distance", 0.002))

        samp = cfg.get("sampling", {}) or {}
        self.min_distance_m = float(samp.get("min_distance_m", 200.0))
        self.allow_same_node_dest = bool(samp.get("allow_same_node_destination", False))

        cls = s.get("classification", {}) or {}
        self.residential_origin_prob = float(cls.get("residential_origin_prob", 0.7))
        self.simulate_immediate_arrival = bool(cls.get("simulate_immediate_arrival", False))

        out = cfg.get("output", {}) or {}
        self.save_csv = bool(out.get("save_spawn_events_csv", True))
        self.csv_name = out.get("spawn_events_filename", "iteration_spawns.csv")
        self.save_mode = out.get("save_all_timesteps_as", "append")
        self.save_json = bool(out.get("save_json", False))
        self.json_name = out.get("json_filename", "iteration_spawns.json")
        self.save_gpkg = bool(out.get("save_gpkg", False))
        self.gpkg_name = out.get("gpkg_filename", "iteration_passengers.gpkg")
        self.gpkg_prefix = out.get("gpkg_layers_prefix", "iter")

        # runtime containers
        self.nodes_gdf: gpd.GeoDataFrame | None = None
        self.edges_gdf: gpd.GeoDataFrame | None = None
        self.G: nx.Graph | None = None
        self.passengers: dict[str, Passenger] = {}
        self._res_idx: list[int] = []
        self._nonres_idx: list[int] = []

        # performance helpers
        self._despawn_heap: list[tuple[float, str]] = []   # heap of (expiry_time_seconds, pid)
        self._cached_weights: Optional[pd.Series] = None
        self._weights_last_update: Optional[float] = None
        # how long to keep cached weights (seconds). configurable via YAML under spawner.weights_cache_seconds
        self.weights_cache_seconds = float(self.cfg.get("spawner", {}).get("weights_cache_seconds", 600.0))  # default 10 minutes

    # ---------- IO ----------
    def load_nodes_edges(self) -> None:
        # Nodes
        if self.nodes_path.exists():
            if self.nodes_path.suffix.lower() in (".gpkg", ".shp", ".geojson"):
                self.nodes_gdf = gpd.read_file(self.nodes_path)
            else:
                df = pd.read_csv(self.nodes_path)
                if "geometry" in df.columns:
                    try:
                        self.nodes_gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
                    except Exception:
                        self.nodes_gdf = gpd.GeoDataFrame(df)
                else:
                    self.nodes_gdf = gpd.GeoDataFrame(df)
        else:
            self.nodes_gdf = self._synthetic_nodes()

        # idx -> node_id common case
        if "node_id" not in self.nodes_gdf.columns and "idx" in self.nodes_gdf.columns:
            self.nodes_gdf = self.nodes_gdf.rename(columns={"idx": "node_id"})

        # geometry creation if needed
        if "geometry" not in self.nodes_gdf.columns:
            if {"x", "y"}.issubset(self.nodes_gdf.columns):
                geom = [Point(xy) for xy in zip(self.nodes_gdf["x"].astype(float), self.nodes_gdf["y"].astype(float))]
            elif {"lon", "lat"}.issubset(self.nodes_gdf.columns):
                geom = [Point(xy) for xy in zip(self.nodes_gdf["lon"].astype(float), self.nodes_gdf["lat"].astype(float))]
            else:
                raise ValueError("nodes must contain geometry or x/y or lon/lat")
            self.nodes_gdf = gpd.GeoDataFrame(self.nodes_gdf, geometry=geom)

        if "node_id" not in self.nodes_gdf.columns and "id" in self.nodes_gdf.columns:
            self.nodes_gdf = self.nodes_gdf.rename(columns={"id": "node_id"})
        if "node_id" not in self.nodes_gdf.columns:
            self.nodes_gdf = self.nodes_gdf.reset_index(drop=True)
            self.nodes_gdf["node_id"] = self.nodes_gdf.index.astype(int)

        if self.nodes_gdf.crs is None:
            try:
                self.nodes_gdf.set_crs(epsg=4326, inplace=True)
            except Exception:
                pass

        # Edges (optional)
        if self.edges_path.exists():
            if self.edges_path.suffix.lower() in (".gpkg", ".shp", ".geojson"):
                self.edges_gdf = gpd.read_file(self.edges_path)
            else:
                self.edges_gdf = gpd.GeoDataFrame(pd.read_csv(self.edges_path))
        else:
            self.edges_gdf = None

        self._build_graph()
        self._classify_nodes()
        self._save_config_snapshot()

    def _synthetic_nodes(self) -> gpd.GeoDataFrame:
        # small grid for tests
        center_lon, center_lat = 124.231, 8.229
        pts, ids = [], []
        for i in range(3):
            for j in range(3):
                pts.append(Point(center_lon + 0.001 * (j - 1), center_lat + 0.001 * (i - 1)))
                ids.append(len(ids))
        gdf = gpd.GeoDataFrame({"node_id": ids}, geometry=pts, crs="EPSG:4326")
        gdf["pop_density"] = 0; gdf["poi_count"] = 0
        return gdf

    def _build_graph(self) -> None:
        if self.edges_gdf is None:
            self.G = None; return
        length_col = next((c for c in ("length", "length_m", "weight") if c in self.edges_gdf.columns), None)
        if {"u", "v"}.issubset(self.edges_gdf.columns):
            G = nx.Graph()
            for _, r in self.nodes_gdf.iterrows():
                G.add_node(int(r["node_id"]), geometry=r.geometry)
            for _, r in self.edges_gdf.iterrows():
                try:
                    u, v = int(r["u"]), int(r["v"])
                except Exception:
                    continue
                length = float(r.get(length_col, r.get("length", 1.0))) if length_col else float(r.get("length", 1.0))
                G.add_edge(u, v, length=length)
            self.G = G
        else:
            self.G = None

    # ---------- classification & weights ----------
    def _classify_nodes(self) -> None:
        df = self.nodes_gdf
        col = next((c for c in ("node_type", "landuse", "class", "type", "category", "amenity") if c in df.columns), None)
        if col:
            def m(x):
                if pd.isna(x): return "unknown"
                s = str(x).lower()
                if "resid" in s or "house" in s: return "residential"
                if any(k in s for k in ("commercial", "retail", "industrial", "center", "centre", "market", "downtown")): return "non_residential"
                return "unknown"
            df["node_class"] = df[col].apply(m)
        elif "pop_density" in df.columns or "poi_count" in df.columns:
            pop = df.get("pop_density", pd.Series(0)).fillna(0)
            poi = df.get("poi_count", pd.Series(0)).fillna(0)
            comb = (pop - pop.min()) / (pop.max() - pop.min() + 1e-12) + (poi - poi.min()) / (poi.max() - poi.min() + 1e-12)
            thresh = comb.median()
            df["node_class"] = ["non_residential" if v >= thresh else "residential" for v in comb]
        else:
            # centroid heuristic
            proj = df.to_crs(epsg=3857)
            cx, cy = proj.geometry.x.mean(), proj.geometry.y.mean()
            d = proj.geometry.apply(lambda p: ((p.x - cx) ** 2 + (p.y - cy) ** 2) ** 0.5)
            med = d.median()
            df["node_class"] = ["non_residential" if dv < med else "residential" for dv in d]

        self._res_idx = list(df[df["node_class"] == "residential"].index)
        self._nonres_idx = list(df[df["node_class"] == "non_residential"].index)

    def compute_node_weights(self) -> pd.Series:
        gm = self.nodes_gdf.to_crs(epsg=3857)
        coords = np.vstack([gm.geometry.x.values, gm.geometry.y.values]).T
        kde = KernelDensity(bandwidth=self.kde_bandwidth_m).fit(coords)
        dens = np.exp(kde.score_samples(coords))
        dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
        w = self.w_kde * dens_norm
        if "pop_density" in self.nodes_gdf.columns:
            pop = self.nodes_gdf["pop_density"].fillna(0).to_numpy()
            pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-12)
            w += self.w_pop * pop
        if "poi_count" in self.nodes_gdf.columns:
            poi = self.nodes_gdf["poi_count"].fillna(0).to_numpy()
            poi = (poi - poi.min()) / (poi.max() - poi.min() + 1e-12)
            w += self.w_poi * poi
        if w.sum() <= 0: w += 1.0
        return pd.Series(w, index=self.nodes_gdf.index)

    def get_node_weights(self, sim_time_seconds: float, force: bool = False) -> pd.Series:
        """Return cached node weights unless cache expired or force=True."""
        if (not force) and (self._cached_weights is not None) and (self._weights_last_update is not None):
            if (sim_time_seconds - self._weights_last_update) < self.weights_cache_seconds:
                return self._cached_weights
        # recompute and cache
        self._cached_weights = self.compute_node_weights()
        self._weights_last_update = sim_time_seconds
        return self._cached_weights

    # ---------- sampling ----------
    def _weighted_choice_index(self, weights: pd.Series, k: int = 1):
        p = weights.values.astype(float); p = p / (p.sum() + 1e-12)
        idx = np.arange(len(p))
        chosen = np.random.choice(idx, size=k, replace=(k > len(p)), p=p)
        return self.nodes_gdf.index[chosen]

    def _distance_array_from_origin(self, origin_node_id: int):
        if self.G is not None and origin_node_id in self.G.nodes:
            try:
                lengths = nx.single_source_dijkstra_path_length(self.G, origin_node_id, weight="length")
                arr = [lengths.get(int(r["node_id"]), float("inf")) for _, r in self.nodes_gdf.iterrows()]
                node_ids = list(self.nodes_gdf["node_id"].values)
                return np.array(arr, dtype=float), node_ids
            except Exception:
                pass
        gm = self.nodes_gdf.to_crs(epsg=3857)
        origin_row = self.nodes_gdf[self.nodes_gdf["node_id"] == origin_node_id]
        if origin_row.empty:
            return np.full(len(gm), np.inf), list(self.nodes_gdf["node_id"].values)
        origin_proj = gpd.GeoSeries([origin_row.geometry.values[0]], crs=self.nodes_gdf.crs).to_crs(epsg=3857).iloc[0]
        xs, ys = gm.geometry.x.values, gm.geometry.y.values
        dists = np.sqrt((xs - origin_proj.x) ** 2 + (ys - origin_proj.y) ** 2)
        return dists, list(self.nodes_gdf["node_id"].values)

    def _sample_from_class(self, class_name: str, node_weights: pd.Series):
        idx_list = self._res_idx if class_name == "residential" else self._nonres_idx if class_name == "non_residential" else []
        if not idx_list: return None
        pos_map = {idx: pos for pos, idx in enumerate(self.nodes_gdf.index)}
        pos_list = [pos_map[i] for i in idx_list if i in pos_map]
        if not pos_list: return None
        w = node_weights.reindex(self.nodes_gdf.index).fillna(0).values
        mask = np.zeros_like(w, dtype=bool); mask[pos_list] = True
        w_masked = np.where(mask, w, 0.0)
        if w_masked.sum() <= 0:
            chosen_pos = int(np.random.choice(pos_list)); return self.nodes_gdf.index[chosen_pos]
        probs = w_masked / (w_masked.sum() + 1e-12)
        chosen_pos = np.random.choice(len(probs), p=probs); return self.nodes_gdf.index[chosen_pos]

    def sample_origin_destination_monocentric(self, node_weights: pd.Series):
        if np.random.rand() < self.residential_origin_prob:
            origin_class, dest_class = "residential", "non_residential"
        else:
            origin_class, dest_class = "non_residential", "residential"
        origin_idx = self._sample_from_class(origin_class, node_weights) or self._weighted_choice_index(node_weights, k=1)[0]
        dest_idx = self._sample_from_class(dest_class, node_weights) or self._weighted_choice_index(node_weights, k=1)[0]
        # enforce min distance
        for _ in range(5):
            o_nid = int(self.nodes_gdf.loc[origin_idx]["node_id"]); d_nid = int(self.nodes_gdf.loc[dest_idx]["node_id"])
            dists, node_ids = self._distance_array_from_origin(o_nid)
            try:
                pos = node_ids.index(d_nid); val = float(dists[pos])
            except ValueError:
                val = float("inf")
            if (val >= self.min_distance_m) and (self.allow_same_node_dest or origin_idx != dest_idx):
                break
            dest_idx = self._sample_from_class(dest_class, node_weights) or self._weighted_choice_index(node_weights, k=1)[0]
        return int(origin_idx), int(dest_idx)

    # ---------- spawn / despawn ----------
    def spawn_step(self, sim_time_seconds: float):
        lam = float(self.lambda_per_timestep) if self.lambda_per_timestep is not None else (self.base_rate_per_hour / 60.0) * self.dt_minutes
        n = np.random.poisson(lam)
        if n == 0: return []
        # use cached weights (fast)
        weights = self.get_node_weights(sim_time_seconds)
        out = []
        for _ in range(n):
            try:
                o_idx, d_idx = self.sample_origin_destination_monocentric(weights)
            except Exception:
                o_idx = self._weighted_choice_index(weights, k=1)[0]; d_idx = self._weighted_choice_index(weights, k=1)[0]
            pid = str(uuid.uuid4())
            p = Passenger(pid=str(pid), origin=int(self.nodes_gdf.loc[o_idx]["node_id"]),
                          destination=int(self.nodes_gdf.loc[d_idx]["node_id"]), spawn_time=sim_time_seconds,
                          ttl_minutes=self.max_wait_minutes, metadata={"origin_index": int(o_idx), "dest_index": int(d_idx)})
            if self.simulate_immediate_arrival:
                p.status = "done"; p.state = "DONE"
                # we still add to store in case the caller wants to persist immediate arrivals
            self.passengers[p.pid] = p
            # push expiry time into heap for fast despawn checks
            expiry = sim_time_seconds + p.ttl_minutes * 60.0
            heapq.heappush(self._despawn_heap, (expiry, p.pid))
            out.append(p)
        logger.info("Spawned %d passengers t=%.1f λ=%.3f", len(out), sim_time_seconds, lam)
        return out

    def despawn_step(self, sim_time_seconds: float):
        removed = []
        # pop all expired items from heap
        while self._despawn_heap and self._despawn_heap[0][0] <= sim_time_seconds:
            expiry, pid = heapq.heappop(self._despawn_heap)
            p = self.passengers.pop(pid, None)
            if p is None:
                continue  # already removed (e.g., marked arrived earlier)
            # if passenger was already marked done externally, keep that status but remove
            if p.status == "done" or p.state == "DONE":
                removed.append(p)
            else:
                p.status = "timed_out"; p.state = "DONE"; removed.append(p)
        if removed:
            logger.info("Despawning %d at t=%.1f", len(removed), sim_time_seconds)
        return removed

    # ---------- persistence ----------
    def save_iteration(self, spawned, iter_filename=None, timestep=None):
        if not spawned: return
        recs = [p.to_record() for p in spawned]
        df = pd.DataFrame.from_records(recs)
        # CSV append
        if self.save_csv:
            csv_name = iter_filename or self.csv_name
            out = self.out_folder / csv_name
            header = not out.exists()
            if self.save_mode == "append":
                df.to_csv(out, mode="a", index=False, header=header)
            elif self.save_mode == "per_timestep" and timestep is not None:
                df.to_csv(self.out_folder / f"{Path(csv_name).stem}_t{timestep}{Path(csv_name).suffix}", index=False)
            else:
                df.to_csv(out, index=False)
        # JSON NDJSON
        if self.save_json:
            out_json = self.out_folder / (self.json_name if self.save_mode == "append" else (f"{Path(self.json_name).stem}_t{timestep}{Path(self.json_name).suffix}" if (self.save_mode == "per_timestep" and timestep is not None) else self.json_name))
            meta = {"seed": int(self.seed), "base_rate_per_hour": float(self.base_rate_per_hour), "dt_minutes": float(self.dt_minutes), "timestep": timestep, "timestamp_utc": datetime.utcnow().isoformat()+"Z"}
            if self.save_mode == "append":
                with open(out_json, "a", encoding="utf-8") as fh:
                    for r in recs:
                        r["_experiment"] = meta; fh.write(_json.dumps(r, ensure_ascii=False) + "")
            else:
                with open(out_json, "w", encoding="utf-8") as fh:
                    _json.dump({"experiment": meta, "spawned": recs}, fh, ensure_ascii=False, indent=2)
        # GPKG optional (kept minimal)
        if self.save_gpkg:
            node_geom = dict(zip(self.nodes_gdf["node_id"], self.nodes_gdf.geometry))
            origins, lines = [], []
            for r in recs:
                o, d = r["origin"], r["destination"]
                og, dg = node_geom.get(o), node_geom.get(d)
                if og is None or dg is None: continue
                origins.append({"pid": r["pid"], "node_id": o, "spawn_time": r["spawn_time"], "geometry": og})
                try: lines.append({"pid": r["pid"], "node_origin": o, "node_destination": d, "spawn_time": r["spawn_time"], "geometry": LineString([og, dg])})
                except Exception: pass
            gdf_o = gpd.GeoDataFrame(origins, geometry="geometry", crs=self.nodes_gdf.crs) if origins else gpd.GeoDataFrame(columns=["pid","node_id","spawn_time","geometry"], geometry="geometry")
            gdf_l = gpd.GeoDataFrame(lines, geometry="geometry", crs=self.nodes_gdf.crs) if lines else gpd.GeoDataFrame(columns=["pid","node_origin","node_destination","spawn_time","geometry"], geometry="geometry")
            out_gpkg = self.out_folder / (f"{Path(self.gpkg_name).stem}_t{timestep}.gpkg" if (self.save_mode=="per_timestep" and timestep is not None) else self.gpkg_name)
            try:
                gdf_o.to_file(out_gpkg, layer="origins", driver="GPKG")
                gdf_l.to_file(out_gpkg, layer="od_lines", driver="GPKG")
            except Exception as exc:
                logger.warning("GPKG write failed: %s", exc)

    # ---------- run bounded simulation & event logging ----------
    def append_event_rows(self, recs: List[Dict], out_name: Optional[str] = None):
        """Append event records to CSV in out_folder. Uses pandas for convenience."""
        if not recs:
            return
        csv_name = out_name or "passenger_events.csv"
        out = self.out_folder / csv_name
        df = pd.DataFrame.from_records(recs)
        header = not out.exists()
        df.to_csv(out, mode="a", index=False, header=header)

    def run_simulation(self, sim_hours: Optional[float] = None, n_steps: Optional[int] = None, out_name: str = "passenger_events.csv") -> Path:
        """Run a bounded simulation, write spawn and despawn events to CSV, and return path to CSV.

        Provide either sim_hours (float) OR n_steps (int). If both provided, n_steps takes precedence.
        """
        if n_steps is None and sim_hours is None:
            raise ValueError("Either sim_hours or n_steps must be provided")
        # prepare CSV
        out = self.out_folder / out_name
        if out.exists():
            # don't overwrite; append by default but user may want clean file
            out.unlink()
        header = ["pid","event_type","origin_node_id","dest_node_id","spawn_time","event_time",
                  "ttl_minutes","status","state","origin_index","dest_index"]
        pd.DataFrame(columns=header).to_csv(out, index=False)

        # temporarily disable internal spawn CSV to avoid duplicates
        prev_save_csv = getattr(self, "save_csv", False)
        self.save_csv = False

        dt = self.dt_minutes * 60.0
        t = 0.0
        steps = n_steps if n_steps is not None else int(round((sim_hours * 3600.0) / dt))

        for step in range(steps):
            spawned = self.spawn_step(t)
            removed = self.despawn_step(t)

            rows = []
            for p in spawned:
                rows.append({
                    "pid": p.pid,
                    "event_type": "spawn",
                    "origin_node_id": p.origin,
                    "dest_node_id": p.destination,
                    "spawn_time": p.spawn_time,
                    "event_time": p.spawn_time,
                    "ttl_minutes": p.ttl_minutes,
                    "status": p.status,
                    "state": p.state,
                    "origin_index": p.metadata.get("origin_index") if p.metadata else None,
                    "dest_index": p.metadata.get("dest_index") if p.metadata else None,
                })
            for p in removed:
                etype = "arrive" if p.status == "done" else "timeout"
                rows.append({
                    "pid": p.pid,
                    "event_type": etype,
                    "origin_node_id": p.origin,
                    "dest_node_id": p.destination,
                    "spawn_time": p.spawn_time,
                    "event_time": t,
                    "ttl_minutes": p.ttl_minutes,
                    "status": p.status,
                    "state": p.state,
                    "origin_index": p.metadata.get("origin_index") if p.metadata else None,
                    "dest_index": p.metadata.get("dest_index") if p.metadata else None,
                })
            if rows:
                self.append_event_rows(rows, out_name=out_name)

            t += dt
        # restore save_csv
        self.save_csv = prev_save_csv
        logger.info("Simulation finished: steps=%d, events_file=%s", steps, str(out))
        return out

    # ---------- validation & reproducibility ----------
    def validate_spatial_weighting(self, n_timesteps=100, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        origins = []
        t = 0.0
        for _ in range(n_timesteps):
            spawned = self.spawn_step(t); origins.extend([s.origin for s in spawned]); self.despawn_step(t); t += self.dt_minutes*60.0
        from collections import Counter
        cnt = Counter(origins)
        weights = self.compute_node_weights()
        freq = np.array([cnt.get(int(nid), 0) for nid in self.nodes_gdf["node_id"]])
        w = weights.values
        mask = (w > 0) | (freq > 0)
        try:
            corr = float(np.corrcoef(w[mask], freq[mask])[0,1]) if mask.any() else float("nan")
        except Exception:
            corr = float("nan")
        return {"n_timesteps": n_timesteps, "correlation": corr, "total_spawned": len(origins), "top_origins": dict(Counter(origins).most_common(10))}

    def mark_arrived(self, pid: str):
        p = self.passengers.pop(pid, None)
        if p is not None:
            p.status = "done"; p.state = "DONE"
            return p
        return None

    def _save_config_snapshot(self):
        try:
            meta = {"timestamp_utc": datetime.utcnow().isoformat()+"Z", "seed": int(self.seed), "base_rate_per_hour": float(self.base_rate_per_hour), "dt_minutes": float(self.dt_minutes)}
            out = self.out_folder / f"config_used_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            with open(out, "w", encoding="utf-8") as fh:
                _json.dump({"config": self.cfg, "experiment_metadata": meta}, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Could not save config snapshot: %s", exc)
