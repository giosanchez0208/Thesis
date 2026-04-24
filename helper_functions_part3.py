# Helper functions for part 3 batch generation and live table updates
import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path

def route_key(route) -> tuple[int, ...]:
    """Create a unique signature for a route based on its node path."""
    return tuple(int(node_id) for node_id in route.path_node_ids)

def load_existing_library(csv_path: Path) -> pd.DataFrame:
    """Load the route library CSV, or return empty dataframe if not found."""
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Parse JSON columns
            for col in ['route_node_ids', 'anchor_node_ids', 'ordered_anchor_node_ids', 'path_latlon', 'anchor_latlon']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            return df
        except Exception as e:
            print(f"Warning: Could not load CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def append_records_to_csv(records: list[dict], csv_path: Path) -> None:
    """Append route records to CSV, creating file if it doesn't exist."""
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert JSON fields to strings for CSV storage
    rows_to_write = []
    for record in records:
        row = record.copy()
        for json_col in ['route_node_ids', 'anchor_node_ids', 'ordered_anchor_node_ids', 'path_latlon', 'anchor_latlon']:
            if json_col in row and isinstance(row[json_col], (list, dict)):
                row[json_col] = json.dumps(row[json_col], ensure_ascii=False)
        rows_to_write.append(row)

    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()
    fieldnames = list(rows_to_write[0].keys()) if rows_to_write else []

    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows_to_write)

def flush_pending_records(library_records: list[dict], pending: list[dict]) -> list[dict]:
    """Add pending records to the in-memory library and return updated list."""
    # Convert JSON strings back to Python objects for the in-memory cache
    for record in pending:
        for json_col in ['route_node_ids', 'anchor_node_ids', 'ordered_anchor_node_ids', 'path_latlon', 'anchor_latlon']:
            if json_col in record and isinstance(record[json_col], str):
                try:
                    record[json_col] = json.loads(record[json_col])
                except:
                    pass
    library_records.extend(pending)
    return library_records

def live_top_percentile_summary(library_records: list[dict], percentile: float = 95.0) -> pd.DataFrame:
    """Create a summary table of top percentile routes from in-memory records."""
    if not library_records:
        return pd.DataFrame({
            'routes_logged': [0],
            'p95_threshold': [np.nan],
            'routes_at_or_above_p95': [0],
            'top_p95_mean_score': [np.nan],
            'top_p95_std_score': [np.nan],
            'best_route_id': ['—'],
            'best_score': [np.nan],
        })

    df = pd.DataFrame(library_records)
    if 'screening_risk_adjusted_score' not in df.columns:
        return pd.DataFrame()

    scores = df['screening_risk_adjusted_score'].to_numpy(dtype=float)
    threshold = float(np.percentile(scores, percentile))
    top_mask = scores >= threshold
    top_routes = df[top_mask]

    summary_df = pd.DataFrame({
        'routes_logged': [len(df)],
        'p95_threshold': [threshold],
        'routes_at_or_above_p95': [int(top_mask.sum())],
        'top_p95_mean_score': [float(top_routes['screening_risk_adjusted_score'].mean()) if len(top_routes) > 0 else np.nan],
        'top_p95_std_score': [float(top_routes['screening_risk_adjusted_score'].std()) if len(top_routes) > 1 else np.nan],
        'best_route_id': [df.iloc[0]['route_id'] if len(df) > 0 else '—'],
        'best_score': [df.iloc[0]['screening_risk_adjusted_score'] if len(df) > 0 else np.nan],
    })
    return summary_df

def write_library_snapshot(library_df: pd.DataFrame, session_stats: dict) -> pd.DataFrame:
    """Save library state to YAML manifest and return the updated dataframe."""
    global OUTPUT_YAML, GOOD_PERCENTILE

    if OUTPUT_YAML is None:
        return library_df

    # Summarize library statistics
    scores = library_df.get('screening_risk_adjusted_score', pd.Series()).to_numpy(dtype=float)
    if len(scores) > 0:
        threshold = float(np.percentile(scores, GOOD_PERCENTILE))
        top_mask = scores >= threshold
        top_routes = library_df[top_mask]
    else:
        threshold = None
        top_routes = library_df

    manifest = {
        'experiment': 'B4_good_route_library',
        'higher_is_better': True,
        'run_seed': RUN_SEED,
        'risk_aversion_k': RISK_AVERSION_K,
        'robustness_cv_ceiling': ROBUSTNESS_CV_CEILING,
        'summary': {
            'route_count': int(len(library_df)),
            'mean_validation_average_gtc': None,
            'mean_validation_std_gtc': None,
            'mean_validation_risk_adjusted_score': None,
            'std_validation_risk_adjusted_score': None,
            'best_route_id': str(library_df.iloc[0]['route_id']) if len(library_df) > 0 else None,
            'best_validation_risk_adjusted_score': None,
            'best_validation_average_gtc': None,
            'best_validation_std_gtc': None,
        },
        'session': session_stats,
    }

    import yaml
    OUTPUT_YAML.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False), encoding='utf-8')
    return library_df
