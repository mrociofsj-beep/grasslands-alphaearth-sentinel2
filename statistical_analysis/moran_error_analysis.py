"""
Moran's I analysis of classification errors

This script:
1. Loads one or more CSV files containing evaluation points exported from GEE
2. Computes nearest-neighbor distance statistics for the point coordinates
3. Builds a binary distance-based spatial weights matrix
4. Computes Moran's I for the binary classification error variable
5. Prints a concise summary for each input file

Expected input:
- CSV files containing at least:
    - x
    - y
    - error

Typical use case:
- Spatial autocorrelation analysis of classification errors
  extracted from external evaluation points
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import DistanceBand
from sklearn.neighbors import NearestNeighbors


# ======================================================================
# Configuration
# ======================================================================
BASE_DIR = Path("path/to/your/project")

INPUT_FILES = [
    "Eval_points_Moran_AE_DEM_PHENO_2023.csv",
    "Eval_points_Moran_S2_DEM_PHENO_2023.csv",
]

DISTANCE_THRESHOLD = 200.0  # meters
K_NEIGHBORS = 2             # self + nearest neighbor


# ======================================================================
# Helper functions
# ======================================================================
def load_point_table(csv_path: Path) -> pd.DataFrame:
    """Load a Moran input table and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"x", "y", "error"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing_columns}"
        )

    return df


def summarize_nearest_neighbor_distances(coords: np.ndarray) -> dict[str, float]:
    """
    Compute nearest-neighbor distance summary statistics.

    Uses n_neighbors=2 because the first neighbor is each point itself
    at distance 0, and the second is the true nearest neighbor.
    """
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS)
    nbrs.fit(coords)

    distances, _ = nbrs.kneighbors(coords)
    nearest = distances[:, 1]

    return {
        "min_distance": float(nearest.min()),
        "mean_distance": float(nearest.mean()),
        "p05_distance": float(np.percentile(nearest, 5)),
        "p50_distance": float(np.percentile(nearest, 50)),
        "p95_distance": float(np.percentile(nearest, 95)),
    }


def run_moran_analysis(csv_path: Path, threshold: float) -> None:
    """Run Moran's I analysis for a single CSV file."""
    df = load_point_table(csv_path)

    coords = df[["x", "y"]].to_numpy()
    error = df["error"].to_numpy()

    nn_summary = summarize_nearest_neighbor_distances(coords)

    print("\n===================================")
    print(f"File: {csv_path.name}")
    print("===================================")
    print(f"Number of points: {len(df)}")
    print(f"Mean error rate: {error.mean():.6f}")
    print(f"Nearest-neighbor minimum distance: {nn_summary['min_distance']:.6f}")
    print(f"Nearest-neighbor mean distance: {nn_summary['mean_distance']:.6f}")
    print(f"Nearest-neighbor 5th percentile: {nn_summary['p05_distance']:.6f}")
    print(f"Nearest-neighbor 50th percentile: {nn_summary['p50_distance']:.6f}")
    print(f"Nearest-neighbor 95th percentile: {nn_summary['p95_distance']:.6f}")
    print(f"Distance threshold used: {threshold:.2f}")

    weights = DistanceBand(coords, threshold=threshold, binary=True)
    weights.transform = "r"

    moran = Moran(error, weights)

    print(f"Moran's I: {moran.I:.6f}")
    print(f"Permutation p-value: {moran.p_sim:.6f}")
    print(f"Permutation z-score: {moran.z_sim:.6f}")


# ======================================================================
# Main
# ======================================================================
for filename in INPUT_FILES:
    run_moran_analysis(BASE_DIR / filename, threshold=DISTANCE_THRESHOLD)