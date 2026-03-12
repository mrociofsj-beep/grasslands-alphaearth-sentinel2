"""
Spatial spacing and Moran's I analysis of reference and evaluation points.

This script performs two complementary analyses:

1. Training point spacing analysis
   - Computes nearest-neighbor distance statistics for the full pool
     of reference points used to generate Random Forest training subsets.

2. Spatial autocorrelation of classification errors
   - Computes Moran's I for evaluation errors across multiple
     distance thresholds using a binary distance-based spatial weights matrix.

Main steps:
1. Load CSV files containing training or evaluation points exported from GEE
2. Extract coordinates (from x/y columns or GeoJSON geometry if needed)
3. Compute nearest-neighbor distance statistics
4. Build spatial weights matrices for multiple distance thresholds
5. Compute Moran's I for classification errors
6. Print summary statistics for each dataset

Expected input:

Training CSV:
- x
- y
or
- .geo column containing GeoJSON Point geometry

Evaluation CSV:
- x
- y
- error

Typical use case:
Spatial diagnostics for reference datasets used in remote sensing
classification workflows (e.g., checking point spacing and spatial
autocorrelation of classification errors).
"""
"""
Spatial spacing and Moran's I analysis of reference and evaluation points.

This script performs two complementary analyses:

1. Training-point spacing analysis
   - Computes nearest-neighbour distance statistics for the full pool of
     reference points used to generate Random Forest training subsets.

2. Spatial autocorrelation of classification errors
   - Computes Moran's I for evaluation errors across multiple distance
     thresholds using a binary distance-based spatial weights matrix.

Main steps
----------
1. Load CSV files containing training or evaluation points exported from GEE.
2. Extract coordinates from `x`/`y` columns or from a GeoJSON geometry column.
3. Compute nearest-neighbour distance statistics.
4. Build spatial weights matrices for multiple distance thresholds.
5. Compute Moran's I for classification errors.
6. Print summary statistics for each dataset.

Expected input
--------------
Training CSV:
- `x`
- `y`
or
- `.geo` column containing GeoJSON Point geometry

Evaluation CSV:
- `x`
- `y`
- `error`

Typical use case
----------------
Spatial diagnostics for reference datasets used in remote sensing
classification workflows, including point-spacing evaluation and
spatial autocorrelation analysis of classification errors.
"""

import json
import os

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import DistanceBand
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors

# ============================================================================
# CONFIGURATION
# ============================================================================

THRESHOLDS = [50, 75, 100, 150, 250, 500, 750, 1000]

# Coordinate transformation: WGS84 -> UTM 21S
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)

# Evaluation datasets (must contain columns: x, y, error)
EVAL_FILES = [
    "evaluation_points_model1.csv",
    "evaluation_points_model2.csv",
]

# Training reference dataset
TRAIN_FILES = [
    "training_reference_points.csv"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_file_exists(csv_path):
    """Raise an error if the input file does not exist."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")


def read_csv_safe(csv_path):
    """Read a CSV file after checking that it exists."""
    check_file_exists(csv_path)
    return pd.read_csv(csv_path)


def require_columns(df, required_cols, csv_path):
    """Check that all required columns are present in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def extract_xy_from_geo_column(df, geo_col=".geo"):
    """
    Extract x/y coordinates from a GeoJSON geometry column if x/y
    are not already present.

    Expected geometry format:
    {"geodesic": false, "type": "Point", "coordinates": [lon, lat]}
    """
    if "x" in df.columns and "y" in df.columns:
        return df

    if geo_col not in df.columns:
        raise ValueError(
            f"Columns 'x' and 'y' were not found, and '{geo_col}' is also missing.\n"
            f"Available columns: {list(df.columns)}"
        )

    def parse_geo(value):
        if pd.isna(value):
            return np.nan, np.nan
        try:
            obj = json.loads(value)
            coords = obj.get("coordinates", [np.nan, np.nan])
            if len(coords) >= 2:
                return coords[0], coords[1]
            return np.nan, np.nan
        except Exception:
            return np.nan, np.nan

    xy = df[geo_col].apply(parse_geo)
    df = df.copy()
    df["x"] = xy.apply(lambda t: t[0])
    df["y"] = xy.apply(lambda t: t[1])
    return df


def lonlat_to_utm(df):
    """Convert longitude/latitude coordinates to UTM 21S."""
    lon = df["x"].to_numpy()
    lat = df["y"].to_numpy()
    x_utm, y_utm = TRANSFORMER.transform(lon, lat)
    return np.column_stack((x_utm, y_utm))


def nearest_neighbor_distances(coords):
    """Compute nearest-neighbour distances for a set of projected coordinates."""
    if len(coords) < 2:
        raise ValueError("At least 2 points are required to compute nearest-neighbour distances.")

    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return distances[:, 1]


def significance_label(p_value):
    """Return a significance label based on the simulated p-value."""
    if p_value < 0.01:
        return "*** (p < 0.01)"
    if p_value < 0.05:
        return "** (p < 0.05)"
    if p_value < 0.10:
        return "* (p < 0.10)"
    return "ns"


def strength_label(moran_i):
    """Return a qualitative label for Moran's I magnitude."""
    abs_i = abs(moran_i)
    if abs_i < 0.1:
        return "negligible"
    if abs_i < 0.3:
        return "weak"
    if abs_i < 0.5:
        return "moderate"
    return "strong"


# ============================================================================
# TRAINING-POINT SPACING ANALYSIS
# ============================================================================

def load_training_points(csv_path):
    """
    Load training points, extract coordinates, remove duplicate locations,
    and return the cleaned DataFrame, projected coordinates, and number
    of duplicates removed.
    """
    df = read_csv_safe(csv_path)

    # Use x/y directly when available; otherwise extract them from .geo
    df = extract_xy_from_geo_column(df, geo_col=".geo")
    require_columns(df, ["x", "y"], csv_path)

    df = df.dropna(subset=["x", "y"]).copy()

    n_before = len(df)
    df = df.drop_duplicates(subset=["x", "y"], keep="first").copy()
    n_after = len(df)
    n_duplicates = n_before - n_after

    coords_utm = lonlat_to_utm(df)
    return df, coords_utm, n_duplicates


def analyze_training_spacing(csv_path, nominal_spacing=50):
    """Print nearest-neighbour distance statistics for a training dataset."""
    print("\n" + "=" * 70)
    print("TRAINING-POINT SPACING ANALYSIS")
    print(f"File: {csv_path}")
    print("=" * 70)

    df, coords_utm, n_duplicates = load_training_points(csv_path)

    if n_duplicates > 0:
        print(f"Duplicate point locations removed: {n_duplicates}")

    nearest = nearest_neighbor_distances(coords_utm)

    print(f"\nPoints analysed: {len(df)}")
    print(f"Nominal spacing: {nominal_spacing} m")

    print("\n--- ORIGINAL COORDINATES ---")
    print(df[["x", "y"]].head())

    print("\n--- NEAREST-NEIGHBOUR DISTANCE SUMMARY (m) ---")
    print(f"Minimum: {nearest.min():.2f}")
    print(f"Mean: {nearest.mean():.2f}")
    print(f"Percentile 5: {np.percentile(nearest, 5):.2f}")
    print(f"Percentile 25: {np.percentile(nearest, 25):.2f}")
    print(f"Median: {np.percentile(nearest, 50):.2f}")
    print(f"Percentile 75: {np.percentile(nearest, 75):.2f}")
    print(f"Percentile 95: {np.percentile(nearest, 95):.2f}")
    print(f"Maximum: {nearest.max():.2f}")

    n_lt_25 = np.sum(nearest < 25)
    n_lt_50 = np.sum(nearest < 50)
    n_lt_75 = np.sum(nearest < 75)
    n_lt_100 = np.sum(nearest < 100)

    print("\n--- THRESHOLD COUNTS ---")
    print(f"Points with nearest neighbour < 25 m: {n_lt_25} ({100 * n_lt_25 / len(df):.1f}%)")
    print(f"Points with nearest neighbour < 50 m: {n_lt_50} ({100 * n_lt_50 / len(df):.1f}%)")
    print(f"Points with nearest neighbour < 75 m: {n_lt_75} ({100 * n_lt_75 / len(df):.1f}%)")
    print(f"Points with nearest neighbour < 100 m: {n_lt_100} ({100 * n_lt_100 / len(df):.1f}%)")

    if "class_id" in df.columns:
        print("\n--- CLASS DISTRIBUTION ---")
        print(df["class_id"].value_counts().sort_index())


# ============================================================================
# MORAN'S I ANALYSIS OF EVALUATION ERRORS
# ============================================================================

def load_evaluation_points(csv_path):
    """
    Load evaluation points, remove duplicate locations, and return the
    cleaned DataFrame, projected coordinates, error values, and number
    of duplicates removed.
    """
    df = read_csv_safe(csv_path)
    require_columns(df, ["x", "y", "error"], csv_path)

    df = df.dropna(subset=["x", "y", "error"]).copy()

    n_before = len(df)
    df = df.drop_duplicates(subset=["x", "y"], keep="first").copy()
    n_after = len(df)
    n_duplicates = n_before - n_after

    error = df["error"].to_numpy()
    coords_utm = lonlat_to_utm(df)

    return df, coords_utm, error, n_duplicates


def check_evaluation_coordinates(csv_path):
    """Print coordinate diagnostics for an evaluation dataset."""
    print("\n" + "#" * 35)
    print("EVALUATION COORDINATE CHECK")
    print(f"File: {csv_path}")
    print("#" * 35)

    df, coords_utm, error, n_duplicates = load_evaluation_points(csv_path)

    if n_duplicates > 0:
        print(f"Duplicate point locations removed: {n_duplicates}")

    print("\nOriginal coordinates (lon/lat), first rows:")
    print(df[["x", "y"]].head())

    print("\nOriginal X range (lon):", df["x"].min(), "to", df["x"].max())
    print("Original Y range (lat):", df["y"].min(), "to", df["y"].max())

    preview = pd.DataFrame(coords_utm[:5], columns=["x_utm", "y_utm"])
    print("\nProjected coordinates (UTM 21S), first rows:")
    print(preview)

    print("\nProjected X range:", coords_utm[:, 0].min(), "to", coords_utm[:, 0].max())
    print("Projected Y range:", coords_utm[:, 1].min(), "to", coords_utm[:, 1].max())

    print("\nMean error:", error.mean())


def summarize_evaluation_distances(csv_path):
    """Print nearest-neighbour distance statistics for evaluation points."""
    print("\n" + "-" * 40)
    print("EVALUATION NEAREST-NEIGHBOUR DISTANCE SUMMARY")
    print(f"File: {csv_path}")
    print("-" * 40)

    df, coords_utm, _, _ = load_evaluation_points(csv_path)
    nearest = nearest_neighbor_distances(coords_utm)

    print(f"Points analysed: {len(df)}")
    print(f"Minimum distance (m): {nearest.min():.2f}")
    print(f"Mean distance (m): {nearest.mean():.2f}")
    print(f"Percentile 5 (m): {np.percentile(nearest, 5):.2f}")
    print(f"Percentile 25 (m): {np.percentile(nearest, 25):.2f}")
    print(f"Percentile 50 (m): {np.percentile(nearest, 50):.2f}")
    print(f"Percentile 75 (m): {np.percentile(nearest, 75):.2f}")
    print(f"Percentile 95 (m): {np.percentile(nearest, 95):.2f}")


def run_moran_analysis(csv_path):
    """Compute and print Moran's I for evaluation errors across distance thresholds."""
    print("\n" + "=" * 65)
    print("MORAN'S I ANALYSIS OF CLASSIFICATION ERRORS")
    print(f"File: {csv_path}")
    print("=" * 65)

    df, coords_utm, error, n_duplicates = load_evaluation_points(csv_path)

    print(f"Valid points: {len(df)}")
    if n_duplicates > 0:
        print(f"Duplicate point locations removed: {n_duplicates}")
    print(f"Mean error: {error.mean():.6f}")
    print(f"Error SD: {error.std():.6f}")

    print("\n----- Moran's I by distance threshold -----")

    for threshold in THRESHOLDS:
        try:
            weights = DistanceBand(
                coords_utm,
                threshold=threshold,
                binary=True,
                silence_warnings=True,
            )
            weights.transform = "r"

            neighbor_counts = np.array([len(v) for v in weights.neighbors.values()])
            mean_neighbors = neighbor_counts.mean()
            min_neighbors = neighbor_counts.min()
            max_neighbors = neighbor_counts.max()
            islands = np.sum(neighbor_counts == 0)

            print(f"\nDistance threshold: {threshold} m")

            if islands == len(df):
                print("  Moran's I could not be computed: no points have neighbours at this threshold.")
                continue

            if mean_neighbors < 1:
                print(f"  Warning: very low mean number of neighbours ({mean_neighbors:.2f})")
                print("  Interpretation at this distance should be treated with caution.")

            moran = Moran(error, weights)

            direction = "positive" if moran.I > 0 else "negative"
            strength = strength_label(moran.I)
            significance = significance_label(moran.p_sim)

            print(f"  Mean neighbours: {mean_neighbors:.2f}")
            print(f"  Neighbour range: {int(min_neighbors)} - {int(max_neighbors)}")
            print(f"  Points without neighbours: {int(islands)} ({100 * islands / len(df):.1f}%)")
            print(f"  Moran's I: {moran.I:.4f} ({direction}, {strength})")
            print(f"  p-value: {moran.p_sim:.4f} {significance}")
            print(f"  z-score: {moran.z_sim:.2f}")

        except Exception as exc:
            print(f"  Error at {threshold} m: {exc}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run training-point spacing and Moran's I analyses."""
    print("\n" + "=" * 80)
    print("A) TRAINING-POINT SPACING ANALYSIS")
    print("=" * 80)

    for csv_file in TRAIN_FILES:
        try:
            analyze_training_spacing(csv_file, nominal_spacing=50)
        except Exception as exc:
            print(f"\n[ERROR] Could not analyse training file: {csv_file}\n{exc}\n")

    print("\n" + "=" * 80)
    print("B) MORAN'S I ANALYSIS OF EVALUATION ERRORS")
    print("=" * 80)

    for csv_file in EVAL_FILES:
        try:
            check_evaluation_coordinates(csv_file)
            summarize_evaluation_distances(csv_file)
            run_moran_analysis(csv_file)
        except Exception as exc:
            print(f"\n[ERROR] Could not analyse evaluation file: {csv_file}\n{exc}\n")

        print("\n" + "=" * 65)


if __name__ == "__main__":
    main()

