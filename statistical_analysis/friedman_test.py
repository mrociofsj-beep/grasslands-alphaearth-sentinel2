"""
Friedman test and Kendall's W for comparing classification setups

This script:
1. Loads a CSV file containing replicate-level performance metrics
2. Reshapes the data into a wide table (replicates x setups)
3. Runs the Friedman test for each selected metric
4. Computes Kendall's W as an effect size measure
5. Exports overview, summary, and wide-format tables

Expected input:
- A CSV file with at least:
    - rep
    - setup
    - one or more metric columns (e.g., OA, Kappa, MacroF1)

Typical use case:
- Comparing multiple Random Forest predictor setups across repeated runs
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import friedmanchisquare


# ======================================================================
# Configuration
# ======================================================================
BASE_DIR = Path("path/to/your/project")
INPUT_CSV_NAME = "RF_replicas_all_setups_2023.csv"
INPUT_CSV = BASE_DIR / INPUT_CSV_NAME

OUT_PREFIX = BASE_DIR / "friedman_results"

METRICS: List[str] = ["OA", "Kappa", "MacroF1"]
ALPHA = 0.05


# ======================================================================
# Robust reader for Google Earth Engine CSV exports with .geo
# ======================================================================
def load_gee_export_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV exported from Google Earth Engine, handling cases where a
    trailing .geo column contains JSON with internal commas.

    Strategy:
    - Read the header normally
    - Assume .geo, if present, is the last column
    - For each row, remove the .geo field before splitting on commas
    - Return a DataFrame without the .geo column
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found:\n{filepath}")

    rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
        columns = header.split(",")

        if len(columns) < 3:
            raise ValueError("Header is too short. The CSV does not appear to be valid.")

        has_geo = columns[-1] == ".geo"
        columns_no_geo = columns[:-1] if has_geo else columns

        for line in f:
            line = line.rstrip("\n")

            if line.startswith('"'):
                line = line[1:]

            if has_geo:
                cut = line.rfind(',""{')
                if cut == -1:
                    cut = line.rfind(',"{')
                left = line[:cut] if cut != -1 else line
            else:
                left = line

            parts = left.split(",")

            if len(parts) != len(columns_no_geo):
                continue

            rows.append(parts)

    return pd.DataFrame(rows, columns=columns_no_geo)


# ======================================================================
# Load data
# ======================================================================
df = load_gee_export_csv(INPUT_CSV)

df.columns = [col.strip() for col in df.columns]

required_columns = {"rep", "setup"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

df["setup"] = df["setup"].astype(str).str.strip()
df = df[
    df["setup"].notna()
    & (df["setup"] != "")
    & (df["setup"].str.lower() != "setup")
].copy()

df["rep"] = (
    df["rep"]
    .astype(str)
    .str.strip()
    .str.replace(",", ".", regex=False)
)
df["rep"] = pd.to_numeric(df["rep"], errors="coerce")

n_bad_rep = df["rep"].isna().sum()
if n_bad_rep > 0:
    print(f"[WARN] 'rep': {n_bad_rep} rows with empty/non-numeric replicate IDs were removed.")
    df = df.dropna(subset=["rep"]).copy()

df["rep"] = df["rep"].astype(int)

print("\n[DEBUG] Setup counts:")
print(df["setup"].value_counts())

if df["setup"].nunique() < 2:
    raise ValueError(
        "At least two setups are required to run the Friedman test.\n"
        "If this fails, the CSV likely does not contain multiple setups."
    )


# ======================================================================
# Friedman test + Kendall's W
# ======================================================================
results = []

for metric in METRICS:
    if metric not in df.columns:
        print(f"[WARN] Metric '{metric}' not found. Skipping.")
        continue

    df_metric = df.copy()
    df_metric[metric] = (
        df_metric[metric]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df_metric[metric] = pd.to_numeric(df_metric[metric], errors="coerce")

    wide = df_metric.pivot(index="rep", columns="setup", values=metric)

    n_before = wide.shape[0]
    wide = wide.dropna(axis=0, how="any")
    n_after = wide.shape[0]

    if n_after == 0:
        print(f"[WARN] '{metric}': no complete replicate blocks remained. Skipping.")
        continue

    if n_after < n_before:
        print(f"[WARN] '{metric}': {n_before - n_after} incomplete replicate blocks were removed.")

    if wide.shape[1] < 2:
        print(f"[WARN] '{metric}': fewer than 2 setups with valid data. Skipping.")
        continue

    arrays = [wide[col].to_numpy() for col in wide.columns]
    chi2, p_value = friedmanchisquare(*arrays)

    n_reps = wide.shape[0]
    n_setups = wide.shape[1]
    kendall_w = chi2 / (n_reps * (n_setups - 1))

    summary = pd.DataFrame({
        "mean": wide.mean(),
        "sd": wide.std(ddof=1),
        "median": wide.median(),
        "q25": wide.quantile(0.25),
        "q75": wide.quantile(0.75),
        "n_reps": n_reps,
    }).sort_values("mean", ascending=False)

    summary.to_csv(f"{OUT_PREFIX}_{metric}_summary.csv")
    wide.to_csv(f"{OUT_PREFIX}_{metric}_wide.csv")

    results.append({
        "metric": metric,
        "friedman_chi2": chi2,
        "friedman_p": p_value,
        "kendall_w": kendall_w,
        "n_reps_used": n_reps,
        "n_setups": n_setups,
        "significant_alpha_0.05": p_value < ALPHA,
    })

    print(f"\n=== {metric} ===")
    print(f"Friedman chi-square = {chi2:.4f}")
    print(f"p-value             = {p_value:.4g}")
    print(f"Kendall's W         = {kendall_w:.3f}")
    print(f"Replicates used     = {n_reps}")
    print(f"Setups compared     = {n_setups}")
    print(f"Significant (alpha = 0.05): {'YES' if p_value < ALPHA else 'NO'}")

if not results:
    raise RuntimeError("No valid metrics were processed. No output files were generated.")

overview = pd.DataFrame(results).sort_values("friedman_p")
overview.to_csv(f"{OUT_PREFIX}_overview.csv", index=False)

print("\n==============================")
print("ANALYSIS COMPLETED")
print(f"- Input CSV: {INPUT_CSV}")
print(f"- Overview: {OUT_PREFIX}_overview.csv")
print(f"- Metric summaries: {OUT_PREFIX}_<METRIC>_summary.csv")
print(f"- Wide tables: {OUT_PREFIX}_<METRIC>_wide.csv")
print("==============================")