"""
Wilcoxon signed-rank post hoc tests with Holm correction

This script:
1. Loads wide-format metric tables produced after the Friedman test
2. Performs pairwise Wilcoxon signed-rank tests between setups
3. Applies Holm correction for multiple comparisons
4. Exports one CSV file per metric and one combined CSV file

Expected input:
- Wide-format CSV files with:
    - rows = replicate blocks
    - columns = predictor setups

Typical use case:
- Post hoc pairwise comparisons after a significant Friedman test
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ======================================================================
# Configuration
# ======================================================================
BASE_DIR = Path("path/to/your/project")

WIDE_FILES = {
    "OA": "friedman_results_OA_wide.csv",
    "Kappa": "friedman_results_Kappa_wide.csv",
    "MacroF1": "friedman_results_MacroF1_wide.csv",
}

OUT_PREFIX = BASE_DIR / "wilcoxon_holm"
ALPHA = 0.05


# ======================================================================
# Run pairwise Wilcoxon tests for each metric
# ======================================================================
all_results = []

for metric, wide_filename in WIDE_FILES.items():
    wide_path = BASE_DIR / wide_filename

    if not wide_path.exists():
        raise FileNotFoundError(
            f"Wide-format input file for {metric} was not found:\n{wide_path}"
        )

    wide = pd.read_csv(wide_path, index_col=0)
    wide = wide.dropna(axis=0, how="any")

    if wide.shape[0] < 2:
        print(f"[WARN] '{metric}': too few complete replicate blocks ({wide.shape[0]}). Skipping.")
        continue

    setup_names = list(wide.columns)
    if len(setup_names) < 2:
        print(f"[WARN] '{metric}': fewer than 2 setups. Skipping.")
        continue

    pairs = list(combinations(setup_names, 2))

    raw_p_values = []
    test_statistics = []
    valid_pairs = []

    for setup_a, setup_b in pairs:
        try:
            result = wilcoxon(
                wide[setup_a],
                wide[setup_b],
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
            )
            raw_p_values.append(result.pvalue)
            test_statistics.append(result.statistic)
            valid_pairs.append((setup_a, setup_b))

        except ValueError as exc:
            print(
                f"[WARN] '{metric}': Wilcoxon test failed for "
                f"{setup_a} vs {setup_b}: {exc}. Pair skipped."
            )

    if not raw_p_values:
        print(f"[WARN] '{metric}': no valid setup pairs remained.")
        continue

    reject, p_adjusted, _, _ = multipletests(
        raw_p_values,
        alpha=ALPHA,
        method="holm",
    )

    posthoc = pd.DataFrame({
        "metric": metric,
        "setup_A": [a for a, b in valid_pairs],
        "setup_B": [b for a, b in valid_pairs],
        "wilcoxon_W": test_statistics,
        "p_raw": raw_p_values,
        "p_holm": p_adjusted,
        "significant_holm": reject,
    }).sort_values("p_holm")

    output_csv = f"{OUT_PREFIX}_{metric}.csv"
    posthoc.to_csv(output_csv, index=False)

    print(f"\n=== {metric} ===")
    print(f"Saved: {output_csv}")
    print(posthoc.head(10))

    all_results.append(posthoc)


# ======================================================================
# Export combined results
# ======================================================================
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_output = BASE_DIR / "wilcoxon_holm_all_metrics.csv"
    combined_results.to_csv(combined_output, index=False)
    print(f"\n[OK] Combined results saved: {combined_output}")

print("\nWILCOXON + HOLM POST HOC ANALYSIS COMPLETED")