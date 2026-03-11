"""
Permutation-based variable importance (MDA) for global accuracy and class-specific F1

This script:
1. Loads setup-specific TRAINPOOL and EVAL tables exported from Google Earth Engine
2. Recreates replicate-specific stratified training subsets
3. Trains one Random Forest model per replicate
4. Computes permutation importance using:
   - global accuracy
   - class-specific F1 for grassland
   - class-specific F1 for rocky grassland
5. Exports replicate-level metrics and variable-importance summaries

Outputs:
- rep_metrics_<SETUP>_<YEAR>.csv
- MDA_global_acc_summary_<SETUP>_<YEAR>.csv
- MDA_F1_grassland_summary_<SETUP>_<YEAR>.csv
- MDA_F1_rocky_grassland_summary_<SETUP>_<YEAR>.csv
- replicate parquet files for each importance type
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


# ======================================================================
# Configuration
# ======================================================================
YEAR = 2023
SETUPS = ["AE_DEM_PHENO", "S2_DEM_PHENO"]

N_REPLICATES = 100
BASE_SEED = 2025

TRAIN_FRACTION = 0.70
MIN_TRAIN_PER_CLASS = 150
MAX_TRAIN_PER_CLASS = 4000

PERMUTATION_REPEATS = 10
NODATA = -9999

ROCKY_GRASSLAND_ID = 1
GRASSLAND_ID = 2

OUTPUT_DIR = Path("MDA_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================================================================
# Random Forest hyperparameters by setup
# ======================================================================
RF_PARAMS_BY_SETUP = {
    "S2_DEM_PHENO": dict(
        n_estimators=500,
        max_features=8,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=0.60,
        n_jobs=-1,
    ),
    "AE_DEM_PHENO": dict(
        n_estimators=500,
        max_features=9,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=0.60,
        n_jobs=-1,
    ),
}


# ======================================================================
# Helper functions
# ======================================================================
def read_csv_clean(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read a CSV exported from GEE, remove geometry if present, convert all
    predictors to numeric, replace NODATA with NaN, and drop incomplete rows.
    """
    df = pd.read_csv(path)

    if "class_id" not in df.columns:
        raise ValueError(f"Missing 'class_id' column in: {path}")

    df = df.drop(columns=[".geo"], errors="ignore")

    y = pd.to_numeric(df["class_id"], errors="coerce").astype("Int64")

    X = df.drop(columns=["class_id"], errors="ignore").copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace(NODATA, np.nan)
    X = X.dropna(axis=1, how="all")

    out = pd.concat([X, y.rename("class_id")], axis=1).dropna()
    out["class_id"] = out["class_id"].astype(int)

    feature_columns = [col for col in out.columns if col != "class_id"]
    return out, feature_columns


def stratified_subset(df_pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Build a replicate-specific stratified training subset.

    Logic:
    - take TRAIN_FRACTION per class
    - if the subset is smaller than the minimum, use the full class
    - apply the final cap per class
    """
    rng = np.random.default_rng(seed)
    parts = []

    for class_id, group in df_pool.groupby("class_id"):
        n_total = len(group)
        n_take = int(np.floor(n_total * TRAIN_FRACTION))

        if n_take < MIN_TRAIN_PER_CLASS:
            n_take = n_total

        n_take = min(n_take, MAX_TRAIN_PER_CLASS, n_total)

        selected_idx = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        parts.append(df_pool.loc[selected_idx])

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed)
    return out


def make_class_f1_scorer(class_id: int) -> Callable:
    """Return a scorer for one-vs-rest F1 for a given class."""
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        y_true_bin = (y == class_id).astype(int)
        y_pred_bin = (y_pred == class_id).astype(int)
        return f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return scorer


def save_summary_and_replicates(
    feature_names: list[str],
    replicate_importances: list[np.ndarray],
    tag: str,
    setup_name: str,
    year: int,
) -> None:
    """
    Save:
    1. A summary CSV with mean and SD per feature
    2. A parquet file with replicate-level importance values
    """
    importance_array = np.vstack(replicate_importances)

    mean_values = importance_array.mean(axis=0)
    sd_values = importance_array.std(axis=0, ddof=1)

    summary_df = pd.DataFrame({
        "feature": feature_names,
        f"{tag}_mean": mean_values,
        f"{tag}_sd": sd_values,
    }).sort_values(by=f"{tag}_mean", ascending=False).reset_index(drop=True)

    summary_df.to_csv(
        OUTPUT_DIR / f"{tag}_summary_{setup_name}_{year}.csv",
        index=False,
    )

    replicate_df = pd.DataFrame(importance_array, columns=feature_names)
    replicate_df.insert(0, "rep", np.arange(1, importance_array.shape[0] + 1))

    replicate_df.to_parquet(
        OUTPUT_DIR / f"rep_{tag}_{setup_name}_{year}.parquet",
        index=False,
    )


def run_setup(setup_name: str) -> None:
    if setup_name not in RF_PARAMS_BY_SETUP:
        raise KeyError(f"No Random Forest hyperparameters defined for {setup_name}")

    train_path = Path(f"TRAINPOOL_{setup_name}_{YEAR}.csv")
    eval_path = Path(f"EVAL_{setup_name}_{YEAR}.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Could not find: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Could not find: {eval_path}")

    df_pool, feature_pool = read_csv_clean(train_path)
    df_eval, feature_eval = read_csv_clean(eval_path)

    common_features = sorted(set(feature_pool).intersection(feature_eval))
    if len(common_features) < 5:
        raise ValueError(
            f"Too few common predictors in {setup_name}: {len(common_features)}"
        )

    X_eval = df_eval[common_features].astype("float32").to_numpy()
    y_eval = df_eval["class_id"].to_numpy()

    train_classes = set(df_pool["class_id"].unique())
    eval_classes = set(df_eval["class_id"].unique())
    if train_classes != eval_classes:
        print(f"[WARN] {setup_name}: different classes in TRAINPOOL and EVAL.")
        print(f"  TRAINPOOL: {sorted(train_classes)}")
        print(f"  EVAL     : {sorted(eval_classes)}")

    if GRASSLAND_ID not in eval_classes or ROCKY_GRASSLAND_ID not in eval_classes:
        raise ValueError(
            f"{setup_name}: target classes are missing in EVAL. "
            f"Present classes: {sorted(eval_classes)}"
        )

    rf_params = RF_PARAMS_BY_SETUP[setup_name]
    print(f"\n[INFO] Setup: {setup_name}")
    print(f"[INFO] RF parameters: {rf_params}")
    print(f"[INFO] Number of common predictors: {len(common_features)}")

    grassland_scorer = make_class_f1_scorer(GRASSLAND_ID)
    rocky_grassland_scorer = make_class_f1_scorer(ROCKY_GRASSLAND_ID)

    replicate_metrics = []
    replicate_mda_global = []
    replicate_mda_grassland = []
    replicate_mda_rocky = []

    for rep in range(1, N_REPLICATES + 1):
        seed = BASE_SEED + rep

        df_train = stratified_subset(df_pool, seed=seed)
        X_train = df_train[common_features].astype("float32").to_numpy()
        y_train = df_train["class_id"].to_numpy()

        classifier = RandomForestClassifier(random_state=seed, **rf_params)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_eval)
        oa = accuracy_score(y_eval, y_pred)
        kappa = cohen_kappa_score(y_eval, y_pred)
        macro_f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)

        f1_grassland = f1_score(
            (y_eval == GRASSLAND_ID).astype(int),
            (y_pred == GRASSLAND_ID).astype(int),
            zero_division=0,
        )

        f1_rocky = f1_score(
            (y_eval == ROCKY_GRASSLAND_ID).astype(int),
            (y_pred == ROCKY_GRASSLAND_ID).astype(int),
            zero_division=0,
        )

        perm_global = permutation_importance(
            classifier,
            X_eval,
            y_eval,
            scoring="accuracy",
            n_repeats=PERMUTATION_REPEATS,
            random_state=seed,
            n_jobs=-1,
        )

        perm_grassland = permutation_importance(
            classifier,
            X_eval,
            y_eval,
            scoring=grassland_scorer,
            n_repeats=PERMUTATION_REPEATS,
            random_state=seed,
            n_jobs=-1,
        )

        perm_rocky = permutation_importance(
            classifier,
            X_eval,
            y_eval,
            scoring=rocky_grassland_scorer,
            n_repeats=PERMUTATION_REPEATS,
            random_state=seed,
            n_jobs=-1,
        )

        replicate_metrics.append({
            "setup": setup_name,
            "rep": rep,
            "seed": seed,
            "nTrainUsed": int(len(df_train)),
            "nEvalUsed": int(len(df_eval)),
            "OA": float(oa),
            "Kappa": float(kappa),
            "MacroF1": float(macro_f1),
            "F1_grassland": float(f1_grassland),
            "F1_rocky_grassland": float(f1_rocky),
        })

        replicate_mda_global.append(perm_global.importances_mean)
        replicate_mda_grassland.append(perm_grassland.importances_mean)
        replicate_mda_rocky.append(perm_rocky.importances_mean)

        if rep % 10 == 0 or rep == 1:
            print(
                f"[{setup_name}] rep {rep}/{N_REPLICATES} | "
                f"OA={oa:.4f} | "
                f"F1_grassland={f1_grassland:.4f} | "
                f"F1_rocky={f1_rocky:.4f} | "
                f"nTrain={len(df_train)}"
            )

    metrics_df = pd.DataFrame(replicate_metrics)
    metrics_df.to_csv(
        OUTPUT_DIR / f"rep_metrics_{setup_name}_{YEAR}.csv",
        index=False,
    )

    save_summary_and_replicates(
        common_features,
        replicate_mda_global,
        "MDA_global_acc",
        setup_name,
        YEAR,
    )

    save_summary_and_replicates(
        common_features,
        replicate_mda_grassland,
        "MDA_F1_grassland",
        setup_name,
        YEAR,
    )

    save_summary_and_replicates(
        common_features,
        replicate_mda_rocky,
        "MDA_F1_rocky_grassland",
        setup_name,
        YEAR,
    )

    print(f"\n[OK] {setup_name}: outputs saved in {OUTPUT_DIR}/")
    print(f"  - rep_metrics_{setup_name}_{YEAR}.csv")
    print(f"  - MDA_global_acc_summary_{setup_name}_{YEAR}.csv")
    print(f"  - MDA_F1_grassland_summary_{setup_name}_{YEAR}.csv")
    print(f"  - MDA_F1_rocky_grassland_summary_{setup_name}_{YEAR}.csv")


if __name__ == "__main__":
    for setup in SETUPS:
        run_setup(setup)