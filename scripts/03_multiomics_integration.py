# -*- coding: utf-8 -*-
"""
TsD Multi-omics Integration Analysis (Microbiome + Metabolome)

This script performs multi-omics integration for predicting Time Since Deposition (TsD)
under strict anti-leakage nested cross-validation:
- Outer Loop: Leave-One-Donor-Out CV (LODO-CV) to evaluate donor-wise generalization.
- Inner Loop: 5-Fold CV via GridSearchCV for hyperparameter tuning.

Integration Strategies:
1. Single-omics Baseline (Microbiome-only, Metabolome-only)
2. Early Integration (Feature-level concatenation + Single learner)
3. Latent Integration (Block-wise PLS score extraction + Ridge regression)
4. Late Integration (Stacking ensemble with block-specific base learners)

"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

import matplotlib.pyplot as plt
import seaborn as sns

# Optional acceleration libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Global Constants
RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# Main algorithm for evaluation (e.g., "RF", "KNN", "XGB", "LGBM")
FIXED_ALGO = "RF"

# File Paths
GENUS_FILE = "28 genus.txt"
METAB_FILE = "Final_Important_Metabolites_31.txt"
META_FILE  = "metadata.txt"


# ============================================================
# 1) Data Preprocessing & Transformations
# ============================================================
def clr_transform(df: pd.DataFrame, pseudo: float = 1e-6) -> pd.DataFrame:
    """
    Centered Log-Ratio (CLR) transformation for compositional data.
    This transformation operates sample-wise (no population parameters fit),
    making it safe from data leakage during cross-validation.
    """
    vals = df.values.astype(float) + pseudo
    vals = vals / vals.sum(axis=1, keepdims=True)
    log_vals = np.log(vals)
    geom_mean = log_vals.mean(axis=1, keepdims=True)
    clr_vals = log_vals - geom_mean
    return pd.DataFrame(clr_vals, index=df.index, columns=df.columns)


def load_and_align_data(meta_path: str, genus_path: str, metab_path: str):
    """Load metadata, microbiome genus table, and metabolome table, aligning sample IDs."""
    meta = pd.read_csv(meta_path, sep="\t")
    meta = meta.drop_duplicates(subset=["SampleID"]).set_index("SampleID")
    meta["Donor"] = meta.index.to_series().str.extract(r"(D\d+)", expand=False)

    genus = pd.read_csv(genus_path, sep="\t").set_index("genus").T
    genus.index.name = "SampleID"

    metab = pd.read_csv(metab_path, sep="\t").set_index("Metabolite").T
    metab.index.name = "SampleID"

    common_samples = meta.index.intersection(genus.index).intersection(metab.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples found across metadata, microbiome, and metabolome tables.")

    meta = meta.loc[common_samples].copy()
    genus = genus.loc[common_samples].copy()
    metab = metab.loc[common_samples].copy()

    y = meta["TsD"].values.astype(float)
    donors = meta["Donor"].values

    print(f"[Data I/O] Aligned Samples: {len(common_samples)} | "
          f"Microbiome Genera: {genus.shape[1]} | Metabolites: {metab.shape[1]}")

    return genus, metab, y, donors


# Preprocessing Pipelines
micro_pre = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

metab_pre = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler()),
])


# ============================================================
# 2) Model Definitions and Parameter Grids
# ============================================================
def get_models_and_grids():
    models, grids = {}, {}

    models["RF"] = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grids["RF"] = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }

    models["KNN"] = KNeighborsRegressor()
    grids["KNN"] = {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    if HAS_XGB:
        models["XGB"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_estimators=500,
            n_jobs=-1,
        )
        grids["XGB"] = {
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.7, 1.0],
        }

    if HAS_LGBM:
        models["LGBM"] = LGBMRegressor(
            random_state=RANDOM_STATE,
            n_estimators=500,
            n_jobs=-1,
            verbosity=-1
        )
        grids["LGBM"] = {
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [-1, 5],
        }

    return models, grids


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute standard regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_true - y_pred))  # Bias = True - Pred
    return r2, rmse, mae, bias


# ============================================================
# 3) Latent Integration Estimator (Block PLS + Ridge)
# ============================================================
class BlockPLSRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Block-wise Partial Least Squares (PLS) feature extractor followed by Ridge Regression.
    Fully compatible with scikit-learn Pipeline and GridSearchCV.
    """
    def __init__(self, micro_idx, metab_idx, n_components=2, alpha=1.0, random_state=RANDOM_STATE):
        self.micro_idx = micro_idx
        self.metab_idx = metab_idx
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        Xm = X[:, self.micro_idx]
        Xb = X[:, self.metab_idx]

        self.micro_pre_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        self.metab_pre_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ])

        Xm_p = self.micro_pre_.fit_transform(Xm, y)
        Xb_p = self.metab_pre_.fit_transform(Xb, y)

        self.pls_micro_ = PLSRegression(n_components=self.n_components)
        self.pls_metab_ = PLSRegression(n_components=self.n_components)

        Tm, _ = self.pls_micro_.fit_transform(Xm_p, y)
        Tb, _ = self.pls_metab_.fit_transform(Xb_p, y)

        Z = np.concatenate([Tm, Tb], axis=1)

        self.ridge_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.ridge_.fit(Z, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        Xm = X[:, self.micro_idx]
        Xb = X[:, self.metab_idx]

        Xm_p = self.micro_pre_.transform(Xm)
        Xb_p = self.metab_pre_.transform(Xb)

        Tm = self.pls_micro_.transform(Xm_p)
        Tb = self.pls_metab_.transform(Xb_p)
        Z = np.concatenate([Tm, Tb], axis=1)

        return self.ridge_.predict(Z)


# ============================================================
# 4) Cross-Validation Pipelines
# ============================================================
def nested_cv_single_block(X, y, donors, pipe, param_grid, strategy_label,
                           n_inner_splits=5, random_state=RANDOM_STATE):
    results = []
    unique_donors = np.unique(donors)

    for donor in unique_donors:
        test_mask = (donors == donor)
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(pipe, param_grid, scoring="r2", cv=inner_cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        y_pred = gs.best_estimator_.predict(X_test)
        r2, rmse, mae, bias = calc_metrics(y_test, y_pred)

        results.append({
            "donor": donor,
            "StrategyLabel": strategy_label,
            "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias,
            "best_params": gs.best_params_
        })
        print(f"[{strategy_label}] Donor={donor} | R2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} Bias={bias:.3f}")

    return results


def nested_cv_early_integration(X_micro, X_metab_raw, y, donors, model, model_grid,
                                n_inner_splits=5, random_state=RANDOM_STATE):
    X_all = np.concatenate([X_micro, X_metab_raw], axis=1)
    n_micro = X_micro.shape[1]
    micro_idx = np.arange(n_micro)
    metab_idx = np.arange(n_micro, X_all.shape[1])

    pre = ColumnTransformer(transformers=[
        ("micro", micro_pre, micro_idx),
        ("metab", metab_pre, metab_idx),
    ])

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    results = []
    unique_donors = np.unique(donors)

    for donor in unique_donors:
        test_mask = (donors == donor)
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(pipe, model_grid, scoring="r2", cv=inner_cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        y_pred = gs.best_estimator_.predict(X_test)
        r2, rmse, mae, bias = calc_metrics(y_test, y_pred)

        label = f"Early-integration ({FIXED_ALGO})"
        results.append({
            "donor": donor,
            "StrategyLabel": label,
            "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias,
            "best_params": gs.best_params_
        })
        print(f"[{label}] Donor={donor} | R2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} Bias={bias:.3f}")

    return results


def nested_cv_latent_integration(X_micro, X_metab_raw, y, donors,
                                 n_components_list=(2, 3, 4),
                                 alpha_list=(0.1, 1.0, 10.0),
                                 n_inner_splits=5,
                                 random_state=RANDOM_STATE):
    X_all = np.concatenate([X_micro, X_metab_raw], axis=1)
    n_micro = X_micro.shape[1]
    micro_idx = np.arange(n_micro)
    metab_idx = np.arange(n_micro, X_all.shape[1])

    est = BlockPLSRidgeRegressor(micro_idx=micro_idx, metab_idx=metab_idx)
    param_grid = {
        "n_components": list(n_components_list),
        "alpha": list(alpha_list),
    }

    results = []
    unique_donors = np.unique(donors)

    for donor in unique_donors:
        test_mask = (donors == donor)
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring="r2", cv=inner_cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        y_pred = gs.best_estimator_.predict(X_test)
        r2, rmse, mae, bias = calc_metrics(y_test, y_pred)

        results.append({
            "donor": donor,
            "StrategyLabel": "Latent (PLS+Ridge)",
            "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias,
            "best_params": gs.best_params_
        })
        bp = gs.best_params_
        print(f"[Latent] Donor={donor} | best_k={bp['n_components']} alpha={bp['alpha']} "
              f"| R2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} Bias={bias:.3f}")

    return results


def nested_cv_late_integration(X_micro, X_metab_raw, y, donors,
                               n_inner_splits=5,
                               random_state=RANDOM_STATE,
                               tune=True):
    X_all = np.concatenate([X_micro, X_metab_raw], axis=1)
    n_micro = X_micro.shape[1]
    micro_idx = np.arange(n_micro)
    metab_idx = np.arange(n_micro, X_all.shape[1])

    base_micro = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    base_metab = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    micro_pipe = Pipeline(steps=[
        ("selector", FunctionTransformer(lambda X: X[:, micro_idx], validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", base_micro),
    ])

    metab_pipe = Pipeline(steps=[
        ("selector", FunctionTransformer(lambda X: X[:, metab_idx], validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
        ("model", base_metab),
    ])

    final_est = Ridge(alpha=1.0, random_state=random_state)

    stack = StackingRegressor(
        estimators=[("micro_model", micro_pipe), ("metab_model", metab_pipe)],
        final_estimator=final_est,
        cv=n_inner_splits,
        n_jobs=-1,
        passthrough=False
    )

    param_grid = {
        "micro_model__model__n_estimators": [200, 500],
        "micro_model__model__max_depth": [None, 10],
        "micro_model__model__min_samples_split": [2, 5],
        "metab_model__model__n_estimators": [200, 500],
        "metab_model__model__max_depth": [None, 10],
        "metab_model__model__min_samples_split": [2, 5],
        "final_estimator__alpha": [0.1, 1.0, 10.0],
    }

    results = []
    unique_donors = np.unique(donors)

    for donor in unique_donors:
        test_mask = (donors == donor)
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if tune:
            inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
            gs = GridSearchCV(stack, param_grid, scoring="r2", cv=inner_cv, n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best = stack.fit(X_train, y_train)
            best_params = {}

        y_pred = best.predict(X_test)
        r2, rmse, mae, bias = calc_metrics(y_test, y_pred)

        results.append({
            "donor": donor,
            "StrategyLabel": "Late (Stacking RF)",
            "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias,
            "best_params": best_params
        })
        print(f"[Late(Stacking)] Donor={donor} | R2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} Bias={bias:.3f}")

    return results


# ============================================================
# 5) Main Execution Flow
# ============================================================
def main():
    genus, metab, y, donors = load_and_align_data(META_FILE, GENUS_FILE, METAB_FILE)

    # Microbiome sample-wise CLR transformation
    genus_clr = clr_transform(genus)
    X_micro = genus_clr.values.astype(float)
    X_metab_raw = metab.values.astype(float)

    base_models, base_grids = get_models_and_grids()
    if FIXED_ALGO not in base_models:
        raise ValueError(f"FIXED_ALGO='{FIXED_ALGO}' not available. Options: {list(base_models.keys())}")

    all_results = []

    # 1. Microbiome-only
    print("\n--- Running Microbiome-only Pipeline ---")
    pipe_micro = Pipeline(steps=[("pre", micro_pre), ("model", base_models[FIXED_ALGO])])
    all_results.extend(nested_cv_single_block(
        X_micro, y, donors, pipe_micro, base_grids[FIXED_ALGO],
        strategy_label=f"Microbiome-only ({FIXED_ALGO})"
    ))

    # 2. Metabolome-only
    print("\n--- Running Metabolome-only Pipeline ---")
    pipe_metab = Pipeline(steps=[("pre", metab_pre), ("model", base_models[FIXED_ALGO])])
    all_results.extend(nested_cv_single_block(
        X_metab_raw, y, donors, pipe_metab, base_grids[FIXED_ALGO],
        strategy_label=f"Metabolome-only ({FIXED_ALGO})"
    ))

    # 3. Early Integration
    print("\n--- Running Early Integration Pipeline ---")
    all_results.extend(nested_cv_early_integration(
        X_micro, X_metab_raw, y, donors,
        model=base_models[FIXED_ALGO], model_grid=base_grids[FIXED_ALGO]
    ))

    # 4. Latent Integration
    print("\n--- Running Latent Integration Pipeline ---")
    all_results.extend(nested_cv_latent_integration(
        X_micro, X_metab_raw, y, donors,
        n_components_list=(2, 3, 4), alpha_list=(0.1, 1.0, 10.0)
    ))

    # 5. Late Integration
    print("\n--- Running Late Integration Pipeline ---")
    all_results.extend(nested_cv_late_integration(
        X_micro, X_metab_raw, y, donors, tune=True
    ))

    # Save detailed CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("TsD_multiomics_LODO_results_detailed_FIXEDALGO.csv", index=False)
    print("\n[Output] Saved detailed fold results to 'TsD_multiomics_LODO_results_detailed_FIXEDALGO.csv'")

    # Summary by Strategy
    summary = (
        results_df.groupby("StrategyLabel")[["R2", "RMSE", "MAE", "Bias"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Strategy", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd",
        "MAE_mean", "MAE_sd", "Bias_mean", "Bias_sd"
    ]
    summary["n_folds"] = results_df.groupby("StrategyLabel").size().values
    summary.to_csv("TsD_multiomics_LODO_summary_by_strategy_FIXEDALGO.csv", index=False)
    print("[Output] Saved strategy summary to 'TsD_multiomics_LODO_summary_by_strategy_FIXEDALGO.csv'")

    # ============================================================
    # 6) Figure 8A Visualization
    # ============================================================
    plot_df = summary.copy().sort_values("R2_mean", ascending=False)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    metrics = [
        ("R2_mean", "R²", True),
        ("RMSE_mean", "RMSE (days)", False),
        ("MAE_mean", "MAE (days)", False)
    ]

    palette = sns.color_palette("deep", n_colors=len(plot_df))
    order = plot_df["Strategy"].tolist()

    for ax, (col, label, is_r2) in zip(axes, metrics):
        sns.barplot(
            x="Strategy", y=col, data=plot_df,
            order=order, hue="Strategy", legend=False,
            ax=ax, palette=palette, edgecolor="black", linewidth=0.8
        )

        err_col = col.replace("_mean", "_sd")
        ax.errorbar(
            x=np.arange(len(plot_df)),
            y=plot_df[col].values,
            yerr=plot_df[err_col].values,
            fmt="none", ecolor="black", elinewidth=1, capsize=3
        )

        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_xticklabels(order, rotation=35, ha="right")
        sns.despine(ax=ax, top=True, right=True)

        if is_r2:
            ax.set_ylim(0, 1.0)

    fig.suptitle("Figure 8A  Multi-omics TsD prediction performance under different strategies",
                 y=1.03, fontsize=14, fontweight="bold")

    plt.savefig("Figure_8A_TsD_multiomics_performance_FIXEDALGO.png", dpi=600, bbox_inches="tight")
    plt.savefig("Figure_8A_TsD_multiomics_performance_FIXEDALGO.pdf", bbox_inches="tight")
    plt.show()

    print("[Output] Saved publication-ready figure: Figure_8A_TsD_multiomics_performance_FIXEDALGO.(png/pdf)")


if __name__ == "__main__":
    main()