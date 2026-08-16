"""
Code 5 (Part 1/2): Donor-LODO Regression, Stratified Diagnostics, Direct Classification, and Sensitivity Analysis
------------------------------------------------------------------------------------------------------------------
This script performs:
1. Leakage-free preprocessing (train-fit / test-apply for CLR and log-zscore).
2. Donor-LODO (Leave-One-Donor-Out) Random Forest regression & environment-stratified performance analysis.
3. Post-regression time-window evaluation & direct multiclass RF ROC analysis (one-vs-rest).
4. Sensitivity analysis across multiple preprocessing/subsetting scenarios.

Data Requirements (in ../data/):
- 28 genus.txt
- Final_Important_Metabolites_31.txt
- metadata.txt
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, roc_curve, auc
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------
# 0. File Paths & Data Loading
# ---------------------------------------------------------------------
DATA_DIR = os.path.join("..", "data")
GENUS_FILE = os.path.join(DATA_DIR, "28 genus.txt")
METAB_FILE = os.path.join(DATA_DIR, "Final_Important_Metabolites_31.txt")
META_FILE  = os.path.join(DATA_DIR, "metadata.txt")

meta = pd.read_csv(META_FILE, sep="\t").drop_duplicates(subset=["SampleID"]).set_index("SampleID")
required_cols = ["TsD", "Donor", "Env"]
for c in required_cols:
    if c not in meta.columns:
        raise ValueError(f"metadata.txt must contain column: {c}")

genus = pd.read_csv(GENUS_FILE, sep="\t").set_index("genus").T
genus.index.name = "SampleID"

metab = pd.read_csv(METAB_FILE, sep="\t").set_index("Metabolite").T
metab.index.name = "SampleID"

common_samples = meta.index.intersection(genus.index).intersection(metab.index)
meta  = meta.loc[common_samples].copy()
genus = genus.loc[common_samples].copy()
metab = metab.loc[common_samples].copy()

genus = genus.apply(pd.to_numeric, errors="coerce")
metab = metab.apply(pd.to_numeric, errors="coerce")

y          = meta["TsD"].astype(float).values
donors     = meta["Donor"].astype(str).values
envs       = meta["Env"].astype(str).values
sample_ids = meta.index.astype(str).values

micro_feature_names = genus.columns.astype(str).tolist()
metab_feature_names = metab.columns.astype(str).tolist()
feature_names_full  = micro_feature_names + metab_feature_names

# ---------------------------------------------------------------------
# 1. Helper Preprocessing Functions (Train-Fit / Test-Apply)
# ---------------------------------------------------------------------
def clr_transform_df(df: pd.DataFrame, pseudo: float = 1e-6) -> pd.DataFrame:
    X = df.fillna(0.0).astype(float).values + pseudo
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum
    logX = np.log(X)
    gm = logX.mean(axis=1, keepdims=True)
    clr = logX - gm
    return pd.DataFrame(clr, index=df.index, columns=df.columns)

def fit_zscore(train_df: pd.DataFrame):
    mu = train_df.mean(axis=0)
    sd = train_df.std(axis=0).replace(0, 1.0)
    return mu, sd

def apply_zscore(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return (df - mu) / sd

def fit_metab_log1p_zscore(train_df: pd.DataFrame):
    med = train_df.median(axis=0, skipna=True)
    train_imp = train_df.fillna(med)
    train_log = np.log1p(train_imp.astype(float))
    mu = train_log.mean(axis=0)
    sd = train_log.std(axis=0).replace(0, 1.0)
    train_z = (train_log - mu) / sd
    params = {"med": med, "mu": mu, "sd": sd}
    return train_z, params

def apply_metab_log1p_zscore(test_df: pd.DataFrame, params: dict):
    med, mu, sd = params["med"], params["mu"], params["sd"]
    test_imp = test_df.fillna(med)
    test_log = np.log1p(test_imp.astype(float))
    test_z = (test_log - mu) / sd
    return test_z

def fit_transform_omics(train_ids, test_ids, micro_norm="clr", metab_norm="logz"):
    g_tr, g_te = genus.loc[train_ids], genus.loc[test_ids]
    m_tr, m_te = metab.loc[train_ids], metab.loc[test_ids]

    if micro_norm == "clr":
        g_tr_t = clr_transform_df(g_tr)
        g_te_t = clr_transform_df(g_te)
    elif micro_norm == "rel_arcsin":
        rel_tr = g_tr.div(g_tr.sum(axis=1), axis=0).fillna(0.0)
        rel_te = g_te.div(g_te.sum(axis=1), axis=0).fillna(0.0)
        g_tr_t = pd.DataFrame(np.arcsin(np.sqrt(rel_tr)), index=g_tr.index, columns=g_tr.columns)
        g_te_t = pd.DataFrame(np.arcsin(np.sqrt(rel_te)), index=g_te.index, columns=g_te.columns)
    else:
        raise ValueError("micro_norm must be 'clr' or 'rel_arcsin'.")

    g_mu, g_sd = fit_zscore(g_tr_t)
    g_tr_z = apply_zscore(g_tr_t, g_mu, g_sd)
    g_te_z = apply_zscore(g_te_t, g_mu, g_sd)

    if metab_norm == "logz":
        m_tr_z, params = fit_metab_log1p_zscore(m_tr)
        m_te_z = apply_metab_log1p_zscore(m_te, params)
    elif metab_norm == "log_only":
        med = m_tr.median(axis=0, skipna=True)
        m_tr_z = np.log1p(m_tr.fillna(med).astype(float))
        m_te_z = np.log1p(m_te.fillna(med).astype(float))
    else:
        raise ValueError("metab_norm must be 'logz' or 'log_only'.")

    X_tr = np.hstack([g_tr_z.values, m_tr_z.values])
    X_te = np.hstack([g_te_z.values, m_te_z.values])
    return X_tr, X_te, feature_names_full

# ---------------------------------------------------------------------
# 2. Pipeline Factories
# ---------------------------------------------------------------------
def make_rf_reg_pipeline():
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=500, n_jobs=-1)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    param_grid = {
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5],
        "rf__max_features": ["sqrt", 0.5, 0.8]
    }
    return pipe, param_grid

def make_rf_clf_pipeline():
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=500, n_jobs=-1)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    param_grid = {
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5],
        "rf__max_features": ["sqrt", 0.5, 0.8]
    }
    return pipe, param_grid

def group_inner_cv(groups_train):
    ug = np.unique(groups_train)
    return GroupKFold(n_splits=min(3, len(ug))) if len(ug) >= 3 else LeaveOneGroupOut()

def assign_time_window(tsd_days: float) -> str:
    if tsd_days <= 1.0:
        return "0-24h"
    elif tsd_days <= 7.0:
        return "1-7d"
    else:
        return ">7d"

# ---------------------------------------------------------------------
# 3. Donor-LODO Full Regression
# ---------------------------------------------------------------------
logo = LeaveOneGroupOut()
reg_metrics_records = []
reg_pred_records = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(sample_ids, y, groups=donors)):
    test_donor = donors[test_idx][0]
    train_ids, test_ids = sample_ids[train_idx], sample_ids[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = donors[train_idx]

    X_train, X_test, _ = fit_transform_omics(train_ids, test_ids, micro_norm="clr", metab_norm="logz")

    pipe, param_grid = make_rf_reg_pipeline()
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=group_inner_cv(groups_train).split(X_train, y_train, groups=groups_train),
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    reg_metrics_records.append({
        "Donor": test_donor, "R2": r2, "RMSE": rmse, "MAE": mae, "Best_params": gs.best_params_
    })

    for sid_i, yt, yp, env_i in zip(test_ids, y_test, y_pred, envs[test_idx]):
        reg_pred_records.append({
            "SampleID": sid_i, "Donor": test_donor, "Env": str(env_i),
            "y_true": float(yt), "y_pred": float(yp), "Bias": float(yt - yp),
            "TimeWindow_true": assign_time_window(float(yt)),
            "TimeWindow_pred": assign_time_window(float(yp))
        })

reg_metrics_df = pd.DataFrame(reg_metrics_records)
reg_pred_df    = pd.DataFrame(reg_pred_records)

reg_metrics_df.to_csv("TsD_RF_LODO_regression_metrics_by_donor.csv", index=False)
reg_pred_df.to_csv("TsD_LODO_regression_predictions_by_donor.csv", index=False)

# Env-stratified Summary
env_summary = []
for e in sorted(reg_pred_df["Env"].unique()):
    df_e = reg_pred_df[reg_pred_df["Env"] == e]
    rmse_e = np.sqrt(np.mean((df_e["y_true"] - df_e["y_pred"]) ** 2))
    mae_e  = np.mean(np.abs(df_e["y_true"] - df_e["y_pred"]))
    env_summary.append({"Env": e, "n": len(df_e), "RMSE": rmse_e, "MAE": mae_e, "Bias_mean": np.mean(df_e["Bias"])})
env_metrics_df = pd.DataFrame(env_summary)
env_metrics_df.to_csv("TsD_LODO_regression_metrics_by_env.csv", index=False)

# Figure 8B
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.scatterplot(data=reg_pred_df, x="y_true", y="y_pred", hue="Env", style="Env", s=55, ax=ax)
minv = min(reg_pred_df["y_true"].min(), reg_pred_df["y_pred"].min())
maxv = max(reg_pred_df["y_true"].max(), reg_pred_df["y_pred"].max())
ax.plot([minv, maxv], [minv, maxv], linestyle="--", color="black", linewidth=1)
ax.set_xlabel("Observed TsD (days)")
ax.set_ylabel("Predicted TsD (days)")
ax.set_title("Figure 8B Donor-LODO TsD prediction stratified by environment")
sns.despine(fig=fig)
plt.tight_layout()
plt.savefig("Figure_8B_DonorLODO_scatter_byEnv.pdf", dpi=600, bbox_inches="tight")
plt.savefig("Figure_8B_DonorLODO_scatter_byEnv.png", dpi=600, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 4. Time Window Derived Classification & Direct ROC
# ---------------------------------------------------------------------
windows = ["0-24h", "1-7d", ">7d"]
cm = confusion_matrix(reg_pred_df["TimeWindow_true"], reg_pred_df["TimeWindow_pred"], labels=windows)
pd.DataFrame(cm, index=windows, columns=windows).to_csv("TsD_timewindow_confusion_matrix.csv")

# Direct Multiclass ROC Classifier
y_cls = np.array([assign_time_window(v) for v in y])
class_to_int = {c: i for i, c in enumerate(windows)}
y_int = np.array([class_to_int[c] for c in y_cls])

y_true_all, y_proba_all = [], []
for train_idx, test_idx in logo.split(sample_ids, y_int, groups=donors):
    train_ids, test_ids = sample_ids[train_idx], sample_ids[test_idx]
    y_train, y_test = y_int[train_idx], y_int[test_idx]
    groups_train = donors[train_idx]

    X_train, X_test, _ = fit_transform_omics(train_ids, test_ids)
    pipe, param_grid = make_rf_clf_pipeline()
    gs = GridSearchCV(
        estimator=pipe, param_grid=param_grid, scoring="accuracy",
        cv=group_inner_cv(groups_train).split(X_train, y_train, groups=groups_train),
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    y_true_all.append(y_test)
    y_proba_all.append(gs.best_estimator_.predict_proba(X_test))

y_true_all = np.concatenate(y_true_all)
y_proba_all = np.vstack(y_proba_all)
y_bin = label_binarize(y_true_all, classes=list(range(len(windows))))

fig, ax = plt.subplots(figsize=(5.2, 4.2))
for i, c in enumerate(windows):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_all[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f"{c} (AUC={roc_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="black")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Figure 8D Donor-LODO ROC (one-vs-rest)")
ax.legend(loc="lower right")
sns.despine(fig=fig)
plt.tight_layout()
plt.savefig("Figure_8D_TimeWindow_ROC.pdf", dpi=600, bbox_inches="tight")
plt.savefig("Figure_8D_TimeWindow_ROC.png", dpi=600, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 5. Sensitivity Analysis Scenarios
# ---------------------------------------------------------------------
def eval_regression_scenario(sample_subset, micro_norm="clr", metab_norm="logz", feature_filter=False):
    meta_sub = meta.loc[sample_subset]
    y_sub, donors_sub, sids_sub = meta_sub["TsD"].values, meta_sub["Donor"].values, meta_sub.index.values
    fold_metrics = []

    for train_idx, test_idx in logo.split(sids_sub, y_sub, groups=donors_sub):
        X_train, X_test, _ = fit_transform_omics(sids_sub[train_idx], sids_sub[test_idx], micro_norm, metab_norm)
        if feature_filter:
            pvals = [spearmanr(X_train[:, j], y_sub[train_idx])[1] for j in range(X_train.shape[1])]
            reject, _, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), alpha=0.01, method="fdr_bh")
            keep_idx = np.where(reject)[0]
            if len(keep_idx) < 5:
                keep_idx = np.arange(X_train.shape[1])
            X_train, X_test = X_train[:, keep_idx], X_test[:, keep_idx]

        pipe, param_grid = make_rf_reg_pipeline()
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, scoring="r2",
            cv=group_inner_cv(donors_sub[train_idx]).split(X_train, y_sub[train_idx], groups=donors_sub[train_idx]),
            n_jobs=-1
        )
        gs.fit(X_train, y_sub[train_idx])
        y_pred = gs.best_estimator_.predict(X_test)
        fold_metrics.append([
            r2_score(y_sub[test_idx], y_pred),
            np.sqrt(mean_squared_error(y_sub[test_idx], y_pred)),
            mean_absolute_error(y_sub[test_idx], y_pred)
        ])
    fm = np.array(fold_metrics)
    return fm.mean(axis=0), fm.std(axis=0)

sensitivity_results = []
m_mean, m_sd = eval_regression_scenario(common_samples)
sensitivity_results.append({
    "Scenario": "Baseline (all samples, CLR+logz)",
    "R2_mean": m_mean[0], "R2_sd": m_sd[0], "RMSE_mean": m_mean[1], "RMSE_sd": m_sd[1], "MAE_mean": m_mean[2], "MAE_sd": m_sd[2]
})

pd.DataFrame(sensitivity_results).to_csv("TsD_model_sensitivity_analysis.csv", index=False)
print("Part 1 complete. Proceed to Part 2.")