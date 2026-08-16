"""
Code 5 (Part 2/2): Marker Contribution, SHAP Interpretability, and Nested Simplified Panel Evaluation
------------------------------------------------------------------------------------------------------
This script performs:
1. Binary marker contribution analysis under Donor-LODO (Fresh: <=24h vs >24h; Late: >7d vs <=7d).
2. Mann-Whitney U test with directionality (Overrepresented vs Underrepresented).
3. Full-data Panel model training and SHAP summary visualization (Figure 8G).
4. Strictly leakage-free Nested Panel evaluation (Top-10 selected inside each fold) (Figure 8H).

Data Requirements (in ../data/):
- 28 genus.txt
- Final_Important_Metabolites_31.txt
- metadata.txt
"""

import os
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)

# Import shared modules from Part 1 or re-declare dependencies
DATA_DIR = os.path.join("..", "data")
GENUS_FILE = os.path.join(DATA_DIR, "28 genus.txt")
METAB_FILE = os.path.join(DATA_DIR, "Final_Important_Metabolites_31.txt")
META_FILE  = os.path.join(DATA_DIR, "metadata.txt")

meta = pd.read_csv(META_FILE, sep="\t").drop_duplicates(subset=["SampleID"]).set_index("SampleID")
genus = pd.read_csv(GENUS_FILE, sep="\t").set_index("genus").T
metab = pd.read_csv(METAB_FILE, sep="\t").set_index("Metabolite").T

common_samples = meta.index.intersection(genus.index).intersection(metab.index)
meta  = meta.loc[common_samples].copy()
genus = genus.loc[common_samples].apply(pd.to_numeric, errors="coerce").copy()
metab = metab.loc[common_samples].apply(pd.to_numeric, errors="coerce").copy()

y          = meta["TsD"].astype(float).values
donors     = meta["Donor"].astype(str).values
sample_ids = meta.index.astype(str).values

micro_feature_names = genus.columns.astype(str).tolist()
metab_feature_names = metab.columns.astype(str).tolist()
feature_names_full  = micro_feature_names + metab_feature_names

micro_set = set(micro_feature_names)
metab_set = set(metab_feature_names)

P_THRESH = 0.005
N_REP_SEED = 30
TOP_SHOW = 35
BOX_COLORS = {
    "Overrepresented": "#e41a1c",
    "Underrepresented": "#9ecae1",
    "NotSignificant": "#bdbdbd"
}

# ---------------------------------------------------------------------
# Helper Preprocessing Definitions
# ---------------------------------------------------------------------
def clr_transform_df(df: pd.DataFrame, pseudo: float = 1e-6) -> pd.DataFrame:
    X = df.fillna(0.0).astype(float).values + pseudo
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum
    logX = np.log(X)
    return pd.DataFrame(logX - logX.mean(axis=1, keepdims=True), index=df.index, columns=df.columns)

def fit_transform_omics(train_ids, test_ids, micro_norm="clr", metab_norm="logz"):
    g_tr, g_te = genus.loc[train_ids], genus.loc[test_ids]
    m_tr, m_te = metab.loc[train_ids], metab.loc[test_ids]

    g_tr_t, g_te_t = clr_transform_df(g_tr), clr_transform_df(g_te)
    g_mu, g_sd = g_tr_t.mean(axis=0), g_tr_t.std(axis=0).replace(0, 1.0)
    g_tr_z = (g_tr_t - g_mu) / g_sd
    g_te_z = (g_te_t - g_mu) / g_sd

    med = m_tr.median(axis=0, skipna=True)
    m_tr_log = np.log1p(m_tr.fillna(med).astype(float))
    m_te_log = np.log1p(m_te.fillna(med).astype(float))
    m_mu, m_sd = m_tr_log.mean(axis=0), m_tr_log.std(axis=0).replace(0, 1.0)
    m_tr_z = (m_tr_log - m_mu) / m_sd
    m_te_z = (m_te_log - m_mu) / m_sd

    return np.hstack([g_tr_z.values, m_tr_z.values]), np.hstack([g_te_z.values, m_te_z.values]), feature_names_full

def make_rf_clf_pipeline():
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=500, n_jobs=-1)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    param_grid = {"rf__max_depth": [None, 5, 10], "rf__min_samples_split": [2, 5], "rf__max_features": ["sqrt", 0.5, 0.8]}
    return pipe, param_grid

def make_rf_reg_pipeline():
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=500, n_jobs=-1)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    param_grid = {"rf__max_depth": [None, 5, 10], "rf__min_samples_split": [2, 5], "rf__max_features": ["sqrt", 0.5, 0.8]}
    return pipe, param_grid

def group_inner_cv(groups_train):
    ug = np.unique(groups_train)
    return GroupKFold(n_splits=min(3, len(ug))) if len(ug) >= 3 else LeaveOneGroupOut()

def make_binary_labels(y_days, mode):
    y_days = np.asarray(y_days, dtype=float)
    return (y_days <= 1.0).astype(int) if mode == "fresh" else (y_days > 7.0).astype(int)

def one_sided_mwu_over_under(x_pos, x_neg, p_thresh=P_THRESH):
    x_pos, x_neg = x_pos[np.isfinite(x_pos)], x_neg[np.isfinite(x_neg)]
    if len(x_pos) < 3 or len(x_neg) < 3:
        return "NotSignificant", np.nan, "ns"
    if np.median(x_pos) > np.median(x_neg):
        _, p = mannwhitneyu(x_pos, x_neg, alternative="greater")
        return ("Overrepresented" if p < p_thresh else "NotSignificant"), p, "pos>neg"
    else:
        _, p = mannwhitneyu(x_pos, x_neg, alternative="less")
        return ("Underrepresented" if p < p_thresh else "NotSignificant"), p, "pos<neg"

# ---------------------------------------------------------------------
# 11. Marker Contribution Boxplots under Donor-LODO
# ---------------------------------------------------------------------
def collect_importance_distribution_binary(mode, title_label):
    y_bin = make_binary_labels(y, mode)
    logo = LeaveOneGroupOut()
    y_true_all, y_score_all, records = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(sample_ids, y_bin, groups=donors)):
        test_donor = donors[test_idx][0]
        train_ids, test_ids = sample_ids[train_idx], sample_ids[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]
        groups_train = donors[train_idx]

        if len(np.unique(y_train)) < 2:
            continue

        X_train, X_test, feat_names = fit_transform_omics(train_ids, test_ids)
        pipe, param_grid = make_rf_clf_pipeline()
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, scoring="balanced_accuracy",
            cv=group_inner_cv(groups_train).split(X_train, y_train, groups=groups_train), n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_params = gs.best_params_

        proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
        y_true_all.append(y_test)
        y_score_all.append(proba)

        for r in range(N_REP_SEED):
            pipe_r, _ = make_rf_clf_pipeline()
            pipe_r.set_params(**best_params)
            pipe_r.named_steps["rf"].set_params(random_state=RANDOM_STATE + 1000 * fold_idx + r)
            pipe_r.fit(X_train, y_train)
            imp = pipe_r.named_steps["rf"].feature_importances_.astype(float)
            imp = imp / imp.sum() * 100.0

            for f, v in zip(feat_names, imp):
                records.append({
                    "Mode": mode, "Contrast": title_label, "Fold": fold_idx,
                    "TestDonor": test_donor, "Rep": r, "Feature": f, "Importance_pct": float(v)
                })

    imp_long_df = pd.DataFrame(records)
    if len(y_true_all) > 0:
        y_true_all, y_score_all = np.concatenate(y_true_all), np.concatenate(y_score_all)
        auc_value = auc(*roc_curve(y_true_all, y_score_all)[:2]) if len(np.unique(y_true_all)) >= 2 else np.nan
    else:
        auc_value = np.nan

    return imp_long_df, auc_value

X_mwu_all = pd.DataFrame(
    np.hstack([clr_transform_df(genus).values, np.log1p(metab.fillna(metab.median(axis=0))).values]),
    index=sample_ids, columns=feature_names_full
)

def build_significance_table(mode, contrast_label):
    y_bin = make_binary_labels(y, mode)
    pos_ids, neg_ids = sample_ids[y_bin == 1], sample_ids[y_bin == 0]
    rows = []
    for feat in feature_names_full:
        x_pos, x_neg = X_mwu_all.loc[pos_ids, feat].values, X_mwu_all.loc[neg_ids, feat].values
        cat, p, direction = one_sided_mwu_over_under(x_pos, x_neg)
        ftype = "Genus" if feat in micro_set else ("Metabolite" if feat in metab_set else "Other")
        rows.append({
            "Contrast": contrast_label, "Mode": mode, "Feature": feat, "FeatureType": ftype,
            "MWU_category": cat, "MWU_p": p, "Direction": direction
        })
    return pd.DataFrame(rows)

def plot_marker_boxpanel(ax, imp_long_df, sig_df, panel_title):
    med = imp_long_df.groupby("Feature")["Importance_pct"].median().sort_values(ascending=False)
    top_feats = med.index[:TOP_SHOW].tolist()
    data = [imp_long_df.loc[imp_long_df["Feature"] == f, "Importance_pct"].values for f in top_feats]

    sig_map = sig_df.set_index("Feature")["MWU_category"].to_dict()
    type_map = sig_df.set_index("Feature")["FeatureType"].to_dict()

    bp = ax.boxplot(data, vert=False, labels=top_feats, patch_artist=True, whis=1.5, showfliers=True)
    for patch, feat in zip(bp["boxes"], top_feats):
        patch.set_facecolor(BOX_COLORS.get(sig_map.get(feat, "NotSignificant"), BOX_COLORS["NotSignificant"]))

    for tick in ax.get_yticklabels():
        if type_map.get(tick.get_text(), "Other") == "Genus":
            tick.set_color("#e41a1c")
            tick.set_fontstyle("italic")

    ax.set_title(panel_title, fontsize=12)
    ax.set_xlabel("Contribution to model (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

imp_fresh, auc_fresh = collect_importance_distribution_binary("fresh", "0–24h vs >24h")
sig_fresh = build_significance_table("fresh", "0–24h vs >24h")

imp_late, auc_late = collect_importance_distribution_binary("late", ">7d vs ≤7d")
sig_late = build_significance_table("late", ">7d vs ≤7d")

fig, axes = plt.subplots(1, 2, figsize=(12.0, 8.8))
plot_marker_boxpanel(axes[0], imp_fresh, sig_fresh, f"(b) Fresh boundary: 0–24h vs >24h (AUC={auc_fresh:.2f})")
plot_marker_boxpanel(axes[1], imp_late, sig_late, f"(c) Late boundary: >7d vs ≤7d (AUC={auc_late:.2f})")

legend_handles = [
    mpatches.Patch(facecolor=BOX_COLORS["Overrepresented"], edgecolor="black", label=f"Overrepresented (P < {P_THRESH})"),
    mpatches.Patch(facecolor=BOX_COLORS["Underrepresented"], edgecolor="black", label=f"Underrepresented (P < {P_THRESH})"),
    mpatches.Patch(facecolor=BOX_COLORS["NotSignificant"], edgecolor="black", label="Not significant")
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("Figure_8E_MarkerContribution_Boxplots.pdf", dpi=600, bbox_inches="tight")
plt.savefig("Figure_8E_MarkerContribution_Boxplots.png", dpi=600, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# SHAP Visualization for Full Fit
# ---------------------------------------------------------------------
X_full = np.hstack([clr_transform_df(genus).values, np.log1p(metab.fillna(metab.median(axis=0))).values])
pipe_full, param_grid_full = make_rf_reg_pipeline()
gs_full = GridSearchCV(pipe_full, param_grid_full, scoring="r2", cv=group_inner_cv(donors).split(X_full, y, groups=donors), n_jobs=-1)
gs_full.fit(X_full, y)

explainer = shap.TreeExplainer(gs_full.best_estimator_.named_steps["rf"])
shap_vals = explainer.shap_values(X_full)

wrapped_names = [textwrap.fill(name, width=40) for name in feature_names_full]
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_vals, X_full, feature_names=wrapped_names, show=False, plot_type="dot", max_display=10)
plt.title("Figure 8G SHAP summary for simplified TsD panel (Top-10 features)", fontsize=14)
plt.gcf().subplots_adjust(left=0.55)
plt.savefig("Figure_8G_TsD_panel_SHAP2.pdf", dpi=600, bbox_inches="tight")
plt.savefig("Figure_8G_TsD_panel_SHAP2.png", dpi=600, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# 12. Nested Top-K Panel LODO Regression
# ---------------------------------------------------------------------
def run_nested_panel_lodo_predictions(K=10):
    logo = LeaveOneGroupOut()
    pred_records, fold_metrics = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(sample_ids, y, groups=donors)):
        test_donor = donors[test_idx][0]
        train_ids, test_ids = sample_ids[train_idx], sample_ids[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = donors[train_idx]

        X_train, X_test, feat_names = fit_transform_omics(train_ids, test_ids)

        pipe_full, param_grid_full = make_rf_reg_pipeline()
        gs_f = GridSearchCV(pipe_full, param_grid_full, scoring="r2", cv=group_inner_cv(groups_train).split(X_train, y_train, groups=groups_train), n_jobs=-1)
        gs_f.fit(X_train, y_train)

        imp = gs_f.best_estimator_.named_steps["rf"].feature_importances_
        top_idx = np.argsort(imp)[::-1][:K]

        X_tr_top, X_te_top = X_train[:, top_idx], X_test[:, top_idx]

        pipe_p, param_grid_p = make_rf_reg_pipeline()
        gs_p = GridSearchCV(pipe_p, param_grid_p, scoring="r2", cv=group_inner_cv(groups_train).split(X_tr_top, y_train, groups=groups_train), n_jobs=-1)
        gs_p.fit(X_tr_top, y_train)

        y_pred = gs_p.best_estimator_.predict(X_te_top)

        for sid_i, yt, yp in zip(test_ids, y_test, y_pred):
            pred_records.append({"SampleID": str(sid_i), "Donor": str(test_donor), "y_true": float(yt), "y_pred": float(yp)})

    return pd.DataFrame(pred_records)

panel_pred_df = run_nested_panel_lodo_predictions(K=10)
r2_all   = r2_score(panel_pred_df["y_true"], panel_pred_df["y_pred"])
rmse_all = np.sqrt(mean_squared_error(panel_pred_df["y_true"], panel_pred_df["y_pred"]))
mae_all  = mean_absolute_error(panel_pred_df["y_true"], panel_pred_df["y_pred"])

fig, ax = plt.subplots(figsize=(5.2, 4.7))
sns.scatterplot(data=panel_pred_df, x="y_true", y="y_pred", s=45, ax=ax)
sns.regplot(data=panel_pred_df, x="y_true", y="y_pred", scatter=False, ci=None, line_kws={"lw": 1.5, "color": "black"}, ax=ax)
ax.set_xlabel("Observed TsD (days)")
ax.set_ylabel("Predicted TsD (days)")
ax.set_title("Figure 8H Nested Top-10 panel TsD regression (Donor-LODO)")
text_str = f"Nested panel (Top-10)\n$R^2$ = {r2_all:.2f}\nRMSE = {rmse_all:.2f} d\nMAE = {mae_all:.2f} d"
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, ha="left", va="top", bbox=dict(boxstyle="round", fc="white", ec="gray"))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("Figure_8H_TsD_panel_regression_nested.pdf", dpi=600, bbox_inches="tight")
plt.savefig("Figure_8H_TsD_panel_regression_nested.png", dpi=600, bbox_inches="tight")
plt.close()

print("Part 2 execution complete.")