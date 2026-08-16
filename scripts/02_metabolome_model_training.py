#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_metabolome_model_training.py

Metabolomics-Based Post-Mortem / Post-Ex Vivo Interval (TsD) Prediction Model Training.
Evaluates multiple machine learning algorithms using Leave-One-Donor-Out Cross-Validation (LODO-CV)
and Bayesian Optimization for Hyperparameter Tuning.

"""

import argparse
import os
import sys
import warnings
from typing import Dict, Any, Tuple, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
import xgboost as xgb

# Global Warnings Configuration
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set Global Plotting Style for Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train machine learning models with Bayesian Optimization for TsD prediction."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="Final_Important_Metabolites_31.txt",
        help="Path to the feature matrix file (e.g., Final_Important_Metabolites_31.txt)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.txt",
        help="Path to the metadata file (e.g., metadata.txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save output figures and metrics CSV files"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="Number of iterations for Bayesian optimization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_and_preprocess_data(features_path: str, metadata_path: str):
    """Load metabolomics feature matrix and metadata, then align samples."""
    print("Loading datasets...")

    # Auto-detect separator for robustness
    metabolites_df = pd.read_csv(features_path, sep=r'\s+|,', engine='python', index_col=0)
    metadata_df = pd.read_csv(metadata_path, sep=r'\s+|,', engine='python')

    # Transpose metabolite matrix (Samples in rows, Features in columns)
    X_df = metabolites_df.T
    X_df.index.name = 'SampleID'
    X_df = X_df.reset_index()

    # Merge features with metadata
    data = pd.merge(X_df, metadata_df, on='SampleID', how='inner')

    # Extract Donor metadata if not explicitly provided
    if 'Donor' not in data.columns:
        data['Donor'] = data['SampleID'].str.extract(r'(D\d+)')[0]
    if 'Timepoint' not in data.columns:
        data['Timepoint'] = data['SampleID'].str.extract(r'T(\d+)')[0]
    if 'Replicate' not in data.columns:
        data['Replicate'] = data['SampleID'].str.extract(r'([A-C])$')[0]

    print(f"Dataset successfully loaded.")
    print(f"Total Samples: {data.shape[0]}")
    print(f"Total Donors: {data['Donor'].nunique()} ({list(data['Donor'].unique())})")
    print(f"Timepoints (TsD): {sorted(data['TsD'].unique())}")

    # Extract feature columns
    excluded_cols = ['SampleID', 'TsD', 'Donor', 'Timepoint', 'Replicate']
    feature_cols = [col for col in data.columns if col not in excluded_cols]
    print(f"Extracted Features Count: {len(feature_cols)}")

    X = data[feature_cols].values
    y = data['TsD'].values
    groups = data['Donor'].values
    sample_ids = data['SampleID'].values

    return X, y, groups, sample_ids, data, feature_cols


def define_models_and_spaces(seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Define machine learning models and hyperparameter search spaces."""
    models = {
        "RF": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "KNN": KNeighborsRegressor(n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(eval_metric="rmse", random_state=seed, n_jobs=-1),
        "CatBoost": CatBoostRegressor(verbose=0, random_seed=seed, allow_writing_files=False),
        "SVM": SVR(),
        "LightGBM": LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1)
    }

    param_spaces = {
        "RF": {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2'])
        },
        "KNN": {
            'n_neighbors': Integer(3, 15),
            'weights': Categorical(['uniform', 'distance']),
            'p': Integer(1, 2)
        },
        "XGBoost": {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.2, prior='uniform'),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
            'gamma': Real(0, 5, prior='uniform'),
            'reg_alpha': Real(0, 5, prior='uniform'),
            'reg_lambda': Real(0, 5, prior='uniform')
        },
        "CatBoost": {
            'iterations': Integer(50, 300),
            'depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.2, prior='uniform'),
            'l2_leaf_reg': Real(1, 10, prior='uniform')
        },
        "SVM": {
            'C': Real(0.1, 100, prior='log-uniform'),
            'gamma': Categorical(['scale', 'auto']),
            'kernel': Categorical(['linear', 'rbf']),
            'epsilon': Real(0.01, 0.5, prior='uniform')
        },
        "LightGBM": {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.1, prior='uniform'),
            'num_leaves': Integer(20, 100),
            'max_depth': Integer(3, 10),
            'min_child_samples': Integer(5, 50),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.5, 1.0, prior='uniform')
        }
    }

    return models, param_spaces


def train_with_bayes_search(X_train, y_train, model, param_space, n_iter=15, cv=5, seed=42):
    """Execute Bayesian search CV for parameter optimization."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=kfold,
        n_jobs=-1,
        verbose=0,
        scoring='neg_mean_squared_error',
        random_state=seed
    )

    opt.fit(X_train, y_train)

    cv_r2_scores = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        best_clf = opt.best_estimator_
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        best_clf.fit(X_tr, y_tr)
        y_pred = best_clf.predict(X_val)
        cv_r2_scores.append(r2_score(y_val, y_pred))

    avg_cv_r2 = float(np.mean(cv_r2_scores)) if cv_r2_scores else 0.0
    return opt.best_estimator_, opt.best_params_, avg_cv_r2, cv_r2_scores


def run_lodo_cv(X, y, groups, sample_ids, models, param_spaces, n_iter=15, seed=42):
    """Execute Leave-One-Donor-Out Cross Validation."""
    print("\n" + "=" * 65)
    print(" Starting Leave-One-Donor-Out Cross-Validation (LODO-CV)")
    print("=" * 65)

    logo = LeaveOneGroupOut()
    model_results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating Model: [{model_name}]...")

        model_fold_predictions = []
        model_fold_train_r2, model_fold_train_rmse, model_fold_train_mae = [], [], []
        model_fold_test_r2, model_fold_test_rmse, model_fold_test_mae = [], [], []
        model_fold_cv_r2 = []
        model_fold_best_params = []

        fold = 0
        for train_idx, test_idx in logo.split(X, y, groups):
            fold += 1
            test_donor = groups[test_idx][0]
            print(f"  --> Fold {fold}: Test Donor = {test_donor}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardization inside CV loop to avoid data leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                best_model, best_params, cv_r2, _ = train_with_bayes_search(
                    X_train_scaled, y_train, model, param_spaces[model_name], n_iter=n_iter, seed=seed
                )
                model_fold_cv_r2.append(cv_r2)
                model_fold_best_params.append(best_params)
            except Exception as e:
                print(f"      Optimization fallback trigger: {str(e)[:80]}")
                best_model = model
                best_model.fit(X_train_scaled, y_train)
                model_fold_cv_r2.append(0.0)
                model_fold_best_params.append({})

            y_train_pred = best_model.predict(X_train_scaled)
            y_test_pred = best_model.predict(X_test_scaled)

            # Metrics calculation
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)

            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)

            model_fold_train_r2.append(train_r2)
            model_fold_train_rmse.append(train_rmse)
            model_fold_train_mae.append(train_mae)

            model_fold_test_r2.append(test_r2)
            model_fold_test_rmse.append(test_rmse)
            model_fold_test_mae.append(test_mae)

            for i, idx in enumerate(test_idx):
                model_fold_predictions.append({
                    'Model': model_name,
                    'SampleID': sample_ids[idx],
                    'Donor': groups[idx],
                    'Actual': y_test[i],
                    'Predicted': y_test_pred[i],
                    'Fold': fold,
                    'TestDonor': test_donor
                })

            print(f"      Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

        model_predictions_df = pd.DataFrame(model_fold_predictions)

        # Overall dataset metrics
        overall_test_r2 = r2_score(model_predictions_df['Actual'], model_predictions_df['Predicted'])
        overall_test_rmse = np.sqrt(
            mean_squared_error(model_predictions_df['Actual'], model_predictions_df['Predicted']))
        overall_test_mae = mean_absolute_error(model_predictions_df['Actual'], model_predictions_df['Predicted'])

        model_results[model_name] = {
            'Overall_Train_R2': np.mean(model_fold_train_r2),
            'Overall_Train_RMSE': np.mean(model_fold_train_rmse),
            'Overall_Train_MAE': np.mean(model_fold_train_mae),
            'Overall_Test_R2': overall_test_r2,
            'Overall_Test_RMSE': overall_test_rmse,
            'Overall_Test_MAE': overall_test_mae,
            'Avg_Fold_Test_R2': np.mean(model_fold_test_r2),
            'Avg_Fold_Test_RMSE': np.mean(model_fold_test_rmse),
            'Avg_Fold_Test_MAE': np.mean(model_fold_test_mae),
            'Fold_Test_R2': model_fold_test_r2,
            'Fold_Test_RMSE': model_fold_test_rmse,
            'Fold_CV_R2': model_fold_cv_r2,
            'Fold_Best_Params': model_fold_best_params,
            'Predictions': model_predictions_df
        }

    return model_results


def plot_model_comparison(df_perf: pd.DataFrame, best_model_name: str, output_dir: str):
    """Plot multi-metric bar charts for model benchmark comparison."""
    models_list = ["Bayes-\n" + m for m in df_perf['Model']]
    train_r2 = df_perf['Train_R2'].values
    test_r2 = df_perf['Test_R2'].values
    cv_r2 = df_perf['Avg_Fold_CV_R2'].values
    train_rmse = df_perf['Train_RMSE'].values
    test_rmse = df_perf['Test_RMSE'].values

    fig, ax1 = plt.subplots(figsize=(18, 9), dpi=300)
    ax2 = ax1.twinx()

    x = np.arange(len(models_list))
    width = 0.15
    offsets = [-2, -1, 0, 1, 2]

    c_train_r2, c_test_r2, c_cv_r2 = '#2EB8B1', '#6BD4C8', '#429E9D'
    c_train_rmse, c_test_rmse = '#FF6F61', '#FFAD60'

    rects1 = ax1.bar(x + offsets[0] * width, train_r2, width, label='Train R²', color=c_train_r2, edgecolor='white')
    rects2 = ax1.bar(x + offsets[1] * width, test_r2, width, label='Test R²', color=c_test_r2, edgecolor='white')
    rects3 = ax1.bar(x + offsets[2] * width, cv_r2, width, label='CV R²', color=c_cv_r2, edgecolor='white')

    rects4 = ax2.bar(x + offsets[3] * width, train_rmse, width, label='Train RMSE', color=c_train_rmse, alpha=0.9,
                     edgecolor='white')
    rects5 = ax2.bar(x + offsets[4] * width, test_rmse, width, label='Test RMSE', color=c_test_rmse, hatch='///',
                     edgecolor='white')

    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel('R²', fontsize=20, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='both', labelsize=16)

    rmse_max = max(train_rmse.max(), test_rmse.max())
    ax2.set_ylim(0, rmse_max * 1.35)
    ax2.set_ylabel('RMSE (days)', fontsize=20, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='y', labelsize=16)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, fontsize=16, fontweight='bold')
    ax1.set_xlim(-0.6, len(models_list) - 0.4)

    def autolabel(rects, ax, is_rmse=False, color='black'):
        for rect in rects:
            height = rect.get_height()
            val_text = f'{height:.3f}' if not is_rmse else f'{height:.2f}'
            xy_pos = (rect.get_x() + rect.get_width() / 2, height)
            text_color = color if not is_rmse else '#D81B60'
            ax.annotate(val_text, xy=xy_pos, xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color=text_color)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)
    autolabel(rects4, ax2, is_rmse=True)
    autolabel(rects5, ax2, is_rmse=True, color='#D81B60')

    # Highlight optimal model box
    try:
        clean_name = best_model_name.replace("Bayes-\n", "")
        idx = df_perf[df_perf['Model'] == clean_name].index[0]
        x_left = (idx + offsets[0] * width) - width / 2 - 0.05
        x_right = (idx + offsets[4] * width) + width / 2 + 0.05
        rect = patches.Rectangle((x_left, 0), x_right - x_left, 1.10,
                                 linewidth=2, edgecolor='#6495ED', facecolor='none',
                                 linestyle='--', alpha=0.8)
        ax1.add_patch(rect)
    except IndexError:
        pass

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False,
               prop={'size': 16, 'weight': 'bold'}, columnspacing=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    save_path = os.path.join(output_dir, "Model_Performance_Comparison.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_best_model_scatter(best_predictions: pd.DataFrame, best_model_name: str,
                            metrics: Dict[str, float], output_dir: str, seed: int = 42):
    """Plot Joint scatter and density plot for the optimal prediction model."""
    title = f'Bayes-{best_model_name} (Leave-One-Donor-Out)'

    np.random.seed(seed)
    jitter_amount = 0.15

    y_test_actual = best_predictions['Actual'].values
    y_test_pred = best_predictions['Predicted'].values

    y_test_actual_jittered = y_test_actual + np.random.normal(0, jitter_amount, len(y_test_actual))
    y_test_pred_jittered = y_test_pred + np.random.normal(0, jitter_amount, len(y_test_pred))

    g = sns.JointGrid()

    sns.scatterplot(x=y_test_actual_jittered, y=y_test_pred_jittered, s=90, color='#4B61D1',
                    ax=g.ax_joint, alpha=0.85, edgecolor='black', linewidth=0.8, label=f'Test (n={len(y_test_actual)})')

    sns.regplot(x=y_test_actual, y=y_test_pred, scatter=False, ci=None, color='grey',
                ax=g.ax_joint, line_kws={'linewidth': 2, 'linestyle': '--'})

    sns.histplot(x=y_test_actual, ax=g.ax_marg_x, color='#4B61D1', edgecolor='white', kde=True, bins=10)
    sns.histplot(y=y_test_pred, ax=g.ax_marg_y, color='#4B61D1', edgecolor='white', kde=True, bins=10)

    max_val = max(np.max(y_test_actual), np.max(y_test_pred))
    g.ax_joint.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Perfect Line')

    g.set_axis_labels("Actual TsD (days)", "Predicted TsD (days)", fontsize=16, fontweight='bold')
    g.ax_joint.tick_params(axis='both', labelsize=14)
    g.ax_joint.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.8)
    g.fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    g.ax_joint.text(
        0.95, 0.05,
        f"Train R²: {metrics['train_r2']:.4f}\n"
        f"Train RMSE: {metrics['train_rmse']:.2f} d\n"
        f"Train MAE: {metrics['train_mae']:.2f} d\n"
        f"Test R²: {metrics['test_r2']:.4f}\n"
        f"Test RMSE: {metrics['test_rmse']:.2f} d\n"
        f"Test MAE: {metrics['test_mae']:.2f} d",
        fontsize=11, ha='right', va='bottom', transform=g.ax_joint.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', edgecolor='#CCCCCC', facecolor='white', alpha=0.9)
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"Best_Model_Bayes_{best_model_name}_LODO.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Scatter plot saved to: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    X, y, groups, sample_ids, data, feature_cols = load_and_preprocess_data(args.features, args.metadata)

    # 2. Setup Models
    models, param_spaces = define_models_and_spaces(seed=args.seed)

    # 3. Execution LODO-CV Optimization
    model_results = run_lodo_cv(
        X, y, groups, sample_ids, models, param_spaces, n_iter=args.n_iter, seed=args.seed
    )

    # 4. Summarize Benchmark Results
    performance_summary = []
    for model_name in models.keys():
        res = model_results[model_name]
        performance_summary.append({
            'Model': model_name,
            'Train_R2': res['Overall_Train_R2'],
            'Train_RMSE': res['Overall_Train_RMSE'],
            'Train_MAE': res['Overall_Train_MAE'],
            'Test_R2': res['Overall_Test_R2'],
            'Test_RMSE': res['Overall_Test_RMSE'],
            'Test_MAE': res['Overall_Test_MAE'],
            'Avg_Fold_Test_R2': np.mean(res['Fold_Test_R2']),
            'Std_Fold_Test_R2': np.std(res['Fold_Test_R2']),
            'Avg_Fold_CV_R2': np.mean(res['Fold_CV_R2'])
        })

    df_perf = pd.DataFrame(performance_summary).sort_values('Test_R2', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print(" Performance Summary (Ranked by Test R²)")
    print("=" * 65)
    print(df_perf[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Train_R2', 'Avg_Fold_Test_R2']].to_string(index=False))

    best_model_name = df_perf.iloc[0]['Model']
    print(f"\nOptimal Model Selected: [{best_model_name}] (Test R² = {df_perf.iloc[0]['Test_R2']:.4f})")

    # 5. Save Summary Tables
    perf_csv = os.path.join(args.output_dir, "Model_Performance_LODO.csv")
    df_perf.to_csv(perf_csv, index=False)

    best_preds_df = model_results[best_model_name]['Predictions']
    preds_csv = os.path.join(args.output_dir, f"Best_Model_{best_model_name}_Predictions.csv")
    best_preds_df.to_csv(preds_csv, index=False)

    # 6. Plotting
    plot_model_comparison(df_perf, best_model_name, args.output_dir)

    metrics = {
        'train_r2': model_results[best_model_name]['Overall_Train_R2'],
        'train_rmse': model_results[best_model_name]['Overall_Train_RMSE'],
        'train_mae': model_results[best_model_name]['Overall_Train_MAE'],
        'test_r2': df_perf.iloc[0]['Test_R2'],
        'test_rmse': df_perf.iloc[0]['Test_RMSE'],
        'test_mae': df_perf.iloc[0]['Test_MAE']
    }
    plot_best_model_scatter(best_preds_df, best_model_name, metrics, args.output_dir, seed=args.seed)

    print(f"\nAll task files successfully saved to: '{os.path.abspath(args.output_dir)}'")


if __name__ == "__main__":
    main()