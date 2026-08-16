"""
01_metabolome_feature_selection.py

Two-step feature selection strategy for metabolomics data:
1. Recursive Feature Addition (RFA) via ExtraTreesRegressor.
2. Stepwise Elimination Feature Selection (SEFE) via XGBRegressor.

"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Metabolome Feature Selection Pipeline (RFA + SEIFE)")
    parser.add_argument('-m', '--metabolite', type=str, default='76.txt', help='Path to metabolite intensity file.')
    parser.add_argument('-s', '--sample_meta', type=str, default='metadata.txt', help='Path to metadata file.')
    parser.add_argument('--mock', action='store_true', help='Generate synthetic mock dataset for reproduction testing.')
    return parser.parse_args()


def generate_mock_data():
    """Generates a reproducible synthetic dataset with matching data structures."""
    print("[INFO] Generating synthetic mock dataset for pipeline demonstration...")
    np.random.seed(42)
    
    # Generate Sample IDs matching the experimental design
    donors = ['D1', 'D2', 'D3', 'D4']
    time_points = [('T0', 0), ('T1A', 0.5), ('T1B', 0.5), ('T1C', 0.5), 
                   ('T2A', 1), ('T2B', 1), ('T2C', 1), ('T3A', 2), ('T3B', 2), ('T3C', 2),
                   ('T4A', 7), ('T4B', 7), ('T4C', 7), ('T5A', 14), ('T5B', 14), ('T5C', 14),
                   ('T6A', 28), ('T6B', 28), ('T6C', 28)]
    
    samples = []
    tsd_list = []
    for d in donors:
        for tp, tsd in time_points:
            samples.append(f"{d}{tp}")
            tsd_list.append(tsd)
            
    metadata = pd.DataFrame({'SampleID': samples, 'TsD': tsd_list})
    
    # Generate mock features (120 metabolites across samples)
    n_samples = len(samples)
    n_features = 120
    
    metab_ids = [f"M{i+1:04d}" for i in range(n_features)]
    feature_names = [f"Metabolite_Feature_{i+1}" for i in range(n_features)]
    
    # Synthetic continuous intensity matrix
    data_matrix = np.random.uniform(10, 1000, size=(n_features, n_samples))
    
    df_metab = pd.DataFrame(data_matrix, columns=samples)
    df_metab.insert(0, 'metab_id', metab_ids)
    df_metab.index = feature_names
    
    return df_metab, metadata


def main():
    args = parse_args()

    # Load real or synthetic data
    if args.mock or not (os.path.exists(args.metabolite) and os.path.exists(args.sample_meta)):
        if not args.mock:
            print(f"[WARNING] Input files ({args.metabolite}, {args.sample_meta}) not found. Fallback to mock data.")
        metabolite_data, metadata = generate_mock_data()
    else:
        print(f"[INFO] Loading real dataset from {args.metabolite} and {args.sample_meta}...")
        metabolite_data = pd.read_csv(args.metabolite, sep='\t', index_col=0)
        metadata = pd.read_csv(args.sample_meta, sep='\t')

    # Extract ID mappings and matrix values
    metab_id_mapping = metabolite_data.iloc[:, 0]
    metabolite_values = metabolite_data.iloc[:, 1:]

    # Transpose matrix (Samples x Features)
    X_total = metabolite_values.T
    X_total_clean = X_total.copy()
    X_total_clean.columns = [metab_id_mapping[name] for name in X_total.columns]

    # Target variable: TsD
    Y_total = metadata.set_index('SampleID')['TsD']

    # Align samples
    X_total = X_total.loc[Y_total.index]
    X_total_clean = X_total_clean.loc[Y_total.index]

    print(f"\nData Matrix Dimension: Samples={X_total.shape[0]}, Features={X_total.shape[1]}")
    print(f"TsD Timepoints Range: {Y_total.min()} - {Y_total.max()} Days")

    # ============================================================================
    # Part 1: Recursive Feature Addition (RFA)
    # ============================================================================
    print("\n" + "=" * 60)
    print("Part 1: Recursive Feature Addition (RFA) using ExtraTreesRegressor")
    print("=" * 60)

    reg = ExtraTreesRegressor(random_state=50, n_estimators=100)
    reg.fit(X_total, Y_total)

    feature_importances = reg.feature_importances_
    names = X_total.columns
    importances_ranking = sorted(zip(map(lambda x: round(x, 4), feature_importances), names), reverse=True)

    importance_df = pd.DataFrame(importances_ranking, columns=['importance', 'feature'])
    sorted_features = importance_df['feature'].tolist()
    X_sorted = X_total[sorted_features]

    cv_r2_scores, cv_rmse_scores = [], []
    selected_features, selected_metab_ids, contributions = [], [], []

    max_features_to_test = min(80, len(sorted_features))
    print(f"Testing top {max_features_to_test} features via 10-fold Cross Validation...")

    for i in range(1, max_features_to_test + 1):
        X_subset = X_sorted.iloc[:, :i]
        kf = KFold(n_splits=10, shuffle=True, random_state=10)

        r2_scores = cross_val_score(reg, X_subset, Y_total, cv=kf, scoring='r2')
        rmse_scores = cross_val_score(reg, X_subset, Y_total, cv=kf, scoring='neg_mean_squared_error')

        cv_r2_scores.append(np.mean(r2_scores))
        cv_rmse_scores.append(np.mean(np.sqrt(-rmse_scores)))

        selected_features.append(sorted_features[i - 1])
        selected_metab_ids.append(metab_id_mapping[sorted_features[i - 1]])
        contributions.append(feature_importances[list(names).index(sorted_features[i - 1])])

    feature_selection_rfa = pd.DataFrame({
        'feature_num': range(1, len(cv_r2_scores) + 1),
        'cv_r2': cv_r2_scores,
        'cv_rmse': cv_rmse_scores,
        'selected_features': selected_features,
        'selected_metab_ids': selected_metab_ids,
        'contribution': contributions
    })

    best_r2_idx_rfa = feature_selection_rfa['cv_r2'].idxmax()
    best_r2_rfa = feature_selection_rfa.loc[best_r2_idx_rfa, 'cv_r2']
    best_feature_num_rfa = best_r2_idx_rfa + 1
    print(f"RFA Best R² Score: {best_r2_rfa:.4f} with Top {best_feature_num_rfa} Features.")

    # Viz 1: RFA Feature Importance
    plt.figure(figsize=(15, 6))
    top_n = min(50, len(feature_selection_rfa))
    top_features_rfa = feature_selection_rfa.head(top_n)

    plt.bar(top_features_rfa['selected_metab_ids'], top_features_rfa['contribution'], color='orange')
    plt.ylabel('Importance score', fontsize=18, fontweight='bold')
    plt.xlabel('Metabolite ID', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, rotation=90, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("RFA_Importance_score.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Viz 2: RFA Feature Curve
    plt.figure(figsize=(15, 6))
    plt.plot(feature_selection_rfa['feature_num'], feature_selection_rfa['cv_r2'],
             color='orange', marker='o', linestyle='--', markersize=6)
    plt.scatter(best_feature_num_rfa, best_r2_rfa, color='green', s=180, marker='*')
    plt.xlabel('Number of Features', fontsize=18, fontweight='bold')
    plt.ylabel('Average R² (10-fold CV)', fontsize=18, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("RFA_Number_of_Features.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # ============================================================================
    # Part 2: Stepwise Elimination Feature Selection (SEFE)
    # ============================================================================
    print("\n" + "=" * 60)
    print("Part 2: Stepwise Elimination Feature Selection (SEFE) via XGBoost")
    print("=" * 60)

    top_k_for_sefe = min(best_feature_num_rfa, len(feature_selection_rfa))
    rfa_selected_metab_ids = feature_selection_rfa['selected_metab_ids'].head(top_k_for_sefe).tolist()
    rfa_selected_features = feature_selection_rfa['selected_features'].head(top_k_for_sefe).tolist()

    metab_id_to_name = {m_id: name for name, m_id in zip(rfa_selected_features, rfa_selected_metab_ids)}
    X_rfa_selected_clean = X_total_clean[rfa_selected_metab_ids]

    kf = KFold(n_splits=10, shuffle=True, random_state=10)

    def evaluate_model_regression(X, Y):
        all_y_true, all_y_pred = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            model = XGBRegressor(random_state=50, n_estimators=50, max_depth=6, learning_rate=0.1)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)

            all_y_true.extend(Y_test)
            all_y_pred.extend(y_pred)

        r2 = r2_score(all_y_true, all_y_pred)
        rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
        return r2, rmse

    initial_r2, initial_rmse = evaluate_model_regression(X_rfa_selected_clean, Y_total)
    print(f"Initial Baseline Model - R²: {initial_r2:.4f}, RMSE: {initial_rmse:.4f}")

    model_xgb = XGBRegressor(random_state=50, n_estimators=50, max_depth=6, learning_rate=0.1)
    model_xgb.fit(X_rfa_selected_clean, Y_total)

    feature_importances_xgb = model_xgb.feature_importances_
    sorted_idx = np.argsort(feature_importances_xgb)

    best_columns_metab = list(X_rfa_selected_clean.columns)
    best_columns_names = rfa_selected_features[:]
    current_r2 = initial_r2

    for i in sorted_idx:
        metab_id_to_remove = X_rfa_selected_clean.columns[i]
        current_columns_metab = [col for col in best_columns_metab if col != metab_id_to_remove]

        if len(current_columns_metab) == 0:
            break

        X_current = X_rfa_selected_clean[current_columns_metab]
        new_r2, new_rmse = evaluate_model_regression(X_current, Y_total)

        if new_r2 > current_r2:
            current_r2 = new_r2
            best_columns_metab = current_columns_metab
            best_columns_names = [metab_id_to_name[col] for col in best_columns_metab]

    # ============================================================================
    # Part 3: Final Model Evaluation & Exports
    # ============================================================================
    print("\n" + "=" * 60)
    print("Part 3: Final Feature Panel Evaluation")
    print("=" * 60)

    X_final = X_total[best_columns_names]
    reg_final = ExtraTreesRegressor(random_state=50, n_estimators=100)
    reg_final.fit(X_final, Y_total)

    feature_importances_final = reg_final.feature_importances_
    names_final = X_final.columns

    important_metabolites_final = pd.DataFrame({
        'Metabolite_Name': best_columns_names,
        'Metabolite_ID': best_columns_metab,
        'Importance_Score': [feature_importances_final[list(names_final).index(feat)] for feat in best_columns_names]
    }).sort_values('Importance_Score', ascending=False)

    important_metabolites_final.to_csv('Final_Important_Metabolites_RFA_SEFE.csv', index=False)
    print(f"Final Selected Panel Size: {len(best_columns_names)} metabolites.")
    print(f"Exported panel result to 'Final_Important_Metabolites_RFA_SEFE.csv'.")
    print("Pipeline execution completed successfully.")


if __name__ == '__main__':
    main()