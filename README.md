# Semen-TsD-Multiomics
Source code and analysis pipeline for semen time since deposition (TsD) inference using microbiome and metabolomics data.

# Forensic Semen TsD Inference Pipeline

This repository contains the complete computational workflow and analysis source code for the manuscript: 
**"Inference of Time Since Deposition (TsD) of Human Semen Stains Using Integrated Microbiome and Metabolomics Approaches under Diverse Environmental Conditions"**.

## Directory & Script Overview

- `01_metabolome_feature_selection.py`: Two-step feature selection (RFA + SEIFE) and hyperparameter tuning (BayesSearchCV) for single-omics models.
- `02_multiomics_integration.py`: Comparative pipeline for Early, Latent (PLS), and Late (Stacking) multi-omics integration models under LODO-CV.
- `03_environmental_correction_LMM.R`: Env-LOO cross-validation, PCA projections, Linear Mixed-Effects Models (lme4/lmerTest), and 2D Bias Landscape interpolation (akima).
- `04_donor_lodo_shap_panel.py`: Nested feature selection (Top-K), SHAP interpretability, window classification (0-24h, 1-7d, >7d), and statistical evaluations.
- `05_picrust2_functional_sankey.R`: PICRUSt2 functional transformation, pathway enrichment (ORA), topology analysis, and interactive Sankey diagram construction (networkD3).

## Environment & Dependencies

### Python (v3.10+)
- `scikit-learn` == 1.5.2
- `scikit-optimize` == 0.10.2
- `statsmodels` == 0.14.5
- `xgboost`, `lightgbm`, `catboost`, `shap`

### R (v4.2+)
- `lme4`, `lmerTest`, `akima` (v0.6.3.6)
- `networkD3` (v0.4.1), `htmlwidgets` (v1.6.4), `ggplot2`

## Reproducibility Notice
All stochastic machine learning iterations and data split folds are initialized with a fixed random seed (`random_state = 42`).
