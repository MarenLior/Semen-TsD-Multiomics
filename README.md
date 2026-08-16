# Semen-TsD-Multiomics
Source code and analysis pipeline for semen time since deposition (TsD) inference using microbiome and metabolomics data.

# Forensic Semen TsD Inference Pipeline

This repository contains the complete computational workflow and analysis source code for the manuscript: 
**"Inference of Time Since Deposition (TsD) of Human Semen Stains Using Integrated Microbiome and Metabolomics Approaches under Diverse Environmental Conditions"**.

## Directory & Script Overview

- `01_metabolome_feature_selection.py`: Two-step feature selection (Recursive Feature Addition / RFA + SEIFE algorithm) for identifying key metabolomic markers.
- `02_metabolome_model_training.py`: Single-omics machine learning model training, Bayesian hyperparameter optimization (BayesSearchCV), and performance evaluation.
- `03_multiomics_integration.py`: Comparative evaluation of Early, Latent (PLS), and Late (Stacking) multi-omics integration models using Leave-One-Donor-Out (LODO) cross-validation.
- `04_environment_correction_lmm.R`: Env-LOO cross-validation, PCA projections, Linear Mixed-Effects Models (lme4/lmerTest), and 2D Bias Landscape interpolation (akima).
- `05_donor_lodo_markers_shap_roc1.py`: Part 1 of Donor-LODO analysis — leakage-free preprocessing, RF regression, environment-stratified diagnostics, post-regression time-window classification (0–24h, 1–7d, >7d), direct multi-class ROC analysis, and scenario sensitivity testing.
- `05_donor_lodo_markers_shap_roc2.R`: Part 2 of Donor-LODO analysis — binary marker contribution analysis (Mann–Whitney U test with over/underrepresented directionality), SHAP summary interpretability, and nested Top-K panel evaluation.

## Data Availability Statement

Due to raw signal confidentiality and repository compliance, raw biological data are deposited in recognized global public repositories and are not directly hosted inside this code repository:

- **16S rRNA Microbiome Sequencing Data**: NCBI Sequence Read Archive (SRA) BioProject [PRJNA1433698](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1433698).
- **Untargeted Metabolomics Data**: EMBL-EBI MetaboLights database with accession [MTBLS14027](https://www.ebi.ac.uk/metabolights/MTBLS14027). 
