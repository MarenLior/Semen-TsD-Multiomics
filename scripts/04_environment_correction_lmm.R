#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
"""
Script Name: 04_environment_correction_lmm.R
Description: Env-LOO Cross-Validation & Linear Mixed Models (LMM) for
             Environmental Correction in TsD (Time Since Deposition) Inference.

Models Evaluated:
  - m0       : TsD ~ PC1 + PC2 + (1|Donor)                            (Baseline omics model)
  - m_env    : TsD ~ PC1 + PC2 + Temperature + Humidity + (1|Donor)  (Deployable correction model)
  - m_oracle : TsD ~ PC1 + PC2 + ADD_true + Humidity + (1|Donor)     (Oracle upper bound)
  - m_add    : ADD_true ~ PC1 + PC2 + (1|Donor)                       (Biological-time target)

Outputs:
  - Figure7_EnvLOO_DeepCorrection.pdf / .png
  - Table_S_envLOO_predictions_deep.csv
  - Table_S_envLOO_model_comparison_by_env_deep.csv
  - Table_S_envLOO_fixed_effects_m_env.csv

"""

# Clear workspace environment
rm(list = ls())

# Load required libraries quietly
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(lme4)
  library(lmerTest)
  library(ggplot2)
  library(viridis)
  library(patchwork)
  library(akima)
})

set.seed(2025)

# ===============================================================
# 0) Input Files & Directory Verification
# ===============================================================
meta_file  <- "metadata.txt"
genus_file <- "28 genus.txt"
metab_file <- "Final_Important_Metabolites_31.txt"

missing_files <- c(meta_file, genus_file, metab_file)[!file.exists(c(meta_file, genus_file, metab_file))]
if (length(missing_files) > 0) {
  stop("[Error] Required input files missing in working directory: ", paste(missing_files, collapse = ", "))
}

# ===============================================================
# 1) Metadata Processing & ADD_true Calculation
# ===============================================================
meta <- read.delim(meta_file, header = TRUE, sep = "\t", check.names = FALSE)

req_cols <- c("SampleID", "Env", "Donor", "Temperature", "Humidity", "TsD")
missing_cols <- setdiff(req_cols, colnames(meta))
if (length(missing_cols) > 0) {
  stop("[Error] metadata.txt is missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Calculate Accumulated Degree Days (ADD_true) using trapezoidal integration per Donor-Env group
meta <- meta %>%
  mutate(
    Env         = as.factor(Env),
    Donor       = as.factor(Donor),
    Temperature = as.numeric(Temperature),
    Humidity    = as.numeric(Humidity),
    TsD         = as.numeric(TsD)
  ) %>%
  group_by(Donor, Env) %>%
  arrange(TsD, .by_group = TRUE) %>%
  mutate(
    delta_t  = c(0, diff(TsD)),
    Temp_lag = dplyr::lag(Temperature, default = first(Temperature)),
    Temp_avg = (Temp_lag + Temperature) / 2,
    ADD_true = cumsum(Temp_avg * delta_t)
  ) %>%
  ungroup() %>%
  select(-Temp_lag, -Temp_avg)

# ===============================================================
# 2) Omics Data Loading & Alignment
# ===============================================================
genus_raw <- read.delim(genus_file, sep = "\t", header = TRUE, check.names = FALSE)
metab_raw <- read.delim(metab_file, sep = "\t", header = TRUE, check.names = FALSE)

if (!("genus" %in% colnames(genus_raw))) stop("[Error] '28 genus.txt' must contain a 'genus' column.")
if (!("Metabolite" %in% colnames(metab_raw))) stop("[Error] 'Final_Important_Metabolites_31.txt' must contain a 'Metabolite' column.")

genus_names <- genus_raw$genus
metab_names <- metab_raw$Metabolite

genus_wide <- genus_raw %>%
  column_to_rownames("genus") %>%
  t() %>% as.data.frame() %>%
  rownames_to_column("SampleID")

metab_wide <- metab_raw %>%
  column_to_rownames("Metabolite") %>%
  t() %>% as.data.frame() %>%
  rownames_to_column("SampleID")

full <- meta %>%
  inner_join(genus_wide, by = "SampleID") %>%
  inner_join(metab_wide, by = "SampleID")

message(sprintf("[Data I/O] Aligned Samples: %d | Microbiome Genera: %d | Metabolites: %d",
                nrow(full), length(genus_names), length(metab_names)))

# ===============================================================
# 3) Helper Functions for Preprocessing & Metrics
# ===============================================================
to_num_mat <- function(df) {
  m <- sapply(df, function(x) as.numeric(as.character(x)))
  m <- as.matrix(m)
  colnames(m) <- colnames(df)
  m
}

# Centered Log-Ratio (Sample-wise; safe from CV leakage)
clr_transform <- function(X, pseudo = 1e-6) {
  X <- as.matrix(X)
  X[is.na(X)] <- 0
  X <- X + pseudo
  rs <- rowSums(X); rs[rs == 0] <- 1
  X <- X / rs
  logX <- log(X)
  sweep(logX, 1, rowMeans(logX), "-")
}

impute_median_with_ref <- function(X, ref_medians) {
  X <- as.matrix(X)
  for (j in seq_len(ncol(X))) X[is.na(X[, j]), j] <- ref_medians[j]
  X
}

fit_log1p_zscore <- function(X_train_raw) {
  X_train_raw <- as.matrix(X_train_raw)
  meds <- apply(X_train_raw, 2, median, na.rm = TRUE)
  X_imp <- impute_median_with_ref(X_train_raw, meds)
  X_log <- log1p(X_imp)
  mu <- colMeans(X_log)
  sdv <- apply(X_log, 2, sd); sdv[sdv == 0 | is.na(sdv)] <- 1
  X_z <- sweep(sweep(X_log, 2, mu, "-"), 2, sdv, "/")
  list(X_train = X_z, medians = meds, mu = mu, sd = sdv)
}

apply_log1p_zscore <- function(X_test_raw, meds, mu, sdv) {
  X_test_raw <- as.matrix(X_test_raw)
  X_imp <- impute_median_with_ref(X_test_raw, meds)
  X_log <- log1p(X_imp)
  sdv[sdv == 0 | is.na(sdv)] <- 1
  sweep(sweep(X_log, 2, mu, "-"), 2, sdv, "/")
}

fit_zscore <- function(X_train) {
  X_train <- as.matrix(X_train)
  mu <- colMeans(X_train)
  sdv <- apply(X_train, 2, sd); sdv[sdv == 0 | is.na(sdv)] <- 1
  X_z <- sweep(sweep(X_train, 2, mu, "-"), 2, sdv, "/")
  list(X_train = X_z, mu = mu, sd = sdv)
}

apply_zscore <- function(X_test, mu, sdv) {
  X_test <- as.matrix(X_test)
  sdv[sdv == 0 | is.na(sdv)] <- 1
  sweep(sweep(X_test, 2, mu, "-"), 2, sdv, "/")
}

safe_rmse <- function(err) sqrt(mean(err^2, na.rm = TRUE))
safe_mae  <- function(err) mean(abs(err), na.rm = TRUE)

# ===============================================================
# 4) Env-LOO Cross-Validation & LMM Fitting
# ===============================================================
env_levels <- sort(unique(full$Env))
pred_list <- list()
comp_list <- list()
coef_list <- list()

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

message("\n--- Running Env-LOO Cross-Validation ---")
for (e in env_levels) {
  
  train <- full %>% filter(Env != e)
  test  <- full %>% filter(Env == e)
  
  # 4.1 Omics Preprocessing (Train-fit / Test-apply)
  Xmicro_train_raw <- to_num_mat(train %>% select(all_of(genus_names)))
  Xmicro_test_raw  <- to_num_mat(test  %>% select(all_of(genus_names)))
  
  Xmicro_train_clr <- clr_transform(Xmicro_train_raw)
  Xmicro_test_clr  <- clr_transform(Xmicro_test_raw)
  
  micro_scale     <- fit_zscore(Xmicro_train_clr)
  Xmicro_train_z  <- micro_scale$X_train
  Xmicro_test_z   <- apply_zscore(Xmicro_test_clr, micro_scale$mu, micro_scale$sd)
  
  Xmetab_train_raw <- to_num_mat(train %>% select(all_of(metab_names)))
  Xmetab_test_raw  <- to_num_mat(test  %>% select(all_of(metab_names)))
  
  metab_scale     <- fit_log1p_zscore(Xmetab_train_raw)
  Xmetab_train_z  <- metab_scale$X_train
  Xmetab_test_z   <- apply_log1p_zscore(
    Xmetab_test_raw,
    meds = metab_scale$medians,
    mu   = metab_scale$mu,
    sdv  = metab_scale$sd
  )
  
  # 4.2 PCA Inside Fold (Train-fit / Test-project)
  X_train_omics <- cbind(Xmicro_train_z, Xmetab_train_z)
  X_test_omics  <- cbind(Xmicro_test_z,  Xmetab_test_z)
  
  pca <- prcomp(X_train_omics, center = FALSE, scale. = FALSE)
  
  PC_train <- predict(pca, newdata = X_train_omics)
  PC_test  <- predict(pca, newdata = X_test_omics)
  
  train$PC1 <- PC_train[, 1]; train$PC2 <- PC_train[, 2]
  test$PC1  <- PC_test[, 1];  test$PC2  <- PC_test[, 2]
  
  # 4.3 Fit Linear Mixed Models (LMMs)
  m0       <- lmer(TsD ~ PC1 + PC2 + (1|Donor), data = train, REML = TRUE, control = ctrl)
  m_env    <- lmer(TsD ~ PC1 + PC2 + Temperature + Humidity + (1|Donor), data = train, REML = TRUE, control = ctrl)
  m_oracle <- lmer(TsD ~ PC1 + PC2 + ADD_true + Humidity + (1|Donor), data = train, REML = TRUE, control = ctrl)
  m_add    <- lmer(ADD_true ~ PC1 + PC2 + (1|Donor), data = train, REML = TRUE, control = ctrl)
  
  # 4.4 Fixed-Effect Prediction Only (for forensic deployment to unknown donors)
  test$Pred_m0       <- predict(m0,       newdata = test, re.form = NA, allow.new.levels = TRUE)
  test$Pred_m_env    <- predict(m_env,    newdata = test, re.form = NA, allow.new.levels = TRUE)
  test$Pred_m_oracle <- predict(m_oracle, newdata = test, re.form = NA, allow.new.levels = TRUE)
  test$ADD_hat       <- predict(m_add,    newdata = test, re.form = NA, allow.new.levels = TRUE)
  
  # Compute Bias (Observed - Predicted)
  test$Bias_m0       <- test$TsD - test$Pred_m0
  test$Bias_m_env    <- test$TsD - test$Pred_m_env
  test$Bias_m_oracle <- test$TsD - test$Pred_m_oracle
  test$ADD_err       <- test$ADD_true - test$ADD_hat
  
  # 4.5 Metrics Summary per Environment
  comp <- tibble(
    Env            = as.character(e),
    n              = nrow(test),
    RMSE_m0        = safe_rmse(test$Bias_m0),
    MAE_m0         = safe_mae(test$Bias_m0),
    RMSE_m_env     = safe_rmse(test$Bias_m_env),
    MAE_m_env      = safe_mae(test$Bias_m_env),
    RMSE_m_oracle  = safe_rmse(test$Bias_m_oracle),
    MAE_m_oracle   = safe_mae(test$Bias_m_oracle),
    ADD_RMSE       = safe_rmse(test$ADD_err),
    ADD_MAE        = safe_mae(test$ADD_err)
  )
  
  # Fixed Effects Summary
  cf <- summary(m_env)$coefficients
  pcol <- if ("Pr(>|t|)" %in% colnames(cf)) "Pr(>|t|)" else NA
  
  coef_list[[as.character(e)]] <- tibble(
    Train_excluding_env = as.character(e),
    term                = rownames(cf),
    estimate            = cf[, 1],
    se                  = cf[, 2],
    p                   = if (!is.na(pcol)) cf[, pcol] else NA_real_
  )
  
  pred_list[[as.character(e)]] <- test
  comp_list[[as.character(e)]] <- comp
  
  message(sprintf("Left-out Env: %-10s | RMSE(m0): %6.3f | RMSE(m_env): %6.3f | RMSE(m_oracle): %6.3f",
                  e, comp$RMSE_m0, comp$RMSE_m_env, comp$RMSE_m_oracle))
}

pred_envloo <- bind_rows(pred_list)
comp_envloo <- bind_rows(comp_list)
coef_envloo <- bind_rows(coef_list)

# Export Tables
write.csv(pred_envloo, "Table_S_envLOO_predictions_deep.csv", row.names = FALSE)
write.csv(comp_envloo, "Table_S_envLOO_model_comparison_by_env_deep.csv", row.names = FALSE)
write.csv(coef_envloo, "Table_S_envLOO_fixed_effects_m_env.csv", row.names = FALSE)

message("\n[Output] Tables saved successfully:")
message("  - Table_S_envLOO_predictions_deep.csv")
message("  - Table_S_envLOO_model_comparison_by_env_deep.csv")
message("  - Table_S_envLOO_fixed_effects_m_env.csv")

# ===============================================================
# 5) Figure 7 Generation
# ===============================================================
# Panel A: Environmental Bias Landscape (m0)
bias_df <- pred_envloo %>%
  filter(!is.na(Temperature), !is.na(Humidity), !is.na(Bias_m0))

n_unique_pairs <- bias_df %>%
  transmute(pair = paste0(round(Temperature, 3), "_", round(Humidity, 3))) %>%
  distinct() %>% nrow()

temp_sd <- sd(bias_df$Temperature, na.rm = TRUE)
hum_sd  <- sd(bias_df$Humidity, na.rm = TRUE)

if (nrow(bias_df) >= 10 && n_unique_pairs >= 10 && temp_sd > 0.2 && hum_sd > 0.5) {
  interp_grid <- with(
    bias_df,
    akima::interp(
      x = Temperature, y = Humidity, z = Bias_m0,
      duplicate = "median", nx = 80, ny = 80
    )
  )
  
  grid_df <- data.frame(
    Temperature = rep(interp_grid$x, times = length(interp_grid$y)),
    Humidity    = rep(interp_grid$y, each  = length(interp_grid$x)),
    Bias        = as.vector(interp_grid$z)
  ) %>% filter(!is.na(Bias))
  
  p1 <- ggplot(grid_df, aes(Temperature, Humidity)) +
    geom_raster(aes(fill = Bias), interpolate = TRUE) +
    stat_contour(aes(z = Bias), colour = "white", alpha = 0.6, bins = 10, linewidth = 0.3) +
    scale_fill_viridis(option = "A") +
    theme_classic(base_size = 12) +
    labs(
      title = "A) Environmental Bias Landscape (m0)",
      x = "Temperature (°C)", y = "Humidity (%)",
      fill = "Bias (days)"
    )
} else {
  p1 <- ggplot(bias_df, aes(Temperature, Humidity)) +
    geom_point(aes(colour = Bias_m0), size = 2.5, alpha = 0.85) +
    scale_color_viridis(option = "A") +
    theme_classic(base_size = 12) +
    labs(
      title = "A) Environmental Bias (m0)",
      x = "Temperature (°C)", y = "Humidity (%)",
      colour = "Bias (days)"
    )
}

# Panel B: RMSE Comparison Across Models
comp_long <- comp_envloo %>%
  select(Env, RMSE_m0, RMSE_m_env, RMSE_m_oracle) %>%
  pivot_longer(-Env, names_to = "Model", values_to = "RMSE") %>%
  mutate(Model = recode(Model,
                        RMSE_m0       = "m0 (Omics only)",
                        RMSE_m_env    = "m_env (Temp+Hum)",
                        RMSE_m_oracle = "m_oracle (ADD_true+Hum)"))

p2 <- ggplot(comp_long, aes(Env, RMSE, fill = Model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, colour = "white") +
  scale_fill_viridis_d(option = "D", end = 0.85) +
  theme_classic(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(title = "B) Env-LOO RMSE Comparison", x = "Environment", y = "RMSE (days)", fill = "Model")

# Panel C & D: Bias vs TsD Curves
p3 <- ggplot(pred_envloo, aes(TsD, Bias_m0)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(alpha = 0.7, size = 1.8, colour = "#1f77b4") +
  geom_smooth(method = "loess", se = TRUE, colour = "#1f77b4") +
  theme_classic(base_size = 12) +
  labs(title = "C) Bias vs TsD (m0 baseline)", x = "Observed TsD (days)", y = "Bias (days)")

p4 <- ggplot(pred_envloo, aes(TsD, Bias_m_env)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(alpha = 0.7, size = 1.8, colour = "#d62728") +
  geom_smooth(method = "loess", se = TRUE, colour = "#d62728") +
  theme_classic(base_size = 12) +
  labs(title = "D) Bias vs TsD (m_env deployable)", x = "Observed TsD (days)", y = "Bias (days)")

# Combine Panels via Patchwork
final_plot <- (p1 + p2) / (p3 + p4) +
  plot_annotation(
    title = "Figure 7: Deep Environmental Correction under Env-LOO Cross-Validation",
    theme = theme(plot.title = element_text(size = 14, face = "bold"))
  )

# Save PDF & High-resolution PNG
ggsave("Figure7_EnvLOO_DeepCorrection.pdf", final_plot, width = 13, height = 8.5)
ggsave("Figure7_EnvLOO_DeepCorrection.png", final_plot, width = 13, height = 8.5, dpi = 600)

message("\n[Output] Figure 7 saved successfully: Figure7_EnvLOO_DeepCorrection.(pdf/png)")
message("Script completed successfully.")