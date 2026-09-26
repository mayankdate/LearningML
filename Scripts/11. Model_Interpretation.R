# ============================================================ #
# 11 — Model Interpretation: Opening the Black Box
# ============================================================ #
# Purpose: Teach model-agnostic interpretation methods.
#   ML models like XGBoost make great predictions, but HOW?
#   This script trains its own XGBoost, then applies every
#   major interpretation tool to dissect its predictions.
#
# We cover:
#   1. Feature importance (global: which features matter overall?)
#   2. SHAP values (global + local: why did THIS patient get THIS prediction?)
#   3. Partial Dependence Plots (how does changing one feature affect predictions?)
#   4. LIME (local explanation: approximate the model around one patient)
#   5. Calibration curves (are the predicted probabilities trustworthy?)
#   6. Decision curve analysis (is the model useful for clinical decisions?)
#
# This script is SELF-CONTAINED — trains its own XGBoost.
# No dependency on Script 04.
# Target: heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis,
  caret, pROC,
  xgboost,           # Train the model we'll interpret
  SHAPforxgboost,    # SHAP values for XGBoost
  pdp,               # Partial dependence plots
  lime                # Local Interpretable Model-agnostic Explanations
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Model_Interpretation", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Model_Interpretation"

# ---- Global Plot Theme ----
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    legend.position = "bottom"
  )

theme_annotation <- theme(
  plot.title    = element_text(face = "bold", size = 14),
  plot.subtitle = element_text(color = "grey40", size = 11)
)

col_neg  <- "#2196F3"
col_pos  <- "#F44336"
col_good <- "#4CAF50"


# ============================================================ #
# SECTION 1: WHY MODEL INTERPRETATION MATTERS
# ============================================================ #
#
# THE PROBLEM:
#   XGBoost tells you "this patient has a 73% chance of heart disease."
#   But a doctor asks: "WHY? Which features drove that prediction?
#   Should I trust this number? Would the prediction change if the
#   patient lost weight?"
#
#   Without interpretation, ML models are BLACK BOXES — accurate
#   but opaque. In healthcare, opaque isn't acceptable.
#
# THE TOOLS (from least to most detailed):
#
#   1. FEATURE IMPORTANCE (Global)
#      "Which features matter most OVERALL?"
#      One bar chart. Quick overview. Doesn't explain individual patients.
#
#   2. PARTIAL DEPENDENCE PLOTS (Global)
#      "How does the model's prediction change as ONE feature varies?"
#      Shows the average effect. Doesn't show individual variation.
#
#   3. SHAP VALUES (Global + Local)
#      "For EACH patient, how much did EACH feature push the
#       prediction up or down?"
#      The gold standard. Explains both the model overall AND
#      individual predictions. Computationally expensive.
#
#   4. LIME (Local only)
#      "For THIS ONE patient, what's a simple explanation?"
#      Fits a simple interpretable model (like linear regression)
#      in the neighborhood of one patient. Fast, approximate.
#
#   5. CALIBRATION (Model reliability)
#      "When the model says 30%, does it really mean 30%?"
#      If the model says 10 patients each have 30% risk,
#      roughly 3 of them should actually have heart disease.
#
#   6. DECISION CURVE ANALYSIS (Clinical utility)
#      "Is using this model better than treating everyone or nobody?"
#      Measures net benefit at different decision thresholds.
# ============================================================ #


# ============================================================ #
# SECTION 2: TRAIN AN XGBOOST MODEL (SELF-CONTAINED)
# ============================================================ #

# ---- Load & Prep Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")

target   <- "heart_disease"
features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)

ml_data <- ml_data %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College",
                                       "Bachelor's", "Graduate")),
    smoking_status = factor(smoking_status,
                            levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use,
                            levels = c("None", "Light", "Moderate", "Heavy"))
  )

# ---- Train/Test Split ----
set.seed(42)
train_index <- caret::createDataPartition(ml_data[[target]], p = 0.75, list = FALSE)
train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]

# ---- Create XGBoost Matrices ----
xgb_formula  <- as.formula(paste("~", paste(features, collapse = " + ")))
train_matrix <- model.matrix(xgb_formula, data = train_df)[, -1]
test_matrix  <- model.matrix(xgb_formula, data = test_df)[, -1]
train_y      <- as.numeric(train_df[[target]])
test_y       <- as.numeric(test_df[[target]])

dtrain <- xgboost::xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgboost::xgb.DMatrix(data = test_matrix,  label = test_y)

# ---- Train XGBoost ----
xgb_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  learning_rate    = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 10,
  min_split_loss   = 1
)

set.seed(42)
xgb_cv <- xgboost::xgb.cv(
  params = xgb_params, data = dtrain, nrounds = 1500,
  nfold = 5, early_stopping_rounds = 50, verbose = 0
)

cv_log     <- as.data.frame(xgb_cv$evaluation_log)
cv_auc_col <- grep("test.*auc.*mean", colnames(cv_log), value = TRUE)[1]
if (is.na(cv_auc_col)) cv_auc_col <- grep("test.*auc", colnames(cv_log), value = TRUE)[1]
best_iter  <- which.max(cv_log[[cv_auc_col]])

set.seed(42)
xgb_model <- xgboost::xgb.train(
  params = xgb_params, data = dtrain, nrounds = best_iter, verbose = 0
)

test_probs <- predict(xgb_model, newdata = dtest)
roc_obj    <- pROC::roc(test_y, test_probs, quiet = TRUE)
auc_val    <- round(as.numeric(pROC::auc(roc_obj)), 4)

cat("XGBoost trained:", best_iter, "rounds | Test AUC:", auc_val, "\n")
cat("Now we dissect this model.\n\n")


# ============================================================ #
# SECTION 3: FEATURE IMPORTANCE (GLOBAL)
# ============================================================ #
#
# XGBoost tracks three importance metrics:
#   Gain: average improvement in prediction accuracy when a feature is used
#   Cover: average number of patients affected by splits on this feature
#   Frequency: how often the feature is used in splits across all trees
#
# GAIN is usually the most informative — it measures how much each
# feature actually improves predictions when the model uses it.

# ---- 3a: XGBoost Built-in Importance ----
imp_matrix <- xgboost::xgb.importance(model = xgb_model)

# Map dummy names back to parent features
imp_matrix$parent <- dplyr::case_when(
  stringr::str_detect(imp_matrix$Feature, "^sex")             ~ "sex",
  stringr::str_detect(imp_matrix$Feature, "^race")            ~ "race",
  stringr::str_detect(imp_matrix$Feature, "^education")       ~ "education",
  stringr::str_detect(imp_matrix$Feature, "^smoking_status")  ~ "smoking_status",
  stringr::str_detect(imp_matrix$Feature, "^alcohol_use")     ~ "alcohol_use",
  TRUE ~ imp_matrix$Feature
)

parent_imp <- imp_matrix %>%
  dplyr::group_by(parent) %>%
  dplyr::summarise(Gain = sum(Gain), .groups = "drop") %>%
  dplyr::arrange(dplyr::desc(Gain)) %>%
  dplyr::mutate(parent = forcats::fct_inorder(parent))

p_imp <- ggplot(parent_imp, aes(x = Gain, y = forcats::fct_rev(parent))) +
  geom_col(fill = "#1565C0", width = 0.7) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3.5) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "XGBoost Feature Importance (Gain)",
    subtitle = "Gain = how much each feature improves predictions when used in a split.\nHigher = the model relies more on this feature. Quick global overview.",
    x = "Total Gain", y = NULL
  ) +
  theme_ml

print(p_imp)
ggsave(file.path(out_dir, "01_feature_importance.png"), p_imp,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: SHAP VALUES
# ============================================================ #
#
# WHAT ARE SHAP VALUES?
#   For each patient and each feature, the SHAP value tells you:
#   "How much did THIS feature push THIS patient's prediction
#    away from the average prediction?"
#
#   Example: Average prediction = 6% (baseline heart disease rate).
#   For patient #42:
#     age SHAP = +0.15 → being 78 pushed the prediction UP by 15%
#     smoking SHAP = +0.08 → being a current smoker pushed it UP
#     income SHAP = -0.02 → having high income pushed it DOWN slightly
#     Sum of all SHAP values + baseline = final prediction
#
# WHY SHAP IS THE GOLD STANDARD:
#   - Explains BOTH global patterns and individual predictions
#   - Mathematically grounded (Shapley values from game theory)
#   - Additive: SHAP values sum to the prediction
#   - Consistent: if a feature contributes more, its SHAP is larger
#
# TWO VIEWS:
#   Summary plot: shows distribution of SHAP values across ALL patients
#   Waterfall plot: explains ONE patient's prediction step by step

# ---- 4a: Compute SHAP Values ----
# Use a subsample for speed (SHAP is computationally expensive)
set.seed(42)
shap_sample_idx <- sample(1:nrow(test_matrix), min(500, nrow(test_matrix)))

shap_data <- SHAPforxgboost::shap.values(
  xgb_model = xgb_model,
  X_train   = test_matrix[shap_sample_idx, ]
)

cat("SHAP values computed for", length(shap_sample_idx), "patients\n")


# ---- 4b: SHAP Summary Plot (Beeswarm) ----
# Each dot = one patient × one feature.
# X-axis = SHAP value (how much it pushed the prediction).
# Color = feature value (red = high, blue = low).
# Features sorted by total importance (top = most important).
shap_long <- SHAPforxgboost::shap.prep(
  shap_contrib = shap_data$shap_score,
  X_train      = as.data.frame(test_matrix[shap_sample_idx, ])
)

p_shap_summary <- SHAPforxgboost::shap.plot.summary(shap_long) +
  labs(
    title    = "SHAP Summary Plot — How Each Feature Affects Predictions",
    subtitle = "Each dot = one patient. X-axis = SHAP value (positive = increases risk).\nColor = feature value (red = high, blue = low).\nExample: high age (red dots) pushes predictions RIGHT (increases risk)."
  ) +
  theme_ml +
  theme(plot.title = element_text(face = "bold", size = 14))

print(p_shap_summary)
ggsave(file.path(out_dir, "02_shap_summary.png"), p_shap_summary,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 4c: SHAP Dependence Plots ----
# How does the SHAP value for one feature change across its range?
# Shows nonlinear effects and interactions.

# Age dependence
p_shap_age <- SHAPforxgboost::shap.plot.dependence(
  shap_long, x = "age", color_feature = "age",
  smooth = TRUE, size0 = 1, alpha = 0.3
) +
  labs(
    title    = "SHAP Dependence — Age",
    subtitle = "Each dot = one patient. Y-axis = how much age pushed their prediction.\nUpward curve = risk accelerates with age (nonlinear effect)."
  ) +
  theme_ml

ggsave(file.path(out_dir, "03_shap_dependence_age.png"), p_shap_age,
       width = 10, height = 7, dpi = 150, bg = "white")

# Cholesterol dependence (colored by age to show interaction)
# Using a continuous feature avoids the dummy-coding name issue that
# occurs with ordered factors (model.matrix creates .L, .Q, .C columns
# instead of intuitive names like "smoking_statusCurrent").
p_shap_chol <- SHAPforxgboost::shap.plot.dependence(
  shap_long, x = "cholesterol", color_feature = "age",
  smooth = TRUE, size0 = 1, alpha = 0.3
) +
  labs(
    title    = "SHAP Dependence — Cholesterol (Colored by Age)",
    subtitle = "Each dot = one patient. Y-axis = how much cholesterol pushed their prediction.\nColor = age (yellow = young, purple = old). If old patients with high cholesterol\nhave larger SHAP values, there's an age × cholesterol interaction."
  ) +
  theme_ml

ggsave(file.path(out_dir, "04_shap_dependence_cholesterol.png"), p_shap_chol,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 4d: SHAP Waterfall for Individual Patients ----
# Explain specific predictions: a high-risk and a low-risk patient
# Find a high-risk and low-risk patient from the SHAP sample
sample_probs <- test_probs[shap_sample_idx]
high_risk_idx <- which.max(sample_probs)
low_risk_idx  <- which.min(sample_probs)

# Build waterfall data manually (more reliable than package functions)
build_waterfall <- function(shap_values, feature_names, patient_idx, baseline) {
  vals <- as.numeric(shap_values[patient_idx, ])
  df <- tibble(feature = feature_names, shap = vals) %>%
    dplyr::arrange(dplyr::desc(abs(shap))) %>%
    dplyr::slice_head(n = 10)  # Top 10 features
  df$feature <- forcats::fct_inorder(df$feature)
  df$direction <- ifelse(df$shap > 0, "Increases risk", "Decreases risk")
  return(df)
}

shap_matrix <- as.matrix(shap_data$shap_score)
baseline_val <- shap_data$BIAS0[1]
feat_names <- colnames(shap_matrix)

wf_high <- build_waterfall(shap_matrix, feat_names, high_risk_idx, baseline_val)
wf_low  <- build_waterfall(shap_matrix, feat_names, low_risk_idx, baseline_val)

p_wf_high <- ggplot(wf_high, aes(x = shap, y = forcats::fct_rev(feature), fill = direction)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0, color = "grey40") +
  scale_fill_manual(values = c("Increases risk" = col_pos, "Decreases risk" = col_neg)) +
  labs(
    title    = paste0("SHAP Waterfall — High-Risk Patient (Predicted: ",
                      round(sample_probs[high_risk_idx] * 100, 1), "%)"),
    subtitle = "Each bar = how much one feature pushed this patient's prediction.\nRed = increased risk. Blue = decreased risk.",
    x = "SHAP Value", y = NULL, fill = NULL
  ) +
  theme_ml

p_wf_low <- ggplot(wf_low, aes(x = shap, y = forcats::fct_rev(feature), fill = direction)) +
  geom_col(width = 0.7) +
  geom_vline(xintercept = 0, color = "grey40") +
  scale_fill_manual(values = c("Increases risk" = col_pos, "Decreases risk" = col_neg)) +
  labs(
    title    = paste0("SHAP Waterfall — Low-Risk Patient (Predicted: ",
                      round(sample_probs[low_risk_idx] * 100, 1), "%)"),
    subtitle = "Same visualization for a low-risk patient. Blue bars dominate —\nprotective features pushed the prediction down.",
    x = "SHAP Value", y = NULL, fill = NULL
  ) +
  theme_ml

p_waterfall <- p_wf_high / p_wf_low +
  patchwork::plot_annotation(
    title = "Individual Patient Explanations — SHAP Waterfall",
    theme = theme_annotation
  )

print(p_waterfall)
ggsave(file.path(out_dir, "05_shap_waterfall.png"), p_waterfall,
       width = 11, height = 12, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: PARTIAL DEPENDENCE PLOTS (PDP)
# ============================================================ #
#
# WHAT IS A PDP?
#   "If I change ONE feature for EVERY patient while holding
#    all other features constant, how does the average prediction change?"
#
# HOW IT WORKS:
#   1. Pick a feature (e.g., age)
#   2. For each value of age (20, 25, 30, ...):
#      a. Set ALL patients' age to that value
#      b. Predict for all patients
#      c. Record the AVERAGE prediction
#   3. Plot average prediction vs feature value
#
# This shows the MARGINAL EFFECT of each feature — averaging out
# the effects of all other features. It's the "all else equal" view.
#
# LIMITATION: PDPs can be misleading if features are correlated.
# Example: setting age=25 AND cholesterol=280 might never happen
# in real data, but PDP considers it anyway.

# ---- 5a: PDP for Top Features ----
# Use the pdp package which works directly with xgboost models
# pdp::partial() needs a prediction wrapper
predict_xgb_prob <- function(object, newdata) {
  dmat <- xgboost::xgb.DMatrix(data = as.matrix(newdata))
  predict(object, newdata = dmat)
}

# Age PDP
pdp_age <- pdp::partial(
  xgb_model,
  pred.var  = "age",
  train     = as.data.frame(train_matrix),
  pred.fun  = predict_xgb_prob,
  grid.resolution = 30
)

p_pdp_age <- ggplot(pdp_age, aes(x = age, y = yhat)) +
  geom_line(color = "#1565C0", linewidth = 1.2) +
  geom_rug(data = tibble(age = train_matrix[, "age"]),
           aes(x = age), inherit.aes = FALSE, alpha = 0.05, sides = "b") +
  labs(
    title    = "Partial Dependence — Age",
    subtitle = "Average predicted probability as age varies, all else equal.\nSteeper slope = feature has more effect in that range.\nRug plot shows where training data actually exists.",
    x = "Age", y = "Average Predicted P(Heart Disease)"
  ) +
  theme_ml

# BMI PDP
pdp_bmi <- pdp::partial(
  xgb_model,
  pred.var  = "bmi",
  train     = as.data.frame(train_matrix),
  pred.fun  = predict_xgb_prob,
  grid.resolution = 30
)

p_pdp_bmi <- ggplot(pdp_bmi, aes(x = bmi, y = yhat)) +
  geom_line(color = "#E53935", linewidth = 1.2) +
  geom_rug(data = tibble(bmi = train_matrix[, "bmi"]),
           aes(x = bmi), inherit.aes = FALSE, alpha = 0.05, sides = "b") +
  labs(
    title    = "Partial Dependence — BMI",
    subtitle = "Average predicted probability as BMI varies.\nFlat regions = BMI doesn't matter in that range.\nSteep regions = strong effect.",
    x = "BMI", y = "Average Predicted P(Heart Disease)"
  ) +
  theme_ml

# SBP PDP
pdp_sbp <- pdp::partial(
  xgb_model,
  pred.var  = "sbp",
  train     = as.data.frame(train_matrix),
  pred.fun  = predict_xgb_prob,
  grid.resolution = 30
)

p_pdp_sbp <- ggplot(pdp_sbp, aes(x = sbp, y = yhat)) +
  geom_line(color = "#43A047", linewidth = 1.2) +
  geom_rug(data = tibble(sbp = train_matrix[, "sbp"]),
           aes(x = sbp), inherit.aes = FALSE, alpha = 0.05, sides = "b") +
  labs(
    title    = "Partial Dependence — SBP",
    subtitle = "Average predicted probability as systolic blood pressure varies.",
    x = "Systolic Blood Pressure", y = "Average Predicted P(Heart Disease)"
  ) +
  theme_ml

p_pdp_grid <- p_pdp_age | p_pdp_bmi | p_pdp_sbp
p_pdp_grid <- p_pdp_grid + patchwork::plot_annotation(
  title = "Partial Dependence Plots — Top Features",
  subtitle = "How does the average prediction change as one feature varies, all else equal?",
  theme = theme_annotation
)

print(p_pdp_grid)
ggsave(file.path(out_dir, "06_partial_dependence.png"), p_pdp_grid,
       width = 15, height = 6, dpi = 150, bg = "white")


# ---- 5b: 2D PDP — Age × SBP Interaction ----
pdp_2d <- pdp::partial(
  xgb_model,
  pred.var  = c("age", "sbp"),
  train     = as.data.frame(train_matrix),
  pred.fun  = predict_xgb_prob,
  grid.resolution = 25
)

p_pdp_2d <- ggplot(pdp_2d, aes(x = age, y = sbp, fill = yhat)) +
  geom_tile() +
  scale_fill_gradient2(low = col_neg, mid = "white", high = col_pos,
                       midpoint = mean(pdp_2d$yhat), name = "P(Heart\nDisease)") +
  labs(
    title    = "2D Partial Dependence — Age × SBP",
    subtitle = "Shows how the prediction changes across COMBINATIONS of age and SBP.\nRed = higher predicted risk. Blue = lower risk.\nNon-parallel contours would indicate an interaction effect.",
    x = "Age", y = "Systolic Blood Pressure"
  ) +
  theme_ml +
  theme(panel.grid = element_blank())

print(p_pdp_2d)
ggsave(file.path(out_dir, "07_pdp_2d_age_sbp.png"), p_pdp_2d,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: LIME — LOCAL INTERPRETABLE EXPLANATIONS
# ============================================================ #
#
# WHAT IS LIME?
#   "Fit a simple model that explains ONE prediction."
#
# HOW IT WORKS:
#   1. Pick one patient you want to explain
#   2. Generate many SIMILAR patients (perturbed versions)
#   3. Get the black box's prediction for each perturbed patient
#   4. Fit a simple model (like linear regression) to those
#      predictions, weighted by similarity to the original patient
#   5. The simple model's coefficients explain the prediction
#
# LIME vs SHAP:
#   - LIME: fast, approximate, explains one patient at a time
#   - SHAP: exact (for tree models), can explain all patients at once
#   - LIME is useful when SHAP is too slow (very large models)
#   - In practice, SHAP has largely replaced LIME for tree models
#
# We include LIME because it's conceptually important and widely cited.

# ---- 6a: Set Up LIME Explainer ----
# LIME needs a prediction function that returns probabilities
# and the training data as a regular data frame
train_df_for_lime <- as.data.frame(train_matrix)

lime_predict <- function(model, newdata) {
  dmat <- xgboost::xgb.DMatrix(data = as.matrix(newdata))
  probs <- predict(model, newdata = dmat)
  data.frame(No = 1 - probs, Yes = probs)
}

lime_explainer <- lime::lime(
  x     = train_df_for_lime,
  model = xgb_model,
  bin_continuous = TRUE,
  n_bins = 5
)

# ---- 6b: Explain Two Patients ----
# High-risk and low-risk from the test set
test_df_for_lime <- as.data.frame(test_matrix)
test_probs_all   <- predict(xgb_model, newdata = dtest)

high_risk_test <- which.max(test_probs_all)
low_risk_test  <- which.min(test_probs_all)

lime_explanations <- lime::explain(
  x         = test_df_for_lime[c(high_risk_test, low_risk_test), ],
  explainer = lime_explainer,
  n_labels  = 1,
  n_features = 8,
  feature_select = "highest_weights"
)

p_lime <- lime::plot_explanations(lime_explanations) +
  labs(
    title    = "LIME Explanations — High-Risk vs Low-Risk Patient",
    subtitle = "LIME fits a simple model around each patient to explain the prediction.\nGreen = supports the predicted class. Red = contradicts it."
  ) +
  theme_ml +
  theme(axis.text.y = element_text(size = 9))

print(p_lime)
ggsave(file.path(out_dir, "08_lime_explanations.png"), p_lime,
       width = 12, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: CALIBRATION CURVE
# ============================================================ #
#
# WHAT IS CALIBRATION?
#   "When the model says 30% risk, do 30% of those patients
#    actually have heart disease?"
#
# A model can have high AUC (good at RANKING patients) but
# poor calibration (the actual probabilities are wrong).
# Example: a model might rank patients correctly but predict
# 50% when the true risk is only 10%.
#
# HOW TO CHECK:
#   1. Bin patients by predicted probability (0-10%, 10-20%, etc.)
#   2. In each bin, compute the actual % with heart disease
#   3. Plot predicted % vs actual %
#   Perfect calibration = points on the diagonal.
#   Above diagonal = model UNDERESTIMATES risk.
#   Below diagonal = model OVERESTIMATES risk.

# ---- 7a: Build Calibration Curve ----
cal_df <- tibble(
  predicted = test_probs,
  actual    = test_y
)

# Create probability bins
n_bins <- 10
cal_df$bin <- cut(cal_df$predicted,
                  breaks = seq(0, 1, length.out = n_bins + 1),
                  include.lowest = TRUE)

cal_summary <- cal_df %>%
  dplyr::group_by(bin) %>%
  dplyr::summarise(
    mean_predicted = mean(predicted),
    mean_actual    = mean(actual),
    n              = dplyr::n(),
    .groups = "drop"
  )

p_cal <- ggplot(cal_summary, aes(x = mean_predicted, y = mean_actual)) +
  geom_abline(linetype = "dashed", color = "grey50") +
  geom_point(aes(size = n), color = "#1565C0") +
  geom_line(color = "#1565C0", linewidth = 0.8) +
  scale_size_continuous(range = c(2, 8), name = "Patients\nin bin") +
  coord_equal(xlim = c(0, max(cal_summary$mean_predicted) * 1.1),
              ylim = c(0, max(cal_summary$mean_actual) * 1.1)) +
  labs(
    title    = "Calibration Curve — Are Predicted Probabilities Trustworthy?",
    subtitle = "Perfect calibration = points on the dashed diagonal.\nAbove diagonal = model underestimates risk. Below = overestimates.\nLarger dots = more patients in that bin.",
    x = "Mean Predicted Probability",
    y = "Observed Proportion (Actual)"
  ) +
  theme_ml

print(p_cal)
ggsave(file.path(out_dir, "09_calibration_curve.png"), p_cal,
       width = 9, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: DECISION CURVE ANALYSIS (DCA)
# ============================================================ #
#
# WHAT IS DCA?
#   "Is using this model to make clinical decisions better than
#    the alternatives (treat everyone, treat nobody)?"
#
# THE CONCEPT — THRESHOLD PROBABILITY:
#   A doctor has a threshold: "I'll treat if the risk is above X%."
#   Different thresholds reflect different risk tolerances.
#     Aggressive doctor: treat if risk > 5% (low threshold)
#     Conservative doctor: treat if risk > 20% (high threshold)
#
# NET BENEFIT:
#   At each threshold, DCA computes:
#     Net benefit = (true positives / N) - (false positives / N) × (threshold / (1-threshold))
#   The second term penalizes false positives — the penalty increases
#   as the threshold increases (conservative doctors penalize FP more).
#
# THE PLOT:
#   X-axis = threshold probability
#   Y-axis = net benefit
#   Three lines: Model, Treat All, Treat None (always 0)
#   The model is useful where its curve is ABOVE both alternatives.

# ---- 8a: Compute DCA ----
thresholds <- seq(0.01, 0.50, by = 0.01)

dca_df <- purrr::map_dfr(thresholds, function(pt) {
  # Model predictions at this threshold
  pred_positive <- test_probs >= pt
  tp <- sum(pred_positive & test_y == 1)
  fp <- sum(pred_positive & test_y == 0)
  n  <- length(test_y)
  
  # Net benefit for model
  nb_model <- (tp / n) - (fp / n) * (pt / (1 - pt))
  
  # Net benefit for "treat all"
  prevalence <- mean(test_y)
  nb_all <- prevalence - (1 - prevalence) * (pt / (1 - pt))
  
  tibble(threshold = pt, model = nb_model, treat_all = nb_all, treat_none = 0)
})

dca_long <- dca_df %>%
  tidyr::pivot_longer(-threshold, names_to = "strategy", values_to = "net_benefit") %>%
  dplyr::mutate(strategy = dplyr::case_when(
    strategy == "model"      ~ "XGBoost Model",
    strategy == "treat_all"  ~ "Treat All",
    strategy == "treat_none" ~ "Treat None"
  ))

p_dca <- ggplot(dca_long, aes(x = threshold, y = net_benefit, color = strategy)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c(
    "XGBoost Model" = "#1565C0", "Treat All" = "#FF9800", "Treat None" = "grey50"
  )) +
  coord_cartesian(ylim = c(-0.05, max(dca_df$model) * 1.2)) +
  labs(
    title    = "Decision Curve Analysis — Is the Model Clinically Useful?",
    subtitle = "The model is useful where its curve (blue) is ABOVE both alternatives.\nX-axis = how aggressive the clinician is (low threshold = aggressive).\nY-axis = net benefit (higher is better).",
    x = "Threshold Probability",
    y = "Net Benefit",
    color = "Strategy"
  ) +
  theme_ml

print(p_dca)
ggsave(file.path(out_dir, "10_decision_curve.png"), p_dca,
       width = 11, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 9: SAVE RESULTS
# ============================================================ #

results_interp <- list(
  xgb_model      = xgb_model,
  auc            = auc_val,
  feature_imp    = parent_imp,
  calibration    = cal_summary,
  best_iter      = best_iter
)

saveRDS(results_interp, file.path(out_dir, "results_interpretation.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 10: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== MODEL INTERPRETATION — SUMMARY ===\n")
cat("XGBoost AUC:", auc_val, "| Best iteration:", best_iter, "\n")

cat("\nWHAT WE LEARNED:\n")
cat("1.  FEATURE IMPORTANCE gives a quick global ranking of which\n")
cat("    features the model uses most. But it doesn't explain HOW\n")
cat("    or tell you about individual patients.\n")
cat("2.  SHAP VALUES are the gold standard for interpretation.\n")
cat("    They explain EACH prediction by showing how much EACH feature\n")
cat("    pushed it up or down. They're additive: all SHAP values + baseline = prediction.\n")
cat("3.  The SHAP SUMMARY PLOT shows global patterns: which features matter\n")
cat("    and in what direction. High age (red) pushes predictions right (higher risk).\n")
cat("4.  SHAP DEPENDENCE PLOTS reveal nonlinear effects (e.g., age effect\n")
cat("    accelerates after 60) and interactions (e.g., smoking × age).\n")
cat("5.  SHAP WATERFALLS explain individual patients: 'this 78-year-old\n")
cat("    smoker has a 73% predicted risk because age added +15%, smoking\n")
cat("    added +8%, and high income only subtracted -2%.'\n")
cat("6.  PARTIAL DEPENDENCE shows the average effect of changing one feature.\n")
cat("    Flat regions = feature doesn't matter there. Steep = strong effect.\n")
cat("7.  LIME fits a simple model around one patient for a local explanation.\n")
cat("    Faster than SHAP for very large models, but approximate.\n")
cat("8.  CALIBRATION checks if predicted probabilities are trustworthy.\n")
cat("    High AUC ≠ good calibration. A model can rank well but give\n")
cat("    wrong absolute probabilities.\n")
cat("9.  DECISION CURVE ANALYSIS asks: 'Is using this model better than\n")
cat("    treating everyone or nobody?' It measures clinical utility, not\n")
cat("    just statistical performance.\n")
cat("10. No single interpretation tool is enough. Use them together:\n")
cat("    Feature importance → SHAP → PDP → Calibration → DCA.\n")

cat("\nWHEN TO USE EACH TOOL:\n")
cat("  Feature importance: Quick first look, presentations\n")
cat("  SHAP summary:       Detailed global patterns, publications\n")
cat("  SHAP waterfall:     Explaining specific patients to clinicians\n")
cat("  PDP:                'What if?' scenario analysis\n")
cat("  LIME:               When SHAP is too slow (huge models)\n")
cat("  Calibration:        Before deploying any model in practice\n")
cat("  DCA:                Justifying clinical use of the model\n")
cat("===================================\n")

cat("\nScript 11 complete.\n")