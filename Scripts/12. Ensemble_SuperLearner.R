# ============================================================ #
# 12 — Ensemble Methods & Super Learner
# ============================================================ #
# Purpose: Teach how to COMBINE multiple models for better
#   predictions than any single model. We've learned logistic
#   regression, trees, forests, boosting, neural nets, KNN, SVM.
#   Now we ask: "Can we use ALL of them together?"
#
# We cover:
#   1. Simple averaging — average predictions from multiple models
#   2. Weighted averaging — weight models by their CV performance
#   3. Stacking — train a "meta-learner" on top of base models
#   4. Super Learner — the optimal, theoretically-grounded version
#
# This script is SELF-CONTAINED — trains all base models internally.
# Target: heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales,
  caret, pROC,
  glmnet,       # Regularized logistic regression (base learner + meta-learner)
  ranger,       # Fast Random Forest
  xgboost,      # XGBoost
  nnet,         # Neural network
  e1071         # SVM
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Ensemble_SuperLearner", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Ensemble_SuperLearner"

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


# ============================================================ #
# SECTION 1: WHAT IS AN ENSEMBLE?
# ============================================================ #
#
# ANALOGY: Imagine asking 5 doctors to diagnose a patient.
# Each doctor has different training and experience:
#   - Doctor A: specialist in heart disease (very focused)
#   - Doctor B: generalist (broad but less deep)
#   - Doctor C: data-driven (relies on lab numbers)
#   - Doctor D: pattern-matcher (recognizes clinical patterns)
#   - Doctor E: conservative (only flags clear cases)
#
# No single doctor is always right. But if you combine their
# opinions — maybe voting, maybe weighting the specialist higher —
# the group diagnosis is often better than any individual.
#
# That's an ensemble. Different models have different strengths:
#   - Logistic regression: simple, catches linear effects
#   - Random Forest: handles nonlinearity, robust
#   - XGBoost: catches subtle interactions, highest raw accuracy
#   - Neural Net: flexible, learns complex patterns
#   - SVM: good at separating difficult cases
#
# THREE WAYS TO COMBINE:
#
# 1. SIMPLE AVERAGE:
#    Final prediction = mean(all model predictions)
#    Every model gets equal vote. Dead simple. Often surprisingly good.
#
# 2. WEIGHTED AVERAGE:
#    Final prediction = weighted mean, weights = model quality
#    Better models get more say. Weights based on CV performance.
#
# 3. STACKING (Super Learner):
#    Train a META-MODEL that learns how to best combine the base
#    models. The meta-model sees each base model's prediction as
#    a FEATURE and learns which models to trust in which situations.
#    This is the most powerful approach — it can learn "trust XGBoost
#    for old patients but Random Forest for young patients."
#
# WHY ENSEMBLES WORK:
#   Each model makes different mistakes. If the mistakes are
#   UNCORRELATED, averaging them out reduces total error.
#   This is the same principle as Random Forest (averaging trees)
#   but applied at the MODEL level instead of the tree level.
#
# THE SUPER LEARNER GUARANTEE:
#   Under mild conditions, the Super Learner performs asymptotically
#   as well as the BEST possible weighted combination of the base
#   models. You can't do worse than the best individual model
#   (in theory). In practice, it usually does better.
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #

# ---- Load Data ----
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

train_y <- train_df[[target]]
test_y  <- test_df[[target]]

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Prevalence — Train:", round(mean(train_y) * 100, 1),
    "% | Test:", round(mean(test_y) * 100, 1), "%\n")

# ---- Create Model Matrices ----
# Numeric matrices for models that need them (XGBoost, glmnet, nnet)
xgb_formula  <- as.formula(paste("~", paste(features, collapse = " + ")))
train_matrix <- model.matrix(xgb_formula, data = train_df)[, -1]
test_matrix  <- model.matrix(xgb_formula, data = test_df)[, -1]

# Scaled version for nnet and SVM
train_means  <- colMeans(train_matrix)
train_sds    <- apply(train_matrix, 2, sd)
train_sds[train_sds == 0] <- 1
train_scaled <- scale(train_matrix, center = train_means, scale = train_sds)
test_scaled  <- scale(test_matrix,  center = train_means, scale = train_sds)

cat("Feature matrix:", ncol(train_matrix), "columns after dummy coding\n")


# ============================================================ #
# SECTION 3: TRAIN BASE MODELS
# ============================================================ #
# We train 5 diverse base models. Diversity is key — if all
# models make the same predictions, combining them adds nothing.
# We want models with different INDUCTIVE BIASES:
#   - Linear vs nonlinear
#   - Distance-based vs tree-based vs weight-based
#   - Simple vs complex

cat("\n--- Training 5 base models ---\n")

# ---- 3a: Logistic Regression (L2 regularized) ----
# Simple linear baseline. Catches the main effects.
set.seed(42)
glmnet_model <- glmnet::cv.glmnet(
  x = train_matrix, y = train_y,
  family = "binomial", alpha = 0, nfolds = 5
)
glmnet_train_probs <- as.numeric(predict(glmnet_model, newx = train_matrix,
                                         s = "lambda.min", type = "response"))
glmnet_test_probs  <- as.numeric(predict(glmnet_model, newx = test_matrix,
                                         s = "lambda.min", type = "response"))
cat("1. Logistic Regression (glmnet) — AUC:",
    round(as.numeric(pROC::auc(pROC::roc(test_y, glmnet_test_probs, quiet = TRUE))), 4), "\n")


# ---- 3b: Random Forest ----
# Bagging + random features. Robust, handles nonlinearity.
# ranger needs the target as a factor for probability = TRUE to work properly
train_df_rf <- train_df
train_df_rf[[target]] <- factor(train_df_rf[[target]])
test_df_rf <- test_df
test_df_rf[[target]] <- factor(test_df_rf[[target]])

set.seed(42)
rf_model <- ranger::ranger(
  formula        = as.formula(paste(target, "~", paste(features, collapse = " + "))),
  data           = train_df_rf,
  num.trees      = 500,
  mtry           = 4,
  min.node.size  = 10,
  probability    = TRUE,
  seed           = 42
)
# Column 2 = probability of class "1" (heart disease)
rf_train_probs <- predict(rf_model, data = train_df_rf)$predictions[, 2]
rf_test_probs  <- predict(rf_model, data = test_df_rf)$predictions[, 2]
cat("2. Random Forest (ranger) — AUC:",
    round(as.numeric(pROC::auc(pROC::roc(test_y, rf_test_probs, quiet = TRUE))), 4), "\n")


# ---- 3c: XGBoost ----
# Sequential boosting. Usually the strongest individual model.
dtrain <- xgboost::xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgboost::xgb.DMatrix(data = test_matrix,  label = test_y)

xgb_params <- list(
  objective = "binary:logistic", eval_metric = "auc",
  learning_rate = 0.05, max_depth = 4, subsample = 0.8,
  colsample_bytree = 0.8, min_child_weight = 10
)

set.seed(42)
xgb_cv <- xgboost::xgb.cv(params = xgb_params, data = dtrain, nrounds = 1500,
                          nfold = 5, early_stopping_rounds = 50, verbose = 0)
cv_log     <- as.data.frame(xgb_cv$evaluation_log)
cv_auc_col <- grep("test.*auc.*mean", colnames(cv_log), value = TRUE)[1]
if (is.na(cv_auc_col)) cv_auc_col <- grep("test.*auc", colnames(cv_log), value = TRUE)[1]
best_iter  <- which.max(cv_log[[cv_auc_col]])

set.seed(42)
xgb_model <- xgboost::xgb.train(params = xgb_params, data = dtrain,
                                nrounds = best_iter, verbose = 0)
xgb_train_probs <- predict(xgb_model, newdata = dtrain)
xgb_test_probs  <- predict(xgb_model, newdata = dtest)
cat("3. XGBoost —", best_iter, "rounds — AUC:",
    round(as.numeric(pROC::auc(pROC::roc(test_y, xgb_test_probs, quiet = TRUE))), 4), "\n")


# ---- 3d: Neural Network ----
# Flexible nonlinear model. Different learning mechanism from trees.
train_nn <- as.data.frame(train_scaled)
train_nn$target <- factor(train_y, levels = c(0, 1), labels = c("No", "Yes"))
test_nn <- as.data.frame(test_scaled)

set.seed(42)
nn_model <- nnet::nnet(
  target ~ ., data = train_nn,
  size = 10, decay = 0.1, maxit = 100,
  trace = FALSE, linout = FALSE
)
nn_train_probs <- predict(nn_model, newdata = as.data.frame(train_scaled), type = "raw")[, 1]
nn_test_probs  <- predict(nn_model, newdata = as.data.frame(test_scaled), type = "raw")[, 1]
cat("4. Neural Network (nnet) — AUC:",
    round(as.numeric(pROC::auc(pROC::roc(test_y, nn_test_probs, quiet = TRUE))), 4), "\n")


# ---- 3e: SVM (Radial Kernel) ----
# Margin-based, uses kernel trick for nonlinearity.
set.seed(42)
svm_model <- e1071::svm(
  x = train_scaled, y = factor(train_y),
  kernel = "radial", probability = TRUE,
  cost = 1, gamma = 0.01
)
svm_train_probs <- attr(predict(svm_model, train_scaled, probability = TRUE), "probabilities")[, "1"]
svm_test_probs  <- attr(predict(svm_model, test_scaled, probability = TRUE), "probabilities")[, "1"]
cat("5. SVM (radial) — AUC:",
    round(as.numeric(pROC::auc(pROC::roc(test_y, svm_test_probs, quiet = TRUE))), 4), "\n")


# ============================================================ #
# SECTION 4: ENSEMBLE METHOD 1 — SIMPLE AVERAGE
# ============================================================ #
#
# The simplest ensemble: average all predictions with equal weight.
# Surprisingly effective because it cancels out uncorrelated errors.

avg_test_probs <- (glmnet_test_probs + rf_test_probs + xgb_test_probs +
                     nn_test_probs + svm_test_probs) / 5

avg_auc <- round(as.numeric(pROC::auc(pROC::roc(test_y, avg_test_probs, quiet = TRUE))), 4)
cat("\n--- Ensemble Results ---\n")
cat("Simple Average AUC:", avg_auc, "\n")


# ============================================================ #
# SECTION 5: ENSEMBLE METHOD 2 — WEIGHTED AVERAGE
# ============================================================ #
#
# Weight each model by its individual AUC. Better models get more say.
# Normalize weights so they sum to 1.

individual_aucs <- c(
  glmnet = as.numeric(pROC::auc(pROC::roc(test_y, glmnet_test_probs, quiet = TRUE))),
  rf     = as.numeric(pROC::auc(pROC::roc(test_y, rf_test_probs, quiet = TRUE))),
  xgb    = as.numeric(pROC::auc(pROC::roc(test_y, xgb_test_probs, quiet = TRUE))),
  nn     = as.numeric(pROC::auc(pROC::roc(test_y, nn_test_probs, quiet = TRUE))),
  svm    = as.numeric(pROC::auc(pROC::roc(test_y, svm_test_probs, quiet = TRUE)))
)

# Normalize to weights
weights <- individual_aucs / sum(individual_aucs)

cat("Model weights (AUC-proportional):\n")
print(round(weights, 3))

weighted_test_probs <- (weights["glmnet"] * glmnet_test_probs +
                          weights["rf"] * rf_test_probs +
                          weights["xgb"] * xgb_test_probs +
                          weights["nn"] * nn_test_probs +
                          weights["svm"] * svm_test_probs)

weighted_auc <- round(as.numeric(pROC::auc(pROC::roc(test_y, weighted_test_probs, quiet = TRUE))), 4)
cat("Weighted Average AUC:", weighted_auc, "\n")


# ============================================================ #
# SECTION 6: ENSEMBLE METHOD 3 — STACKING (SUPER LEARNER)
# ============================================================ #
#
# WHAT IS STACKING?
#   Train a META-LEARNER that takes base model predictions as inputs
#   and learns the OPTIMAL way to combine them.
#
# THE KEY INSIGHT: We can't use the base models' training predictions
#   to train the meta-learner — that would be data leakage (the base
#   models already saw those patients). Instead we use CROSS-VALIDATED
#   predictions: for each patient, the prediction comes from a model
#   that NEVER saw that patient during training.
#
# HOW IT WORKS (Super Learner algorithm):
#   1. Split training data into K folds
#   2. For each fold k:
#      a. Train each base model on all folds EXCEPT k
#      b. Predict on fold k (out-of-fold predictions)
#   3. Stack all out-of-fold predictions → each patient has a
#      prediction from each model, and that model never saw them
#   4. Train a meta-learner (e.g., logistic regression) on these
#      out-of-fold predictions to learn optimal weights
#   5. For final predictions: train all base models on full training
#      data, get predictions, feed to meta-learner
#
# WHY THIS IS POWERFUL:
#   - Learns which models to trust for which types of patients
#   - Can't do worse than the best individual model (asymptotically)
#   - No manual weight selection needed

# ---- 6a: Generate Cross-Validated Predictions ----
set.seed(42)
n_folds  <- 5
fold_ids <- caret::createFolds(train_y, k = n_folds, list = FALSE, returnTrain = FALSE)

# Matrix to store out-of-fold predictions: n_train × n_models
oof_predictions <- matrix(NA, nrow = nrow(train_df), ncol = 5)
colnames(oof_predictions) <- c("glmnet", "rf", "xgb", "nn", "svm")

cat("\nGenerating cross-validated predictions for stacking...\n")

for (k in 1:n_folds) {
  cat("  Fold", k, "of", n_folds, "\n")
  
  # Split
  fold_train_idx <- fold_ids != k
  fold_val_idx   <- fold_ids == k
  
  fold_train_df <- train_df[fold_train_idx, ]
  fold_val_df   <- train_df[fold_val_idx, ]
  fold_train_y  <- train_y[fold_train_idx]
  fold_val_y    <- train_y[fold_val_idx]
  
  fold_train_mat <- train_matrix[fold_train_idx, ]
  fold_val_mat   <- train_matrix[fold_val_idx, ]
  fold_train_sc  <- train_scaled[fold_train_idx, ]
  fold_val_sc    <- train_scaled[fold_val_idx, ]
  
  # 1. glmnet
  fold_glm <- glmnet::cv.glmnet(fold_train_mat, fold_train_y,
                                family = "binomial", alpha = 0, nfolds = 3)
  oof_predictions[fold_val_idx, "glmnet"] <- as.numeric(
    predict(fold_glm, newx = fold_val_mat, s = "lambda.min", type = "response"))
  
  # 2. Random Forest
  fold_rf_data <- fold_train_df
  fold_rf_data[[target]] <- factor(fold_rf_data[[target]])
  fold_val_rf <- fold_val_df
  fold_val_rf[[target]] <- factor(fold_val_rf[[target]])
  fold_rf <- ranger::ranger(
    as.formula(paste(target, "~", paste(features, collapse = " + "))),
    data = fold_rf_data, num.trees = 300, mtry = 4,
    min.node.size = 10, probability = TRUE, seed = 42
  )
  oof_predictions[fold_val_idx, "rf"] <- predict(fold_rf, data = fold_val_rf)$predictions[, 2]
  
  # 3. XGBoost
  fold_dtrain <- xgboost::xgb.DMatrix(fold_train_mat, label = fold_train_y)
  fold_xgb <- xgboost::xgb.train(
    params = xgb_params, data = fold_dtrain,
    nrounds = min(best_iter, 300), verbose = 0
  )
  fold_dval <- xgboost::xgb.DMatrix(fold_val_mat)
  oof_predictions[fold_val_idx, "xgb"] <- predict(fold_xgb, newdata = fold_dval)
  
  # 4. Neural Network
  fold_nn_df <- as.data.frame(fold_train_sc)
  fold_nn_df$target <- factor(fold_train_y, levels = c(0, 1), labels = c("No", "Yes"))
  fold_nn <- nnet::nnet(target ~ ., data = fold_nn_df,
                        size = 10, decay = 0.1, maxit = 100, trace = FALSE, linout = FALSE)
  oof_predictions[fold_val_idx, "nn"] <- predict(fold_nn,
                                                 newdata = as.data.frame(fold_val_sc), type = "raw")[, 1]
  
  # 5. SVM
  fold_svm <- e1071::svm(fold_train_sc, factor(fold_train_y),
                         kernel = "radial", probability = TRUE, cost = 1, gamma = 0.01)
  oof_predictions[fold_val_idx, "svm"] <- attr(
    predict(fold_svm, fold_val_sc, probability = TRUE), "probabilities")[, "1"]
}

cat("Out-of-fold predictions complete\n")
cat("Correlation between base model OOF predictions:\n")
print(round(cor(oof_predictions), 3))


# ---- 6b: Train Meta-Learner ----
# The meta-learner is a simple model that learns optimal weights.
# We use logistic regression (via glmnet with alpha=0 for stability).
# The INPUTS are the 5 base model predictions.
# The TARGET is the actual outcome.
meta_x <- oof_predictions
meta_y <- train_y

set.seed(42)
meta_model <- glmnet::cv.glmnet(
  x = meta_x, y = meta_y,
  family = "binomial", alpha = 0, nfolds = 5
)

# Extract meta-learner coefficients — these are the learned weights
meta_coefs <- as.numeric(coef(meta_model, s = "lambda.min"))
names(meta_coefs) <- c("(Intercept)", colnames(oof_predictions))
cat("\nMeta-learner coefficients (log-odds scale):\n")
print(round(meta_coefs, 3))


# ---- 6c: Generate Final Stacked Predictions ----
# Use ALL training data to get base model predictions on test set
# (already computed in Section 3), then feed to meta-learner
test_base_preds <- cbind(
  glmnet = glmnet_test_probs,
  rf     = rf_test_probs,
  xgb    = xgb_test_probs,
  nn     = nn_test_probs,
  svm    = svm_test_probs
)

stacked_test_probs <- as.numeric(predict(meta_model, newx = test_base_preds,
                                         s = "lambda.min", type = "response"))
stacked_auc <- round(as.numeric(pROC::auc(pROC::roc(test_y, stacked_test_probs, quiet = TRUE))), 4)
cat("Super Learner (Stacked) AUC:", stacked_auc, "\n")


# ============================================================ #
# SECTION 7: COMPARE ALL APPROACHES
# ============================================================ #

# ---- 7a: AUC Comparison Table ----
all_results <- tibble(
  model = c("Logistic (glmnet)", "Random Forest", "XGBoost",
            "Neural Network", "SVM (radial)",
            "Simple Average", "Weighted Average", "Super Learner (Stacked)"),
  auc = c(round(individual_aucs, 4),
          avg_auc, weighted_auc, stacked_auc),
  type = c(rep("Individual", 5), rep("Ensemble", 3))
)

cat("\n--- Full Comparison ---\n")
print(as.data.frame(all_results))

# Best individual vs best ensemble
best_individual <- all_results %>% dplyr::filter(type == "Individual") %>%
  dplyr::slice_max(auc, n = 1)
best_ensemble   <- all_results %>% dplyr::filter(type == "Ensemble") %>%
  dplyr::slice_max(auc, n = 1)

cat("\nBest individual:", best_individual$model, "(AUC:", best_individual$auc, ")\n")
cat("Best ensemble:  ", best_ensemble$model, "(AUC:", best_ensemble$auc, ")\n")
cat("Improvement:    ", round(best_ensemble$auc - best_individual$auc, 4), "\n")


# ---- 7b: AUC Bar Chart ----
all_results$model <- forcats::fct_reorder(all_results$model, all_results$auc)

p_compare <- ggplot(all_results, aes(x = auc, y = model, fill = type)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = round(auc, 4)), hjust = -0.1, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("Individual" = col_neg, "Ensemble" = col_pos)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title    = "Model Comparison — Individual vs Ensemble",
    subtitle = "Red = ensemble methods. Blue = individual models.\nEnsembles combine diverse models to reduce prediction errors.",
    x = "Test AUC", y = NULL, fill = "Type"
  ) +
  theme_ml

print(p_compare)
ggsave(file.path(out_dir, "01_model_comparison.png"), p_compare,
       width = 11, height = 7, dpi = 150, bg = "white")


# ---- 7c: ROC Curves Overlay ----
roc_data <- dplyr::bind_rows(
  tibble(fpr = 1 - pROC::roc(test_y, glmnet_test_probs, quiet = TRUE)$specificities,
         tpr = pROC::roc(test_y, glmnet_test_probs, quiet = TRUE)$sensitivities,
         model = paste0("Logistic (", round(individual_aucs["glmnet"], 3), ")")),
  tibble(fpr = 1 - pROC::roc(test_y, xgb_test_probs, quiet = TRUE)$specificities,
         tpr = pROC::roc(test_y, xgb_test_probs, quiet = TRUE)$sensitivities,
         model = paste0("XGBoost (", round(individual_aucs["xgb"], 3), ")")),
  tibble(fpr = 1 - pROC::roc(test_y, stacked_test_probs, quiet = TRUE)$specificities,
         tpr = pROC::roc(test_y, stacked_test_probs, quiet = TRUE)$sensitivities,
         model = paste0("Super Learner (", stacked_auc, ")"))
)

p_roc <- ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey60") +
  scale_color_manual(values = c(
    setNames("#90CAF9", grep("Logistic", unique(roc_data$model), value = TRUE)),
    setNames("#1565C0", grep("XGBoost", unique(roc_data$model), value = TRUE)),
    setNames("#E53935", grep("Super", unique(roc_data$model), value = TRUE))
  )) +
  labs(
    title    = "ROC Curves — Logistic vs XGBoost vs Super Learner",
    subtitle = "Shows the simplest model, the best individual model, and the stacked ensemble.\nThe Super Learner curve should be at or above the best individual.",
    x = "False Positive Rate", y = "True Positive Rate", color = "Model"
  ) +
  theme_ml

print(p_roc)
ggsave(file.path(out_dir, "02_roc_comparison.png"), p_roc,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 7d: Base Model Correlation Heatmap ----
# Low correlation between models = more benefit from combining.
# High correlation = models agree, so ensemble adds little.
cor_mat <- cor(oof_predictions)

cor_df <- as.data.frame(as.table(cor_mat))
names(cor_df) <- c("Model1", "Model2", "correlation")

p_cor <- ggplot(cor_df, aes(x = Model1, y = Model2, fill = correlation)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(correlation, 2)), size = 4, fontface = "bold") +
  scale_fill_gradient2(low = col_neg, mid = "white", high = col_pos,
                       midpoint = 0.5, limits = c(0, 1)) +
  labs(
    title    = "Base Model Prediction Correlation",
    subtitle = "Low correlation = models make different mistakes = more ensemble benefit.\nHigh correlation = models agree = less improvement from combining.",
    x = NULL, y = NULL, fill = "Correlation"
  ) +
  theme_ml +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1))

print(p_cor)
ggsave(file.path(out_dir, "03_model_correlation.png"), p_cor,
       width = 9, height = 7, dpi = 150, bg = "white")


# ---- 7e: Meta-Learner Weights Visualization ----
# What weights did the Super Learner assign to each base model?
# This shows which models it trusts most.
weight_df <- tibble(
  model  = names(meta_coefs)[-1],  # Drop intercept
  weight = meta_coefs[-1]
) %>%
  dplyr::mutate(model = forcats::fct_reorder(model, weight))

p_weights <- ggplot(weight_df, aes(x = weight, y = model)) +
  geom_col(aes(fill = weight > 0), width = 0.6) +
  geom_vline(xintercept = 0, color = "grey40") +
  scale_fill_manual(values = c("TRUE" = "#43A047", "FALSE" = "#E53935"),
                    guide = "none") +
  labs(
    title    = "Super Learner Meta-Coefficients",
    subtitle = "How much the meta-learner trusts each base model (log-odds scale).\nPositive = contributes to the ensemble. Negative = partially offsets others.",
    x = "Meta-Learner Coefficient", y = NULL
  ) +
  theme_ml

print(p_weights)
ggsave(file.path(out_dir, "04_meta_learner_weights.png"), p_weights,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 7f: Prediction Scatter — XGBoost vs Super Learner ----
scatter_df <- tibble(
  xgb    = xgb_test_probs,
  stacked = stacked_test_probs,
  actual = factor(test_y, levels = c(0, 1), labels = c("No", "Yes"))
)

p_scatter <- ggplot(scatter_df, aes(x = xgb, y = stacked, color = actual)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_abline(linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c("No" = col_neg, "Yes" = col_pos)) +
  labs(
    title    = "XGBoost vs Super Learner Predictions",
    subtitle = "Points on the diagonal = both models agree. Points off = they disagree.\nThe Super Learner adjusts predictions where the base models differ.",
    x = "XGBoost Predicted Probability",
    y = "Super Learner Predicted Probability",
    color = "Actual"
  ) +
  theme_ml

print(p_scatter)
ggsave(file.path(out_dir, "05_xgb_vs_stacked.png"), p_scatter,
       width = 9, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: SAVE RESULTS
# ============================================================ #

results_ensemble <- list(
  individual_aucs = individual_aucs,
  avg_auc         = avg_auc,
  weighted_auc    = weighted_auc,
  stacked_auc     = stacked_auc,
  meta_coefs      = meta_coefs,
  model_correlation = cor_mat,
  all_results     = all_results
)

saveRDS(results_ensemble, file.path(out_dir, "results_ensemble.rds"))
saveRDS(stacked_test_probs, file.path(out_dir, "test_predictions_stacked.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 9: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== ENSEMBLE & SUPER LEARNER — SUMMARY ===\n")
cat("Individual model AUCs:\n")
for (i in seq_along(individual_aucs)) {
  cat("  ", names(individual_aucs)[i], ":", round(individual_aucs[i], 4), "\n")
}
cat("Ensemble AUCs:\n")
cat("  Simple Average: ", avg_auc, "\n")
cat("  Weighted Average:", weighted_auc, "\n")
cat("  Super Learner:  ", stacked_auc, "\n")

cat("\nWHAT WE LEARNED:\n")
cat("1.  Ensembles combine multiple models to reduce prediction errors.\n")
cat("    The key insight: different models make DIFFERENT mistakes,\n")
cat("    so averaging them out cancels uncorrelated errors.\n")
cat("2.  SIMPLE AVERAGING is surprisingly effective. Just average all\n")
cat("    predictions with equal weight — no tuning needed.\n")
cat("3.  WEIGHTED AVERAGING gives more say to better models. Weights\n")
cat("    are proportional to each model's AUC.\n")
cat("4.  STACKING (Super Learner) trains a META-LEARNER that learns\n")
cat("    the optimal combination. It uses CROSS-VALIDATED predictions\n")
cat("    to avoid data leakage — each patient's prediction comes from\n")
cat("    a model that never saw them during training.\n")
cat("5.  MODEL DIVERSITY matters more than individual model quality.\n")
cat("    Combining 5 models that all agree adds nothing. Combining\n")
cat("    models with different strengths (linear + tree + neural) helps.\n")
cat("6.  The correlation heatmap shows how similar base model predictions\n")
cat("    are. Low correlation = high ensemble benefit.\n")
cat("7.  The Super Learner's meta-coefficients show which models it\n")
cat("    trusts most. This often reveals that one or two models dominate.\n")
cat("8.  In practice, the improvement from ensembling over the best\n")
cat("    individual model is often small (0.5–2% AUC) but consistent.\n")
cat("    For high-stakes applications, every fraction matters.\n")

cat("\nWHEN TO USE ENSEMBLES:\n")
cat("  Simple average: Quick improvement, no extra tuning\n")
cat("  Weighted average: When models have very different quality\n")
cat("  Super Learner: When you need maximum performance, research settings\n")
cat("  Skip ensembles: When interpretability matters more than accuracy\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Scripts 01-06: Each taught one model family. This script\n")
cat("    shows how to combine them all.\n")
cat("  - Script 03 (Random Forest): RF itself is an ensemble of trees.\n")
cat("    Super Learner is an ensemble of MODELS — one level higher.\n")
cat("  - Script 04 (XGBoost): Often the strongest individual model.\n")
cat("    Super Learner can still improve on it by adding diversity.\n")
cat("  - Script 15 (Model Comparison): Will compare all approaches.\n")
cat("===================================\n")

cat("\nScript 12 complete.\n")