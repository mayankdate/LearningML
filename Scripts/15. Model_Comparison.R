# ============================================================ #
# 15 — Model Comparison: The Grand Summary
# ============================================================ #
# Purpose: Compare ALL classification models side-by-side.
#   Every script (01–06, 12) saved its results and test
#   predictions in standardized format. This script loads
#   them all and creates a comprehensive comparison.
#
# We produce:
#   1. Performance table (AUC, accuracy, precision, recall, F1)
#   2. ROC curve overlay (all models on one plot)
#   3. Overfitting comparison (train vs test AUC)
#   4. CV stability comparison
#   5. Strengths/weaknesses summary
#   6. Recommendation framework
#
# This script ONLY loads saved results — it doesn't train anything.
# If a model's results file doesn't exist, it's skipped gracefully.
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales,
  pROC, kableExtra
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Model_Comparison", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Model_Comparison"

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


# ============================================================ #
# SECTION 1: LOAD ALL SAVED RESULTS
# ============================================================ #
# Each classification script saved:
#   results_[model].rds — tibble with AUC, accuracy, precision, etc.
#   test_predictions_[model].rds — predicted probabilities on test set
#
# We search for these files and load whatever exists.

# ---- Define Model Registry ----
# Map each model to its output directory and file names
model_registry <- list(
  list(name = "Logistic Regression",
       dir = "Logistic_Regression",
       results_file = "results_logistic.rds",
       preds_file = "test_predictions_logistic.rds"),
  list(name = "Decision Tree",
       dir = "Decision_Trees",
       results_file = "results_tree.rds",
       preds_file = "test_predictions_tree.rds"),
  list(name = "Random Forest",
       dir = "Random_Forest",
       results_file = "results_rf.rds",
       preds_file = "test_predictions_rf.rds"),
  list(name = "GBM",
       dir = "Boosting",
       results_file = "results_boosting.rds",
       preds_file = "test_predictions_gbm.rds"),
  list(name = "XGBoost",
       dir = "Boosting",
       results_file = "results_boosting.rds",
       preds_file = "test_predictions_xgboost.rds"),
  list(name = "KNN",
       dir = "KNN_SVM",
       results_file = "results_knn.rds",
       preds_file = "test_predictions_knn.rds"),
  list(name = "SVM",
       dir = "KNN_SVM",
       results_file = "results_svm.rds",
       preds_file = "test_predictions_svm.rds"),
  list(name = "Neural Network",
       dir = "Neural_Network",
       results_file = "results_nn.rds",
       preds_file = "test_predictions_nn.rds"),
  list(name = "Super Learner",
       dir = "Ensemble_SuperLearner",
       results_file = "results_ensemble.rds",
       preds_file = "test_predictions_stacked.rds")
)

# ---- Load Results ----
all_results <- list()
all_preds   <- list()

for (m in model_registry) {
  results_path <- file.path("Outputs", m$dir, m$results_file)
  preds_path   <- file.path("Outputs", m$dir, m$preds_file)
  
  if (file.exists(results_path)) {
    res <- readRDS(results_path)
    
    # Handle different result formats:
    # Scripts 01-06 save a tibble with model, auc, accuracy, etc.
    # Script 04 (Boosting) saves both GBM and XGBoost in one tibble.
    # Script 12 (Ensemble) saves a list with individual_aucs and stacked_auc.
    
    if (is.data.frame(res)) {
      # Standard format — may have multiple rows (Boosting has GBM + XGBoost)
      if (m$name == "GBM") {
        row <- res %>% dplyr::filter(model == "GBM")
        if (nrow(row) > 0) all_results[[m$name]] <- row[1, ]
      } else if (m$name == "XGBoost") {
        row <- res %>% dplyr::filter(model == "XGBoost")
        if (nrow(row) > 0) all_results[[m$name]] <- row[1, ]
      } else {
        all_results[[m$name]] <- res[1, ]  # Take first row
      }
    } else if (is.list(res) && "stacked_auc" %in% names(res)) {
      # Ensemble results — build a standardized row
      all_results[[m$name]] <- tibble(
        model     = "Super Learner",
        auc       = res$stacked_auc,
        accuracy  = NA_real_,
        precision = NA_real_,
        recall    = NA_real_,
        f1        = NA_real_,
        cv_auc_mean = NA_real_,
        cv_auc_sd   = NA_real_,
        train_auc   = NA_real_
      )
    }
    cat("Loaded results:", m$name, "\n")
  }
  
  if (file.exists(preds_path)) {
    all_preds[[m$name]] <- readRDS(preds_path)
    cat("Loaded predictions:", m$name, "\n")
  }
}

n_models <- length(all_results)
cat("\nLoaded", n_models, "model results and", length(all_preds), "prediction sets\n")

if (n_models == 0) {
  stop("No model results found. Run the individual scripts (01-12) first.")
}

# ---- Load Test Labels ----
# We need the actual test set labels to compute ROC curves
ml_data <- readRDS("Data/ml_teaching_data.rds")
set.seed(42)
train_index <- caret::createDataPartition(ml_data$heart_disease, p = 0.75, list = FALSE)
test_y <- ml_data$heart_disease[-train_index[, 1]]

cat("Test set:", length(test_y), "patients |",
    round(mean(test_y) * 100, 1), "% heart disease\n")


# ============================================================ #
# SECTION 2: BUILD COMPARISON TABLE
# ============================================================ #

# ---- Combine All Results ----
results_table <- dplyr::bind_rows(all_results, .id = "model_name")

# Standardize column names (some scripts may have slight differences)
if (!"model_name" %in% names(results_table)) {
  results_table$model_name <- results_table$model
}

# Sort by AUC descending
results_table <- results_table %>%
  dplyr::arrange(dplyr::desc(auc))

cat("\n=== MODEL COMPARISON TABLE ===\n")
results_display <- results_table %>%
  dplyr::select(model_name, auc, accuracy, precision, recall, f1,
                cv_auc_mean, cv_auc_sd, train_auc) %>%
  dplyr::mutate(dplyr::across(where(is.numeric), ~round(., 4)))

print(as.data.frame(results_display))

# Save as CSV for easy viewing
write.csv(results_display, file.path(out_dir, "model_comparison_table.csv"),
          row.names = FALSE)


# ============================================================ #
# SECTION 3: VISUALIZATIONS
# ============================================================ #

# ---- 3a: AUC Bar Chart ----
results_table$model_name <- forcats::fct_reorder(results_table$model_name, results_table$auc)

# Color by model family
results_table <- results_table %>%
  dplyr::mutate(family = dplyr::case_when(
    model_name %in% c("Logistic Regression")           ~ "Linear",
    model_name %in% c("Decision Tree")                  ~ "Tree",
    model_name %in% c("Random Forest", "GBM", "XGBoost") ~ "Ensemble (Trees)",
    model_name %in% c("KNN", "SVM")                    ~ "Distance/Margin",
    model_name %in% c("Neural Network")                 ~ "Neural",
    model_name %in% c("Super Learner")                  ~ "Meta-Ensemble",
    TRUE ~ "Other"
  ))

family_colors <- c(
  "Linear"          = "#90CAF9",
  "Tree"            = "#A5D6A7",
  "Ensemble (Trees)" = "#1565C0",
  "Distance/Margin" = "#FF9800",
  "Neural"          = "#E53935",
  "Meta-Ensemble"   = "#8E24AA",
  "Other"           = "grey60"
)

p_auc <- ggplot(results_table, aes(x = auc, y = model_name, fill = family)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = round(auc, 4)), hjust = -0.1, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = family_colors) +
  scale_x_continuous(limits = c(0, max(results_table$auc, na.rm = TRUE) * 1.08),
                     expand = expansion(mult = c(0, 0))) +
  labs(
    title    = "Test AUC — All Models Compared",
    subtitle = "AUC = area under the ROC curve. Higher = better discrimination.\n0.5 = random guessing. 1.0 = perfect. Colors show model families.",
    x = "Test AUC", y = NULL, fill = "Model Family"
  ) +
  theme_ml

print(p_auc)
ggsave(file.path(out_dir, "01_auc_comparison.png"), p_auc,
       width = 11, height = 8, dpi = 150, bg = "white")


# ---- 3b: ROC Curve Overlay ----
# Build ROC curves for all models that have saved predictions
roc_data_list <- list()

for (model_name in names(all_preds)) {
  probs <- all_preds[[model_name]]
  
  # Verify predictions match test set length
  if (length(probs) != length(test_y)) {
    cat("Warning: predictions for", model_name, "don't match test set length. Skipping.\n")
    next
  }
  
  roc_obj <- pROC::roc(test_y, probs, quiet = TRUE)
  auc_val <- round(as.numeric(pROC::auc(roc_obj)), 3)
  
  roc_data_list[[model_name]] <- tibble(
    fpr   = 1 - roc_obj$specificities,
    tpr   = roc_obj$sensitivities,
    model = paste0(model_name, " (", auc_val, ")")
  )
}

roc_all <- dplyr::bind_rows(roc_data_list)

# Use a distinct color palette
n_roc_models <- length(unique(roc_all$model))
roc_colors <- scales::hue_pal()(n_roc_models)
names(roc_colors) <- unique(roc_all$model)

p_roc <- ggplot(roc_all, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(linetype = "dashed", color = "grey60") +
  scale_color_manual(values = roc_colors) +
  labs(
    title    = "ROC Curves — All Models Overlaid",
    subtitle = "Each curve = one model's tradeoff between sensitivity and false positive rate.\nCurves closer to the top-left corner are better. AUC values in legend.",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model (AUC)"
  ) +
  theme_ml +
  theme(legend.text = element_text(size = 9))

print(p_roc)
ggsave(file.path(out_dir, "02_roc_overlay.png"), p_roc,
       width = 12, height = 9, dpi = 150, bg = "white")


# ---- 3c: Overfitting Check — Train vs Test AUC ----
overfit_df <- results_table %>%
  dplyr::filter(!is.na(train_auc) & !is.na(auc)) %>%
  dplyr::select(model_name, train_auc, auc) %>%
  dplyr::mutate(gap = round(train_auc - auc, 4)) %>%
  tidyr::pivot_longer(c(train_auc, auc), names_to = "dataset", values_to = "auc_val") %>%
  dplyr::mutate(dataset = ifelse(dataset == "train_auc", "Train", "Test"))

if (nrow(overfit_df) > 0) {
  p_overfit <- ggplot(overfit_df, aes(x = auc_val, y = model_name, fill = dataset)) +
    geom_col(position = "dodge", width = 0.6) +
    geom_text(aes(label = round(auc_val, 3)),
              position = position_dodge(width = 0.6), hjust = -0.1, size = 3) +
    scale_fill_manual(values = c("Train" = "#1565C0", "Test" = "#E53935")) +
    scale_x_continuous(limits = c(0, 1.08), expand = expansion(mult = c(0, 0))) +
    labs(
      title    = "Overfitting Check — Train vs Test AUC",
      subtitle = "Large gap (train >> test) = overfitting. Small gap = good generalization.\nDecision trees often overfit. Regularized models (XGBoost, glmnet) generalize better.",
      x = "AUC", y = NULL, fill = "Dataset"
    ) +
    theme_ml
  
  print(p_overfit)
  ggsave(file.path(out_dir, "03_overfitting_comparison.png"), p_overfit,
         width = 11, height = 8, dpi = 150, bg = "white")
}


# ---- 3d: CV Stability ----
cv_df <- results_table %>%
  dplyr::filter(!is.na(cv_auc_mean) & !is.na(cv_auc_sd)) %>%
  dplyr::select(model_name, cv_auc_mean, cv_auc_sd) %>%
  dplyr::arrange(dplyr::desc(cv_auc_mean))

if (nrow(cv_df) > 0) {
  cv_df$model_name <- forcats::fct_reorder(cv_df$model_name, cv_df$cv_auc_mean)
  
  p_cv <- ggplot(cv_df, aes(x = cv_auc_mean, y = model_name)) +
    geom_point(size = 4, color = "#1565C0") +
    geom_errorbarh(aes(xmin = cv_auc_mean - cv_auc_sd,
                       xmax = cv_auc_mean + cv_auc_sd),
                   height = 0.3, color = "#1565C0") +
    geom_text(aes(label = paste0(round(cv_auc_mean, 3), " ± ", round(cv_auc_sd, 3))),
              hjust = -0.2, size = 3.5) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.15))) +
    labs(
      title    = "Cross-Validation AUC — Mean ± SD",
      subtitle = "Error bars = 1 SD across CV folds. Shorter bars = more stable.\nHigh mean + short bar = reliably good model.",
      x = "CV AUC (mean ± SD)", y = NULL
    ) +
    theme_ml
  
  print(p_cv)
  ggsave(file.path(out_dir, "04_cv_stability.png"), p_cv,
         width = 11, height = 7, dpi = 150, bg = "white")
}


# ---- 3e: Precision-Recall Tradeoff ----
pr_df <- results_table %>%
  dplyr::filter(!is.na(precision) & !is.na(recall))

if (nrow(pr_df) > 0) {
  p_pr <- ggplot(pr_df, aes(x = recall, y = precision, color = family)) +
    geom_point(size = 5) +
    ggrepel::geom_text_repel(aes(label = model_name), size = 3.5,
                             fontface = "bold", max.overlaps = 15) +
    scale_color_manual(values = family_colors) +
    labs(
      title    = "Precision vs Recall — The Diagnostic Tradeoff",
      subtitle = "Top-right = catches many cases (high recall) AND few false alarms (high precision).\nWith ~6% prevalence, high precision is hard — most flagged patients won't have disease.",
      x = "Recall (sensitivity — fraction of true cases caught)",
      y = "Precision (fraction of flagged patients who truly have disease)",
      color = "Model Family"
    ) +
    theme_ml
  
  print(p_pr)
  ggsave(file.path(out_dir, "05_precision_recall.png"), p_pr,
         width = 10, height = 8, dpi = 150, bg = "white")
}


# ---- 3f: Multi-Metric Radar/Heatmap ----
# Heatmap of all metrics across all models
metric_cols <- c("auc", "accuracy", "precision", "recall", "f1")
available_metrics <- metric_cols[metric_cols %in% names(results_table)]

heatmap_df <- results_table %>%
  dplyr::select(model_name, dplyr::all_of(available_metrics)) %>%
  tidyr::pivot_longer(-model_name, names_to = "metric", values_to = "value") %>%
  dplyr::filter(!is.na(value))

if (nrow(heatmap_df) > 0) {
  heatmap_df$metric <- toupper(heatmap_df$metric)
  
  p_heatmap <- ggplot(heatmap_df, aes(x = metric, y = model_name, fill = value)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = round(value, 3)), size = 3.5, fontface = "bold") +
    scale_fill_gradient(low = "#FFCDD2", high = "#1B5E20", name = "Score") +
    labs(
      title    = "Performance Heatmap — All Models × All Metrics",
      subtitle = "Green = better performance. Numbers show exact values.\nNo single metric tells the whole story — look at the pattern across all five.",
      x = NULL, y = NULL
    ) +
    theme_ml +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(face = "bold", size = 11))
  
  print(p_heatmap)
  ggsave(file.path(out_dir, "06_performance_heatmap.png"), p_heatmap,
         width = 11, height = 8, dpi = 150, bg = "white")
}


# ============================================================ #
# SECTION 4: STRENGTHS & WEAKNESSES GUIDE
# ============================================================ #

# ---- 4a: Model Characteristics Table ----
characteristics <- tibble(
  model = c("Logistic Regression", "Decision Tree", "Random Forest",
            "XGBoost", "KNN", "SVM", "Neural Network", "Super Learner"),
  interpretability = c("High", "Very High", "Medium", "Low",
                       "Low", "Low", "Low", "Very Low"),
  speed_train = c("Very Fast", "Fast", "Medium", "Medium",
                  "Fast", "Slow", "Slow", "Very Slow"),
  speed_predict = c("Very Fast", "Very Fast", "Fast", "Fast",
                    "Slow", "Medium", "Very Fast", "Medium"),
  handles_nonlinear = c("No", "Yes", "Yes", "Yes",
                        "Yes", "Yes", "Yes", "Yes"),
  needs_scaling = c("No", "No", "No", "No",
                    "Yes", "Yes", "Yes", "No"),
  best_for = c("Baseline, publication",
               "Explanation, if-then rules",
               "Robust general-purpose",
               "Max accuracy on tabular data",
               "Small data, few features",
               "Clear margin separation",
               "Complex interactions, large data",
               "Squeezing last % of performance")
)

cat("\n=== MODEL CHARACTERISTICS ===\n")
print(as.data.frame(characteristics))

write.csv(characteristics, file.path(out_dir, "model_characteristics.csv"),
          row.names = FALSE)


# ---- 4b: Characteristics Heatmap ----
char_score <- tibble(
  model = rep(characteristics$model, 4),
  characteristic = rep(c("Interpretability", "Train Speed",
                         "Nonlinear Handling", "Needs Scaling"), each = 8),
  score = c(
    c(5, 5, 3, 2, 2, 2, 2, 1),   # Interpretability
    c(5, 4, 3, 3, 4, 2, 2, 1),   # Train Speed
    c(1, 3, 5, 5, 4, 4, 5, 5),   # Nonlinear Handling
    c(1, 1, 1, 1, 5, 5, 5, 1)    # Needs Scaling (higher = more needed)
  )
)

char_score$model <- factor(char_score$model, levels = characteristics$model)

p_chars <- ggplot(char_score, aes(x = characteristic, y = model, fill = score)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = score), size = 4, fontface = "bold") +
  scale_fill_gradient(low = "#FFCDD2", high = "#1B5E20",
                      name = "Score\n(5 = best)") +
  labs(
    title    = "Model Characteristics — Beyond Accuracy",
    subtitle = "Accuracy isn't everything. Consider interpretability, speed, and requirements.\n5 = best for that characteristic. Choose based on your priorities.",
    x = NULL, y = NULL
  ) +
  theme_ml +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 20, hjust = 1, size = 11, face = "bold"))

print(p_chars)
ggsave(file.path(out_dir, "07_model_characteristics.png"), p_chars,
       width = 11, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: SAVE RESULTS
# ============================================================ #

saveRDS(results_table, file.path(out_dir, "results_all_models.rds"))
cat("\nAll comparison results saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 6: SUMMARY & RECOMMENDATIONS
# ============================================================ #

# Find best model
best_model <- results_table %>% dplyr::slice_max(auc, n = 1)

cat("\n")
cat("==============================================\n")
cat("   MODEL COMPARISON — FINAL SUMMARY\n")
cat("==============================================\n")
cat("\nModels compared:", n_models, "\n")
cat("Best test AUC:", best_model$model_name[1], "—", round(best_model$auc[1], 4), "\n")

if (n_models >= 3) {
  top3 <- results_table %>% dplyr::slice_head(n = 3)
  cat("\nTop 3 models:\n")
  for (i in 1:nrow(top3)) {
    cat("  ", i, ". ", top3$model_name[i], " — AUC: ", round(top3$auc[i], 4), "\n", sep = "")
  }
}

cat("\nWHAT WE LEARNED ACROSS ALL MODELS:\n")
cat("1.  No single model is always best. The 'best' depends on your\n")
cat("    priorities: accuracy, interpretability, speed, or simplicity.\n")
cat("2.  For TABULAR DATA like ours, tree-based ensembles (XGBoost,\n")
cat("    Random Forest) consistently rank near the top. This is a\n")
cat("    well-established finding in ML research.\n")
cat("3.  LOGISTIC REGRESSION is the essential baseline. It's fast,\n")
cat("    fully interpretable, and often surprisingly competitive.\n")
cat("    Never skip it.\n")
cat("4.  SINGLE DECISION TREES underperform but provide unmatched\n")
cat("    interpretability. Use them for explanation, not prediction.\n")
cat("5.  NEURAL NETWORKS on tabular data rarely beat XGBoost.\n")
cat("    They shine on images, text, and sequences instead.\n")
cat("6.  ENSEMBLES (Super Learner) can squeeze out a small extra\n")
cat("    improvement by combining diverse models. Worth it when\n")
cat("    every fraction of AUC matters.\n")
cat("7.  OVERFITTING is visible in the train-vs-test comparison.\n")
cat("    Models with large gaps (train >> test) memorized training\n")
cat("    data. Regularization and CV prevent this.\n")
cat("8.  CV STABILITY matters as much as raw performance. A model\n")
cat("    with AUC 0.82 ± 0.01 is more trustworthy than 0.83 ± 0.05.\n")
cat("9.  With ~6% disease prevalence, PRECISION is naturally low —\n")
cat("    most flagged patients won't have heart disease. This is\n")
cat("    normal for rare-event prediction, not a model failure.\n")

cat("\nRECOMMENDATION FRAMEWORK:\n")
cat("  Need to explain to clinicians?   → Logistic Regression or Decision Tree\n")
cat("  Need maximum accuracy?            → XGBoost or Super Learner\n")
cat("  Need robust, minimal tuning?      → Random Forest\n")
cat("  Need to understand what causes?   → Script 13 (Causal ML), not prediction\n")
cat("  Need to detect unusual patients?  → Script 08 (Isolation Forest)\n")
cat("  Need to predict when, not if?     → Script 09 (Survival ML)\n")
cat("  Need to forecast over time?       → Script 10 (Time Series)\n")

cat("\n==============================================\n")
cat("  THE ML TEACHING PROJECT — R SCRIPTS COMPLETE\n")
cat("  Scripts 16-20 continue in Python notebooks\n")
cat("==============================================\n")

cat("\nScript 15 complete.\n")