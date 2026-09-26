# ============================================================ #
# 03 — Random Forest
# ============================================================ #
# Purpose: Learn how combining hundreds of decision trees into
#   an ensemble creates one of the most powerful and reliable
#   models in all of machine learning.
#
# Target: heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, readr, purrr,
  ggplot2, patchwork, scales, viridis,
  caret,            # Train/test split, CV
  pROC,             # ROC curves
  randomForest,     # The classic RF implementation
  pdp               # Partial dependence plots
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Random_Forest", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Random_Forest"

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
# SECTION 1: WHAT IS A RANDOM FOREST?
# ============================================================ #
#
# SIMPLE ANALOGY — WISDOM OF CROWDS:
#   Imagine you want to know if a patient has heart disease.
#   You could ask ONE doctor — but that doctor might be wrong.
#   Or you could ask 500 doctors, each looking at the patient's
#   chart from a slightly different angle, and take a VOTE.
#   The majority vote is almost always more accurate than any
#   single doctor's opinion.
#
#   A Random Forest does exactly this, but with decision trees
#   instead of doctors.
#
# WHAT IS A DECISION TREE? (Quick recap for this script)
#   A decision tree is a flowchart of yes/no questions:
#     "Is age > 62?" → Yes → "Is cholesterol > 240?" → Yes → HIGH RISK
#   It splits patients into groups based on their features,
#   and predicts based on which group a patient falls into.
#   The problem: a single tree is UNSTABLE — train it on
#   slightly different data and you get a completely different tree.
#
# HOW A RANDOM FOREST WORKS — STEP BY STEP:
#
#   1. GROW MANY TREES (typically 500):
#      Instead of one tree, grow hundreds of them.
#
#   2. MAKE EACH TREE DIFFERENT (two sources of randomness):
#
#      a) BAGGING (Bootstrap AGGregatING):
#         Each tree is trained on a RANDOM SAMPLE of the data.
#         We draw N rows WITH REPLACEMENT from the training set.
#         This means each tree sees ~63% of the data.
#         The ~37% left out become that tree's "Out-of-Bag" (OOB)
#         sample — a free validation set.
#
#      b) FEATURE RANDOMIZATION:
#         At each split, only a RANDOM SUBSET of features is
#         considered. If we have 13 features, each split might
#         only look at 3-4 of them.
#
#         WHY? Without this, every tree would pick "age" as the
#         first split (it's the strongest predictor), making all
#         trees nearly identical. By forcing trees to sometimes
#         ignore age, they discover patterns in other features
#         like cholesterol, smoking, or blood pressure.
#
#   3. AGGREGATE PREDICTIONS:
#      For classification: each tree VOTES (0 or 1).
#        The majority vote wins. If 312 trees say "heart disease"
#        and 188 say "no," the prediction is "heart disease."
#        The proportion (312/500 = 62.4%) is the probability.
#
#      For regression: take the AVERAGE of all tree predictions.
#
# WHY DOES AVERAGING MANY TREES WORK?
#   Each individual tree is noisy and overfit (memorizes its
#   training sample). BUT their errors are DIFFERENT because each
#   tree saw different data and different features.
#
#   When you average many noisy-but-different predictions,
#   the random errors cancel out, leaving the true signal.
#   Think of it like static on a radio — average enough
#   recordings and the static disappears, leaving the music.
#
# WHEN RANDOM FOREST WORKS WELL:
#   - Tabular data (spreadsheets, databases, clinical data)
#   - Medium to large datasets
#   - Mix of numeric and categorical features
#   - When you want strong performance with minimal tuning
#   - Often the FIRST model to try on a new problem
#
# WHEN IT STRUGGLES:
#   - Very high-dimensional sparse data (text, genomics)
#   - When you need a simple, interpretable model
#   - Extrapolation (predicting outside the training range)
#   - Very small datasets (not enough data to split among trees)
#
# KEY HYPERPARAMETERS (settings you can tune):
#   ntree    = number of trees (more = better but slower; 500 is default)
#   mtry     = features considered per split (default = sqrt(total features))
#   nodesize = minimum observations in each leaf (controls tree complexity)
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #

# ---- 2A: Load Data ----

ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Full dataset:", nrow(ml_data), "rows x", ncol(ml_data), "columns\n")
cat("Heart disease prevalence:", round(mean(ml_data$heart_disease) * 100, 1), "%\n")

target <- "heart_disease"

features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)


# ---- 2B: Train/Test Split ----
# WHY DO WE SPLIT THE DATA?
#   We need to know how the model performs on patients it has NEVER
#   seen. If we only tested on the same patients we trained on, the
#   model could just memorize the answers and look artificially perfect.
#
#   The test set simulates deploying the model on new patients in a
#   real clinic. We NEVER touch the test set during training.
#
#   We use 75% for training and 25% for testing.
#   "Stratified" split means both sets have the same heart disease rate.

set.seed(42)

train_index <- caret::createDataPartition(
  ml_data[[target]],
  p    = 0.75,
  list = FALSE
)

train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train HD rate:", round(mean(train_df[[target]]) * 100, 1), "%\n")
cat("Test HD rate: ", round(mean(test_df[[target]]) * 100, 1), "%\n")


# ---- 2C: Preprocessing ----
# GOOD NEWS: Random Forests need ZERO preprocessing!
#
# WHY NOT?
#   - Trees split on thresholds ("is age > 62?"). The actual
#     numeric scale doesn't matter — only the ORDERING matters.
#     So scaling/centering does nothing.
#   - Trees handle categorical features natively (they split on
#     "is sex = Male?"). No need for one-hot encoding.
#   - Trees aren't affected by outliers (an extreme value just
#     ends up in its own leaf).
#   - Trees don't assume any distribution shape.
#
#   This is a major advantage over models like KNN (which measures
#   distance between points, so scale matters) or logistic regression
#   (which assumes linear relationships in log-odds).
#
#   We just need factors to have proper levels.

train_df <- train_df %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")),
    smoking_status = factor(smoking_status, levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use, levels = c("None", "Light", "Moderate", "Heavy"))
  )

test_df <- test_df %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")),
    smoking_status = factor(smoking_status, levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use, levels = c("None", "Light", "Moderate", "Heavy"))
  )

# randomForest requires the target to be a FACTOR for classification.
# WHY? The randomForest package uses the class of the outcome variable to
# decide what kind of model to build:
#   - If the target is numeric (0, 1): it builds a REGRESSION forest
#     (predicts numbers, not classes — wrong for our purpose)
#   - If the target is a factor ("0", "1"): it builds a CLASSIFICATION forest
#     (predicts categories and outputs class probabilities — what we want)
#
# We explicitly set levels = c("0", "1") so that:
#   - "0" (no heart disease) is the first class — the "reference"
#   - "1" (has heart disease) is the second class — the "positive" case
# This matters later when we extract probabilities: predict(..., type="prob")
# returns a matrix with columns named "0" and "1", and we always want column "1".
train_df[[target]] <- factor(train_df[[target]], levels = c("0", "1"))
test_df[[target]]  <- factor(test_df[[target]], levels = c("0", "1"))

model_formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

cat("Data ready — no scaling or encoding needed for tree-based models.\n")


# ============================================================ #
# SECTION 3: FIT THE RANDOM FOREST
# ============================================================ #

# ---- 3A: Train with Default Settings ----
# ntree = 500: grow 500 trees (a solid default)
# mtry = sqrt(13) ≈ 3: consider 3 random features at each split
# importance = TRUE: compute feature importance (slightly slower but essential)
# nodesize = 5: each leaf must have at least 5 patients

set.seed(42)

rf_model <- randomForest::randomForest(
  model_formula,
  data       = train_df,
  ntree      = 500,
  mtry       = floor(sqrt(length(features))),
  importance = TRUE,
  nodesize   = 5,
  keep.forest = TRUE
)

cat("\n--- Random Forest Summary ---\n")
print(rf_model)
cat("mtry (features per split):", rf_model$mtry, "out of", length(features), "total\n")


# ---- 3B: Out-of-Bag (OOB) Error — The "Free" Validation ----
# WHAT IS OOB ERROR?
#   Each tree was trained on a bootstrap sample (~63% of data).
#   The remaining ~37% — the "Out-of-Bag" (OOB) observations —
#   were NOT used to build that tree.
#
#   For each patient, we can collect predictions from ONLY the
#   trees that DIDN'T train on that patient. This gives us a
#   validation estimate computed for free — no separate CV needed!
#
#   OOB error ≈ what we'd get from cross-validation, but it's
#   computed automatically during training. This is one of
#   Random Forest's superpowers.

oob_error <- rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]
cat("\nOOB error rate:", round(oob_error * 100, 2), "%\n")
cat("OOB accuracy:", round((1 - oob_error) * 100, 2), "%\n")


# ============================================================ #
# SECTION 4: HOW MANY TREES DO WE NEED?
# ============================================================ #
# As we add more trees, the OOB error stabilizes. Let's plot
# this to find the point of diminishing returns.
#
# WHAT TO EXPECT:
#   - Error drops sharply with the first 50-100 trees
#   - Then levels off (adding tree #400 barely helps)
#   - Unlike a single tree, MORE trees NEVER hurts (no overfitting
#     from adding more trees — only diminishing returns)
# ============================================================ #

# rf_model$err.rate has OOB error after each tree was added
oob_df <- tibble(
  n_trees   = 1:nrow(rf_model$err.rate),
  oob_error = rf_model$err.rate[, "OOB"],
  class_0   = rf_model$err.rate[, "0"],
  class_1   = rf_model$err.rate[, "1"]
)

p_ntree <- oob_df %>%
  tidyr::pivot_longer(
    cols = c(oob_error, class_0, class_1),
    names_to = "error_type", values_to = "error"
  ) %>%
  dplyr::mutate(
    error_type = dplyr::case_when(
      error_type == "oob_error" ~ "Overall OOB Error",
      error_type == "class_0"   ~ "Error on Non-HD patients",
      error_type == "class_1"   ~ "Error on HD patients"
    )
  ) %>%
  ggplot(aes(x = n_trees, y = error, color = error_type)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = c(
    "Overall OOB Error"        = "#212121",
    "Error on Non-HD patients" = "#2196F3",
    "Error on HD patients"     = "#F44336"
  )) +
  labs(
    title    = "OOB Error vs Number of Trees",
    subtitle = paste(
      "Error stabilizes after ~100-200 trees. More trees never hurts, just slows computation.\n",
      "Notice: error on heart disease patients (red) is much higher — the model struggles\n",
      "with the rare class (only ~6% of patients have heart disease)."
    ),
    x = "Number of Trees", y = "Error Rate", color = NULL
  ) +
  theme_ml

print(p_ntree)
ggsave(file.path(out_dir, "01_oob_error_vs_ntree.png"), p_ntree,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: TUNING mtry — HOW MANY FEATURES PER SPLIT?
# ============================================================ #
# mtry is the most important hyperparameter in Random Forest.
# It controls how "different" the trees are from each other.
#
# LOW mtry (e.g., 2):  Trees are very different (more random),
#   which reduces correlation between trees. BUT each tree is
#   weaker individually because it often can't use the best features.
#
# HIGH mtry (e.g., 13): Trees are more similar (less random),
#   each tree is stronger individually. BUT the trees make similar
#   errors, so averaging helps less.
#
# The sweet spot balances tree strength vs tree diversity.
# Default for classification: sqrt(total features)
# Let's test several values and see which works best.
# ============================================================ #

set.seed(42)

mtry_values <- 2:length(features)  # Test every value from 2 up to 13 (all features)

# HOW THIS LOOP WORKS:
#   map_dfr() runs the code inside for each value of m (2, 3, 4, ... 13).
#   For each m, it:
#     1. Trains a smaller forest (300 trees for speed — enough to get a stable OOB estimate)
#     2. Extracts the final OOB error rate (the last row of err.rate = after all 300 trees)
#     3. Returns a one-row tibble with the mtry value and its OOB error
#   map_dfr() then stacks all 12 one-row tibbles into a single 12-row table.
#   We then find the mtry with the lowest OOB error.

mtry_results <- purrr::map_dfr(mtry_values, function(m) {
  rf_temp <- randomForest::randomForest(
    model_formula,
    data     = train_df,
    ntree    = 300,        # Fewer trees for speed during tuning — 300 is enough
    mtry     = m,          # This is the value we're testing
    nodesize = 5
  )
  
  # err.rate has one row per tree — the last row is after all 300 trees are grown
  # We take the "OOB" column = overall out-of-bag error rate
  oob_err <- rf_temp$err.rate[nrow(rf_temp$err.rate), "OOB"]
  
  tibble(mtry = m, oob_error = oob_err)
})

best_mtry <- mtry_results$mtry[which.min(mtry_results$oob_error)]

p_mtry <- mtry_results %>%
  ggplot(aes(x = mtry, y = oob_error)) +
  geom_line(color = "#5C6BC0", linewidth = 1.2) +
  geom_point(color = "#5C6BC0", size = 3) +
  geom_point(
    data = mtry_results %>% dplyr::filter(mtry == best_mtry),
    color = "#F44336", size = 5, shape = 18
  ) +
  annotate("text",
           x = best_mtry + 0.5, y = min(mtry_results$oob_error),
           label = paste("Best mtry =", best_mtry),
           hjust = 0, fontface = "bold", color = "#F44336"
  ) +
  labs(
    title    = "Tuning mtry — Features Considered Per Split",
    subtitle = paste(
      "Low mtry = diverse but weak trees. High mtry = strong but similar trees.\n",
      "The sweet spot minimizes OOB error."
    ),
    x = "mtry (features per split)", y = "OOB Error Rate"
  ) +
  theme_ml

print(p_mtry)
ggsave(file.path(out_dir, "02_mtry_tuning.png"), p_mtry,
       width = 9, height = 6, dpi = 150, bg = "white")

cat("Best mtry:", best_mtry, "with OOB error:",
    round(min(mtry_results$oob_error) * 100, 2), "%\n")


# ---- 5B: Retrain with Best mtry ----

set.seed(42)

rf_tuned <- randomForest::randomForest(
  model_formula,
  data       = train_df,
  ntree      = 500,
  mtry       = best_mtry,
  importance = TRUE,
  nodesize   = 5,
  keep.forest = TRUE
)

cat("Refit with mtry =", best_mtry, "| OOB error:",
    round(rf_tuned$err.rate[500, "OOB"] * 100, 2), "%\n")


# ============================================================ #
# SECTION 6: PREDICTIONS & EVALUATION
# ============================================================ #

# ---- 6A: Generate Predictions ----
# type = "prob" tells randomForest to return PROBABILITIES, not class labels.
#
# What does predict() actually return here?
#   A matrix with one row per patient and two columns:
#     Column "0": the proportion of the 500 trees that voted "no heart disease"
#     Column "1": the proportion of the 500 trees that voted "heart disease"
#   These two columns always sum to 1.0 for each patient.
#
#   Example for one patient:
#     "0" = 0.74,  "1" = 0.26
#   This means 370 of 500 trees voted "no HD", 130 voted "HD."
#   The probability of heart disease for this patient = 0.26 (26%).
#
# We take column "1" — the probability of heart disease — because that's
# what we need for the ROC curve and threshold selection.
# (We never directly need column "0": it's just 1 - column "1".)
#
# We predict on BOTH training and test data so we can check for overfitting
# in Section 7: if train probability is much higher than test, the model
# memorized the training data rather than learning general patterns.

train_probs <- predict(rf_tuned, newdata = train_df, type = "prob")[, "1"]
test_probs  <- predict(rf_tuned, newdata = test_df, type = "prob")[, "1"]

cat("\nTest predictions — Mean:", round(mean(test_probs), 3),
    "| Range:", round(min(test_probs), 3), "-", round(max(test_probs), 3), "\n")
cat("(Mean should be close to heart disease prevalence of ~6%)\n")


# ---- 6B: ROC Curve & AUC ----
# WHAT IS A ROC CURVE?
#   The model outputs a PROBABILITY for each patient (e.g., 0.35).
#   To make a yes/no decision, we need a THRESHOLD: if P > threshold,
#   predict "heart disease."
#
#   The ROC curve shows what happens at EVERY possible threshold:
#     - Y-axis: True Positive Rate (sensitivity) — of all HD patients,
#       what % did we correctly catch?
#     - X-axis: False Positive Rate — of all healthy patients,
#       what % did we incorrectly flag?
#
#   The curve shows the tradeoff: catching more true cases also
#   means more false alarms.
#
#   AUC (Area Under the Curve) summarizes overall performance:
#     0.5 = random guessing (the diagonal line)
#     0.7-0.8 = acceptable
#     0.8-0.9 = good
#     0.9+    = excellent

roc_obj <- pROC::roc(as.numeric(as.character(test_df[[target]])), test_probs, quiet = TRUE)
auc_val <- round(as.numeric(pROC::auc(roc_obj)), 3)

roc_df <- tibble(
  fpr         = 1 - roc_obj$specificities,
  sensitivity = roc_obj$sensitivities
)

# Optionally load previous model results for comparison
logistic_path <- "Outputs/Logistic_Regression/results_logistic.rds"
tree_path     <- "Outputs/Decision_Trees/results_tree.rds"
has_logistic  <- file.exists(logistic_path)
has_tree      <- file.exists(tree_path)

if (has_logistic) {
  logistic_results    <- readRDS(logistic_path)
  logistic_test_probs <- readRDS("Outputs/Logistic_Regression/test_predictions_logistic.rds")
  logistic_roc        <- pROC::roc(as.numeric(as.character(test_df[[target]])), logistic_test_probs, quiet = TRUE)
  logistic_roc_df     <- tibble(fpr = 1 - logistic_roc$specificities, sensitivity = logistic_roc$sensitivities)
}

if (has_tree) {
  tree_results    <- readRDS(tree_path)
  tree_test_probs <- readRDS("Outputs/Decision_Trees/test_predictions_tree.rds")
  tree_roc        <- pROC::roc(as.numeric(as.character(test_df[[target]])), tree_test_probs, quiet = TRUE)
  tree_roc_df     <- tibble(fpr = 1 - tree_roc$specificities, sensitivity = tree_roc$sensitivities)
}

p_roc <- ggplot() +
  geom_line(data = roc_df, aes(x = fpr, y = sensitivity, color = "Random Forest"),
            linewidth = 1.3) +
  {if (has_logistic)
    geom_line(data = logistic_roc_df,
              aes(x = fpr, y = sensitivity, color = "Logistic Regression"),
              linewidth = 0.9, linetype = "dashed")
  } +
  {if (has_tree)
    geom_line(data = tree_roc_df,
              aes(x = fpr, y = sensitivity, color = "Decision Tree"),
              linewidth = 0.9, linetype = "dotted")
  } +
  geom_abline(linetype = "dotted", color = "grey50") +
  scale_color_manual(values = c(
    "Random Forest"       = "#4CAF50",
    "Logistic Regression" = "#2196F3",
    "Decision Tree"       = "#FF9800"
  )) +
  annotate("text", x = 0.55, y = 0.45,
           label = paste("RF AUC =", auc_val),
           size = 5, fontface = "bold", color = "#4CAF50") +
  {if (has_logistic)
    annotate("text", x = 0.55, y = 0.35,
             label = paste("Logistic AUC =", logistic_results$auc),
             size = 4, color = "#2196F3")
  } +
  {if (has_tree)
    annotate("text", x = 0.55, y = 0.25,
             label = paste("Tree AUC =", tree_results$auc),
             size = 4, color = "#FF9800")
  } +
  labs(
    title    = "ROC Curve — Random Forest",
    subtitle = paste(
      "Closer to the top-left corner = better discrimination.\n",
      "The diagonal = random guessing (AUC = 0.5). A perfect model hugs the top-left."
    ),
    x = "False Positive Rate\n(% of healthy patients incorrectly flagged)",
    y = "True Positive Rate (Sensitivity)\n(% of heart disease patients correctly caught)",
    color = "Model"
  ) +
  theme_ml

print(p_roc)
ggsave(file.path(out_dir, "03_roc_curve.png"), p_roc,
       width = 9, height = 7, dpi = 150, bg = "white")

cat("\nRandom Forest AUC:", auc_val, "\n")
if (has_logistic) cat("Logistic Regression AUC:", logistic_results$auc, "\n")
if (has_tree) cat("Decision Tree AUC:", tree_results$auc, "\n")


# ---- 6C: Threshold Selection & Confusion Matrix ----
# WHAT IS A CONFUSION MATRIX?
#   The model gives probabilities. To make a yes/no decision, we pick
#   a THRESHOLD (e.g., 0.5). Everyone above = predicted positive.
#
#   The confusion matrix shows the 4 possible outcomes:
#     True Positive (TP):  Patient HAS HD, model said YES — correctly caught
#     False Positive (FP): Patient is HEALTHY, model said YES — false alarm
#     True Negative (TN):  Patient is HEALTHY, model said NO — correctly ruled out
#     False Negative (FN): Patient HAS HD, model said NO — MISSED case
#
#   From these we calculate:
#     Precision = TP / (TP + FP) — "Of those we flagged, how many truly have HD?"
#     Recall    = TP / (TP + FN) — "Of those who have HD, how many did we catch?"
#     F1        = harmonic mean of Precision and Recall (balances both)
#
#   Youden's J statistic picks the threshold that maximizes
#   (Sensitivity + Specificity - 1) — a good general-purpose default.

optimal_coords <- pROC::coords(roc_obj, "best",
                               ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold

test_pred_class <- ifelse(test_probs >= optimal_threshold, 1, 0)
test_actual     <- as.numeric(as.character(test_df[[target]]))

cm <- table(Actual = test_actual, Predicted = test_pred_class)
tp <- cm["1", "1"]; fp <- cm["0", "1"]; tn <- cm["0", "0"]; fn <- cm["1", "0"]

accuracy  <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- Classification Metrics (threshold =", round(optimal_threshold, 3), ") ---\n")
cat("Accuracy: ", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), " — Of those flagged, % truly have HD\n")
cat("Recall:   ", round(recall, 3), " — Of HD patients, % we caught\n")
cat("F1 Score: ", round(f1, 3), " — Balance of precision and recall\n")

# Visualize confusion matrix with plain-English labels
cm_df <- tibble(
  actual    = factor(c(0, 0, 1, 1), labels = c("No HD", "Has HD")),
  predicted = factor(c(0, 1, 0, 1), labels = c("No HD", "Has HD")),
  count     = c(tn, fp, fn, tp),
  correct   = c("Correct", "Incorrect", "Incorrect", "Correct"),
  meaning   = c(
    paste0("True Negative\n(correctly ruled out)\nn = ", scales::comma(tn)),
    paste0("False Positive\n(false alarm)\nn = ", scales::comma(fp)),
    paste0("False Negative\n(MISSED case)\nn = ", scales::comma(fn)),
    paste0("True Positive\n(correctly caught)\nn = ", scales::comma(tp))
  )
)

p_confusion <- ggplot(cm_df, aes(x = predicted, y = actual, fill = correct)) +
  geom_tile(alpha = 0.8, color = "white", linewidth = 2) +
  geom_text(aes(label = meaning), size = 4, fontface = "bold", lineheight = 1.1) +
  scale_fill_manual(values = c("Correct" = "#4CAF50", "Incorrect" = "#F44336")) +
  labs(
    title    = "Confusion Matrix — Random Forest",
    subtitle = paste0(
      "Accuracy = ", round(accuracy, 3),
      " | Precision = ", round(precision, 3),
      " | Recall = ", round(recall, 3),
      " | F1 = ", round(f1, 3)
    ),
    x = "What the model PREDICTED", y = "What ACTUALLY happened"
  ) +
  theme_ml +
  theme(legend.position = "none")

print(p_confusion)
ggsave(file.path(out_dir, "04_confusion_matrix.png"), p_confusion,
       width = 8, height = 7, dpi = 150, bg = "white")


# ---- 6D: Predicted Probability Distribution ----

p_prob_dist <- tibble(
  probability = test_probs,
  actual      = factor(test_actual, labels = c("No HD", "Has HD"))
) %>%
  ggplot(aes(x = probability, fill = actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", linewidth = 0.8) +
  scale_fill_manual(values = c("No HD" = "#2196F3", "Has HD" = "#F44336")) +
  labs(
    title    = "Distribution of Predicted Probabilities — Random Forest",
    subtitle = paste(
      "RF produces SMOOTHER probabilities than a single decision tree.\n",
      "A single tree can only predict a few distinct values (one per leaf node),\n",
      "but RF averages 500 trees, creating a more continuous probability spectrum."
    ),
    x = "Predicted P(Heart Disease)", y = "Count", fill = "Actual Status"
  ) +
  theme_ml

print(p_prob_dist)
ggsave(file.path(out_dir, "05_predicted_probabilities.png"), p_prob_dist,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: OVERFITTING CHECK
# ============================================================ #
# WHAT IS OVERFITTING?
#   A model "overfits" when it memorizes the training data —
#   including its noise and quirks — rather than learning the
#   real underlying pattern. An overfit model performs great on
#   training data but poorly on new, unseen data.
#
#   We check for overfitting by comparing TRAINING performance
#   to TEST performance. A big gap = overfitting.
#
# WHY RANDOM FORESTS RESIST OVERFITTING:
#   Each individual tree in the forest IS overfit. But because
#   each tree is overfit in a DIFFERENT WAY (different data,
#   different features), their errors cancel out when averaged.
#   This is the fundamental magic of ensemble methods.
#
#   Adding more trees NEVER causes overfitting in a Random Forest.
#   (It can waste computation, but it won't hurt accuracy.)
# ============================================================ #

train_roc <- pROC::roc(as.numeric(as.character(train_df[[target]])), train_probs, quiet = TRUE)
train_auc <- round(as.numeric(pROC::auc(train_roc)), 3)

cat("\n--- Overfitting Check ---\n")
cat("Train AUC:", train_auc, "\n")
cat("Test AUC: ", auc_val, "\n")
cat("Gap:", round(train_auc - auc_val, 3), "\n")

if (train_auc - auc_val > 0.05) {
  cat("Notable gap — some overfitting. Individual trees memorize training data,\n")
  cat("but the ensemble averages most of this out.\n")
} else {
  cat("Small gap — the ensemble effectively prevents overfitting.\n")
}


# ============================================================ #
# SECTION 8: FEATURE IMPORTANCE
# ============================================================ #
# WHAT IS FEATURE IMPORTANCE?
#   "Which features matter most for prediction?"
#   Feature importance tells us how much each variable contributes
#   to the model's predictions. This is crucial for understanding
#   WHAT the model learned — even if we can't read every tree.
#
# RANDOM FOREST COMPUTES TWO TYPES:
#
#   1. MEAN DECREASE IN ACCURACY (Permutation Importance):
#      Take a feature (e.g., age). Randomly SHUFFLE its values
#      across patients (breaking its relationship with the outcome).
#      How much does the model's accuracy DROP?
#      Big drop = that feature was important.
#      Small drop = the model doesn't rely on it much.
#      This is the most reliable importance measure.
#
#   2. MEAN DECREASE IN GINI (Impurity Importance):
#      How much does each feature improve the "purity" of splits
#      across all trees? Features used more often for better splits
#      get higher Gini importance. Faster to compute but can be
#      biased toward features with many categories.
# ============================================================ #

# Extract both importance types
imp_df <- tibble(
  feature        = rownames(randomForest::importance(rf_tuned)),
  mda            = randomForest::importance(rf_tuned)[, "MeanDecreaseAccuracy"],
  gini           = randomForest::importance(rf_tuned)[, "MeanDecreaseGini"]
)

# Plot Mean Decrease in Accuracy (more reliable)
p_imp_mda <- imp_df %>%
  dplyr::mutate(feature = forcats::fct_reorder(feature, mda)) %>%
  ggplot(aes(x = mda, y = feature)) +
  geom_col(fill = "#5C6BC0", alpha = 0.85) +
  labs(
    title    = "Feature Importance — Mean Decrease in Accuracy",
    subtitle = paste(
      "If we randomly shuffle a feature's values, how much does accuracy drop?\n",
      "Bigger bar = the model relies heavily on this feature."
    ),
    x = "Mean Decrease in Accuracy (higher = more important)", y = NULL
  ) +
  theme_ml

# Plot Gini Importance
p_imp_gini <- imp_df %>%
  dplyr::mutate(feature = forcats::fct_reorder(feature, gini)) %>%
  ggplot(aes(x = gini, y = feature)) +
  geom_col(fill = "#26A69A", alpha = 0.85) +
  labs(
    title    = "Feature Importance — Mean Decrease in Gini",
    subtitle = paste(
      "How much does each feature improve the purity of tree splits?\n",
      "Features used for more/better splits get higher scores."
    ),
    x = "Mean Decrease in Gini (higher = more important)", y = NULL
  ) +
  theme_ml

p_importance <- p_imp_mda | p_imp_gini

print(p_importance)
ggsave(file.path(out_dir, "06_feature_importance.png"), p_importance,
       width = 14, height = 7, dpi = 150, bg = "white")

cat("\n--- Top 5 Features (by Mean Decrease in Accuracy) ---\n")
imp_df %>%
  dplyr::arrange(desc(mda)) %>%
  dplyr::slice_head(n = 5) %>%
  dplyr::mutate(mda = round(mda, 4), gini = round(gini, 2)) %>%
  print()


# ============================================================ #
# SECTION 9: PARTIAL DEPENDENCE PLOTS
# ============================================================ #
# WHAT IS A PARTIAL DEPENDENCE PLOT (PDP)?
#   Feature importance tells us WHICH features matter, but not
#   HOW they affect the prediction.
#
#   A PDP answers: "As age increases from 20 to 90, how does
#   the predicted probability of heart disease change?"
#
# HOW IT WORKS:
#   1. Pick a feature (e.g., age)
#   2. Create a grid of values (age = 20, 25, 30, ..., 90)
#   3. For each grid value, SET every patient's age to that value
#      (leave all other features unchanged)
#   4. Predict heart disease probability for all patients
#   5. AVERAGE the predictions
#   6. Plot average prediction vs the grid value
#
#   This shows the MARGINAL effect of that feature, averaged
#   over all other features. It's like asking: "All else being
#   equal, what does increasing age do to heart disease risk?"
# ============================================================ #

# Compute partial dependence for the top 4 most important features.
# We focus on the top 4 to keep the plot readable — PDPs for all 13 features
# would be overwhelming. The most important features have the most informative PDPs.

top_features <- imp_df %>%
  dplyr::arrange(desc(mda)) %>%
  dplyr::slice_head(n = 4) %>%
  dplyr::pull(feature)

cat("\nComputing partial dependence for:", paste(top_features, collapse = ", "), "\n")

# HOW THIS LOOP WORKS:
#   map() runs the code inside for each feature name in top_features.
#   For each feature, it:
#     1. Checks if the feature is numeric or categorical
#     2. Calls pdp::partial() — this is the function that actually computes
#        the partial dependence by:
#          a. Creating a grid of values for the feature (e.g., age = 20 to 90)
#          b. Setting every patient's age to each grid value (one at a time)
#          c. Predicting heart disease probability for all patients at that age
#          d. Averaging those predictions
#     3. Plots the result as a line (numeric) or bar chart (categorical)
#     4. Returns the plot object
#   map() returns a list of 4 plots, which wrap_plots() then arranges into a grid.
#
# WHY purrr::possibly()?
#   PDPs can occasionally fail for certain feature types (e.g., ordered factors
#   with unusual levels). possibly() wraps the function so that if it errors,
#   it returns NULL instead of crashing the whole script. We filter NULLs out
#   afterward with purrr::compact().

pdp_list <- purrr::map(top_features, function(feat) {
  # Only compute PDP for numeric features (categorical PDPs need special handling below)
  if (is.numeric(train_df[[feat]])) {
    pd <- pdp::partial(
      rf_tuned,
      pred.var = feat,
      train    = train_df,
      prob     = TRUE,       # Output probabilities (not log-odds)
      which.class = "1",     # P(heart_disease = 1)
      grid.resolution = 30
    )
    
    ggplot(pd, aes(x = .data[[feat]], y = yhat)) +
      geom_line(color = "#F44336", linewidth = 1.3) +
      labs(
        title = paste("Partial Dependence —", feat),
        subtitle = "How predicted P(Heart Disease) changes as this feature varies",
        x = feat, y = "Avg Predicted P(HD)"
      ) +
      theme_ml +
      theme(plot.title = element_text(size = 12))
  } else {
    # For categorical features, use a bar plot
    pd <- pdp::partial(
      rf_tuned,
      pred.var = feat,
      train    = train_df,
      prob     = TRUE,
      which.class = "1"
    )
    
    ggplot(pd, aes(x = .data[[feat]], y = yhat)) +
      geom_col(fill = "#F44336", alpha = 0.8) +
      labs(
        title = paste("Partial Dependence —", feat),
        subtitle = "Average predicted risk for each category",
        x = feat, y = "Avg Predicted P(HD)"
      ) +
      theme_ml +
      theme(
        plot.title = element_text(size = 12),
        axis.text.x = element_text(angle = 30, hjust = 1)
      )
  }
})

# Remove any NULLs and combine
pdp_list <- purrr::compact(pdp_list)

if (length(pdp_list) >= 2) {
  p_pdp <- wrap_plots(pdp_list, ncol = 2) +
    plot_annotation(
      title = "Partial Dependence Plots — How Features Affect Predictions",
      subtitle = "Each plot shows the isolated effect of one feature, averaged over all patients.",
      theme = theme_annotation
    )
  
  print(p_pdp)
  ggsave(file.path(out_dir, "07_partial_dependence.png"), p_pdp,
         width = 12, height = 10, dpi = 150, bg = "white")
}


# ============================================================ #
# SECTION 10: CROSS-VALIDATION
# ============================================================ #
# WHAT IS CROSS-VALIDATION?
#   A single train/test split might be "lucky" or "unlucky."
#   Cross-validation repeats the process multiple times:
#     1. Split TRAINING data into 5 equal chunks ("folds")
#     2. Train on 4 folds, evaluate on the 1 held-out fold
#     3. Rotate which fold is held out
#     4. Average the 5 performance scores
#
#   This gives a more STABLE estimate of how good the model
#   really is. If all 5 folds give similar AUC, the model is
#   reliable. If they vary wildly, the model is unstable.
#
#   NOTE: Random Forest already has OOB error (Section 3B),
#   which approximates CV for free. But explicit CV lets us
#   compute AUC and compare directly to other models.
# ============================================================ #

set.seed(42)
cv_folds <- caret::createFolds(train_df[[target]], k = 5, returnTrain = TRUE)

cv_results <- purrr::map_dfr(seq_along(cv_folds), function(i) {
  fold_train <- train_df[cv_folds[[i]], ]
  fold_val   <- train_df[-cv_folds[[i]], ]
  
  fold_rf <- randomForest::randomForest(
    model_formula,
    data     = fold_train,
    ntree    = 300,
    mtry     = best_mtry,
    nodesize = 5
  )
  
  fold_probs  <- predict(fold_rf, newdata = fold_val, type = "prob")[, "1"]
  fold_actual <- as.numeric(as.character(fold_val[[target]]))
  fold_auc    <- as.numeric(pROC::auc(pROC::roc(fold_actual, fold_probs, quiet = TRUE)))
  
  tibble(fold = i, auc = fold_auc)
})

cat("\n--- 5-Fold Cross-Validation AUC ---\n")
print(cv_results)
cat("Mean CV AUC:", round(mean(cv_results$auc), 3),
    "± SD:", round(sd(cv_results$auc), 3), "\n")
cat("Test AUC:", auc_val, "\n")

p_cv <- ggplot(cv_results, aes(x = factor(fold), y = auc)) +
  geom_col(fill = "#4CAF50", alpha = 0.85, width = 0.6) +
  geom_hline(yintercept = mean(cv_results$auc), linetype = "dashed", color = "#F44336") +
  geom_hline(yintercept = auc_val, linetype = "dotted", color = "#212121") +
  annotate("text", x = 5.4, y = mean(cv_results$auc),
           label = paste("CV Mean:", round(mean(cv_results$auc), 3)),
           hjust = 0, color = "#F44336", fontface = "bold", size = 3.5) +
  annotate("text", x = 5.4, y = auc_val,
           label = paste("Test:", auc_val),
           hjust = 0, color = "#212121", fontface = "bold", size = 3.5) +
  coord_cartesian(xlim = c(0.5, 6.5), ylim = c(
    min(cv_results$auc) - 0.02,
    max(c(cv_results$auc, auc_val)) + 0.02
  )) +
  labs(
    title    = "Cross-Validation AUC — Random Forest",
    subtitle = "Consistent bars = stable model. CV mean ≈ test AUC = model generalizes well.",
    x = "Fold", y = "AUC"
  ) +
  theme_ml

print(p_cv)
ggsave(file.path(out_dir, "08_cross_validation.png"), p_cv,
       width = 8, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 11: SAVE RESULTS
# ============================================================ #

results_rf <- tibble(
  model          = "Random Forest",
  target         = "heart_disease",
  auc            = auc_val,
  accuracy       = round(accuracy, 4),
  precision      = round(precision, 4),
  recall         = round(recall, 4),
  f1             = round(f1, 4),
  cv_auc_mean    = round(mean(cv_results$auc), 4),
  cv_auc_sd      = round(sd(cv_results$auc), 4),
  train_auc      = train_auc,
  threshold_used = round(optimal_threshold, 4)
)

saveRDS(results_rf, file.path(out_dir, "results_rf.rds"))
saveRDS(test_probs, file.path(out_dir, "test_predictions_rf.rds"))

cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 12: SUMMARY & KEY TAKEAWAYS
# ============================================================ #

cat("\n")
cat("=== RANDOM FOREST — SUMMARY ===\n")
cat("Test AUC:", auc_val, " — ", dplyr::case_when(
  auc_val >= 0.9 ~ "Excellent discrimination",
  auc_val >= 0.8 ~ "Good discrimination",
  auc_val >= 0.7 ~ "Acceptable discrimination",
  TRUE           ~ "Poor discrimination"
), "\n")
cat("F1:", round(f1, 3), "| Precision:", round(precision, 3), "| Recall:", round(recall, 3), "\n")
cat("CV AUC:", round(mean(cv_results$auc), 3), "±", round(sd(cv_results$auc), 3), "\n")
cat("Best mtry:", best_mtry, "\n")

if (has_logistic | has_tree) {
  cat("\n--- Comparison ---\n")
  if (has_logistic) cat("Logistic Regression AUC:", logistic_results$auc,
                        " — (a simple model that draws a straight decision boundary)\n")
  if (has_tree) cat("Decision Tree AUC:", tree_results$auc,
                    " — (a single tree that splits data into groups)\n")
  cat("Random Forest AUC:", auc_val,
      " — (500 trees averaged together)\n")
}

cat("\n")
cat("WHAT WE LEARNED:\n")
cat("1. A Random Forest grows many trees on random data subsets and random feature subsets.\n")
cat("2. Individual trees overfit, but averaging them cancels out random errors.\n")
cat("3. OOB error is a 'free' validation metric — no separate CV needed.\n")
cat("4. mtry (features per split) is the key hyperparameter to tune.\n")
cat("5. Feature importance tells us WHICH variables matter; PDPs tell us HOW they matter.\n")
cat("6. RF needs NO preprocessing (no scaling, no encoding) — a major practical advantage.\n")
cat("7. RF typically outperforms single trees and often matches or beats logistic regression.\n")
cat("\n")
cat("WHEN TO USE RANDOM FOREST:\n")
cat("- It's an excellent 'first try' model for any tabular dataset.\n")
cat("- When you want strong performance with minimal preprocessing and tuning.\n")
cat("- When interpretability is less critical than accuracy.\n")
cat("\n")
cat("WHAT RANDOM FOREST CANNOT DO:\n")
cat("- It doesn't learn from its mistakes sequentially.\n")
cat("  (Each tree is built independently — none learns from the errors of others.)\n")
cat("- Gradient Boosting and XGBoost fix this by building trees SEQUENTIALLY,\n")
cat("  where each new tree specifically corrects the previous trees' errors.\n")
cat("  This often pushes accuracy even higher.\n")
cat("===================================\n")