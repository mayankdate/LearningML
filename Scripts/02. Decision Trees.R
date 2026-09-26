# ============================================================ #
# 06 — Decision Trees
# ============================================================ #
# Purpose: Learn decision trees from scratch. This is the
#   foundation for Random Forests, Gradient Boosting, and
#   XGBoost — so understanding trees deeply matters.
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
  caret,          # Train/test split, CV
  pROC,           # ROC curves
  rpart,          # Decision tree algorithm
  rpart.plot,     # Beautiful tree visualizations
  vip             # Variable importance plots
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Decision_Trees", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Decision_Trees"

# ---- Global Plot Theme ----
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    legend.position = "bottom"
  )


# ============================================================ #
# SECTION 1: WHAT IS A DECISION TREE?
# ============================================================ #
#
# IMAGINE THIS:
#   You're a doctor. A patient walks in. You need to decide if
#   they're at risk for heart disease. You ask questions:
#
#     "How old are you?"
#       → If over 65, ask: "Do you smoke?"
#         → If yes: HIGH RISK
#         → If no, ask: "What's your blood pressure?"
#           → If over 140: MODERATE RISK
#           → If under 140: LOW RISK
#       → If under 65, ask: "What's your BMI?"
#         → ...and so on
#
#   That's EXACTLY what a decision tree does. It learns a
#   series of yes/no questions that split patients into groups.
#
# HOW IT WORKS STEP BY STEP:
#   1. Start with ALL patients in one group
#   2. Ask: "Which single question best separates heart disease
#      patients from non-heart-disease patients?"
#      → Maybe "Is age > 62?" splits them best
#   3. Now you have TWO groups. For each group, ask AGAIN:
#      "What's the next best question?"
#   4. Keep splitting until you reach a stopping rule
#   5. Each final group (leaf) gets a prediction:
#      → If mostly heart disease patients → predict YES
#      → If mostly non-heart-disease → predict NO
#
# WHAT DOES "BEST SPLIT" MEAN?
#   The tree measures "impurity" — how mixed a group is.
#   A group with 50% heart disease / 50% no disease is
#   MAXIMALLY IMPURE (you can't predict anything).
#   A group with 95% heart disease is very PURE (easy to predict).
#
#   The algorithm picks the question that creates the PUREST
#   possible child groups. The measure we'll use is called
#   "Gini impurity":
#     Gini = 1 - p(yes)^2 - p(no)^2
#     Perfect purity = 0, Maximum impurity = 0.5
#
# KEY DIFFERENCES FROM LOGISTIC REGRESSION:
#   - NO assumption of linearity
#   - Automatically captures INTERACTIONS (e.g., "age > 65 AND smoker")
#   - Splits create rectangular decision regions, not a smooth curve
#   - Very easy to visualize and explain to non-technical people
#   - BUT: single trees tend to OVERFIT (memorize the training data)
#
# WHY LEARN THIS FIRST?
#   Random Forest = many decision trees averaged together
#   XGBoost = many decision trees built sequentially
#   Understanding one tree makes all of these make sense.
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
# GOOD NEWS: Decision trees need ZERO preprocessing!
#
# Why not?
#   - Trees split on "is age > 62?" — the actual numeric VALUE matters,
#     so scaling would change nothing about which split is chosen
#   - Trees handle categories natively (they can split on "is sex = Male?")
#   - Trees don't care about skewed distributions
#   - Trees aren't affected by outliers (an outlier just ends up in a leaf)
#
# This is a major practical advantage over logistic regression,
# KNN, SVM, and neural networks, which all need careful preprocessing.
#
# We just need factors to be proper factors (not character strings).

train_df <- train_df %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")
    ),
    smoking_status = factor(smoking_status, levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use, levels = c("None", "Light", "Moderate", "Heavy"))
  )

test_df <- test_df %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")
    ),
    smoking_status = factor(smoking_status, levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use, levels = c("None", "Light", "Moderate", "Heavy"))
  )

cat("Data ready — no scaling or encoding needed for trees.\n")


# ============================================================ #
# SECTION 3: FIT A FULL (UNPRUNED) TREE
# ============================================================ #
# First, let's see what happens when we let the tree grow as
# deep as it wants. This will OVERFIT — and that's the point.
# We need to see overfitting to understand why we control it.
# ============================================================ #

model_formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))

# ---- 3A: Grow an Unrestricted Tree ----
# cp = 0.001 means "keep splitting as long as the improvement
# is at least 0.1%". This creates a very deep tree.
# minsplit = 20 means "don't split a node with fewer than 20 patients"

tree_full <- rpart::rpart(
  model_formula,
  data    = train_df,
  method  = "class",       # "class" for classification (vs "anova" for regression)
  control = rpart.control(
    cp       = 0.0001,     # Very small complexity parameter → deep tree
    minsplit = 20,         # Minimum observations to attempt a split
    maxdepth = 15          # Allow up to 15 levels deep
  )
)

cat("\nFull tree — Number of splits:", nrow(tree_full$frame[tree_full$frame$var != "<leaf>", ]), "\n")
cat("Number of leaves (terminal nodes):", sum(tree_full$frame$var == "<leaf>"), "\n")


# ---- 3B: Visualize the Full Tree ----
# This will be messy — that's the point!

png(file.path(out_dir, "01_full_tree_overfit.png"), width = 1800, height = 1000, res = 120)
rpart.plot::rpart.plot(
  tree_full,
  type    = 4,        # Label style: shows probability and % of data
  extra   = 106,      # Show predicted class, probability, and % of observations
  under   = TRUE,     # Put text under the node
  fallen.leaves = TRUE,
  roundint = FALSE,
  main    = "Full (Unpruned) Decision Tree — This is OVERFITTING",
  cex     = 0.5       # Shrink text to fit
)
dev.off()
cat("Full tree plot saved\n")


# ---- 3C: Evaluate the Full Tree ----
# Check train vs test performance to SEE overfitting

full_train_probs <- predict(tree_full, newdata = train_df, type = "prob")[, "1"]
full_test_probs  <- predict(tree_full, newdata = test_df, type = "prob")[, "1"]

full_train_auc <- round(as.numeric(pROC::auc(pROC::roc(train_df[[target]], full_train_probs, quiet = TRUE))), 3)
full_test_auc  <- round(as.numeric(pROC::auc(pROC::roc(test_df[[target]], full_test_probs, quiet = TRUE))), 3)

cat("\n--- Full Tree: Overfitting Check ---\n")
cat("Train AUC:", full_train_auc, "\n")
cat("Test AUC: ", full_test_auc, "\n")
cat("Gap:", round(full_train_auc - full_test_auc, 3), "\n")
cat("→ The large gap means the tree MEMORIZED the training data.\n")
cat("  It learned noise and quirks specific to the training set,\n")
cat("  not general patterns that transfer to new patients.\n")
cat("  This is OVERFITTING — the central problem in ML.\n")


# ============================================================ #
# SECTION 4: PRUNING — THE CURE FOR OVERFITTING
# ============================================================ #
# WHAT IS PRUNING?
#   A full tree makes too many splits — some splits capture
#   real patterns, others capture random noise. Pruning cuts
#   back the tree to keep only the meaningful splits.
#
# HOW DOES IT WORK?
#   The "complexity parameter" (cp) controls this:
#   - Each split must improve the model by at least cp
#   - Higher cp = simpler tree (fewer splits)
#   - Lower cp = more complex tree (more splits)
#
#   Think of it like a minimum bar: "don't bother splitting
#   unless it improves prediction by at least this much."
#
# HOW DO WE FIND THE BEST cp?
#   rpart does internal cross-validation automatically.
#   It tries many cp values and picks the one where the
#   cross-validated error is lowest.
#
#   The "1-SE rule" is even more conservative: pick the
#   simplest tree whose error is within 1 standard error
#   of the minimum. This prevents overfitting even more.
# ============================================================ #


# ---- 4A: Look at the CP Table ----
# rpart automatically computed CV error at each cp level

cp_table <- as.data.frame(tree_full$cptable)
colnames(cp_table) <- c("cp", "nsplit", "rel_error", "xerror", "xstd")

cat("\n--- Complexity Parameter Table ---\n")
cat("cp       = complexity parameter (pruning threshold)\n")
cat("nsplit   = number of splits at this cp\n")
cat("rel_error = training error (always decreases — more splits = better fit on training)\n")
cat("xerror   = cross-validated error (what we CARE about — performance on unseen data)\n")
cat("xstd     = standard deviation of xerror\n\n")
print(head(cp_table, 15))


# ---- 4B: Visualize the CP Selection ----
# This plot shows the key insight: training error always goes down,
# but CV error hits a minimum and then rises (overfitting territory).

p_cp <- cp_table %>%
  dplyr::mutate(row_id = dplyr::row_number()) %>%
  tidyr::pivot_longer(
    cols = c(rel_error, xerror),
    names_to = "error_type", values_to = "error"
  ) %>%
  dplyr::mutate(
    error_type = dplyr::case_when(
      error_type == "rel_error" ~ "Training Error",
      error_type == "xerror"    ~ "CV Error (what matters)"
    )
  ) %>%
  ggplot(aes(x = nsplit, y = error, color = error_type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  # Add error bars only for CV error (fully self-contained aes)
  geom_errorbar(
    data = cp_table,
    aes(x = nsplit, ymin = xerror - xstd, ymax = xerror + xstd),
    inherit.aes = FALSE,
    width = 0.5, alpha = 0.4, color = "#F44336"
  ) +
  scale_color_manual(values = c(
    "Training Error"           = "#2196F3",
    "CV Error (what matters)"  = "#F44336"
  )) +
  labs(
    title    = "Training Error vs Cross-Validated Error",
    subtitle = paste(
      "Training error ALWAYS decreases with more splits (the tree memorizes more).\n",
      "CV error hits a minimum then rises — that minimum is our sweet spot."
    ),
    x = "Number of Splits", y = "Relative Error", color = NULL
  ) +
  theme_ml

print(p_cp)
ggsave(file.path(out_dir, "02_cp_selection.png"), p_cp,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 4C: Find Optimal CP ----
# Two common approaches:
#   1. "Minimum CV error" — pick the cp with the lowest xerror
#   2. "1-SE rule" — pick the simplest tree within 1 SE of the minimum
#
# The 1-SE rule is more conservative but can over-prune when the
# outcome is rare (like our ~6% heart disease rate). In that case
# it decides "just predict 0 for everyone" because that's within
# 1 SE of the best. Not useful for learning!
#
# We'll use minimum CV error here, with a safety check.

min_xerror <- min(cp_table$xerror)
min_row    <- which.min(cp_table$xerror)
best_cp    <- cp_table$cp[min_row]

# Safety check: if the best cp produces 0 splits (a stump), relax it.
# This can happen with rare outcomes. Fall back to a reasonable tree.
if (cp_table$nsplit[min_row] == 0) {
  cat("NOTE: Minimum CV error selects 0 splits (just predicts majority class).\n")
  cat("  This happens because heart disease is rare (~6%).\n")
  cat("  Falling back to the cp that gives the best tree with actual splits.\n")
  # Pick the row with the lowest CV error AMONG trees that have splits
  has_splits <- cp_table$nsplit > 0
  if (any(has_splits)) {
    min_row <- which.min(cp_table$xerror[has_splits])
    min_row <- which(has_splits)[min_row]
    best_cp <- cp_table$cp[min_row]
  }
}

cat("\n--- Optimal CP Selection ---\n")
cat("Minimum CV error:", round(min_xerror, 4), "\n")
cat("Selected CP:", round(best_cp, 5), "with", cp_table$nsplit[min_row], "splits\n")


# ---- 4D: Prune the Tree ----

tree_pruned <- rpart::prune(tree_full, cp = best_cp)

cat("\nPruned tree — Number of splits:", nrow(tree_pruned$frame[tree_pruned$frame$var != "<leaf>", ]), "\n")
cat("Number of leaves:", sum(tree_pruned$frame$var == "<leaf>"), "\n")


# ============================================================ #
# SECTION 5: VISUALIZE THE PRUNED TREE
# ============================================================ #
# THIS is the payoff. You can literally READ the tree and
# understand every decision the model makes.
#
# HOW TO READ THE TREE PLOT:
#   - Each box is a NODE (a group of patients)
#   - The TOP of the tree is ALL patients
#   - Each split asks a yes/no question
#     → YES goes LEFT, NO goes RIGHT
#   - Each box shows:
#     → The predicted class (0 or 1)
#     → The probability of heart disease in that group
#     → What % of the data falls in that group
#   - LEAF nodes (bottom) are the final predictions
#   - Darker blue = more non-heart-disease
#   - Darker green/red = more heart disease
# ============================================================ #

png(file.path(out_dir, "03_pruned_tree.png"), width = 1600, height = 1000, res = 120)
rpart.plot::rpart.plot(
  tree_pruned,
  type    = 4,
  extra   = 106,
  under   = TRUE,
  fallen.leaves = TRUE,
  roundint = FALSE,
  main    = "Pruned Decision Tree — Heart Disease Prediction",
  cex     = 0.7,
  box.palette = c("#E3F2FD", "#FFCDD2")  # Blue (low risk) to Red (high risk)
)
dev.off()

cat("Pruned tree plot saved — LOOK AT THIS PLOT CAREFULLY.\n")
cat("It shows the exact decision rules the model learned.\n")


# ============================================================ #
# SECTION 6: HOW THE TREE MAKES PREDICTIONS
# ============================================================ #
# Let's trace through two example patients to see how the
# tree arrives at its prediction.
# ============================================================ #

# Pick one heart disease case and one non-case from test set
example_case    <- test_df %>% dplyr::filter(heart_disease == 1) %>% dplyr::slice(1)
example_noncase <- test_df %>% dplyr::filter(heart_disease == 0) %>% dplyr::slice(1)

cat("\n--- Example Patient 1 (Has Heart Disease) ---\n")
cat("Age:", example_case$age, "| Sex:", as.character(example_case$sex),
    "| BMI:", example_case$bmi, "| SBP:", example_case$sbp,
    "| Smoking:", as.character(example_case$smoking_status),
    "| Cholesterol:", example_case$cholesterol, "\n")

pred_case <- predict(tree_pruned, newdata = example_case, type = "prob")
cat("Predicted P(Heart Disease):", round(pred_case[, "1"], 3), "\n")
cat("→ The tree sent this patient through its yes/no questions and\n")
cat("  landed in a leaf node where", round(pred_case[, "1"] * 100, 1), "% of training\n")
cat("  patients had heart disease.\n")

cat("\n--- Example Patient 2 (No Heart Disease) ---\n")
cat("Age:", example_noncase$age, "| Sex:", as.character(example_noncase$sex),
    "| BMI:", example_noncase$bmi, "| SBP:", example_noncase$sbp,
    "| Smoking:", as.character(example_noncase$smoking_status),
    "| Cholesterol:", example_noncase$cholesterol, "\n")

pred_noncase <- predict(tree_pruned, newdata = example_noncase, type = "prob")
cat("Predicted P(Heart Disease):", round(pred_noncase[, "1"], 3), "\n")


# ============================================================ #
# SECTION 7: EVALUATION
# ============================================================ #


# ---- 7A: Predictions on Test Set ----

test_probs <- predict(tree_pruned, newdata = test_df, type = "prob")[, "1"]
train_probs <- predict(tree_pruned, newdata = train_df, type = "prob")[, "1"]


# ---- 7B: ROC Curve ----

roc_obj <- pROC::roc(test_df[[target]], test_probs, quiet = TRUE)
auc_val <- round(as.numeric(pROC::auc(roc_obj)), 3)

roc_df <- tibble(
  fpr         = 1 - roc_obj$specificities,
  sensitivity = roc_obj$sensitivities
)

# Load logistic regression results for comparison (if available)
logistic_results_path <- "Outputs/05_Logistic_Regression/results_logistic.rds"
has_logistic <- file.exists(logistic_results_path)

if (has_logistic) {
  logistic_results <- readRDS(logistic_results_path)
  logistic_test_probs <- readRDS("Outputs/05_Logistic_Regression/test_predictions_logistic.rds")
  logistic_roc <- pROC::roc(test_df[[target]], logistic_test_probs, quiet = TRUE)
  logistic_roc_df <- tibble(
    fpr         = 1 - logistic_roc$specificities,
    sensitivity = logistic_roc$sensitivities
  )
  logistic_auc <- logistic_results$auc
}

p_roc <- ggplot() +
  geom_line(data = roc_df, aes(x = fpr, y = sensitivity, color = "Decision Tree"),
            linewidth = 1.2) +
  {if (has_logistic)
    geom_line(data = logistic_roc_df, aes(x = fpr, y = sensitivity, color = "Logistic Regression"),
              linewidth = 1, linetype = "dashed")
  } +
  geom_abline(linetype = "dotted", color = "grey50") +
  scale_color_manual(values = c(
    "Decision Tree"        = "#F44336",
    "Logistic Regression"  = "#2196F3"
  )) +
  annotate("text", x = 0.55, y = 0.40,
           label = paste("Tree AUC =", auc_val),
           size = 5, fontface = "bold", color = "#F44336") +
  {if (has_logistic)
    annotate("text", x = 0.55, y = 0.30,
             label = paste("Logistic AUC =", logistic_auc),
             size = 5, fontface = "bold", color = "#2196F3")
  } +
  labs(
    title    = "ROC Curve — Decision Tree vs Logistic Regression",
    subtitle = "Closer to top-left corner = better. Diagonal = random guessing.",
    x = "False Positive Rate\n(% of healthy patients incorrectly flagged)",
    y = "True Positive Rate\n(% of heart disease patients correctly caught)",
    color = "Model"
  ) +
  theme_ml

print(p_roc)
ggsave(file.path(out_dir, "04_roc_curve_comparison.png"), p_roc,
       width = 9, height = 7, dpi = 150, bg = "white")

cat("\nDecision Tree AUC:", auc_val, "\n")
if (has_logistic) cat("Logistic Regression AUC:", logistic_auc, "\n")


# ---- 7C: Confusion Matrix ----

# Use Youden's J for optimal threshold
optimal_coords <- pROC::coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold
test_pred_class <- ifelse(test_probs >= optimal_threshold, 1, 0)

cm <- table(Actual = test_df[[target]], Predicted = test_pred_class)
tp <- cm["1", "1"]; fp <- cm["0", "1"]; tn <- cm["0", "0"]; fn <- cm["1", "0"]

accuracy  <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- Classification Metrics (threshold =", round(optimal_threshold, 3), ") ---\n")
cat("Accuracy: ", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), " — Of those we flagged, how many truly have HD\n")
cat("Recall:   ", round(recall, 3), " — Of those who have HD, how many did we catch\n")
cat("F1 Score: ", round(f1, 3), " — Balance of precision and recall\n")

# Visualize confusion matrix with annotations explaining each cell
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
    title    = "Confusion Matrix — Decision Tree",
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
ggsave(file.path(out_dir, "05_confusion_matrix.png"), p_confusion,
       width = 8, height = 7, dpi = 150, bg = "white")


# ---- 7D: Predicted Probability Distribution ----

p_prob_dist <- tibble(
  probability = test_probs,
  actual      = factor(test_df[[target]], labels = c("No HD", "Has HD"))
) %>%
  ggplot(aes(x = probability, fill = actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", linewidth = 0.8) +
  scale_fill_manual(values = c("No HD" = "#2196F3", "Has HD" = "#F44336")) +
  labs(
    title    = "Predicted Probabilities — Decision Tree",
    subtitle = paste(
      "NOTICE: Trees produce CLUMPY probabilities (only a few distinct values).\n",
      "Each clump = one leaf node. Logistic regression produces smooth probabilities."
    ),
    x = "Predicted P(Heart Disease)", y = "Count", fill = "Actual Status"
  ) +
  theme_ml

print(p_prob_dist)
ggsave(file.path(out_dir, "06_predicted_probabilities.png"), p_prob_dist,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: FEATURE IMPORTANCE
# ============================================================ #
# HOW DO TREES MEASURE FEATURE IMPORTANCE?
#
#   Every time a feature is used to make a split, it improves
#   the purity of the child nodes. The TOTAL improvement across
#   all splits using that feature = its importance.
#
#   A feature used at the top of the tree (where the biggest
#   groups get split) contributes more importance than one
#   used deep in the tree (small groups).
#
#   Think of it as: "How much work did each feature do in
#   separating heart disease from non-heart-disease?"
# ============================================================ #

importance_raw <- tree_pruned$variable.importance

if (length(importance_raw) == 0) {
  cat("WARNING: Pruned tree has no variable importance (too simple).\n")
  cat("Using full tree importance instead.\n")
  importance_raw <- tree_full$variable.importance
}

importance_df <- tibble(
  feature    = names(importance_raw),
  importance = as.numeric(importance_raw)
)

importance_df <- importance_df %>%
  dplyr::mutate(
    pct_importance = round(importance / sum(importance) * 100, 1),
    feature = forcats::fct_reorder(feature, importance)
  )

p_importance <- importance_df %>%
  ggplot(aes(x = importance, y = feature)) +
  geom_col(fill = "#5C6BC0", alpha = 0.85) +
  geom_text(aes(label = paste0(pct_importance, "%")),
            hjust = -0.1, size = 3.5, fontface = "bold") +
  labs(
    title    = "Feature Importance — Decision Tree",
    subtitle = paste(
      "How much each feature contributed to splitting patients into groups.\n",
      "Higher = the tree relied on this feature more for its decisions."
    ),
    x = "Importance (total Gini improvement)", y = NULL
  ) +
  theme_ml

print(p_importance)
ggsave(file.path(out_dir, "07_feature_importance.png"), p_importance,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 9: OVERFITTING DEEP DIVE
# ============================================================ #
# Let's SHOW overfitting visually. We'll train trees at different
# depths and plot train vs test AUC. This is one of the most
# important concepts in all of machine learning.
#
# WHAT WILL WE SEE?
#   - Shallow trees (few splits): UNDERFITTING
#     → Too simple to capture the real pattern
#     → Both train and test performance are mediocre
#
#   - Medium trees: JUST RIGHT
#     → Captures real patterns without memorizing noise
#     → Train and test performance are both good and close together
#
#   - Deep trees (many splits): OVERFITTING
#     → Memorizes training data perfectly
#     → Train performance is excellent, test performance degrades
#     → The GAP between train and test = overfitting
# ============================================================ #

depth_values <- c(1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30)

depth_results <- purrr::map_dfr(depth_values, function(d) {
  # Train a tree at this depth
  tree_d <- rpart::rpart(
    model_formula, data = train_df, method = "class",
    control = rpart.control(maxdepth = d, cp = 0.0001, minsplit = 10)
  )
  
  # Get AUC on train and test
  train_p <- predict(tree_d, newdata = train_df, type = "prob")[, "1"]
  test_p  <- predict(tree_d, newdata = test_df, type = "prob")[, "1"]
  
  train_a <- as.numeric(pROC::auc(pROC::roc(train_df[[target]], train_p, quiet = TRUE)))
  test_a  <- as.numeric(pROC::auc(pROC::roc(test_df[[target]], test_p, quiet = TRUE)))
  
  n_leaves <- sum(tree_d$frame$var == "<leaf>")
  
  tibble(max_depth = d, train_auc = train_a, test_auc = test_a, n_leaves = n_leaves)
})

p_overfit <- depth_results %>%
  tidyr::pivot_longer(
    cols = c(train_auc, test_auc),
    names_to = "set", values_to = "auc"
  ) %>%
  dplyr::mutate(set = ifelse(set == "train_auc", "Training", "Test (unseen data)")) %>%
  ggplot(aes(x = max_depth, y = auc, color = set)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  # Annotate the three zones
  annotate("rect", xmin = 0.5, xmax = 3.5, ymin = -Inf, ymax = Inf,
           alpha = 0.08, fill = "#FF9800") +
  annotate("rect", xmin = 3.5, xmax = 8, ymin = -Inf, ymax = Inf,
           alpha = 0.08, fill = "#4CAF50") +
  annotate("rect", xmin = 8, xmax = 31, ymin = -Inf, ymax = Inf,
           alpha = 0.08, fill = "#F44336") +
  annotate("text", x = 2, y = max(depth_results$train_auc) + 0.005,
           label = "UNDERFITTING\n(too simple)", color = "#FF9800",
           fontface = "bold", size = 3.5, vjust = 0) +
  annotate("text", x = 5.5, y = max(depth_results$train_auc) + 0.005,
           label = "SWEET SPOT", color = "#4CAF50",
           fontface = "bold", size = 3.5, vjust = 0) +
  annotate("text", x = 17, y = max(depth_results$train_auc) + 0.005,
           label = "OVERFITTING\n(memorizing noise)", color = "#F44336",
           fontface = "bold", size = 3.5, vjust = 0) +
  scale_color_manual(values = c("Training" = "#2196F3", "Test (unseen data)" = "#F44336")) +
  labs(
    title    = "The Overfitting Story: Tree Depth vs Performance",
    subtitle = paste(
      "Training AUC always improves with depth (the tree memorizes more).\n",
      "Test AUC peaks then declines — deeper isn't always better."
    ),
    x = "Maximum Tree Depth", y = "AUC", color = NULL
  ) +
  theme_ml

print(p_overfit)
ggsave(file.path(out_dir, "08_overfitting_by_depth.png"), p_overfit,
       width = 11, height = 7, dpi = 150, bg = "white")

cat("\n--- Overfitting by Depth ---\n")
depth_results %>%
  dplyr::mutate(
    gap = round(train_auc - test_auc, 3),
    across(c(train_auc, test_auc), ~ round(.x, 3))
  ) %>%
  print(n = 20)


# ============================================================ #
# SECTION 10: CROSS-VALIDATION
# ============================================================ #

set.seed(42)
cv_folds <- caret::createFolds(train_df[[target]], k = 5, returnTrain = TRUE)

cv_results <- purrr::map_dfr(seq_along(cv_folds), function(i) {
  fold_train <- train_df[cv_folds[[i]], ]
  fold_val   <- train_df[-cv_folds[[i]], ]
  
  fold_tree <- rpart::rpart(
    model_formula, data = fold_train, method = "class",
    control = rpart.control(cp = best_cp)
  )
  
  fold_probs <- predict(fold_tree, newdata = fold_val, type = "prob")[, "1"]
  fold_auc   <- as.numeric(pROC::auc(pROC::roc(fold_val[[target]], fold_probs, quiet = TRUE)))
  
  tibble(fold = i, auc = fold_auc)
})

cat("\n--- 5-Fold Cross-Validation AUC ---\n")
print(cv_results)
cat("Mean CV AUC:", round(mean(cv_results$auc), 3),
    "± SD:", round(sd(cv_results$auc), 3), "\n")
cat("Test AUC:", auc_val, "\n")


# ============================================================ #
# SECTION 11: DECISION BOUNDARIES VISUALIZATION
# ============================================================ #
# Show how a decision tree carves up the feature space into
# RECTANGLES. We'll use the two most important features
# and plot the regions where the tree predicts 0 vs 1.
#
# WHY THIS MATTERS:
#   - Logistic regression creates a STRAIGHT LINE boundary
#   - Decision trees create RECTANGULAR regions (staircase pattern)
#   - This shows why trees can capture nonlinear patterns
#     but also why they can look "blocky"
# ============================================================ #

# Get the top 2 numeric features by importance
top2_features <- importance_df %>%
  dplyr::filter(feature %in% c("age", "sbp", "bmi", "cholesterol",
                               "fasting_glucose", "income", "dbp",
                               "physical_activity")) %>%
  dplyr::slice_max(importance, n = 2) %>%
  dplyr::pull(feature) %>%
  as.character()

cat("\nDecision boundary using:", top2_features[1], "and", top2_features[2], "\n")

# Train a simple tree on JUST these 2 features (so we can plot in 2D)
simple_formula <- as.formula(paste(target, "~", paste(top2_features, collapse = " + ")))

tree_2d <- rpart::rpart(
  simple_formula, data = train_df, method = "class",
  control = rpart.control(maxdepth = 5, cp = 0.005)
)

# Create a grid of points covering the 2D feature space
grid_df <- expand.grid(
  x = seq(min(train_df[[top2_features[1]]]), max(train_df[[top2_features[1]]]), length.out = 200),
  y = seq(min(train_df[[top2_features[2]]]), max(train_df[[top2_features[2]]]), length.out = 200)
)
colnames(grid_df) <- top2_features

# Predict on every grid point
grid_df$pred_prob <- predict(tree_2d, newdata = grid_df, type = "prob")[, "1"]

# Plot the decision regions with actual data points overlaid
p_boundary <- ggplot() +
  # Decision regions (background)
  geom_tile(data = grid_df,
            aes(x = .data[[top2_features[1]]], y = .data[[top2_features[2]]],
                fill = pred_prob), alpha = 0.6) +
  scale_fill_gradient2(low = "#2196F3", mid = "white", high = "#F44336",
                       midpoint = 0.5, name = "P(Heart Disease)") +
  # Actual data points
  geom_point(data = test_df %>% dplyr::sample_n(min(2000, nrow(test_df))),
             aes(x = .data[[top2_features[1]]], y = .data[[top2_features[2]]],
                 color = factor(heart_disease)),
             alpha = 0.3, size = 0.8) +
  scale_color_manual(values = c("0" = "#1565C0", "1" = "#C62828"),
                     labels = c("No HD", "Has HD"), name = "Actual") +
  labs(
    title    = "Decision Tree Boundaries (2 Features Only)",
    subtitle = paste(
      "The RECTANGULAR regions are the tree's decision boundaries.\n",
      "Compare this to logistic regression, which would draw a single straight line."
    ),
    x = top2_features[1], y = top2_features[2]
  ) +
  theme_ml

print(p_boundary)
ggsave(file.path(out_dir, "09_decision_boundaries.png"), p_boundary,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 12: SAVE RESULTS
# ============================================================ #

results_tree <- tibble(
  model          = "Decision Tree",
  target         = "heart_disease",
  auc            = auc_val,
  accuracy       = round(accuracy, 4),
  precision      = round(precision, 4),
  recall         = round(recall, 4),
  f1             = round(f1, 4),
  cv_auc_mean    = round(mean(cv_results$auc), 4),
  cv_auc_sd      = round(sd(cv_results$auc), 4),
  train_auc      = round(as.numeric(pROC::auc(pROC::roc(train_df[[target]], train_probs, quiet = TRUE))), 4),
  threshold_used = round(optimal_threshold, 4)
)

saveRDS(results_tree, file.path(out_dir, "results_tree.rds"))
saveRDS(test_probs, file.path(out_dir, "test_predictions_tree.rds"))


# ============================================================ #
# SECTION 13: SUMMARY & KEY TAKEAWAYS
# ============================================================ #

cat("\n")
cat("=== DECISION TREE — SUMMARY ===\n")
cat("Test AUC:", auc_val, "\n")
cat("F1 Score:", round(f1, 3), "\n")
cat("CV AUC:", round(mean(cv_results$auc), 3), "±", round(sd(cv_results$auc), 3), "\n")
if (has_logistic) {
  cat("\nvs Logistic Regression AUC:", logistic_auc, "\n")
  if (auc_val > logistic_auc) {
    cat("→ Tree BEAT logistic regression! It found nonlinear patterns.\n")
  } else {
    cat("→ Logistic regression still wins. A single tree is often not powerful enough.\n")
    cat("  This is exactly WHY Random Forests exist (next script).\n")
  }
}
cat("\n")
cat("WHAT WE LEARNED:\n")
cat("1. Trees ask yes/no questions to split patients into groups.\n")
cat("2. They find the 'best' split by measuring impurity (Gini).\n")
cat("3. Deep trees OVERFIT — they memorize training data.\n")
cat("4. PRUNING (controlled by cp) prevents overfitting.\n")
cat("5. Trees are incredibly INTERPRETABLE — you can read every decision.\n")
cat("6. Trees capture INTERACTIONS and NONLINEARITY automatically.\n")
cat("7. BUT: a single tree is often not the best predictor.\n")
cat("   → The variance is high (different training data → very different tree).\n")
cat("\n")
cat("PREVIEW — NEXT SCRIPT (Random Forest):\n")
cat("What if instead of ONE tree, we grew HUNDREDS of trees on\n")
cat("slightly different data and let them VOTE? That's a Random Forest.\n")
cat("It keeps the strengths of trees (interactions, no preprocessing)\n")
cat("while fixing the weakness (high variance, overfitting).\n")
cat("===================================\n")