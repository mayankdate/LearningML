# ============================================================ #
# 04 — Gradient Boosting & XGBoost
# ============================================================ #
# Purpose: Learn how boosting builds trees SEQUENTIALLY, where
#   each new tree specifically corrects the mistakes of all
#   previous trees. This is often the highest-performing model
#   for tabular data and dominates ML competitions.
#
# We cover two implementations:
#   1. GBM (Gradient Boosting Machine) — the classic version
#   2. XGBoost — the faster, more powerful evolution
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
  caret,       # Train/test split, CV
  pROC,        # ROC curves
  gbm,         # Gradient Boosting Machine (classic)
  xgboost,     # XGBoost (fast, powerful)
  SHAPforxgboost  # SHAP values for XGBoost interpretation
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Boosting", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Boosting"


# ---- GPU Detection for XGBoost ----
# XGBoost can use your NVIDIA GPU (e.g. RTX 4060) to build trees.
# Instead of the CPU evaluating every possible split one-by-one,
# the GPU evaluates thousands of candidate splits simultaneously
# using its massive parallel cores. On large datasets this is
# 5-10x faster. On our 25k-row data the speedup is modest but
# it's good practice and the code is ready for bigger data.
#
# HOW IT WORKS:
#   tree_method = "hist"      → CPU: builds a histogram of feature
#                                values, then finds the best split
#                                from the histogram bins. Fast and
#                                memory-efficient.
#   tree_method = "gpu_hist"  → GPU: same histogram algorithm but
#                                the binning and split-finding run
#                                on the GPU's thousands of cores.
#
# REQUIREMENTS:
#   1. An NVIDIA GPU (you have an RTX 4060)
#   2. CUDA drivers installed (comes with NVIDIA driver)
#   3. XGBoost built with GPU support. Install via:
#      install.packages("xgboost") — CRAN builds include GPU support
#      on Windows if CUDA is detected. If not, you can install from
#      source or use the pre-built GPU wheel.
#
# The code below tries gpu_hist first. If it fails (no GPU, no CUDA,
# or XGBoost wasn't built with GPU support), it falls back to the
# CPU "hist" method silently. You lose nothing either way.

use_gpu <- FALSE
tryCatch({
  # Try training a tiny 1-round model on GPU to see if it works
  test_mat <- xgboost::xgb.DMatrix(
    data  = matrix(rnorm(100), ncol = 5),
    label = sample(0:1, 20, replace = TRUE)
  )
  xgboost::xgb.train(
    params  = list(tree_method = "gpu_hist", objective = "binary:logistic"),
    data    = test_mat,
    nrounds = 1,
    verbose = 0
  )
  use_gpu <- TRUE
  rm(test_mat)
}, error = function(e) {
  # GPU not available — that's fine, CPU works identically
})

xgb_tree_method <- if (use_gpu) "gpu_hist" else "hist"
cat("XGBoost tree method:", xgb_tree_method,
    if (use_gpu) "— GPU acceleration enabled (RTX 4060)\n"
    else "— CPU mode (GPU not available, install CUDA + GPU-enabled xgboost for acceleration)\n")

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
# SECTION 1: WHAT IS GRADIENT BOOSTING?
# ============================================================ #
#
# SIMPLE ANALOGY — LEARNING FROM YOUR MISTAKES:
#   Imagine a student taking a math test. They get 70%.
#   Instead of retaking the whole test, a tutor says:
#   "Let's focus ONLY on the questions you got wrong."
#   After studying those, the student retakes just those
#   questions and gets most right. But a few are still wrong.
#   So the tutor says: "Now let's focus on THESE remaining
#   mistakes." And so on, each round fixing the previous errors.
#
#   After 10 rounds of this, the student has 95%.
#   That's boosting.
#
# HOW IT WORKS — STEP BY STEP:
#
#   1. Start with a simple prediction (e.g., "predict the average
#      heart disease rate for everyone" — about 6%).
#
#   2. Compute the RESIDUALS — the errors. For each patient:
#      error = actual outcome - current prediction
#      Patients the model gets wrong have large residuals.
#
#   3. Fit a SMALL decision tree to predict the RESIDUALS.
#      This tree learns WHERE the current model is wrong and
#      by HOW MUCH.
#
#   4. ADD this tree's predictions to the running total,
#      scaled by a "learning rate" (a small number like 0.1).
#      New prediction = old prediction + learning_rate × tree prediction
#
#   5. Recompute residuals with the updated prediction.
#      Some errors are now smaller, but new patterns may emerge.
#
#   6. Repeat steps 3-5 for hundreds of rounds.
#      Each new tree chips away at the remaining errors.
#
# HOW IS THIS DIFFERENT FROM RANDOM FOREST?
#
#   RANDOM FOREST:
#     - Trees are built INDEPENDENTLY (in parallel)
#     - Each tree sees random data + random features
#     - Final prediction = AVERAGE of all trees
#     - Each tree tries to solve the whole problem
#
#   GRADIENT BOOSTING:
#     - Trees are built SEQUENTIALLY (one after another)
#     - Each tree specifically targets the ERRORS of prior trees
#     - Final prediction = SUM of all trees' contributions
#     - Each tree only needs to fix a small piece of the puzzle
#     - Trees are intentionally kept SMALL (shallow) — typically
#       3-6 splits deep, called "stumps" or "weak learners"
#
# KEY HYPERPARAMETERS:
#
#   n.trees / nrounds:
#     How many sequential trees to build.
#     More = potentially better, but risk of overfitting.
#
#   learning_rate (eta / shrinkage):
#     How much each tree contributes to the total prediction.
#     Small values (0.01-0.1) = each tree makes a tiny correction.
#     This means you need MORE trees, but the model generalizes
#     better. Think of it as "how cautious is each correction step?"
#     SMALLER IS GENERALLY BETTER (with enough trees).
#
#   max_depth (interaction.depth):
#     How deep each individual tree can grow.
#     Shallow trees (2-4) = simple corrections.
#     Deeper trees (5-8) = can capture interactions between features.
#     Too deep = each tree overfits the residuals.
#
#   subsample (bag.fraction):
#     Fraction of data used to train each tree (like bagging).
#     0.5-0.8 is typical. Adds randomness to prevent overfitting.
#
# WHAT IS XGBoost?
#   XGBoost ("eXtreme Gradient Boosting") is an optimized,
#   faster, more powerful version of gradient boosting.
#   Same fundamental idea, but with:
#     - Built-in regularization (penalties for complexity)
#     - Handling of missing values
#     - Parallel computation (much faster)
#     - Smarter split-finding algorithms
#     - Column subsampling (like Random Forest's mtry)
#   XGBoost dominates Kaggle competitions and is often the
#   go-to model for structured/tabular data.
#
# WHEN BOOSTING WORKS WELL:
#   - Tabular/structured data (the king of spreadsheet ML)
#   - When you want maximum predictive accuracy
#   - Complex feature interactions
#   - Medium to large datasets
#
# WHEN IT STRUGGLES:
#   - Requires more careful tuning than Random Forest
#   - Can overfit if not tuned properly (especially with too many trees)
#   - Slower to train than RF (sequential, not parallel)
#   - Less interpretable than a single tree or logistic regression
#   - Not great for images, text, audio (use neural networks instead)
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
# WHY: We hold out 25% of the data to simulate new, unseen patients.
# If we only evaluated on training data, the model could memorize
# the answers and look perfect — but fail on real patients.
# Stratified split ensures both sets have ~same heart disease rate.

set.seed(42)

train_index <- caret::createDataPartition(
  ml_data[[target]],
  p    = 0.75,
  list = FALSE
)

train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")


# ---- 2C: Preprocessing ----
# GBM (gbm package): handles factors natively, like randomForest.
# XGBoost: requires ALL NUMERIC input — we must one-hot encode factors
# and convert everything to a numeric matrix. This is extra work
# but XGBoost repays it with speed and power.

# Set up factors for GBM
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

model_formula <- as.formula(paste(target, "~", paste(features, collapse = " + ")))


# ---- 2D: Prepare XGBoost Matrices ----
#
# WHY DOES XGBoost NEED A NUMERIC MATRIX?
#   XGBoost is written in C++ under the hood and operates on pure numbers.
#   R data frames can contain factors (categories like "Male"/"Female"),
#   but XGBoost can't split on text — it needs everything as 0s and 1s.
#
# WHAT IS ONE-HOT ENCODING?
#   We convert each categorical variable into a set of binary (0/1) columns.
#   Example — the 'sex' column has two values: "Female" and "Male"
#   After one-hot encoding, it becomes ONE new column:
#     sexMale = 1 if the patient is male, 0 if female
#   (Female is the "reference" category — it's implied when sexMale = 0)
#
#   Example — 'race' has 5 values: Asian, Black, Hispanic, Other, White
#   After encoding, it becomes 4 new columns:
#     raceBlack = 1 if Black, 0 otherwise
#     raceHispanic = 1 if Hispanic, 0 otherwise
#     raceOther = 1 if Other, 0 otherwise
#     raceWhite = 1 if White, 0 otherwise
#   (Asian is the reference — implied when all four columns = 0)
#
#   model.matrix() does this automatically for all factor columns.
#   The [, -1] drops the intercept column that model.matrix() adds by default
#   (a column of all 1s that we don't need here).
#
# WHY DOES THIS MATTER FOR XGBOOST BUT NOT GBM?
#   The gbm package (Part A) handles factors internally — you don't need this.
#   XGBoost requires you to do it yourself. More work upfront, but you have
#   full control over exactly what goes into the model.

xgb_formula <- as.formula(paste("~", paste(features, collapse = " + ")))

# One-hot encode: converts all factors to 0/1 numeric columns
# Result: a plain numeric matrix with no text, no factors
train_matrix <- model.matrix(xgb_formula, data = train_df)[, -1]
test_matrix  <- model.matrix(xgb_formula, data = test_df)[, -1]

# The target must also be plain numeric 0/1 (not a factor)
train_y <- as.numeric(train_df[[target]])
test_y  <- as.numeric(test_df[[target]])

# DMatrix is XGBoost's own optimized data format — like a compressed matrix
# that also stores the labels (outcomes). It makes XGBoost faster internally.
# Think of it as: xgb.DMatrix = matrix of features + the answer key, packaged together.
dtrain <- xgboost::xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgboost::xgb.DMatrix(data = test_matrix, label = test_y)

cat("XGBoost matrices — Train:", nrow(train_matrix), "x", ncol(train_matrix),
    "| Test:", nrow(test_matrix), "x", ncol(test_matrix), "\n")
cat("Note: more columns than original features because factors were one-hot encoded\n")


# ============================================================ #
# PART A: GRADIENT BOOSTING MACHINE (GBM)
# ============================================================ #
# The classic implementation. Slower but easier to understand.
# ============================================================ #


# ---- 3A: Fit GBM ----
# distribution = "bernoulli": binary classification (0/1 outcome)
# n.trees = 1000: build up to 1000 sequential trees
# interaction.depth = 4: each tree has max 4 splits (captures interactions)
# shrinkage = 0.01: learning rate — small steps for better generalization
# bag.fraction = 0.75: each tree trains on 75% of data (adds randomness)
# cv.folds = 5: internal 5-fold CV to find optimal number of trees

set.seed(42)

gbm_model <- gbm::gbm(
  model_formula,
  data              = train_df,
  distribution      = "bernoulli",
  n.trees           = 2000,
  interaction.depth = 4,
  shrinkage         = 0.01,
  bag.fraction      = 0.75,
  cv.folds          = 5,
  n.minobsinnode    = 10,
  verbose           = FALSE
)

# Find optimal number of trees via CV
best_n_trees_gbm <- gbm::gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
cat("\nGBM — Optimal number of trees (CV):", best_n_trees_gbm, "out of 2000\n")


# ---- 3B: Visualize GBM Training Curve ----
# WHAT IS THIS SHOWING?
#   As we add more trees, training error ALWAYS decreases
#   (the model keeps memorizing). But CV error eventually
#   INCREASES — that's where overfitting begins.
#   The optimal tree count is where CV error is minimized.

gbm_perf_df <- tibble(
  n_trees    = 1:gbm_model$n.trees,
  train_loss = gbm_model$train.error,
  cv_loss    = gbm_model$cv.error
)

p_gbm_curve <- gbm_perf_df %>%
  tidyr::pivot_longer(c(train_loss, cv_loss), names_to = "type", values_to = "loss") %>%
  dplyr::mutate(type = ifelse(type == "train_loss", "Training Loss", "CV Loss (what matters)")) %>%
  ggplot(aes(x = n_trees, y = loss, color = type)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = best_n_trees_gbm, linetype = "dashed", color = "grey30") +
  annotate("text", x = best_n_trees_gbm + 50, y = max(gbm_perf_df$cv_loss) * 0.9,
           label = paste("Optimal:", best_n_trees_gbm, "trees"),
           hjust = 0, fontface = "bold", size = 3.5) +
  scale_color_manual(values = c("Training Loss" = "#2196F3", "CV Loss (what matters)" = "#F44336")) +
  labs(
    title    = "GBM Training Curve — When to Stop",
    subtitle = paste(
      "Training loss always decreases (the model memorizes more each round).\n",
      "CV loss hits a minimum then rises — that's OVERFITTING.\n",
      "We stop at the CV minimum. This is called EARLY STOPPING."
    ),
    x = "Number of Trees (Boosting Rounds)", y = "Loss (Deviance)", color = NULL
  ) +
  theme_ml

print(p_gbm_curve)
ggsave(file.path(out_dir, "01_gbm_training_curve.png"), p_gbm_curve,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 3C: GBM Predictions ----
# Now we use the trained GBM to generate predicted probabilities.
#
# type = "response": return probabilities (0 to 1), not raw log-odds.
#   A value of 0.12 means "the model thinks this patient has a 12% chance
#   of heart disease." We always want probabilities for evaluation — not
#   binary yes/no yet, because we need to choose a threshold first.
#
# n.trees = best_n_trees_gbm: use only up to the optimal number of trees.
#   If we used all 2000, we'd be using rounds beyond where CV said to stop,
#   which would give us overfitted (too optimistic) predictions.
#
# We predict on BOTH train and test so we can check for overfitting:
#   If train AUC >> test AUC, the model memorized training data.
#   If they're close, the model is generalizing well.

gbm_train_probs <- predict(gbm_model, newdata = train_df,
                           n.trees = best_n_trees_gbm, type = "response")
gbm_test_probs <- predict(gbm_model, newdata = test_df,
                          n.trees = best_n_trees_gbm, type = "response")

gbm_roc <- pROC::roc(test_df[[target]], gbm_test_probs, quiet = TRUE)
gbm_auc <- round(as.numeric(pROC::auc(gbm_roc)), 3)

gbm_train_roc <- pROC::roc(train_df[[target]], gbm_train_probs, quiet = TRUE)
gbm_train_auc <- round(as.numeric(pROC::auc(gbm_train_roc)), 3)

cat("GBM — Train AUC:", gbm_train_auc, "| Test AUC:", gbm_auc, "\n")
cat("Gap (train - test):", round(gbm_train_auc - gbm_auc, 3),
    "— small gap means model generalizes well\n")


# ============================================================ #
# PART B: XGBoost
# ============================================================ #
# The faster, more powerful evolution. Same idea, better engine.
# ============================================================ #


# ---- 4A: Find Optimal Number of Rounds via Cross-Validation ----
# EARLY STOPPING explained:
#   Instead of guessing how many trees to build, we use
#   cross-validation to track performance as we add more trees.
#   When CV performance stops improving, we stop.
#   This prevents overfitting without needing to guess nrounds.
#
#   xgb.cv does this cleanly: it trains on K folds and returns
#   the train and test AUC at every round.

xgb_params <- list(
  objective        = "binary:logistic",  # Binary classification → output probabilities
  eval_metric      = "auc",              # Optimize for AUC
  tree_method      = xgb_tree_method,    # "gpu_hist" if GPU available, "hist" otherwise
  learning_rate    = 0.05,              # Learning rate — was 'eta' in older XGBoost versions
  max_depth        = 4,                  # Max tree depth (captures interactions)
  subsample        = 0.8,               # Use 80% of data per tree
  colsample_bytree = 0.8,               # Use 80% of features per tree (like mtry in RF)
  min_child_weight = 10,                # Minimum observations per leaf
  min_split_loss   = 1                  # Minimum loss reduction for a split — was 'gamma'
)

set.seed(42)

# Use xgb.cv to find optimal nrounds
xgb_cv <- xgboost::xgb.cv(
  params   = xgb_params,
  data     = dtrain,
  nrounds  = 2000,
  nfold    = 5,
  early_stopping_rounds = 50,
  verbose  = 0
)

# Extract the CV evaluation log
xgb_cv_log <- as.data.frame(xgb_cv$evaluation_log)
cat("\nXGB CV log columns:", paste(colnames(xgb_cv_log), collapse = ", "), "\n")

# Find test AUC column (named test_auc_mean in most versions)
cv_test_col  <- grep("test.*auc.*mean", colnames(xgb_cv_log), value = TRUE)[1]
cv_train_col <- grep("train.*auc.*mean", colnames(xgb_cv_log), value = TRUE)[1]
cv_test_std  <- grep("test.*auc.*std", colnames(xgb_cv_log), value = TRUE)[1]

# If the _mean pattern didn't work, try just test_auc
if (is.na(cv_test_col)) {
  cv_test_col  <- grep("test.*auc", colnames(xgb_cv_log), value = TRUE)[1]
  cv_train_col <- grep("train.*auc", colnames(xgb_cv_log), value = TRUE)[1]
}

best_iter <- which.max(xgb_cv_log[[cv_test_col]])
best_score <- xgb_cv_log[[cv_test_col]][best_iter]

cat("Best iteration:", best_iter, "out of 2000 max\n")
cat("Best CV AUC:", round(best_score, 4), "\n")


# ---- 4B: Visualize XGBoost Training Curve ----

xgb_cv_plot <- tibble(
  iter      = seq_len(nrow(xgb_cv_log)),
  train_auc = xgb_cv_log[[cv_train_col]],
  test_auc  = xgb_cv_log[[cv_test_col]]
)

p_xgb_curve <- xgb_cv_plot %>%
  tidyr::pivot_longer(c(train_auc, test_auc), names_to = "set", values_to = "auc") %>%
  dplyr::mutate(set = ifelse(set == "train_auc", "Training AUC", "CV Test AUC")) %>%
  ggplot(aes(x = iter, y = auc, color = set)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = best_iter, linetype = "dashed", color = "grey30") +
  annotate("text", x = best_iter + 20,
           y = min(xgb_cv_plot$test_auc, na.rm = TRUE) + 0.01,
           label = paste("Optimal:", best_iter, "rounds"),
           hjust = 0, fontface = "bold", size = 3.5) +
  scale_color_manual(values = c("Training AUC" = "#2196F3", "CV Test AUC" = "#F44336")) +
  labs(
    title    = "XGBoost Training Curve — Early Stopping in Action",
    subtitle = paste(
      "Training AUC keeps climbing (the model keeps learning the training data).\n",
      "CV Test AUC peaks then plateaus or drops — that's the overfitting boundary.\n",
      "We stop at the peak to prevent overfitting."
    ),
    x = "Boosting Round", y = "AUC", color = NULL
  ) +
  theme_ml

print(p_xgb_curve)
ggsave(file.path(out_dir, "02_xgb_training_curve.png"), p_xgb_curve,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 4C: Train Final XGBoost Model at Optimal Rounds ----
# Now train on the FULL training set using the optimal nrounds from CV
# Using xgb.train() instead of xgboost() — more explicit, always expects
# a DMatrix + params list, and doesn't try to infer objective from data type.

set.seed(42)

xgb_model <- xgboost::xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = best_iter,
  verbose = 0
)

cat("Final XGBoost model trained with", best_iter, "rounds\n")


# ---- 4D: XGBoost Predictions ----
# predict() on an XGBoost model automatically returns probabilities (0 to 1)
# because we set objective = "binary:logistic" — no need to specify type.
#
# We pass dtrain and dtest (DMatrix objects), not the raw data frames.
# XGBoost works exclusively with its own DMatrix format.
#
# Same logic as GBM: predict on both train and test to check for overfitting.
# A well-tuned XGBoost should have train AUC only slightly above test AUC.
# If train AUC is near 1.0 but test AUC is much lower — overfitting.

xgb_train_probs <- predict(xgb_model, newdata = dtrain)
xgb_test_probs  <- predict(xgb_model, newdata = dtest)

xgb_roc <- pROC::roc(test_y, xgb_test_probs, quiet = TRUE)
xgb_auc <- round(as.numeric(pROC::auc(xgb_roc)), 3)

xgb_train_roc <- pROC::roc(train_y, xgb_train_probs, quiet = TRUE)
xgb_train_auc <- round(as.numeric(pROC::auc(xgb_train_roc)), 3)

cat("XGBoost — Train AUC:", xgb_train_auc, "| Test AUC:", xgb_auc, "\n")
cat("Gap (train - test):", round(xgb_train_auc - xgb_auc, 3),
    "— regularization + early stopping should keep this small\n")


# ============================================================ #
# SECTION 5: THE LEARNING RATE EXPERIMENT
# ============================================================ #
# The learning rate is the most important concept in boosting.
# Let's SHOW its effect visually.
#
# WHAT HAPPENS WITH DIFFERENT LEARNING RATES:
#   High (0.3):  Big corrections each step → reaches good performance
#                fast but overshoots → overfits quickly
#   Medium (0.1): Moderate corrections → good balance
#   Low (0.01):  Tiny corrections each step → needs many more trees
#                but generalizes better → less overfitting
#
# Think of it like walking toward a target:
#   Big steps (high LR) = get there fast but might walk past it
#   Small steps (low LR) = slow but land precisely on target
# ============================================================ #

lr_values <- c(0.3, 0.1, 0.05, 0.01)

lr_results <- purrr::map_dfr(lr_values, function(lr) {
  params_lr <- list(
    objective = "binary:logistic", eval_metric = "auc",
    tree_method = xgb_tree_method,
    learning_rate = lr, max_depth = 4, subsample = 0.8,
    colsample_bytree = 0.8, min_child_weight = 10
  )
  
  xgb_lr <- xgboost::xgb.cv(
    params   = params_lr,
    data     = dtrain,
    nrounds  = 1500,
    nfold    = 5,
    early_stopping_rounds = 50,
    verbose  = 0
  )
  
  lr_log <- as.data.frame(xgb_lr$evaluation_log)
  lr_test_col <- grep("test.*auc.*mean", colnames(lr_log), value = TRUE)[1]
  if (is.na(lr_test_col)) lr_test_col <- grep("test.*auc", colnames(lr_log), value = TRUE)[1]
  
  tibble(
    iter          = seq_len(nrow(lr_log)),
    test_auc      = lr_log[[lr_test_col]],
    learning_rate = paste("LR =", lr)
  )
})

p_lr <- lr_results %>%
  ggplot(aes(x = iter, y = test_auc, color = learning_rate)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = c(
    "LR = 0.3"  = "#F44336",
    "LR = 0.1"  = "#FF9800",
    "LR = 0.05" = "#4CAF50",
    "LR = 0.01" = "#2196F3"
  )) +
  labs(
    title    = "Effect of Learning Rate on XGBoost Performance",
    subtitle = paste(
      "High learning rate (red): learns fast, peaks early, then overfits.\n",
      "Low learning rate (blue): learns slowly, needs more rounds, but reaches\n",
      "a better or equally good peak with less overfitting."
    ),
    x = "Boosting Round", y = "Test AUC", color = NULL
  ) +
  theme_ml

print(p_lr)
ggsave(file.path(out_dir, "03_learning_rate_effect.png"), p_lr,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: EVALUATION — COMPARING GBM vs XGBoost
# ============================================================ #

# ---- 6A: ROC Curves ----
# WHAT IS A ROC CURVE?
#   The model outputs probabilities (e.g. 0.08, 0.23, 0.61).
#   To make a yes/no decision ("does this patient have heart disease?"),
#   we need a THRESHOLD. Any patient above the threshold = predicted positive.
#
#   But which threshold? If we set it at 0.5:
#     - We miss many true cases (heart disease is only ~6% prevalence,
#       so most probabilities are low — even sick patients might score 0.3)
#   If we set it at 0.05:
#     - We catch most cases but create many false alarms
#
#   The ROC curve solves this by showing performance at EVERY possible threshold
#   at once, plotting:
#     - Y axis: True Positive Rate (Sensitivity) — % of real HD cases we catch
#     - X axis: False Positive Rate — % of healthy patients we incorrectly flag
#
#   AUC (Area Under the Curve) summarizes the whole curve in one number:
#     0.5 = as good as random guessing (diagonal line)
#     0.7 = acceptable discrimination
#     0.8 = good
#     0.9+ = excellent
#
#   A model in the TOP-LEFT corner catches most cases with few false alarms.
#   A model on the DIAGONAL is useless — it's just guessing.
#
# YOUDEN'S J — HOW WE PICK THE OPTIMAL THRESHOLD:
#   Among all thresholds on the ROC curve, Youden's J picks the one that
#   maximises (Sensitivity + Specificity - 1), i.e. the point geometrically
#   closest to the top-left corner. This gives the best balance between
#   catching cases and avoiding false alarms.
#   We use this threshold in the confusion matrix below.

gbm_roc_df <- tibble(fpr = 1 - gbm_roc$specificities, sensitivity = gbm_roc$sensitivities)
xgb_roc_df <- tibble(fpr = 1 - xgb_roc$specificities, sensitivity = xgb_roc$sensitivities)

# Load previous models for comparison
logistic_path <- "Outputs/Logistic_Regression/results_logistic.rds"
rf_path       <- "Outputs/Random_Forest/results_rf.rds"
has_logistic  <- file.exists(logistic_path)
has_rf        <- file.exists(rf_path)

if (has_logistic) {
  logistic_results    <- readRDS(logistic_path)
  logistic_test_probs <- readRDS("Outputs/Logistic_Regression/test_predictions_logistic.rds")
  logistic_roc        <- pROC::roc(test_y, logistic_test_probs, quiet = TRUE)
  logistic_roc_df     <- tibble(fpr = 1 - logistic_roc$specificities, sensitivity = logistic_roc$sensitivities)
}

if (has_rf) {
  rf_results    <- readRDS(rf_path)
  rf_test_probs <- readRDS("Outputs/Random_Forest/test_predictions_rf.rds")
  rf_roc        <- pROC::roc(test_y, rf_test_probs, quiet = TRUE)
  rf_roc_df     <- tibble(fpr = 1 - rf_roc$specificities, sensitivity = rf_roc$sensitivities)
}

p_roc <- ggplot() +
  geom_line(data = xgb_roc_df, aes(x = fpr, y = sensitivity, color = "XGBoost"),
            linewidth = 1.3) +
  geom_line(data = gbm_roc_df, aes(x = fpr, y = sensitivity, color = "GBM"),
            linewidth = 1, linetype = "dashed") +
  {if (has_rf)
    geom_line(data = rf_roc_df, aes(x = fpr, y = sensitivity, color = "Random Forest"),
              linewidth = 0.8, linetype = "dotted")
  } +
  {if (has_logistic)
    geom_line(data = logistic_roc_df, aes(x = fpr, y = sensitivity, color = "Logistic Regression"),
              linewidth = 0.8, linetype = "dotted")
  } +
  geom_abline(linetype = "dotted", color = "grey50") +
  scale_color_manual(values = c(
    "XGBoost"             = "#F44336",
    "GBM"                 = "#FF9800",
    "Random Forest"       = "#4CAF50",
    "Logistic Regression" = "#2196F3"
  )) +
  annotate("text", x = 0.55, y = 0.45, label = paste("XGBoost AUC =", xgb_auc),
           size = 4.5, fontface = "bold", color = "#F44336") +
  annotate("text", x = 0.55, y = 0.38, label = paste("GBM AUC =", gbm_auc),
           size = 4, color = "#FF9800") +
  {if (has_rf)
    annotate("text", x = 0.55, y = 0.31, label = paste("RF AUC =", rf_results$auc),
             size = 4, color = "#4CAF50")
  } +
  {if (has_logistic)
    annotate("text", x = 0.55, y = 0.24, label = paste("Logistic AUC =", logistic_results$auc),
             size = 4, color = "#2196F3")
  } +
  labs(
    title    = "ROC Curve — Boosting Models vs Other Approaches",
    subtitle = paste(
      "Closer to the top-left corner = better. Diagonal = random guessing.\n",
      "Boosting models (GBM, XGBoost) learn from their mistakes sequentially."
    ),
    x = "False Positive Rate\n(% of healthy patients incorrectly flagged)",
    y = "True Positive Rate (Sensitivity)\n(% of heart disease patients correctly caught)",
    color = "Model"
  ) +
  theme_ml

print(p_roc)
ggsave(file.path(out_dir, "04_roc_curve_comparison.png"), p_roc,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 6B: Confusion Matrix (XGBoost — best model) ----
# WHAT IS A CONFUSION MATRIX?
#   After choosing a threshold to convert probabilities → yes/no decisions,
#   we get 4 outcomes:
#     True Positive:  Has HD, model said YES — correctly caught
#     False Positive: Healthy, model said YES — false alarm
#     True Negative:  Healthy, model said NO — correctly ruled out
#     False Negative: Has HD, model said NO — MISSED case
#
#   Precision = of those flagged positive, how many truly have HD
#   Recall = of those who have HD, how many did we catch
#   F1 = harmonic mean balancing both

optimal_coords <- pROC::coords(xgb_roc, "best",
                               ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold

xgb_pred_class <- ifelse(xgb_test_probs >= optimal_threshold, 1, 0)

cm <- table(Actual = test_y, Predicted = xgb_pred_class)
tp <- cm["1", "1"]; fp <- cm["0", "1"]; tn <- cm["0", "0"]; fn <- cm["1", "0"]

accuracy  <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- XGBoost Classification Metrics (threshold =", round(optimal_threshold, 3), ") ---\n")
cat("Accuracy: ", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), " — Of those flagged, % truly have HD\n")
cat("Recall:   ", round(recall, 3), " — Of HD patients, % we caught\n")
cat("F1 Score: ", round(f1, 3), "\n")

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
    title    = "Confusion Matrix — XGBoost",
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


# ============================================================ #
# SECTION 7: FEATURE IMPORTANCE
# ============================================================ #
# WHAT IS FEATURE IMPORTANCE?
#   Which features did the model rely on most?
#   XGBoost provides several measures:
#
#   "Gain": The average improvement in accuracy a feature brings
#     when it's used in a split. High gain = the model's predictions
#     get much better when splitting on this feature.
#     This is the most informative measure.
#
#   "Cover": How many observations a feature's splits affect.
#
#   "Frequency": How often a feature is used across all trees.
#     (Can be misleading — a feature might be used often but only
#     for tiny improvements.)
# ============================================================ #

# XGBoost built-in importance
xgb_imp <- xgboost::xgb.importance(
  feature_names = colnames(train_matrix),
  model         = xgb_model
)

p_xgb_imp <- xgb_imp %>%
  dplyr::slice_head(n = 15) %>%
  dplyr::mutate(Feature = forcats::fct_reorder(Feature, Gain)) %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_col(fill = "#F44336", alpha = 0.85) +
  labs(
    title    = "XGBoost Feature Importance (Gain)",
    subtitle = paste(
      "Gain = average improvement in prediction accuracy when this feature is used.\n",
      "Higher gain = the model learns more from this feature."
    ),
    x = "Gain (higher = more important)", y = NULL
  ) +
  theme_ml

print(p_xgb_imp)
ggsave(file.path(out_dir, "06_xgb_feature_importance.png"), p_xgb_imp,
       width = 10, height = 7, dpi = 150, bg = "white")

# GBM importance for comparison
gbm_imp <- summary(gbm_model, n.trees = best_n_trees_gbm, plotit = FALSE)

p_gbm_imp <- gbm_imp %>%
  dplyr::slice_head(n = 15) %>%
  dplyr::mutate(var = forcats::fct_reorder(var, rel.inf)) %>%
  ggplot(aes(x = rel.inf, y = var)) +
  geom_col(fill = "#FF9800", alpha = 0.85) +
  labs(
    title    = "GBM Feature Importance (Relative Influence)",
    subtitle = "How much each feature contributes to the GBM model's predictions.",
    x = "Relative Influence (%)", y = NULL
  ) +
  theme_ml

print(p_gbm_imp)
ggsave(file.path(out_dir, "07_gbm_feature_importance.png"), p_gbm_imp,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: SHAP VALUES — THE GOLD STANDARD OF INTERPRETATION
# ============================================================ #
# WHAT ARE SHAP VALUES?
#   SHAP (SHapley Additive exPlanations) comes from game theory.
#   For EACH patient, it tells you exactly how much each feature
#   PUSHED the prediction up or down from the average.
#
#   Example: For patient #42:
#     Average prediction: 6% (base rate)
#     Age 72: pushes prediction UP by +8%
#     SBP 180: pushes prediction UP by +5%
#     Never smoked: pushes prediction DOWN by -3%
#     Final prediction: 6% + 8% + 5% - 3% + ... = 18%
#
#   SHAP values are additive — they sum to the difference between
#   the individual prediction and the overall average.
#
#   WHY SHAP IS SPECIAL:
#     - Feature importance tells you WHICH features matter ON AVERAGE
#     - SHAP tells you how each feature matters FOR EACH PATIENT
#     - You can see that age matters MORE for older patients
#       and LESS for younger ones
#     - This is crucial in healthcare: "why did the model flag
#       THIS specific patient?"
# ============================================================ #

# Compute SHAP values for a sample of test patients
# (Full dataset is slow; 500 patients gives a good picture)
set.seed(42)
shap_sample_idx <- sample(1:nrow(test_matrix), min(500, nrow(test_matrix)))
shap_data <- SHAPforxgboost::shap.values(
  xgb_model  = xgb_model,
  X_train    = test_matrix[shap_sample_idx, ]
)

# SHAP summary plot — shows distribution of SHAP values per feature
# Each dot is one patient. Color = feature value (red = high, blue = low).
# Position on x-axis = how much that feature pushed the prediction.
shap_long <- SHAPforxgboost::shap.prep(
  shap_contrib = shap_data$shap_score,
  X_train      = as.data.frame(test_matrix[shap_sample_idx, ])
)

p_shap <- SHAPforxgboost::shap.plot.summary(shap_long) +
  labs(
    title    = "SHAP Summary Plot — XGBoost",
    subtitle = paste(
      "Each dot = one patient. Position = how much the feature pushed the prediction.\n",
      "Right (positive SHAP) = increases heart disease risk. Left = decreases risk.\n",
      "Color = feature value (red = high, blue = low).\n",
      "Example: high age (red dots) pushes predictions RIGHT (increases risk)."
    )
  ) +
  theme_ml +
  theme(plot.title = element_text(face = "bold", size = 14))

print(p_shap)
ggsave(file.path(out_dir, "08_shap_summary.png"), p_shap,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 9: CROSS-VALIDATION
# ============================================================ #
# WHAT IS CROSS-VALIDATION?
#   A single train/test split might be lucky or unlucky depending on
#   which patients happened to end up in which set.
#   5-fold CV repeats the experiment 5 times with different splits:
#     1. Split training data into 5 equal chunks ("folds")
#     2. Train on 4 chunks, evaluate on the held-out chunk
#     3. Rotate which chunk is held out — each fold gets a turn
#     4. Average the 5 AUC values
#   Consistent results across folds = reliable model.
#   Wildly varying results = the model is sensitive to which data it sees.
#
# NOTE: xgb.cv (used earlier in Section 4A) already did CV to find the
# optimal number of rounds. This section does an ADDITIONAL CV pass to
# get a clean, stable estimate of out-of-sample AUC using the final model.
# Think of Section 4A CV as "tuning" and this CV as "evaluation."
#
# HOW THE CODE WORKS:
#   createFolds() splits the training row indices into 5 groups.
#   returnTrain = TRUE means each element of cv_folds contains the
#   indices to USE FOR TRAINING (not the held-out fold).
#   map_dfr() loops over each fold (i = 1 to 5), fits a model,
#   evaluates on the held-out fold, and stacks the results into one table.

set.seed(42)
cv_folds <- caret::createFolds(train_y, k = 5, returnTrain = TRUE)

cv_results <- purrr::map_dfr(seq_along(cv_folds), function(i) {
  
  # cv_folds[[i]] = row indices for training in this fold
  # -cv_folds[[i]] = the held-out validation rows (everything NOT in training)
  fold_dtrain <- xgboost::xgb.DMatrix(
    data  = train_matrix[cv_folds[[i]], ],
    label = train_y[cv_folds[[i]]]
  )
  fold_dval <- xgboost::xgb.DMatrix(
    data  = train_matrix[-cv_folds[[i]], ],
    label = train_y[-cv_folds[[i]]]
  )
  
  # Train a fresh XGBoost on this fold's training data
  # Same params and nrounds as the final model — we're evaluating that config
  fold_xgb <- xgboost::xgb.train(
    params  = xgb_params,
    data    = fold_dtrain,
    nrounds = best_iter,
    verbose = 0
  )
  
  # Predict on the held-out validation fold
  fold_probs <- predict(fold_xgb, newdata = fold_dval)
  
  # Compute AUC for this fold
  fold_auc <- as.numeric(pROC::auc(pROC::roc(
    train_y[-cv_folds[[i]]], fold_probs, quiet = TRUE
  )))
  
  # Return a one-row tibble — map_dfr stacks these into 5 rows total
  tibble(fold = i, auc = fold_auc)
})

cat("\n--- 5-Fold Cross-Validation AUC (XGBoost) ---\n")
print(cv_results)
cat("Mean CV AUC:", round(mean(cv_results$auc), 3),
    "± SD:", round(sd(cv_results$auc), 3), "\n")

p_cv <- ggplot(cv_results, aes(x = factor(fold), y = auc)) +
  geom_col(fill = "#F44336", alpha = 0.85, width = 0.6) +
  geom_hline(yintercept = mean(cv_results$auc), linetype = "dashed", color = "#212121") +
  annotate("text", x = 5.4, y = mean(cv_results$auc),
           label = paste("CV Mean:", round(mean(cv_results$auc), 3)),
           hjust = 0, fontface = "bold", size = 3.5) +
  coord_cartesian(xlim = c(0.5, 6.5), ylim = c(
    min(cv_results$auc) - 0.02,
    max(cv_results$auc) + 0.02
  )) +
  labs(
    title    = "Cross-Validation AUC — XGBoost",
    subtitle = "Consistent bars = stable model. Wildly different bars = unreliable.",
    x = "Fold", y = "AUC"
  ) +
  theme_ml

print(p_cv)
ggsave(file.path(out_dir, "09_cross_validation.png"), p_cv,
       width = 8, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 10: SAVE RESULTS
# ============================================================ #

results_boosting <- tibble(
  model          = c("GBM", "XGBoost"),
  target         = "heart_disease",
  auc            = c(gbm_auc, xgb_auc),
  accuracy       = round(accuracy, 4),       # From XGBoost (best model)
  precision      = round(precision, 4),
  recall         = round(recall, 4),
  f1             = round(f1, 4),
  cv_auc_mean    = round(mean(cv_results$auc), 4),
  cv_auc_sd      = round(sd(cv_results$auc), 4),
  train_auc      = c(gbm_train_auc, xgb_train_auc),
  threshold_used = round(optimal_threshold, 4)
)

saveRDS(results_boosting, file.path(out_dir, "results_boosting.rds"))
saveRDS(xgb_test_probs, file.path(out_dir, "test_predictions_xgboost.rds"))
saveRDS(gbm_test_probs, file.path(out_dir, "test_predictions_gbm.rds"))
saveRDS(xgb_model, file.path(out_dir, "xgb_model.rds"))

cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 11: SUMMARY & KEY TAKEAWAYS
# ============================================================ #

cat("\n")
cat("=== GRADIENT BOOSTING & XGBoost — SUMMARY ===\n")
cat("GBM Test AUC:     ", gbm_auc, "\n")
cat("XGBoost Test AUC: ", xgb_auc, "\n")
cat("XGBoost F1:       ", round(f1, 3), "\n")
cat("XGBoost CV AUC:   ", round(mean(cv_results$auc), 3), "±",
    round(sd(cv_results$auc), 3), "\n")
cat("Best n_trees (GBM):", best_n_trees_gbm, "\n")
cat("Best iteration (XGBoost):", best_iter, "\n")

if (has_rf | has_logistic) {
  cat("\n--- Comparison (all predicting heart disease) ---\n")
  if (has_logistic) cat("Logistic Regression:", logistic_results$auc,
                        " — (linear model, assumes straight-line relationships)\n")
  if (has_rf) cat("Random Forest:      ", rf_results$auc,
                  " — (500 independent trees averaged together)\n")
  cat("GBM:                ", gbm_auc,
      " — (sequential trees, each correcting previous errors)\n")
  cat("XGBoost:            ", xgb_auc,
      " — (optimized boosting with regularization)\n")
}

cat("\n")
cat("WHAT WE LEARNED:\n")
cat("1. Boosting builds trees SEQUENTIALLY — each tree fixes the mistakes of prior trees.\n")
cat("2. The learning rate controls how cautious each correction is.\n")
cat("   Small learning rate + many trees = best results.\n")
cat("3. EARLY STOPPING prevents overfitting by watching validation performance.\n")
cat("4. XGBoost adds regularization, column sampling, and speed to basic boosting.\n")
cat("5. SHAP values explain WHY the model made each individual prediction.\n")
cat("6. Boosting models often achieve the highest accuracy on tabular data.\n")
cat("7. The tradeoff: more complex to tune than Random Forest, less interpretable\n")
cat("   than logistic regression, but often the best raw performance.\n")
cat("\n")
cat("WHEN TO USE BOOSTING:\n")
cat("- When you want maximum predictive accuracy on tabular data.\n")
cat("- When you can invest time in tuning hyperparameters.\n")
cat("- XGBoost is the go-to model for most Kaggle competitions.\n")
cat("\n")
cat("WHAT BOOSTING CANNOT DO:\n")
cat("- It doesn't work well on images, text, or audio (use neural networks).\n")
cat("- It needs feature engineering — it can't learn raw pixel patterns.\n")
cat("- It's a 'black box' without tools like SHAP to open it up.\n")
cat("===================================\n")