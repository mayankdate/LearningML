# ============================================================ #
# 06 — Neural Networks for Classification
# ============================================================ #
# Purpose: Teach neural network fundamentals by building a
#          feedforward network to predict heart_disease.
#          Covers: what neurons do, layers, activations,
#          backpropagation intuition, regularization, and
#          how to evaluate a neural net the same way we
#          evaluate any classifier (ROC, confusion matrix, CV).
# Target:  heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, ggplot2, patchwork, pROC, caret,
  nnet,          # Single hidden layer neural network (simple, good for teaching)
  NeuralNetTools, # Visualization of neural network weights and structure
  scales, forcats, tibble, stringr
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Neural_Network", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Neural_Network"

# ---- Global Plot Theme ----
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    legend.position = "bottom"
  )

# For patchwork plot_annotation() — must be a raw theme() object
theme_annotation <- theme(
  plot.title    = element_text(face = "bold", size = 14),
  plot.subtitle = element_text(color = "grey40", size = 11)
)

# Color palette — consistent across all scripts
col_neg  <- "#2196F3"   # Blue  = no disease
col_pos  <- "#F44336"   # Red   = disease
col_good <- "#4CAF50"   # Green = correct

# ---- Load Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")

target   <- "heart_disease"
features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)

cat("Dataset:", nrow(ml_data), "rows |", ncol(ml_data), "columns\n")
cat("Heart disease prevalence:", round(mean(ml_data[[target]]) * 100, 1), "%\n")


# ============================================================ #
# SECTION 1: WHAT IS A NEURAL NETWORK?
# ============================================================ #
#
# ANALOGY: Imagine a group of doctors making a diagnosis together.
# Each doctor looks at the patient's chart and forms a quick
# opinion (a "neuron"). No single doctor sees the full picture —
# one focuses on blood pressure, another on age and smoking.
# Their opinions are combined by a senior doctor, who weighs
# them differently (one doctor's opinion matters more than
# another's). That senior doctor outputs the final diagnosis.
#
# A neural network works the same way:
#
#   INPUT LAYER         HIDDEN LAYER          OUTPUT LAYER
#   (raw features)      (learned combinations) (prediction)
#
#   age ──────────┐
#   sex ──────────┤──→ [Neuron 1] ──┐
#   bmi ──────────┤──→ [Neuron 2] ──┼──→ P(heart disease)
#   sbp ──────────┤──→ [Neuron 3] ──┘
#   smoking ──────┘       ...
#
# Each arrow has a WEIGHT — a number that controls how much
# that input matters. The network LEARNS these weights from data.
#
# HOW IT LEARNS (BACKPROPAGATION — the key idea):
#   1. Forward pass: Feed a patient's data through the network,
#      get a predicted probability.
#   2. Compare prediction to the truth (0 or 1). Calculate ERROR.
#   3. Backward pass: Trace the error back through each weight,
#      asking "how much did YOU contribute to this mistake?"
#   4. Adjust each weight a tiny bit in the direction that
#      reduces the error.
#   5. Repeat for every patient, many times (called "epochs").
#
# This is gradient descent: repeatedly nudging weights downhill
# on an error surface until the network makes good predictions.
#
# ACTIVATION FUNCTIONS — why they matter:
#   Without an activation function, stacking layers is pointless —
#   multiple linear layers collapse into one linear equation.
#   Activation functions introduce nonlinearity, letting the
#   network learn curved, complex decision boundaries.
#
#   Common activations:
#   - Sigmoid: squashes output to 0–1 (used for final probability)
#   - ReLU: max(0, x) — fast, used in hidden layers of deep nets
#   - Logistic (in nnet): similar to sigmoid, default for hidden layer
#
# WHEN NEURAL NETS WORK WELL:
#   - Complex nonlinear relationships between features
#   - Lots of training data (they're data-hungry)
#   - You don't need to explain individual predictions easily
#
# WHEN THEY STRUGGLE:
#   - Small datasets — they overfit quickly
#   - When interpretability is critical (they're "black boxes")
#   - Tabular data with few features (tree ensembles often win)
#
# HOW THIS DIFFERS FROM OTHER MODELS:
#   - Logistic regression: draws ONE straight decision boundary.
#     A neural net with hidden layers can draw curved boundaries.
#   - Decision trees: make yes/no splits on one feature at a time.
#     Neural nets combine ALL features simultaneously in every neuron.
#   - Random forests: average many trees for stability.
#     Neural nets learn a single complex function end-to-end.
#   - XGBoost: builds trees sequentially to fix previous errors.
#     Neural nets adjust all weights simultaneously via gradient descent.
# ============================================================ #


# ---- Train/Test Split ----
# Hold out 25% of patients the model never sees during training.
# We evaluate on these "new patients" to estimate real-world performance.
# Same seed and method as all other scripts for fair comparison.
set.seed(42)
train_index <- caret::createDataPartition(ml_data[[target]], p = 0.75, list = FALSE)
train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train prevalence:", round(mean(train_df[[target]]) * 100, 1), "%\n")
cat("Test prevalence:", round(mean(test_df[[target]]) * 100, 1), "%\n")

# ---- Factor Setup ----
# Convert categorical variables to factors with explicit levels.
# Neural networks need numeric inputs, so caret will dummy-code these
# factors automatically during preprocessing.
factor_setup <- function(df) {
  df %>%
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
}

train_df <- factor_setup(train_df)
test_df  <- factor_setup(test_df)


# ============================================================ #
# SECTION 2: DATA PREPARATION — WHY SCALING IS CRITICAL
# ============================================================ #
#
# Neural networks learn by adjusting weights via gradient descent.
# If features are on very different scales (age: 18–95 vs
# income: 8,000–300,000), the gradients for income will be
# enormous and dominate learning, while age barely moves.
# This makes training unstable and slow.
#
# SOLUTION: Center (subtract mean) and scale (divide by SD) all
# numeric features so they're all roughly on the same scale
# (mean ≈ 0, SD ≈ 1). This is sometimes called "standardization"
# or "z-score normalization."
#
# WHY trees don't need this: Trees split on thresholds
# ("age > 65?"), so absolute scale doesn't matter.
# WHY neural nets DO: They multiply features by weights and sum
# them — large-scale features swamp small-scale ones.
#
# IMPORTANT: We compute mean/SD from TRAINING data only, then
# apply the same transformation to test data. If we used test
# data to compute the scaling parameters, we'd be "peeking" at
# information the model shouldn't have (data leakage).
# ============================================================ #

# ---- Create Model Matrices ----
# caret::dummyVars converts factors to 0/1 indicator columns.
# For example, sex_Male = 1 if Male, 0 if Female.
# This is called "one-hot encoding" — neural nets need numeric input.
dummy_formula <- as.formula(paste("~", paste(features, collapse = " + ")))
dummy_obj     <- caret::dummyVars(dummy_formula, data = train_df)

train_x <- predict(dummy_obj, newdata = train_df)
test_x  <- predict(dummy_obj, newdata = test_df)

cat("After dummy coding:", ncol(train_x), "input features\n")
# NOTE: nnet() has a default limit of 1000 weights (MaxNWts = 1000).
# Total weights = (n_inputs + 1) × size + (size + 1) × 1.
# With 25 inputs and size=40: (26 × 40) + 41 = 1081 → exceeds the cap.
# We raise MaxNWts where needed. This is a common nnet gotcha.

# ---- Scale Numeric Features ----
# Compute mean and SD from training data only
train_means <- colMeans(train_x)
train_sds   <- apply(train_x, 2, sd)

# Prevent division by zero for constant columns (e.g., a dummy with no variance)
train_sds[train_sds == 0] <- 1

# Apply same centering/scaling to both sets
train_x_scaled <- scale(train_x, center = train_means, scale = train_sds)
test_x_scaled  <- scale(test_x,  center = train_means, scale = train_sds)

# Prepare target vectors
train_y <- train_df[[target]]
test_y  <- test_df[[target]]

# Combine into data frames for nnet (it wants a formula interface)
train_nn <- as.data.frame(train_x_scaled)
train_nn$heart_disease <- factor(train_y, levels = c(0, 1), labels = c("No", "Yes"))

test_nn <- as.data.frame(test_x_scaled)
test_nn$heart_disease <- factor(test_y, levels = c(0, 1), labels = c("No", "Yes"))

cat("Preprocessing complete — scaled", ncol(train_x_scaled), "features\n")


# ============================================================ #
# SECTION 3: FIT THE NEURAL NETWORK
# ============================================================ #
#
# We use nnet::nnet() — a single-hidden-layer feedforward network.
# This is the simplest true neural network:
#   Input → One hidden layer → Output
#
# It's not "deep learning" (that needs multiple hidden layers),
# but it teaches all the core concepts: weights, activations,
# backpropagation, overfitting, and regularization.
#
# KEY HYPERPARAMETERS:
#
# size = number of neurons in the hidden layer
#   More neurons = more capacity to learn complex patterns
#   Too many = overfitting (memorizes noise in training data)
#   Too few = underfitting (can't capture the real pattern)
#   Rule of thumb: start between (n_features / 2) and n_features
#
# decay = weight decay (L2 regularization)
#   Adds a penalty for large weights: loss + decay * sum(weights^2)
#   This discourages the network from fitting noise
#   Larger decay = stronger regularization = simpler model
#   Similar idea to the penalty term in Ridge regression
#
# maxit = maximum training iterations (epochs)
#   Each epoch = one full pass through all training data
#   More epochs = more chances to learn, but risk of overfitting
#   nnet stops if it converges before reaching maxit
#
# We'll try multiple configurations to see how these choices
# affect performance.
# ============================================================ #

# ---- 3a: Quick Single Model — See It Work ----
# Before tuning, let's train ONE network to see what happens.
# This takes ~15 seconds and lets us inspect the object.
set.seed(42)
nn_demo <- nnet::nnet(
  heart_disease ~ .,
  data    = train_nn,
  size    = 10,       # 10 hidden neurons
  decay   = 0.1,      # Moderate regularization
  maxit   = 100,      # 100 iterations is enough for nnet to converge on this data
  trace   = FALSE,    # Suppress iteration-by-iteration output
  linout  = FALSE     # FALSE = classification (logistic output)
)

probs_demo <- predict(nn_demo, newdata = test_nn, type = "raw")[, 1]
roc_demo   <- pROC::roc(test_y, probs_demo, quiet = TRUE)
cat("Demo model (10 neurons, decay=0.1) — Test AUC:", round(auc(roc_demo), 4), "\n")

# ---- 3b: Why nnet Is Slow (and What You'd Use in Practice) ----
# nnet is a 1990s pure-R implementation — good for teaching, slow
# for production. For real work, you'd use:
#   - R: keras3 or torch — same GPU backends as Python (your RTX 4060)
#   - Python: PyTorch or TensorFlow — the industry standard
# Tree-based models (XGBoost, ranger) use compiled C++ and are
# already fast in R — no Python needed for those.
# We use nnet here because it's simple to install and debug,
# and the teaching goal is concepts, not speed.


# ============================================================ #
# SECTION 4: HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# ============================================================ #
#
# CROSS-VALIDATION: Instead of a single train/test split, we
# divide the training data into K folds (here K=5). We train on
# 4 folds, test on the 5th, and rotate. Each patient appears in
# the test fold exactly once. This gives us 5 honest performance
# estimates — their average is more reliable than a single split.
#
# We'll use caret::train() to search over a grid of:
#   - size: number of hidden neurons
#   - decay: regularization strength
#
# RUNTIME NOTE: nnet is slow in R. We keep the grid small (6 combos
# × 5 folds = 30 fits) and cap iterations at 100. This should run
# in ~2-4 minutes. The teaching point is the SIZE vs DECAY tradeoff,
# which is clear even from a small grid.
# ============================================================ #

# ---- Tuning Grid ----
nn_grid <- expand.grid(
  size  = c(5, 10, 20),
  decay = c(0.01, 0.1)
)

cat("Testing", nrow(nn_grid), "hyperparameter combinations via 5-fold CV\n")
cat("(6 combos × 5 folds = 30 model fits — should take ~2-4 minutes)\n")

# ---- Cross-Validation Setup ----
set.seed(42)
cv_control <- caret::trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,             # Need probabilities for AUC
  summaryFunction = twoClassSummary,  # Reports ROC (AUC), Sens, Spec
  verboseIter     = TRUE              # Show progress so you know it's working
)

# ---- Train with Grid Search ----
set.seed(42)
nn_tuned <- caret::train(
  heart_disease ~ .,
  data      = train_nn,
  method    = "nnet",
  tuneGrid  = nn_grid,
  trControl = cv_control,
  metric    = "ROC",         # Optimize for AUC
  maxit     = 100,           # 100 iterations is sufficient for convergence
  trace     = FALSE,
  linout    = FALSE,
  MaxNWts   = 2000           # nnet cap is 1000 by default; raise for larger architectures
)

cat("Best tuned parameters: size =", nn_tuned$bestTune$size,
    "| decay =", nn_tuned$bestTune$decay, "\n")
cat("Best CV AUC:", round(max(nn_tuned$results$ROC), 4), "\n")

# ---- Get Final Predictions from Tuned Model ----
probs_tuned <- predict(nn_tuned, newdata = test_nn, type = "prob")[, "Yes"]
roc_tuned   <- pROC::roc(test_y, probs_tuned, quiet = TRUE)

cat("Tuned model Test AUC:", round(auc(roc_tuned), 4), "\n")


# ============================================================ #
# SECTION 5: EVALUATE — OPTIMAL THRESHOLD
# ============================================================ #
#
# The neural network outputs a probability (0 to 1). We need a
# THRESHOLD to convert that to a yes/no prediction.
# Default is 0.5, but with ~6% prevalence, 0.5 is too high —
# almost nobody gets flagged.
#
# Youden's J statistic finds the threshold that maximizes
# (sensitivity + specificity - 1), balancing detection of disease
# cases (sensitivity) with avoiding false alarms (specificity).
# ============================================================ #

# Use the tuned model as our final model
final_probs <- probs_tuned
final_roc   <- roc_tuned

optimal_coords <- pROC::coords(final_roc, "best",
                               ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold

cat("Optimal threshold (Youden's J):", round(optimal_threshold, 4), "\n")
cat("  Sensitivity at threshold:", round(optimal_coords$sensitivity, 4),
    "— catches this fraction of true cases\n")
cat("  Specificity at threshold:", round(optimal_coords$specificity, 4),
    "— correctly rules out this fraction of non-cases\n")

# ---- Apply Threshold ----
pred_class <- ifelse(final_probs >= optimal_threshold, 1, 0)
pred_factor <- factor(pred_class, levels = c(0, 1), labels = c("No", "Yes"))
test_factor <- factor(test_y,     levels = c(0, 1), labels = c("No", "Yes"))

# ---- Confusion Matrix ----
cm <- caret::confusionMatrix(pred_factor, test_factor, positive = "Yes")

accuracy  <- cm$overall["Accuracy"]
precision <- cm$byClass["Pos Pred Value"]
recall    <- cm$byClass["Sensitivity"]
f1        <- 2 * (precision * recall) / (precision + recall)

cat("\nConfusion Matrix Metrics:\n")
cat("  Accuracy: ", round(accuracy, 4),  "— fraction of all predictions correct\n")
cat("  Precision:", round(precision, 4), "— of those we flagged, how many truly have disease\n")
cat("  Recall:   ", round(recall, 4),    "— of true cases, how many we caught\n")
cat("  F1 Score: ", round(f1, 4),        "— harmonic mean of precision and recall\n")


# ============================================================ #
# SECTION 6: VISUALIZATIONS
# ============================================================ #

# ---- 6a: ROC Curve ----
roc_df <- tibble(
  sensitivity = final_roc$sensitivities,
  specificity = final_roc$specificities,
  fpr         = 1 - final_roc$specificities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = sensitivity)) +
  geom_line(color = col_pos, linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "grey60") +
  geom_point(
    aes(x = 1 - optimal_coords$specificity, y = optimal_coords$sensitivity),
    size = 3, color = col_pos, shape = 16
  ) +
  annotate("text",
           x = 1 - optimal_coords$specificity + 0.05,
           y = optimal_coords$sensitivity - 0.05,
           label = paste0("Threshold: ", round(optimal_threshold, 3)),
           size = 3.5, color = col_pos
  ) +
  annotate("text", x = 0.75, y = 0.15,
           label = paste0("AUC = ", round(auc(final_roc), 3)),
           size = 5, fontface = "bold"
  ) +
  labs(
    title    = "ROC Curve — Neural Network",
    subtitle = "Closer to top-left corner = better. Diagonal = random guessing.",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_ml

ggsave(file.path(out_dir, "06_roc_curve.png"), p_roc,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 6b: ROC Comparison with Previous Models (if available) ----
# Load previous model predictions for visual comparison.
# Each previous model saved test_predictions_[model].rds containing
# predicted probabilities on the SAME test set.
prev_models <- list(
  list(name = "Logistic Regression", dir = "Logistic_Regression", file = "test_predictions_logistic.rds"),
  list(name = "Decision Tree",       dir = "Decision_Trees",      file = "test_predictions_tree.rds"),
  list(name = "Random Forest",       dir = "Random_Forest",       file = "test_predictions_rf.rds"),
  list(name = "XGBoost",             dir = "Boosting",            file = "test_predictions_xgb.rds"),
  list(name = "KNN",                 dir = "KNN_SVM",             file = "test_predictions_knn.rds"),
  list(name = "SVM",                 dir = "KNN_SVM",             file = "test_predictions_svm.rds")
)

# Build comparison ROC data
roc_compare_list <- list(
  tibble(fpr = 1 - final_roc$specificities,
         tpr = final_roc$sensitivities,
         model = paste0("Neural Network (", round(auc(final_roc), 3), ")"))
)

for (m in prev_models) {
  fpath <- file.path("Outputs", m$dir, m$file)
  if (file.exists(fpath)) {
    prev_probs <- readRDS(fpath)
    prev_roc   <- pROC::roc(test_y, prev_probs, quiet = TRUE)
    roc_compare_list[[length(roc_compare_list) + 1]] <- tibble(
      fpr   = 1 - prev_roc$specificities,
      tpr   = prev_roc$sensitivities,
      model = paste0(m$name, " (", round(auc(prev_roc), 3), ")")
    )
  }
}

roc_compare_df <- dplyr::bind_rows(roc_compare_list)

# Color palette — neural network highlighted
n_models <- length(unique(roc_compare_df$model))
model_colors <- scales::hue_pal()(n_models)
names(model_colors) <- unique(roc_compare_df$model)
# Make the neural network line bold red
nn_model_name <- grep("Neural Network", names(model_colors), value = TRUE)
model_colors[nn_model_name] <- col_pos

p_roc_compare <- ggplot(roc_compare_df, aes(x = fpr, y = tpr, color = model)) +
  geom_line(aes(linewidth = ifelse(grepl("Neural Network", model), "bold", "normal"))) +
  geom_abline(linetype = "dashed", color = "grey60") +
  scale_linewidth_manual(values = c("bold" = 1.4, "normal" = 0.8), guide = "none") +
  scale_color_manual(values = model_colors) +
  labs(
    title    = "ROC Comparison — Neural Network vs Previous Models",
    subtitle = "Each curve shows one model's tradeoff between catching cases and false alarms.\nAUC values in legend — higher is better.",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  theme_ml +
  theme(legend.text = element_text(size = 9))

ggsave(file.path(out_dir, "06_roc_comparison.png"), p_roc_compare,
       width = 12, height = 8, dpi = 150, bg = "white")


# ---- 6c: Confusion Matrix Heatmap ----
cm_table <- as.data.frame(cm$table)
cm_table$Label <- c(
  "True Negative\n(correctly ruled out)",
  "False Positive\n(false alarm)",
  "False Negative\n(MISSED case)",
  "True Positive\n(correctly caught)"
)

# Reorder so that positive class is bottom-right
cm_table$Prediction <- factor(cm_table$Prediction, levels = c("No", "Yes"))
cm_table$Reference  <- factor(cm_table$Reference,  levels = c("No", "Yes"))

cm_fill <- c(
  "True Negative\n(correctly ruled out)"  = col_good,
  "False Positive\n(false alarm)"         = "#FF9800",
  "False Negative\n(MISSED case)"         = col_pos,
  "True Positive\n(correctly caught)"     = col_neg
)

p_cm <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Label)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = paste0(Label, "\n\nn = ", Freq)),
            size = 4, color = "white", fontface = "bold") +
  scale_fill_manual(values = cm_fill, guide = "none") +
  labs(
    title    = "Confusion Matrix — Neural Network",
    subtitle = paste0("Threshold: ", round(optimal_threshold, 3),
                      " | Green = correct, Orange/Red = errors"),
    x = "Actual",
    y = "Predicted"
  ) +
  theme_ml +
  theme(panel.grid = element_blank())

ggsave(file.path(out_dir, "06_confusion_matrix.png"), p_cm,
       width = 8, height = 7, dpi = 150, bg = "white")


# ---- 6d: Predicted Probability Distribution ----
# Shows how the model separates disease vs no-disease cases.
# Ideally, the two distributions are far apart (good separation).
# Overlapping distributions = the model can't distinguish well
# in that probability range.
prob_df <- tibble(
  probability    = final_probs,
  heart_disease  = factor(test_y, levels = c(0, 1), labels = c("No", "Yes"))
)

p_prob <- ggplot(prob_df, aes(x = probability, fill = heart_disease)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity", color = "white") +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", linewidth = 0.8) +
  annotate("text", x = optimal_threshold + 0.02, y = Inf, vjust = 2,
           label = paste0("Threshold: ", round(optimal_threshold, 3)),
           size = 3.5, fontface = "italic") +
  scale_fill_manual(values = c("No" = col_neg, "Yes" = col_pos)) +
  labs(
    title    = "Predicted Probability Distribution — Neural Network",
    subtitle = "Left of threshold → predicted 'No'. Right → predicted 'Yes'.\nGood models push these histograms apart.",
    x = "Predicted Probability of Heart Disease",
    y = "Count",
    fill = "Actual"
  ) +
  theme_ml

ggsave(file.path(out_dir, "06_probability_distribution.png"), p_prob,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 6e: Hyperparameter Tuning Heatmap ----
# Visualize how different combinations of size and decay affected AUC.
# This teaches the fundamental tradeoff: more capacity needs more
# regularization, and there's usually a sweet spot.
tune_results <- nn_tuned$results %>%
  dplyr::select(size, decay, ROC) %>%
  dplyr::mutate(
    size  = factor(size),
    decay = factor(decay)
  )

p_tune <- ggplot(tune_results, aes(x = decay, y = size, fill = ROC)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(ROC, 3)), size = 3.5, fontface = "bold") +
  scale_fill_gradient(low = "#FFCDD2", high = "#1B5E20") +
  labs(
    title    = "Hyperparameter Tuning — CV AUC by Size and Decay",
    subtitle = "Green = higher AUC. The best combo balances model capacity (size)\nwith regularization strength (decay).",
    x = "Weight Decay (Regularization)",
    y = "Hidden Layer Size (Neurons)",
    fill = "CV AUC"
  ) +
  theme_ml +
  theme(panel.grid = element_blank())

ggsave(file.path(out_dir, "06_tuning_heatmap.png"), p_tune,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 6f: Network Architecture Diagram ----
# NeuralNetTools::plotnet() draws the network structure showing
# connections between input, hidden, and output neurons.
# Black lines = positive weights, grey lines = negative weights.
# Thicker lines = stronger connections.
png(file.path(out_dir, "06_network_diagram.png"),
    width = 12, height = 8, units = "in", res = 150)
NeuralNetTools::plotnet(nn_tuned$finalModel,
                        y_names  = "Heart Disease",
                        pos_col  = "#1565C0",
                        neg_col  = "#E53935",
                        alpha_val = 0.5,
                        circle_cex = 3,
                        cex_val  = 0.7)
title(main = "Neural Network Architecture",
      sub  = paste0("Input (", ncol(train_x_scaled),
                    ") → Hidden (", nn_tuned$bestTune$size,
                    ") → Output (1)"),
      cex.main = 1.3, font.main = 2)
dev.off()


# ---- 6g: Overfitting Check — Train vs Test AUC ----
# If the model performs much better on training data than test data,
# it's memorizing training patterns instead of learning general ones.
# We compare AUC on training data vs test data.
train_probs_tuned <- predict(nn_tuned, newdata = train_nn, type = "prob")[, "Yes"]
roc_train         <- pROC::roc(train_y, train_probs_tuned, quiet = TRUE)
train_auc         <- round(auc(roc_train), 4)
test_auc          <- round(auc(final_roc), 4)
auc_gap           <- round(train_auc - test_auc, 4)

cat("\nOverfitting check:\n")
cat("  Train AUC:", train_auc, "\n")
cat("  Test AUC: ", test_auc, "\n")
cat("  Gap:      ", auc_gap)
if (auc_gap > 0.05) {
  cat(" — Some overfitting (train is notably better than test)\n")
} else {
  cat(" — Minimal overfitting (good generalization)\n")
}

overfit_df <- tibble(
  dataset = c("Train", "Test"),
  auc     = c(train_auc, test_auc)
)

p_overfit <- ggplot(overfit_df, aes(x = dataset, y = auc, fill = dataset)) +
  geom_col(width = 0.5, color = "white") +
  geom_text(aes(label = round(auc, 3)), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Train" = "#1565C0", "Test" = "#E53935")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Overfitting Check — Train vs Test AUC",
    subtitle = paste0("Gap = ", auc_gap,
                      if (auc_gap > 0.05) " — model may be memorizing training data"
                      else " — model generalizes well to unseen patients"),
    x = NULL, y = "AUC"
  ) +
  theme_ml +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "06_overfitting_check.png"), p_overfit,
       width = 7, height = 6, dpi = 150, bg = "white")


# ---- 6h: Variable Importance (Garson's Algorithm) ----
# Neural nets don't have built-in "feature importance" like trees.
# Garson's algorithm decomposes the connection weights to estimate
# each input's relative contribution to the output.
# This is an approximation — not as clean as tree-based importance.
garson_imp <- NeuralNetTools::garson(nn_tuned$finalModel, bar_plot = FALSE)

# garson() returns a data.frame with columns: rel_imp, x_names
# Map dummy-coded variable names back to original feature names
imp_df <- tibble(
  feature    = garson_imp$x_names,
  importance = garson_imp$rel_imp
)

# Group dummy variables back to their parent feature
# e.g., "sexMale" → "sex", "raceBlack" → "race"
imp_df <- imp_df %>%
  dplyr::mutate(
    parent_feature = dplyr::case_when(
      stringr::str_detect(feature, "^sex")            ~ "sex",
      stringr::str_detect(feature, "^race")           ~ "race",
      stringr::str_detect(feature, "^education")      ~ "education",
      stringr::str_detect(feature, "^smoking_status")  ~ "smoking_status",
      stringr::str_detect(feature, "^alcohol_use")     ~ "alcohol_use",
      TRUE ~ feature
    )
  )

# Aggregate importance by parent feature (sum of dummy importances)
parent_imp <- imp_df %>%
  dplyr::group_by(parent_feature) %>%
  dplyr::summarise(importance = sum(importance), .groups = "drop") %>%
  dplyr::arrange(dplyr::desc(importance))

parent_imp <- parent_imp %>%
  dplyr::mutate(parent_feature = forcats::fct_reorder(parent_feature, importance))

p_importance <- ggplot(parent_imp, aes(x = importance, y = parent_feature)) +
  geom_col(fill = "#1565C0", width = 0.7) +
  geom_text(aes(label = round(importance, 2)), hjust = -0.2, size = 3.5) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Variable Importance — Neural Network (Garson's Algorithm)",
    subtitle = "Relative importance of each feature based on connection weight decomposition.\nHigher = the network relies more on this feature for predictions.",
    x = "Relative Importance",
    y = NULL
  ) +
  theme_ml

ggsave(file.path(out_dir, "06_variable_importance.png"), p_importance,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 6i: Decision Boundary Visualization (2D Slice) ----
# Neural nets can draw complex, curved decision boundaries.
# We visualize a 2D slice: age vs sbp, holding other features
# at their median/mode values. The colored surface shows the
# network's predicted probability at each point.
# Compare this to logistic regression (straight line) or
# decision trees (rectangular blocks).

# Build a grid of age × sbp values
age_seq <- seq(min(train_df$age), max(train_df$age), length.out = 100)
sbp_seq <- seq(min(train_df$sbp), max(train_df$sbp), length.out = 100)
grid_2d <- expand.grid(age = age_seq, sbp = sbp_seq)

# Set other features to median/mode from training data
grid_2d$sex              <- factor("Male", levels = levels(train_df$sex))
grid_2d$race             <- factor("White", levels = levels(train_df$race))
grid_2d$education        <- factor("Some College", levels = levels(train_df$education))
grid_2d$income           <- median(train_df$income)
grid_2d$smoking_status   <- factor("Never", levels = levels(train_df$smoking_status))
grid_2d$alcohol_use      <- factor("None", levels = levels(train_df$alcohol_use))
grid_2d$physical_activity <- median(train_df$physical_activity)
grid_2d$bmi              <- median(train_df$bmi)
grid_2d$dbp              <- median(train_df$dbp)
grid_2d$cholesterol      <- median(train_df$cholesterol)
grid_2d$fasting_glucose  <- median(train_df$fasting_glucose)

# Dummy-code and scale using the same objects from training
grid_x        <- predict(dummy_obj, newdata = grid_2d)
grid_x_scaled <- scale(grid_x, center = train_means, scale = train_sds)
grid_nn       <- as.data.frame(grid_x_scaled)

# Predict probabilities across the grid
grid_2d$prob <- predict(nn_tuned, newdata = grid_nn, type = "prob")[, "Yes"]

p_boundary <- ggplot(grid_2d, aes(x = age, y = sbp, fill = prob)) +
  geom_tile() +
  scale_fill_gradient2(
    low = col_neg, mid = "white", high = col_pos,
    midpoint = optimal_threshold,
    limits = c(0, 1),
    name = "P(Heart Disease)"
  ) +
  geom_contour(aes(z = prob), breaks = optimal_threshold,
               color = "black", linewidth = 0.8, linetype = "dashed") +
  labs(
    title    = "Decision Boundary — Neural Network (Age × SBP Slice)",
    subtitle = "Colors show predicted probability. Dashed line = threshold.\nUnlike trees (rectangular) or logistic regression (straight), neural nets learn curved boundaries.",
    x = "Age", y = "Systolic Blood Pressure (SBP)"
  ) +
  theme_ml

ggsave(file.path(out_dir, "06_decision_boundary.png"), p_boundary,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: CROSS-VALIDATION RESULTS
# ============================================================ #
# We already ran CV during tuning. Extract fold-level AUC
# for the best hyperparameter combination to report variability.

# caret stores resample-level results
cv_resamples <- nn_tuned$resample
cv_auc_mean  <- round(mean(cv_resamples$ROC), 4)
cv_auc_sd    <- round(sd(cv_resamples$ROC), 4)

cat("\n5-Fold CV Results (best hyperparameters):\n")
cat("  Mean AUC:", cv_auc_mean, "± SD:", cv_auc_sd, "\n")
cat("  Fold AUCs:", paste(round(cv_resamples$ROC, 4), collapse = ", "), "\n")

# CV fold plot
cv_fold_df <- tibble(
  fold = paste0("Fold ", 1:nrow(cv_resamples)),
  auc  = cv_resamples$ROC
)

p_cv <- ggplot(cv_fold_df, aes(x = fold, y = auc)) +
  geom_col(fill = "#1565C0", width = 0.6) +
  geom_hline(yintercept = cv_auc_mean, linetype = "dashed", color = col_pos) +
  geom_text(aes(label = round(auc, 3)), vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = 0.5, y = cv_auc_mean + 0.005,
           label = paste0("Mean: ", cv_auc_mean),
           hjust = 0, size = 3.5, color = col_pos, fontface = "italic") +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "5-Fold Cross-Validation — Neural Network",
    subtitle = "Each bar = AUC from one fold. Consistent bars = stable model.\nHigh variance across folds would suggest the model is sensitive to which patients are in the training set.",
    x = NULL, y = "AUC"
  ) +
  theme_ml

ggsave(file.path(out_dir, "06_cv_folds.png"), p_cv,
       width = 9, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: SAVE RESULTS
# ============================================================ #

# Standardized results for model comparison script (Script 15)
results_nn <- tibble(
  model          = "Neural Network",
  target         = "heart_disease",
  auc            = round(as.numeric(auc(final_roc)), 4),
  accuracy       = round(accuracy, 4),
  precision      = round(precision, 4),
  recall         = round(recall, 4),
  f1             = round(f1, 4),
  cv_auc_mean    = cv_auc_mean,
  cv_auc_sd      = cv_auc_sd,
  train_auc      = train_auc,
  threshold_used = round(optimal_threshold, 4)
)

saveRDS(results_nn, file.path(out_dir, "results_nn.rds"))
saveRDS(final_probs, file.path(out_dir, "test_predictions_nn.rds"))

cat("\nResults saved to:", out_dir, "\n")
print(results_nn)


# ============================================================ #
# SECTION 9: SUMMARY & WHAT WE LEARNED
# ============================================================ #
#
# KEY METRICS:
cat("\n")
cat("========================================\n")
cat("NEURAL NETWORK — FINAL SUMMARY\n")
cat("========================================\n")
cat("Architecture: Input (", ncol(train_x_scaled), ") → Hidden (",
    nn_tuned$bestTune$size, ") → Output (1)\n")
cat("Regularization: decay =", nn_tuned$bestTune$decay, "\n")
cat("Test AUC:    ", round(as.numeric(auc(final_roc)), 4), "\n")
cat("CV AUC:      ", cv_auc_mean, "±", cv_auc_sd, "\n")
cat("Accuracy:    ", round(accuracy, 4), "\n")
cat("Precision:   ", round(precision, 4), "\n")
cat("Recall:      ", round(recall, 4), "\n")
cat("F1:          ", round(f1, 4), "\n")
cat("Threshold:   ", round(optimal_threshold, 4), "\n")
#
# WHAT WE LEARNED:
#
# 1. A neural network is a function that transforms inputs through
#    layers of weighted connections. Each neuron combines inputs,
#    applies an activation function, and passes the result forward.
#
# 2. BACKPROPAGATION is how the network learns: compare prediction
#    to truth, trace the error backward through every weight, and
#    nudge each weight to reduce the error. Repeat thousands of times.
#
# 3. SCALING is critical for neural networks. Without it, features
#    with large values (income) dominate features with small values
#    (physical_activity), making training unstable and slow.
#
# 4. The SIZE-DECAY tradeoff: more neurons = more capacity to learn
#    complex patterns, but also more risk of memorizing noise.
#    Weight decay (regularization) penalizes large weights, forcing
#    the network to find simpler solutions. The tuning heatmap shows
#    the sweet spot between capacity and regularization.
#
# 5. Neural networks learn CURVED decision boundaries, unlike
#    logistic regression (straight) or decision trees (rectangular).
#    This flexibility is powerful but makes them harder to interpret.
#
# 6. For tabular clinical data with ~25k rows and 13 features,
#    neural networks often perform similarly to tree-based methods
#    (Random Forest, XGBoost). Neural nets shine more with very
#    large datasets, many features, or complex feature interactions.
#
# 7. Interpretability is the main tradeoff: tree-based models tell
#    you "age > 65 AND smoking = Current → high risk." Neural nets
#    give you a probability but don't easily explain WHY. Tools like
#    SHAP (Script 11) can help, but it's an approximation.
#
# WHEN TO USE NEURAL NETWORKS:
#   - Large datasets with complex nonlinear relationships
#   - When you have resources to tune architecture and regularization
#   - When prediction accuracy matters more than interpretability
#   - As one model in an ensemble (Script 12)
#
# WHEN TO PREFER TREE-BASED METHODS:
#   - Small to medium tabular datasets (like this one)
#   - When you need to explain predictions to clinicians
#   - When training speed matters (trees are much faster)
#   - When you want built-in feature importance
#
# HOW THIS COMPARES TO OTHER MODELS:
#   - Logistic regression: Simple, fully interpretable, draws a
#     straight decision boundary. Good baseline. Neural nets can
#     capture nonlinear patterns logistic regression misses.
#   - Decision trees: Highly interpretable (if-then rules), but
#     a single tree usually underfits. Neural nets learn more
#     complex patterns but are opaque.
#   - Random Forest: Averages many trees — strong performance,
#     handles tabular data well, gives clean feature importance.
#     Often matches or beats simple neural nets on tabular data.
#   - XGBoost: Builds trees sequentially, correcting mistakes.
#     Current champion for tabular data. Often outperforms neural
#     nets here unless the dataset is very large.
#   - KNN: Simple distance-based method. Neural nets learn
#     representations internally rather than relying on raw distance.
#   - SVM: Finds optimal margins. Can use kernels for nonlinearity.
#     Neural nets learn their own feature transformations.
# ============================================================ #

cat("\nScript 06 complete.\n")