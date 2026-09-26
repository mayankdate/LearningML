# ============================================================ #
# 05 — Logistic Regression (Baseline Model)
# ============================================================ #
# Purpose: Use logistic regression — a model you already know —
#   to establish a BASELINE for predicting heart disease.
#   Every future model gets compared against this.
#
# Why start here:
#   - You understand the mechanics (log-odds, coefficients)
#   - It forces us to set up evaluation properly (ROC, confusion
#     matrix, train/test) BEFORE adding complexity
#   - If a fancy model can't beat logistic regression, it's not
#     worth the complexity
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
  caret,     # Train/test split, CV
  pROC,      # ROC curves and AUC
  broom      # Tidy model output
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Logistic_Regression", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Logistic_Regression"

# ---- Global Plot Theme ----
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    legend.position = "bottom"
  )


# ============================================================ #
# SECTION 1: WHAT IS LOGISTIC REGRESSION?
# ============================================================ #
# You already know this, but let's frame it in ML terms:
#
# INTUITION:
#   Logistic regression models the PROBABILITY of an event.
#   It takes a linear combination of features (like linear
#   regression) but passes it through a sigmoid function to
#   squeeze the output between 0 and 1.
#
#   P(heart_disease = 1) = sigmoid(b0 + b1*age + b2*bmi + ...)
#
#   sigmoid(x) = 1 / (1 + exp(-x))
#
# IN ML TERMS:
#   - It's a LINEAR classifier: the decision boundary is a
#     straight line (or hyperplane) in feature space
#   - It assumes features contribute ADDITIVELY to log-odds
#   - It CANNOT learn interactions or nonlinear patterns
#     unless you manually create interaction terms
#   - This is both its strength (interpretable) and weakness
#     (limited expressiveness)
#
# WHEN IT WORKS WELL:
#   - Relationships are roughly linear in log-odds
#   - Features are somewhat independent
#   - You need interpretable coefficients
#   - You have a moderate number of features
#
# WHEN IT STRUGGLES:
#   - Complex interactions between variables
#   - Nonlinear relationships (e.g., U-shaped BMI effect)
#   - Very high-dimensional data
#   - Highly correlated features (multicollinearity)
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #


# ---- 2A: Load and Inspect ----

ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Full dataset:", nrow(ml_data), "rows x", ncol(ml_data), "columns\n")
cat("Heart disease prevalence:", round(mean(ml_data$heart_disease) * 100, 1), "%\n")


# ---- 2B: Define Features and Target ----
# For heart disease prediction, we use demographics + lifestyle + clinical measures.
# We EXCLUDE other disease outcomes (diabetes, cancer) — in a real scenario
# you might not have those diagnoses at the time of prediction.

target <- "heart_disease"

features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)


# ---- 2C: Train/Test Split ----
# WHY: We hold out 25% of data that the model NEVER sees during training.
# This simulates deploying the model on new patients.
# STRATIFIED: ensures both sets have ~same heart disease rate.

set.seed(42)

train_index <- caret::createDataPartition(
  ml_data[[target]],
  p     = 0.75,       # 75% train, 25% test
  list  = FALSE
)

train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]

cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")
cat("Train HD rate:", round(mean(train_df[[target]]) * 100, 1), "%",
    "| Test HD rate:", round(mean(test_df[[target]]) * 100, 1), "%\n")


# ---- 2D: Preprocessing for Logistic Regression ----
# Logistic regression in R (glm) handles factors natively — it creates
# dummy variables internally. So we DON'T need manual one-hot encoding.
#
# We DO want to check:
#   1. Factor levels are sensible (reference category matters)
#   2. Numeric variables are on reasonable scales
#
# NOTE: Unlike tree models, logistic regression IS sensitive to scale
# for regularized versions (glmnet). For base glm, scale doesn't
# affect coefficients — but we'll scale anyway for clean interpretation.

# Set meaningful reference levels for factors
train_df <- train_df %>%
  dplyr::mutate(
    sex            = relevel(factor(sex), ref = "Female"),
    race           = relevel(factor(race), ref = "White"),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")
    ),
    smoking_status = factor(smoking_status,
                            levels = c("Never", "Former", "Current")
    ),
    alcohol_use    = factor(alcohol_use,
                            levels = c("None", "Light", "Moderate", "Heavy")
    )
  )

# Apply same factor structure to test set
test_df <- test_df %>%
  dplyr::mutate(
    sex            = relevel(factor(sex), ref = "Female"),
    race           = relevel(factor(race), ref = "White"),
    education      = factor(education,
                            levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")
    ),
    smoking_status = factor(smoking_status,
                            levels = c("Never", "Former", "Current")
    ),
    alcohol_use    = factor(alcohol_use,
                            levels = c("None", "Light", "Moderate", "Heavy")
    )
  )

cat("Factor levels set. Reference categories: Female, White, Less than HS, Never, None\n")


# ============================================================ #
# SECTION 3: FIT THE MODEL
# ============================================================ #


# ---- 3A: Build the Formula ----
# Logistic regression uses a formula interface: target ~ feature1 + feature2 + ...

model_formula <- as.formula(
  paste(target, "~", paste(features, collapse = " + "))
)

cat("\nFormula:", deparse(model_formula), "\n")


# ---- 3B: Fit Logistic Regression ----
# glm() with family = binomial fits logistic regression.
# The "binomial" family tells R the outcome is binary (0/1)
# and to use the logit link function: log(p / (1-p)) = Xb

fit_logistic <- glm(
  model_formula,
  data   = train_df,
  family = binomial(link = "logit")
)

cat("\nModel fitted. Deviance:", round(fit_logistic$deviance, 1), "\n")
cat("AIC:", round(fit_logistic$aic, 1), "\n")


# ---- 3C: Examine Coefficients ----
# Each coefficient represents the change in LOG-ODDS of heart disease
# for a one-unit increase in that feature, holding all others constant.
#
# LOG-ODDS → ODDS RATIO:
#   exp(coefficient) = Odds Ratio (OR) — much easier to interpret.
#
# HOW TO READ AN ODDS RATIO:
#   OR = 1.0 → no effect (baseline)
#   OR = 2.0 → twice the odds of heart disease compared to the reference
#   OR = 0.5 → half the odds (protective)
#
# CONCRETE EXAMPLE — what does OR = 1.8 for "smoking = Current" mean?
#   A current smoker has 1.8x the ODDS of heart disease compared to
#   a never-smoker, holding all other features constant.
#   In plain language: "Current smokers have 80% higher odds of heart
#   disease than never-smokers with the same age, BMI, blood pressure, etc."
#
# IMPORTANT: Odds ratio ≠ risk ratio.
#   With ~6% prevalence, OR ≈ risk ratio (they're close when the outcome is rare).
#   At higher prevalence, an OR of 2.0 doesn't mean "twice the probability."
#
# p-value < 0.05 = "statistically significant" — we're confident this
# association isn't just random noise in the data.

# Tidy the coefficients into a clean data frame
coef_table <- broom::tidy(fit_logistic, conf.int = TRUE, exponentiate = FALSE) %>%
  dplyr::mutate(
    odds_ratio = exp(estimate),
    or_lower   = exp(conf.low),
    or_upper   = exp(conf.high),
    significant = ifelse(p.value < 0.05, "Yes", "No")
  ) %>%
  dplyr::arrange(p.value)

cat("\n--- Top 10 Most Significant Predictors ---\n")
coef_table %>%
  dplyr::slice_head(n = 10) %>%
  dplyr::select(term, estimate, odds_ratio, p.value, significant) %>%
  dplyr::mutate(
    estimate   = round(estimate, 4),
    odds_ratio = round(odds_ratio, 3),
    p.value    = formatC(p.value, format = "e", digits = 2)
  ) %>%
  print(n = 10)


# ---- 3D: Visualize Coefficients ----
# Forest plot of odds ratios — the most intuitive way to read logistic regression

p_coefs <- coef_table %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::mutate(term = forcats::fct_reorder(term, odds_ratio)) %>%
  ggplot(aes(x = odds_ratio, y = term, color = significant)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = or_lower, xmax = or_upper), height = 0.2) +
  scale_color_manual(values = c("Yes" = "#F44336", "No" = "grey60")) +
  labs(
    title    = "Odds Ratios — Logistic Regression for Heart Disease",
    subtitle = "OR > 1 = increased risk | OR < 1 = decreased risk | Dashed line = no effect",
    x = "Odds Ratio (95% CI)", y = NULL, color = "p < 0.05"
  ) +
  theme_ml

print(p_coefs)
ggsave(file.path(out_dir, "01_odds_ratios.png"), p_coefs,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: PREDICTIONS
# ============================================================ #
# The model outputs PROBABILITIES, not classes.
# We must choose a THRESHOLD to convert probabilities → 0/1.
# Default is 0.5, but this isn't always optimal.
# ============================================================ #


# ---- 4A: Generate Predictions ----
# predict() for glm can return different things depending on 'type':
#
#   type = "link"     → returns LOG-ODDS (the raw linear predictor)
#                       e.g. -2.3, 0.8, 1.4 (unbounded, can be any number)
#
#   type = "response" → returns PROBABILITIES (0 to 1) — what we want
#                       The sigmoid function converts log-odds → probability:
#                       P = 1 / (1 + exp(-log_odds))
#                       e.g. log-odds of -2.3 → probability of 0.09 (9%)
#
# We always use type = "response" for evaluation because:
#   - Probabilities are interpretable ("12% chance of heart disease")
#   - ROC curves require probabilities
#   - We can compare probabilities across models
#
# We predict on BOTH train and test:
#   - test_probs → evaluate real-world performance
#   - train_probs → check for overfitting (Section 8)

train_probs <- predict(fit_logistic, newdata = train_df, type = "response")
test_probs  <- predict(fit_logistic, newdata = test_df, type = "response")

cat("\nPrediction distributions:\n")
cat("Train — Mean:", round(mean(train_probs), 3),
    "| Range:", round(min(train_probs), 3), "-", round(max(train_probs), 3), "\n")
cat("Test  — Mean:", round(mean(test_probs), 3),
    "| Range:", round(min(test_probs), 3), "-", round(max(test_probs), 3), "\n")
cat("(Mean should be close to heart disease prevalence of ~6%)\n")


# ---- 4B: Visualize Predicted Probabilities ----
# Good model: the two distributions (actual 0 vs actual 1) should be SEPARATED.
# Bad model: the distributions overlap completely.

p_prob_dist <- tibble(
  probability = test_probs,
  actual      = factor(test_df[[target]], labels = c("No HD", "Has HD"))
) %>%
  ggplot(aes(x = probability, fill = actual)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "black", linewidth = 0.8) +
  scale_fill_manual(values = c("No HD" = "#2196F3", "Has HD" = "#F44336")) +
  labs(
    title    = "Distribution of Predicted Probabilities (Test Set)",
    subtitle = "Good separation = model can distinguish cases from non-cases | Dashed line = 0.5 threshold",
    x = "Predicted P(Heart Disease)", y = "Count", fill = "Actual Status"
  ) +
  theme_ml

print(p_prob_dist)
ggsave(file.path(out_dir, "02_predicted_probabilities.png"), p_prob_dist,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: EVALUATION METRICS
# ============================================================ #
# Now we measure HOW GOOD the model is.
# Each metric captures a different aspect of performance.
# ============================================================ #


# ---- 5A: ROC Curve and AUC ----
# The ROC curve shows the tradeoff between:
#   - Sensitivity (True Positive Rate): of all actual cases, how many did we catch?
#   - Specificity (True Negative Rate): of all actual non-cases, how many did we correctly exclude?
#
# AUC (Area Under the Curve):
#   - 0.5 = random guessing (diagonal line)
#   - 0.7-0.8 = acceptable discrimination
#   - 0.8-0.9 = good discrimination
#   - 0.9+ = excellent (possibly too good — check for leakage)

roc_obj <- pROC::roc(test_df[[target]], test_probs, quiet = TRUE)
auc_val <- round(as.numeric(pROC::auc(roc_obj)), 3)

roc_df <- tibble(
  specificity = roc_obj$specificities,
  sensitivity = roc_obj$sensitivities,
  fpr         = 1 - specificity    # False Positive Rate
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = sensitivity)) +
  geom_line(color = "#F44336", linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.55, y = 0.35,
           label = paste("AUC =", auc_val),
           size = 6, fontface = "bold", color = "#F44336") +
  labs(
    title    = "ROC Curve — Logistic Regression",
    subtitle = "Shows tradeoff: catching more cases (sensitivity) vs more false alarms (FPR)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_ml

print(p_roc)
ggsave(file.path(out_dir, "03_roc_curve.png"), p_roc,
       width = 8, height = 7, dpi = 150, bg = "white")

cat("\nAUC:", auc_val, "\n")


# ---- 5B: Choose a Classification Threshold ----
# The model outputs probabilities (e.g. 0.04, 0.18, 0.63).
# To make a yes/no clinical decision, we need a THRESHOLD:
#   anyone with P(HD) >= threshold → flag as positive.
#
# WHY NOT JUST USE 0.5?
#   With only ~6% heart disease prevalence, most patients have
#   probabilities well below 0.5. Even true cases might only
#   score 0.2 or 0.3. Using 0.5 would miss nearly everyone.
#
# YOUDEN'S J STATISTIC:
#   Picks the threshold that maximizes (Sensitivity + Specificity - 1).
#   Geometrically, this is the point on the ROC curve closest to
#   the top-left corner — the best balance between catching cases
#   and avoiding false alarms.
#   pROC::coords() finds this automatically from the ROC object.

optimal_coords    <- pROC::coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold

cat("Optimal threshold (Youden's J):", round(optimal_threshold, 3), "\n")
cat("At this threshold — Sensitivity:", round(optimal_coords$sensitivity, 3),
    "| Specificity:", round(optimal_coords$specificity, 3), "\n")
cat("Sensitivity = % of HD patients correctly caught\n")
cat("Specificity = % of healthy patients correctly cleared\n")


# ---- 5C: Confusion Matrix (at optimal threshold) ----
# The confusion matrix shows the 4 outcomes of classification:
#
#                    Predicted Negative    Predicted Positive
# Actual Negative    True Negative (TN)    False Positive (FP) — "false alarm"
# Actual Positive    False Negative (FN)   True Positive (TP)  — "correctly caught"
#
# From these we derive:
#   Precision = TP / (TP + FP)  — "Of those we flagged, how many actually have HD?"
#   Recall    = TP / (TP + FN)  — "Of those who have HD, how many did we catch?"
#   F1        = harmonic mean of Precision and Recall

test_pred_class <- ifelse(test_probs >= optimal_threshold, 1, 0)

# Build confusion matrix
cm <- table(
  Actual    = test_df[[target]],
  Predicted = test_pred_class
)

cat("\nConfusion Matrix (threshold =", round(optimal_threshold, 3), "):\n")
print(cm)

# Calculate metrics
tp <- cm["1", "1"]
fp <- cm["0", "1"]
tn <- cm["0", "0"]
fn <- cm["1", "0"]

accuracy  <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)
recall    <- tp / (tp + fn)
f1        <- 2 * precision * recall / (precision + recall)

cat("\n--- Classification Metrics ---\n")
cat("Accuracy: ", round(accuracy, 3), " — Overall correct rate\n")
cat("Precision:", round(precision, 3), " — Of predicted positives, % actually positive\n")
cat("Recall:   ", round(recall, 3), " — Of actual positives, % we caught\n")
cat("F1 Score: ", round(f1, 3), " — Balance of precision and recall\n")
cat("AUC:      ", auc_val, " — Overall discrimination ability\n")


# Visualize confusion matrix
cm_df <- tibble(
  actual    = factor(c(0, 0, 1, 1), labels = c("No HD", "Has HD")),
  predicted = factor(c(0, 1, 0, 1), labels = c("No HD", "Has HD")),
  count     = c(tn, fp, fn, tp),
  correct   = c("Correct", "Incorrect", "Incorrect", "Correct")
)

p_confusion <- ggplot(cm_df, aes(x = predicted, y = actual, fill = correct)) +
  geom_tile(alpha = 0.8, color = "white", linewidth = 2) +
  geom_text(aes(label = scales::comma(count)), size = 8, fontface = "bold") +
  scale_fill_manual(values = c("Correct" = "#4CAF50", "Incorrect" = "#F44336")) +
  labs(
    title    = "Confusion Matrix — Logistic Regression",
    subtitle = paste0(
      "Threshold = ", round(optimal_threshold, 3),
      " | Accuracy = ", round(accuracy, 3),
      " | F1 = ", round(f1, 3)
    ),
    x = "Predicted", y = "Actual"
  ) +
  theme_ml +
  theme(legend.position = "none")

print(p_confusion)
ggsave(file.path(out_dir, "04_confusion_matrix.png"), p_confusion,
       width = 7, height = 6, dpi = 150, bg = "white")


# ---- 5D: Precision-Recall Tradeoff Across Thresholds ----
# The confusion matrix above used ONE threshold (Youden's J).
# But what if you had a different priority — like "catch 90% of cases
# even if it means more false alarms"?
#
# This loop calculates precision and recall at EVERY possible threshold
# from 0.05 to 0.95, so you can see the full tradeoff curve.
#
# HOW THE LOOP WORKS:
#   map_dfr() runs the code for each threshold value.
#   For each threshold, it:
#     1. Converts probabilities → 0/1 predictions using that threshold
#     2. Counts TP, FP, FN from those predictions
#     3. Calculates precision and recall at that threshold
#     4. Returns a one-row tibble
#   The result is a table with one row per threshold — 91 rows total.
#
# WHAT TO LOOK FOR IN THE PLOT:
#   - As threshold DECREASES: recall goes up (catch more cases)
#     but precision drops (more false alarms)
#   - As threshold INCREASES: precision goes up (fewer false alarms)
#     but recall drops (miss more cases)
#   - The optimal threshold (vertical dashed line) is where Youden's J
#     balances both. Whether that's actually optimal depends on clinical
#     priorities: missing a case (false negative) costs more than a
#     false alarm in most medical settings.

thresholds <- seq(0.05, 0.95, by = 0.01)

threshold_metrics <- purrr::map_dfr(thresholds, function(thresh) {
  pred <- ifelse(test_probs >= thresh, 1, 0)
  tp_t <- sum(pred == 1 & test_df[[target]] == 1)
  fp_t <- sum(pred == 1 & test_df[[target]] == 0)
  fn_t <- sum(pred == 0 & test_df[[target]] == 1)
  
  # ifelse guards against dividing by zero at extreme thresholds
  prec <- ifelse(tp_t + fp_t > 0, tp_t / (tp_t + fp_t), NA)
  rec  <- ifelse(tp_t + fn_t > 0, tp_t / (tp_t + fn_t), NA)
  
  tibble(threshold = thresh, precision = prec, recall = rec)
})

p_pr_tradeoff <- threshold_metrics %>%
  tidyr::pivot_longer(c(precision, recall), names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = threshold, y = value, color = metric)) +
  geom_line(linewidth = 1.2) +
  geom_vline(xintercept = optimal_threshold, linetype = "dashed", color = "grey30") +
  annotate("text", x = optimal_threshold + 0.02, y = 0.2,
           label = paste("Optimal:", round(optimal_threshold, 2)),
           hjust = 0, size = 3.5) +
  scale_color_manual(values = c("precision" = "#FF9800", "recall" = "#2196F3")) +
  labs(
    title    = "Precision vs Recall Across Thresholds",
    subtitle = "Lower threshold → more recall (catch more cases) but less precision (more false alarms)",
    x = "Classification Threshold", y = "Metric Value", color = "Metric"
  ) +
  theme_ml

print(p_pr_tradeoff)
ggsave(file.path(out_dir, "05_precision_recall_tradeoff.png"), p_pr_tradeoff,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: MODEL INTERPRETATION
# ============================================================ #
# Logistic regression's biggest advantage: every coefficient
# has a clear, direct interpretation.
# ============================================================ #


# ---- 6A: Feature Importance via Absolute Z-values ----
# The z-value (Wald statistic) measures how many standard errors
# the coefficient is from zero. Larger |z| = more statistically
# important. This is a simple proxy for "feature importance."

p_importance <- coef_table %>%
  dplyr::filter(term != "(Intercept)") %>%
  dplyr::mutate(
    abs_z = abs(statistic),
    term  = forcats::fct_reorder(term, abs_z)
  ) %>%
  dplyr::slice_max(abs_z, n = 15) %>%
  ggplot(aes(x = abs_z, y = term, fill = significant)) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(values = c("Yes" = "#5C6BC0", "No" = "grey70")) +
  labs(
    title    = "Feature Importance — Logistic Regression",
    subtitle = "Based on |z-value| (Wald statistic). Larger = stronger evidence of association.",
    x = "Absolute z-value", y = NULL, fill = "p < 0.05"
  ) +
  theme_ml

print(p_importance)
ggsave(file.path(out_dir, "06_feature_importance.png"), p_importance,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 6B: Predicted Probability by Key Risk Factors ----
# WHAT IS A MARGINAL EFFECT PLOT?
#   A coefficient tells you the direction of an effect (positive/negative)
#   but not how the probability actually changes. An OR of 1.05 per year
#   of age sounds small — but what does that mean going from age 30 to 80?
#
#   Marginal effect plots answer: "As age increases from 20 to 90, what
#   happens to the predicted probability of heart disease?"
#
# HOW WE BUILD THE GRID:
#   We create a synthetic "template patient" with all features held at
#   their median or most common value. Then we vary just ONE feature
#   across its full range while leaving everything else constant.
#   This isolates the effect of that single feature on risk.
#
#   This is the logistic regression equivalent of a partial dependence
#   plot — you'll see the same technique for Random Forest in Script 03.
#
# WHY HOLD OTHER FEATURES AT MEDIAN?
#   The marginal effect of age DEPENDS on the other feature values.
#   By holding them at median, we're asking:
#   "For a typical patient (median BMI, median BP, never smoked, etc.),
#   how does risk change with age?"
#   This gives the most representative picture of the age effect.

# Age effect: vary age from 20 to 90, hold everything else at median/reference
age_grid <- tibble(
  age              = seq(20, 90, by = 1),
  sex              = factor("Male", levels = levels(train_df$sex)),
  race             = factor("White", levels = levels(train_df$race)),
  education        = factor("Some College", levels = levels(train_df$education)),
  income           = median(train_df$income),
  smoking_status   = factor("Never", levels = levels(train_df$smoking_status)),
  alcohol_use      = factor("Light", levels = levels(train_df$alcohol_use)),
  physical_activity = median(train_df$physical_activity),
  bmi              = median(train_df$bmi),
  sbp              = median(train_df$sbp),
  dbp              = median(train_df$dbp),
  cholesterol      = median(train_df$cholesterol),
  fasting_glucose  = median(train_df$fasting_glucose)
)

age_grid$pred_prob <- predict(fit_logistic, newdata = age_grid, type = "response")

p_age_effect <- ggplot(age_grid, aes(x = age, y = pred_prob)) +
  geom_line(color = "#F44336", linewidth = 1.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(
    title    = "Predicted Heart Disease Probability by Age",
    subtitle = "All other features held at median/reference. Shows the S-curve (sigmoid) shape.",
    x = "Age", y = "P(Heart Disease)"
  ) +
  theme_ml

# Same for SBP — build fresh grid (different length than age_grid)
sbp_grid <- tibble(
  age              = 55,
  sex              = factor("Male", levels = levels(train_df$sex)),
  race             = factor("White", levels = levels(train_df$race)),
  education        = factor("Some College", levels = levels(train_df$education)),
  income           = median(train_df$income),
  smoking_status   = factor("Never", levels = levels(train_df$smoking_status)),
  alcohol_use      = factor("Light", levels = levels(train_df$alcohol_use)),
  physical_activity = median(train_df$physical_activity),
  bmi              = median(train_df$bmi),
  sbp              = seq(90, 200, by = 1),
  dbp              = median(train_df$dbp),
  cholesterol      = median(train_df$cholesterol),
  fasting_glucose  = median(train_df$fasting_glucose)
)
sbp_grid$pred_prob <- predict(fit_logistic, newdata = sbp_grid, type = "response")

p_sbp_effect <- ggplot(sbp_grid, aes(x = sbp, y = pred_prob)) +
  geom_line(color = "#2196F3", linewidth = 1.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(
    title    = "Predicted Heart Disease Probability by Systolic BP",
    subtitle = "Age fixed at 55, other features at median/reference.",
    x = "Systolic Blood Pressure (mmHg)", y = "P(Heart Disease)"
  ) +
  theme_ml

p_effects <- p_age_effect / p_sbp_effect +
  plot_annotation(
    title = "How Key Features Drive Predicted Risk",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

print(p_effects)
ggsave(file.path(out_dir, "07_feature_effects.png"), p_effects,
       width = 10, height = 10, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: CROSS-VALIDATION
# ============================================================ #
# A single train/test split can be lucky or unlucky.
# 5-fold cross-validation repeats the process 5 times:
#   - Split training data into 5 chunks
#   - Train on 4 chunks, evaluate on the held-out chunk
#   - Rotate which chunk is held out
#   - Average the 5 AUC values → more stable estimate
#
# If CV AUC ≈ test AUC, the model is stable.
# If CV AUC >> test AUC, the model may be overfitting.
# ============================================================ #

set.seed(42)

cv_folds <- caret::createFolds(train_df[[target]], k = 5, returnTrain = TRUE)

cv_results <- purrr::map_dfr(seq_along(cv_folds), function(i) {
  fold_train <- train_df[cv_folds[[i]], ]
  fold_val   <- train_df[-cv_folds[[i]], ]
  
  # Fit on fold training data
  fold_fit <- glm(model_formula, data = fold_train, family = binomial)
  
  # Predict on fold validation data
  fold_probs <- predict(fold_fit, newdata = fold_val, type = "response")
  
  # Calculate AUC
  fold_roc <- pROC::roc(fold_val[[target]], fold_probs, quiet = TRUE)
  fold_auc <- as.numeric(pROC::auc(fold_roc))
  
  tibble(fold = i, auc = fold_auc)
})

cat("\n--- 5-Fold Cross-Validation AUC ---\n")
print(cv_results)
cat("Mean CV AUC:", round(mean(cv_results$auc), 3),
    "± SD:", round(sd(cv_results$auc), 3), "\n")
cat("Test set AUC:", auc_val, "\n")
cat("Difference:", round(abs(mean(cv_results$auc) - auc_val), 3),
    "— small difference suggests stable model\n")


# ---- Visualize CV Results ----

p_cv <- ggplot(cv_results, aes(x = factor(fold), y = auc)) +
  geom_col(fill = "#5C6BC0", alpha = 0.85, width = 0.6) +
  geom_hline(yintercept = mean(cv_results$auc), linetype = "dashed", color = "#F44336") +
  geom_hline(yintercept = auc_val, linetype = "dotted", color = "#4CAF50") +
  annotate("text", x = 5, y = mean(cv_results$auc) + 0.003,
           label = paste("CV Mean:", round(mean(cv_results$auc), 3)),
           hjust = 1, color = "#F44336", fontface = "bold") +
  annotate("text", x = 5, y = auc_val - 0.003,
           label = paste("Test:", auc_val),
           hjust = 1, color = "#4CAF50", fontface = "bold") +
  labs(
    title    = "Cross-Validation AUC — Logistic Regression",
    subtitle = "Each bar = one fold. Consistent bars = stable model.",
    x = "Fold", y = "AUC"
  ) +
  coord_cartesian(ylim = c(
    min(cv_results$auc) - 0.02,
    max(cv_results$auc) + 0.02
  )) +
  theme_ml

print(p_cv)
ggsave(file.path(out_dir, "08_cross_validation.png"), p_cv,
       width = 8, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: TRAIN vs TEST COMPARISON
# ============================================================ #
# Check for overfitting: if train performance >> test performance,
# the model memorized training data instead of learning patterns.
# Logistic regression rarely overfits (it's too simple), so
# we expect train ≈ test.
# ============================================================ #

train_roc <- pROC::roc(train_df[[target]], train_probs, quiet = TRUE)
train_auc <- round(as.numeric(pROC::auc(train_roc)), 3)

cat("\n--- Overfitting Check ---\n")
cat("Train AUC:", train_auc, "\n")
cat("Test AUC: ", auc_val, "\n")
cat("Gap:", round(train_auc - auc_val, 3), "\n")

if (abs(train_auc - auc_val) < 0.02) {
  cat("Minimal gap — no evidence of overfitting.\n")
} else if (train_auc - auc_val > 0.05) {
  cat("Notable gap — possible overfitting (unusual for logistic regression).\n")
} else {
  cat("Small gap — model generalizes well.\n")
}


# ============================================================ #
# SECTION 9: SAVE RESULTS FOR COMPARISON
# ============================================================ #
# We'll save the key metrics so the final comparison script
# can pull results from all models.
# ============================================================ #

results_logistic <- tibble(
  model          = "Logistic Regression",
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

saveRDS(results_logistic, file.path(out_dir, "results_logistic.rds"))
saveRDS(test_probs, file.path(out_dir, "test_predictions_logistic.rds"))

cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 10: SUMMARY & TAKEAWAYS
# ============================================================ #

cat("\n")
cat("=== LOGISTIC REGRESSION SUMMARY ===\n")
cat("Target: Heart Disease\n")
cat("AUC:", auc_val, " — ", dplyr::case_when(
  auc_val >= 0.9 ~ "Excellent discrimination",
  auc_val >= 0.8 ~ "Good discrimination",
  auc_val >= 0.7 ~ "Acceptable discrimination",
  TRUE           ~ "Poor discrimination"
), "\n")
cat("F1:", round(f1, 3), "\n")
cat("CV AUC:", round(mean(cv_results$auc), 3), "±", round(sd(cv_results$auc), 3), "\n")
cat("\n")
cat("KEY LESSONS:\n")
cat("1. Logistic regression gives us a BASELINE to beat.\n")
cat("2. The model is fully interpretable — every coefficient tells a story.\n")
cat("3. It assumes linear relationships in log-odds.\n")
cat("4. It CANNOT learn interactions unless you manually add them.\n")
cat("5. Next model (Decision Trees) will learn nonlinear patterns automatically.\n")
cat("\n")
cat("QUESTION FOR NEXT SCRIPT:\n")
cat("Can a decision tree — which splits data into groups without assuming\n")
cat("linearity — beat this logistic regression baseline?\n")
cat("===================================\n")