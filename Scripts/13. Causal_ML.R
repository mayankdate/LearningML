# ============================================================ #
# 13 — Causal ML: From Prediction to Causation
# ============================================================ #
# Purpose: Teach the fundamental difference between PREDICTION
#   and CAUSATION. All previous scripts asked "can we predict
#   heart disease?" This script asks "does smoking CAUSE heart
#   disease, or does it just predict it?"
#
# We cover:
#   1. Prediction vs causation — why they're different
#   2. Confounding — why naive associations are misleading
#   3. Propensity scores — balancing treated vs untreated groups
#   4. Inverse Probability Weighting (IPW) — causal effect estimation
#   5. Causal Forests (grf) — heterogeneous treatment effects
#   6. Double/Debiased ML concepts
#
# Treatment: smoking (Current vs Never)
# Outcome: heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales,
  caret, pROC,
  grf,           # Generalized Random Forests (Causal Forest)
  glmnet         # For propensity score estimation
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Causal_ML", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Causal_ML"

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

col_treat   <- "#E53935"   # Red = treated (smokers)
col_control <- "#1565C0"   # Blue = control (never smokers)
col_effect  <- "#43A047"   # Green = causal effect


# ============================================================ #
# SECTION 1: PREDICTION VS CAUSATION
# ============================================================ #
#
# THE KEY DISTINCTION:
#
# PREDICTION (Scripts 01–12):
#   "Given a patient's features, what is the probability of
#    heart disease?" → Used for: screening, risk scores, triage.
#   We don't care WHY the prediction works, only that it's accurate.
#   Age predicts heart disease. Wearing reading glasses predicts it too
#   (because old people wear them). Both are useful for prediction.
#
# CAUSATION (this script):
#   "If we INTERVENED and made this patient stop smoking, would
#    their heart disease risk decrease?" → Used for: treatment
#    decisions, policy, clinical guidelines.
#   We care about the MECHANISM. Reading glasses don't CAUSE heart
#   disease — removing them wouldn't help. Smoking might CAUSE it —
#   quitting might help. Causation answers "what if we did X?"
#
# WHY IS THIS HARD?
#   We can't observe what WOULD HAVE happened if a smoker hadn't smoked.
#   This is the "fundamental problem of causal inference" — we can
#   never observe both potential outcomes for the same person.
#
# THE NAIVE APPROACH (AND WHY IT FAILS):
#   Compare heart disease rates: smokers vs non-smokers.
#   If smokers have higher rates → "smoking causes heart disease"?
#   NOT NECESSARILY. Smokers might also be older, poorer, less
#   educated, heavier drinkers. These CONFOUNDERS make the
#   naive comparison misleading. The difference we see is a mix
#   of the causal effect of smoking AND the effects of confounders.
#
# CONFOUNDERS:
#   Variables that affect BOTH the treatment (smoking) AND the
#   outcome (heart disease). If we don't account for them, we
#   get a biased estimate of the causal effect.
#   Example: Age → affects both smoking likelihood AND heart disease risk.
#   If older people smoke more AND get more heart disease, the naive
#   smoking-heart disease association is inflated by age.
#
# THE SOLUTION — THREE APPROACHES:
#
# 1. PROPENSITY SCORE / IPW:
#    Model P(smoking | confounders) for each patient.
#    Reweight the data so treated and control groups look similar
#    on confounders. Then the reweighted comparison is causal.
#
# 2. CAUSAL FOREST (grf):
#    A modified Random Forest that estimates HETEROGENEOUS treatment
#    effects — how the causal effect VARIES across patients.
#    "Smoking increases risk by 5% on average, but by 12% for
#    elderly patients with high cholesterol."
#
# 3. DOUBLE/DEBIASED ML:
#    Use ML to flexibly model BOTH the treatment and outcome,
#    then estimate the causal effect from the residuals.
#    Conceptually: "After ML removes confounding, what's left
#    is the causal effect."
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #

# ---- Load Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")

# Focus on Current Smokers vs Never Smokers (drop Former for clean comparison)
# This makes the "treatment" binary: smoked (1) vs never smoked (0)
ml_data <- ml_data %>%
  dplyr::filter(smoking_status %in% c("Never", "Current")) %>%
  dplyr::mutate(
    treatment = as.integer(smoking_status == "Current"),
    sex       = factor(sex),
    race      = factor(race),
    education = factor(education,
                       levels = c("Less than HS", "High School", "Some College",
                                  "Bachelor's", "Graduate")),
    alcohol_use = factor(alcohol_use,
                         levels = c("None", "Light", "Moderate", "Heavy"))
  )

cat("After filtering to Current vs Never smokers:", nrow(ml_data), "patients\n")
cat("Treatment (Current smoker):", sum(ml_data$treatment), "(",
    round(mean(ml_data$treatment) * 100, 1), "%)\n")
cat("Heart disease rate — Smokers:", round(mean(ml_data$heart_disease[ml_data$treatment == 1]) * 100, 1),
    "% | Non-smokers:", round(mean(ml_data$heart_disease[ml_data$treatment == 0]) * 100, 1), "%\n")

# Confounders: variables that could affect both smoking and heart disease
confounders <- c("age", "sex", "race", "education", "income",
                 "alcohol_use", "physical_activity",
                 "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose")

outcome <- "heart_disease"


# ============================================================ #
# SECTION 3: THE NAIVE ESTIMATE (BIASED)
# ============================================================ #
# Just compare outcomes between groups — ignoring confounders.

naive_diff <- mean(ml_data$heart_disease[ml_data$treatment == 1]) -
  mean(ml_data$heart_disease[ml_data$treatment == 0])

cat("\n--- Naive Estimate (unadjusted) ---\n")
cat("Heart disease rate difference (smoker - non-smoker):",
    round(naive_diff * 100, 2), "percentage points\n")
cat("This is the ASSOCIATION, not the causal effect.\n")
cat("It's biased by confounders (age, income, education, etc.)\n")


# ============================================================ #
# SECTION 4: CHECK FOR CONFOUNDING
# ============================================================ #
# Show that smokers and non-smokers differ on confounders.
# If they're identical on confounders, naive comparison = causal.
# If they differ, we need adjustment.

# ---- 4a: Covariate Balance Before Adjustment ----
numeric_confounders <- c("age", "bmi", "sbp", "dbp", "cholesterol",
                         "fasting_glucose", "income", "physical_activity")

balance_before <- purrr::map_dfr(numeric_confounders, function(var) {
  treated_mean <- mean(ml_data[[var]][ml_data$treatment == 1])
  control_mean <- mean(ml_data[[var]][ml_data$treatment == 0])
  pooled_sd    <- sd(ml_data[[var]])
  smd <- (treated_mean - control_mean) / pooled_sd  # Standardized mean difference
  tibble(variable = var, treated = treated_mean, control = control_mean,
         smd = round(smd, 3))
})

cat("\nCovariate balance (before adjustment):\n")
print(as.data.frame(balance_before))
cat("SMD > 0.1 suggests meaningful imbalance\n")

p_balance_before <- ggplot(balance_before, aes(x = abs(smd), y = variable)) +
  geom_point(size = 3, color = col_treat) +
  geom_vline(xintercept = 0.1, linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.1, y = 0.5, label = "SMD = 0.1 threshold",
           hjust = -0.1, size = 3, color = "grey40") +
  labs(
    title    = "Covariate Balance — Before Adjustment",
    subtitle = "Standardized Mean Difference (SMD) between smokers and non-smokers.\nSMD > 0.1 (right of dashed line) = meaningful imbalance = confounding.",
    x = "|Standardized Mean Difference|", y = NULL
  ) +
  theme_ml

print(p_balance_before)
ggsave(file.path(out_dir, "01_balance_before.png"), p_balance_before,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: PROPENSITY SCORES
# ============================================================ #
#
# WHAT IS A PROPENSITY SCORE?
#   P(treatment = 1 | confounders) — the probability of being a
#   smoker given a patient's other characteristics.
#
# WHY IS THIS USEFUL?
#   If two patients have the same propensity score (same probability
#   of being a smoker), then ANY difference in their outcomes is
#   not due to confounders — because the confounders are balanced
#   at that propensity score level.
#
# HOW WE USE IT:
#   Inverse Probability Weighting (IPW):
#     Smokers get weight = 1 / P(smoking)
#     Non-smokers get weight = 1 / (1 - P(smoking))
#   This upweights smokers who look like non-smokers (rare smokers)
#   and non-smokers who look like smokers (rare non-smokers),
#   creating a "pseudo-population" where confounders are balanced.

# ---- 5a: Estimate Propensity Score ----
# Use logistic regression (via glmnet) to predict treatment from confounders
ps_formula <- as.formula(paste("~", paste(confounders, collapse = " + ")))
ps_matrix  <- model.matrix(ps_formula, data = ml_data)[, -1]

set.seed(42)
ps_model <- glmnet::cv.glmnet(
  x = ps_matrix, y = ml_data$treatment,
  family = "binomial", alpha = 0, nfolds = 5
)

ml_data$propensity_score <- as.numeric(predict(ps_model, newx = ps_matrix,
                                               s = "lambda.min", type = "response"))

cat("\nPropensity score distribution:\n")
cat("  Smokers — Mean:", round(mean(ml_data$propensity_score[ml_data$treatment == 1]), 3),
    "| SD:", round(sd(ml_data$propensity_score[ml_data$treatment == 1]), 3), "\n")
cat("  Non-smokers — Mean:", round(mean(ml_data$propensity_score[ml_data$treatment == 0]), 3),
    "| SD:", round(sd(ml_data$propensity_score[ml_data$treatment == 0]), 3), "\n")


# ---- 5b: Propensity Score Distribution ----
p_ps <- ggplot(ml_data, aes(x = propensity_score,
                            fill = factor(treatment, labels = c("Never", "Current")))) +
  geom_histogram(bins = 50, alpha = 0.6, position = "identity", color = "white") +
  scale_fill_manual(values = c("Never" = col_control, "Current" = col_treat)) +
  labs(
    title    = "Propensity Score Distribution by Treatment Group",
    subtitle = "OVERLAP is critical — we can only estimate causal effects where both\ngroups have similar propensity scores. Regions with no overlap are dropped.",
    x = "Propensity Score (probability of being a smoker)",
    y = "Count",
    fill = "Smoking Status"
  ) +
  theme_ml

print(p_ps)
ggsave(file.path(out_dir, "02_propensity_scores.png"), p_ps,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: IPW — INVERSE PROBABILITY WEIGHTING
# ============================================================ #
#
# IPW creates a "pseudo-population" where confounders are balanced.
# Each patient gets a weight:
#   Treated:   w = 1 / P(treatment)
#   Control:   w = 1 / (1 - P(treatment))
#
# Then the weighted difference in outcomes = Average Treatment Effect (ATE).

# ---- 6a: Compute IPW Weights ----
# Clip propensity scores to avoid extreme weights (standard practice)
ps_clipped <- pmax(pmin(ml_data$propensity_score, 0.95), 0.05)

ml_data$ipw_weight <- ifelse(
  ml_data$treatment == 1,
  1 / ps_clipped,
  1 / (1 - ps_clipped)
)

cat("\nIPW weights — Mean:", round(mean(ml_data$ipw_weight), 2),
    "| Max:", round(max(ml_data$ipw_weight), 2), "\n")

# ---- 6b: IPW Estimate of Average Treatment Effect ----
# Weighted mean outcome for treated - weighted mean for control
ipw_treated <- weighted.mean(
  ml_data$heart_disease[ml_data$treatment == 1],
  ml_data$ipw_weight[ml_data$treatment == 1]
)
ipw_control <- weighted.mean(
  ml_data$heart_disease[ml_data$treatment == 0],
  ml_data$ipw_weight[ml_data$treatment == 0]
)
ipw_ate <- ipw_treated - ipw_control

cat("\n--- IPW Causal Estimate (Average Treatment Effect) ---\n")
cat("Weighted heart disease rate — Smokers:", round(ipw_treated * 100, 2), "%\n")
cat("Weighted heart disease rate — Non-smokers:", round(ipw_control * 100, 2), "%\n")
cat("IPW ATE:", round(ipw_ate * 100, 2), "percentage points\n")
cat("Compare to naive:", round(naive_diff * 100, 2), "percentage points\n")
cat("Difference:", round((naive_diff - ipw_ate) * 100, 2),
    "pp — this is confounding bias removed by IPW\n")


# ---- 6c: Check Balance After IPW ----
balance_after <- purrr::map_dfr(numeric_confounders, function(var) {
  treated_wmean <- weighted.mean(ml_data[[var]][ml_data$treatment == 1],
                                 ml_data$ipw_weight[ml_data$treatment == 1])
  control_wmean <- weighted.mean(ml_data[[var]][ml_data$treatment == 0],
                                 ml_data$ipw_weight[ml_data$treatment == 0])
  pooled_sd     <- sd(ml_data[[var]])
  smd <- (treated_wmean - control_wmean) / pooled_sd
  tibble(variable = var, smd_before = balance_before$smd[balance_before$variable == var],
         smd_after = round(smd, 3))
})

cat("\nBalance after IPW:\n")
print(as.data.frame(balance_after))

# ---- 6d: Balance Love Plot (Before vs After) ----
balance_long <- balance_after %>%
  tidyr::pivot_longer(c(smd_before, smd_after), names_to = "timing", values_to = "smd") %>%
  dplyr::mutate(timing = ifelse(timing == "smd_before", "Before IPW", "After IPW"))

p_love <- ggplot(balance_long, aes(x = abs(smd), y = variable, color = timing, shape = timing)) +
  geom_point(size = 3) +
  geom_vline(xintercept = 0.1, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("Before IPW" = col_treat, "After IPW" = col_effect)) +
  labs(
    title    = "Love Plot — Covariate Balance Before and After IPW",
    subtitle = "Good IPW moves all points LEFT of the dashed line (SMD < 0.1).\nGreen = after weighting. Red = before. Smaller SMD = better balance.",
    x = "|Standardized Mean Difference|", y = NULL,
    color = NULL, shape = NULL
  ) +
  theme_ml

print(p_love)
ggsave(file.path(out_dir, "03_love_plot.png"), p_love,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: CAUSAL FOREST (grf)
# ============================================================ #
#
# WHAT IS A CAUSAL FOREST?
#   A modified Random Forest that estimates HETEROGENEOUS treatment
#   effects — how the causal effect VARIES across patients.
#
#   Regular prediction: "P(heart disease) = 12% for this patient"
#   Causal Forest:     "Smoking increases this patient's risk by 4%"
#
# HOW IS IT DIFFERENT FROM A REGULAR RANDOM FOREST?
#   Regular RF: splits to minimize prediction error on the outcome.
#   Causal Forest: splits to maximize DIFFERENCE in treatment effects
#     between the two child nodes. It finds subgroups where the
#     causal effect of smoking is different.
#
# THE OUTPUT:
#   For each patient, a CONDITIONAL AVERAGE TREATMENT EFFECT (CATE):
#     τ(x) = E[Y(1) - Y(0) | X = x]
#   "If this patient smoked vs didn't, how much would their
#    heart disease risk change?"
#
# KEY CONCEPT — HONESTY:
#   Causal Forests use "honest estimation": one subsample finds
#   the splits, another estimates the treatment effects in each leaf.
#   This prevents overfitting the treatment effect estimates.

# ---- 7a: Prepare Data for grf ----
# grf wants a numeric matrix X, binary treatment W, binary outcome Y
X_causal <- model.matrix(ps_formula, data = ml_data)[, -1]
W_causal <- ml_data$treatment
Y_causal <- ml_data$heart_disease

cat("\nCausal Forest data:", nrow(X_causal), "patients |",
    ncol(X_causal), "features\n")

# ---- 7b: Fit Causal Forest ----
set.seed(42)
cf_model <- grf::causal_forest(
  X          = X_causal,
  Y          = Y_causal,
  W          = W_causal,
  num.trees  = 2000,
  honesty    = TRUE,       # Honest splitting (prevents overfitting effects)
  seed       = 42
)

cat("Causal Forest trained: 2000 trees, honest estimation\n")

# ---- 7c: Average Treatment Effect (ATE) from Causal Forest ----
ate_estimate <- grf::average_treatment_effect(cf_model, target.sample = "all")

cat("\n--- Causal Forest ATE ---\n")
cat("ATE:", round(ate_estimate[1], 4),
    "± SE:", round(ate_estimate[2], 4), "\n")
cat("95% CI: [", round(ate_estimate[1] - 1.96 * ate_estimate[2], 4), ",",
    round(ate_estimate[1] + 1.96 * ate_estimate[2], 4), "]\n")
cat("Interpretation: Smoking increases heart disease risk by ~",
    round(ate_estimate[1] * 100, 1), " percentage points on average\n")


# ---- 7d: Individual Treatment Effects (CATE) ----
# Each patient gets a personalized causal effect estimate
cate_predictions <- predict(cf_model, estimate.variance = TRUE)
ml_data$cate    <- cate_predictions$predictions
ml_data$cate_se <- sqrt(cate_predictions$variance.estimates)

cat("\nCate distribution:\n")
cat("  Mean:", round(mean(ml_data$cate), 4), "\n")
cat("  SD:  ", round(sd(ml_data$cate), 4), "\n")
cat("  Min: ", round(min(ml_data$cate), 4),
    "| Max:", round(max(ml_data$cate), 4), "\n")


# ---- 7e: CATE Distribution ----
p_cate_dist <- ggplot(ml_data, aes(x = cate)) +
  geom_histogram(bins = 60, fill = col_effect, color = "white", alpha = 0.8) +
  geom_vline(xintercept = ate_estimate[1], linetype = "dashed",
             linewidth = 1, color = "black") +
  annotate("text", x = ate_estimate[1] + 0.005, y = Inf, vjust = 2,
           label = paste0("ATE = ", round(ate_estimate[1] * 100, 1), "pp"),
           fontface = "bold", size = 4) +
  labs(
    title    = "Distribution of Individual Causal Effects (CATE)",
    subtitle = "Each patient gets a personalized estimate of how much smoking\naffects THEIR heart disease risk. Spread = effect heterogeneity.\nDashed line = average treatment effect (ATE).",
    x = "Conditional Average Treatment Effect (CATE)",
    y = "Count"
  ) +
  theme_ml

print(p_cate_dist)
ggsave(file.path(out_dir, "04_cate_distribution.png"), p_cate_dist,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 7f: CATE by Age ----
# Does the causal effect of smoking differ by age?
p_cate_age <- ggplot(ml_data, aes(x = age, y = cate)) +
  geom_point(alpha = 0.2, size = 0.8, color = "grey50") +
  geom_smooth(method = "loess", color = col_effect, linewidth = 1.2, se = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  labs(
    title    = "Causal Effect of Smoking by Age",
    subtitle = "Each dot = one patient's estimated causal effect. Green line = trend.\nAbove 0 = smoking increases risk for that patient.\nIf the line slopes up, smoking is more harmful for older patients.",
    x = "Age", y = "CATE (effect of smoking on heart disease risk)"
  ) +
  theme_ml

print(p_cate_age)
ggsave(file.path(out_dir, "05_cate_by_age.png"), p_cate_age,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 7g: CATE by BMI ----
p_cate_bmi <- ggplot(ml_data, aes(x = bmi, y = cate)) +
  geom_point(alpha = 0.2, size = 0.8, color = "grey50") +
  geom_smooth(method = "loess", color = col_treat, linewidth = 1.2, se = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  labs(
    title    = "Causal Effect of Smoking by BMI",
    subtitle = "Does the harmful effect of smoking depend on body weight?\nIf the line is flat, the effect is the same regardless of BMI.",
    x = "BMI", y = "CATE (effect of smoking)"
  ) +
  theme_ml

print(p_cate_bmi)
ggsave(file.path(out_dir, "06_cate_by_bmi.png"), p_cate_bmi,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 7h: Variable Importance for Treatment Effect Heterogeneity ----
# Which variables drive DIFFERENCES in the causal effect across patients?
# This is NOT the same as predictive importance — it's asking
# "what determines WHO is most affected by smoking?"
cf_vimp <- grf::variable_importance(cf_model)
vimp_df <- tibble(
  feature    = colnames(X_causal),
  importance = as.numeric(cf_vimp)
) %>%
  dplyr::arrange(dplyr::desc(importance)) %>%
  dplyr::slice_head(n = 15) %>%
  dplyr::mutate(feature = forcats::fct_inorder(feature))

p_vimp <- ggplot(vimp_df, aes(x = importance, y = forcats::fct_rev(feature))) +
  geom_col(fill = col_effect, width = 0.7) +
  labs(
    title    = "Variable Importance — What Drives Effect Heterogeneity?",
    subtitle = "NOT predictive importance. This shows which variables determine\nWHO is most affected by smoking. High = the causal effect varies\na lot along this variable.",
    x = "Importance (fraction of splits)", y = NULL
  ) +
  theme_ml

print(p_vimp)
ggsave(file.path(out_dir, "07_causal_variable_importance.png"), p_vimp,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: COMPARISON — NAIVE vs IPW vs CAUSAL FOREST
# ============================================================ #

estimates_df <- tibble(
  method   = c("Naive (unadjusted)", "IPW (weighted)", "Causal Forest"),
  ate      = c(naive_diff, ipw_ate, ate_estimate[1]),
  se       = c(NA, NA, ate_estimate[2]),
  type     = c("Biased", "Adjusted", "Adjusted")
)

estimates_df$method <- forcats::fct_inorder(estimates_df$method)

p_comparison <- ggplot(estimates_df, aes(x = ate * 100, y = method, fill = type)) +
  geom_col(width = 0.5) +
  geom_errorbarh(aes(xmin = (ate - 1.96 * se) * 100,
                     xmax = (ate + 1.96 * se) * 100),
                 height = 0.2, na.rm = TRUE) +
  geom_text(aes(label = paste0(round(ate * 100, 2), "pp")),
            hjust = -0.2, fontface = "bold", size = 4) +
  scale_fill_manual(values = c("Biased" = col_treat, "Adjusted" = col_effect)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(
    title    = "Causal Effect Estimates — Naive vs Adjusted",
    subtitle = "Naive = just compare group means (biased by confounders).\nIPW and Causal Forest adjust for confounders.\nDifference between naive and adjusted = confounding bias.",
    x = "Average Treatment Effect (percentage points)",
    y = NULL, fill = NULL
  ) +
  theme_ml

print(p_comparison)
ggsave(file.path(out_dir, "08_estimate_comparison.png"), p_comparison,
       width = 11, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 9: SAVE RESULTS
# ============================================================ #

results_causal <- list(
  naive_ate    = naive_diff,
  ipw_ate      = ipw_ate,
  cf_ate       = ate_estimate,
  cate_summary = summary(ml_data$cate),
  balance      = balance_after,
  cf_vimp      = vimp_df
)

saveRDS(results_causal, file.path(out_dir, "results_causal_ml.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 10: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== CAUSAL ML — SUMMARY ===\n")
cat("Treatment: Current smoking vs Never smoking\n")
cat("Outcome: Heart disease\n")
cat("Naive ATE:         ", round(naive_diff * 100, 2), "pp\n")
cat("IPW ATE:           ", round(ipw_ate * 100, 2), "pp\n")
cat("Causal Forest ATE: ", round(ate_estimate[1] * 100, 2), "pp ± ",
    round(ate_estimate[2] * 100, 2), "\n")

cat("\nWHAT WE LEARNED:\n")
cat("1.  PREDICTION ≠ CAUSATION. ML models predict outcomes but don't\n")
cat("    tell you what CAUSES them. Correlation ≠ causation.\n")
cat("2.  CONFOUNDERS bias naive comparisons. Smokers and non-smokers\n")
cat("    differ on age, income, education etc. The naive difference\n")
cat("    mixes the smoking effect with confounder effects.\n")
cat("3.  PROPENSITY SCORES model the probability of treatment given\n")
cat("    confounders. They're used for reweighting (IPW) to create\n")
cat("    balanced pseudo-populations.\n")
cat("4.  IPW (Inverse Probability Weighting) reweights patients so\n")
cat("    treated and control groups are comparable on confounders.\n")
cat("    The weighted difference = causal effect estimate.\n")
cat("5.  The LOVE PLOT checks balance: after IPW, all SMDs should\n")
cat("    be < 0.1, meaning confounders are balanced.\n")
cat("6.  CAUSAL FORESTS estimate HETEROGENEOUS effects: how the\n")
cat("    causal effect varies across patient subgroups. Not just\n")
cat("    'does smoking cause heart disease?' but 'for WHOM is it\n")
cat("    most harmful?'\n")
cat("7.  Causal Forest variable importance shows which features drive\n")
cat("    DIFFERENCES in the causal effect, not which features predict\n")
cat("    the outcome. These are fundamentally different questions.\n")
cat("8.  HONEST estimation in Causal Forests prevents overfitting:\n")
cat("    one subsample finds splits, another estimates effects.\n")

cat("\nWHEN TO USE CAUSAL ML:\n")
cat("  IPW:           Simple treatment effect estimation, publications\n")
cat("  Causal Forest: When effects may vary across patients, personalized medicine\n")
cat("  Don't use:     When you only need prediction, not 'what if?' answers\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Scripts 01-06 (Prediction): Answer 'what will happen?'\n")
cat("    Causal ML answers 'what would happen IF we intervened?'\n")
cat("  - Script 03 (Random Forest): Causal Forest modifies the RF\n")
cat("    splitting criterion to maximize treatment effect differences.\n")
cat("  - Script 11 (Interpretation): SHAP explains predictions.\n")
cat("    CATE explains causal effects. Different questions.\n")
cat("===================================\n")

cat("\nScript 13 complete.\n")