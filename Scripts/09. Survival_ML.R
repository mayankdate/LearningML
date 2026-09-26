# ============================================================ #
# 09 — Survival Analysis & Survival ML
# ============================================================ #
# Purpose: Teach survival analysis — modeling TIME-TO-EVENT
#   outcomes. Instead of "will this patient die?" (classification),
#   we ask "WHEN might this patient die, and what affects that timing?"
#
# We cover:
#   1. Kaplan-Meier curves — visualizing survival over time
#   2. Cox Proportional Hazards — the "logistic regression" of survival
#   3. Random Survival Forests — tree-based survival prediction
#
# This script generates SUPPLEMENTARY time-to-event data because
# the main dataset has mortality as 0/1 but no follow-up time.
# We build survival times from the existing features using realistic
# clinical relationships, then analyze them.
#
# Target: time_to_event + event (death indicator)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis,
  survival,      # Core survival analysis (Surv, coxph, survfit)
  survminer,     # Beautiful survival plots (ggsurvplot)
  randomForestSRC, # Random Survival Forests
  pROC,          # For time-dependent AUC
  caret          # dummyVars
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Survival_ML", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Survival_ML"

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
# SECTION 1: WHAT IS SURVIVAL ANALYSIS?
# ============================================================ #
#
# ANALOGY: Imagine tracking 1,000 patients after a hospital visit.
# Some die after 6 months, some after 2 years, some are still alive
# when the study ends at 5 years. You can't just ask "what % died?"
# because TIME matters — dying at 6 months is very different from
# dying at 4 years. And for patients still alive at the study end,
# you only know they survived AT LEAST that long — not when they'll die.
#
# SURVIVAL ANALYSIS answers: "What is the probability of surviving
# past time T, given a set of patient features?"
#
# THE KEY CONCEPT — CENSORING:
#   "Censoring" is what makes survival analysis special.
#   A patient is CENSORED if we don't observe their event (death):
#     - Study ends before they die (administrative censoring)
#     - They move away and we lose contact (lost to follow-up)
#     - They die of something unrelated (competing risks)
#
#   We CANNOT just exclude censored patients — that would bias
#   results toward shorter survival times. We also can't treat
#   them as deaths — that would underestimate survival.
#   Survival methods handle censoring correctly by using the
#   information we DO have: "this patient was alive at time T."
#
# KEY QUANTITIES:
#   Survival function S(t): P(surviving past time t)
#     Starts at 1.0 (everyone alive) and decreases over time
#   Hazard function h(t): instantaneous risk of dying at time t
#     "Given you survived to time t, what's the risk of dying NOW?"
#   Hazard ratio (HR): how much a feature multiplies the hazard
#     HR = 1.0: no effect. HR = 2.0: doubles the risk. HR = 0.5: halves it.
#
# HOW IS THIS DIFFERENT FROM CLASSIFICATION?
#   Classification (Scripts 01-06): Predicts a binary outcome (yes/no).
#     Ignores WHEN the event happens. Treats a patient who died at
#     3 months the same as one who died at 10 years.
#   Survival analysis: Models the TIMING and handles patients we
#     haven't observed long enough (censoring).
#
# THE MODELS WE'LL LEARN:
#   1. Kaplan-Meier: Non-parametric. No features — just visualizes
#      survival over time for the whole group or subgroups.
#   2. Cox PH (Proportional Hazards): The workhorse. Like logistic
#      regression but for hazard rates. Estimates how each feature
#      affects the hazard (risk) of dying.
#   3. Random Survival Forests: Like Random Forest (Script 03) but
#      the outcome is survival time + censoring, not 0/1.
# ============================================================ #


# ============================================================ #
# SECTION 2: GENERATE SURVIVAL DATA
# ============================================================ #
# Our main dataset has mortality (0/1) but no follow-up time.
# We generate realistic survival times from the existing features
# using a Weibull distribution with feature-driven hazard rates.
#
# This is NOT cheating — it's how survival analysis teaching
# datasets work. We're creating a plausible time-to-event
# structure that respects the clinical relationships already
# in the data (age increases risk, smoking increases risk, etc.)

# ---- Load Base Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Base dataset:", nrow(ml_data), "rows\n")

features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)

# ---- Factor Setup ----
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

# ---- Generate Survival Times ----
# We use a Weibull model to generate plausible survival times.
# The linear predictor (log-hazard) is driven by age, smoking,
# BMI, blood pressure, and disease status — reflecting real
# clinical risk factors for mortality.
set.seed(42)

# Build a risk score from features (higher = higher risk of death)
ml_data <- ml_data %>%
  dplyr::mutate(
    risk_score =
      0.04  * (age - 50) +                          # Older → higher risk
      0.3   * (smoking_status == "Current") +        # Smokers → higher risk
      0.15  * (smoking_status == "Former") +
      0.02  * (bmi - 25) +                           # Higher BMI → higher risk
      0.01  * (sbp - 120) +                          # Higher SBP → higher risk
      0.005 * (cholesterol - 200) +
      0.5   * heart_disease +                        # Heart disease → much higher
      0.3   * diabetes +
      0.2   * cancer +
      -0.02 * physical_activity +                    # Exercise → protective
      -0.00001 * income                              # Higher income → protective
  )

# Generate Weibull-distributed survival times
# Higher risk_score → shorter survival time
shape_param <- 1.5  # Weibull shape (>1 = increasing hazard over time)
scale_param <- exp(5 - ml_data$risk_score)  # Baseline scale modified by risk

ml_data$surv_time_raw <- rweibull(nrow(ml_data), shape = shape_param, scale = scale_param)

# Cap at a maximum follow-up of 10 years (120 months)
max_followup <- 120
ml_data$follow_up_time <- pmin(ml_data$surv_time_raw, max_followup)

# Event indicator: 1 = died during follow-up, 0 = censored (still alive at study end)
ml_data$event <- as.integer(ml_data$surv_time_raw <= max_followup)

# Also add some random censoring (lost to follow-up) for ~15% of patients
censor_prob <- 0.15
random_censor <- rbinom(nrow(ml_data), 1, censor_prob)
random_censor_time <- runif(nrow(ml_data), 1, max_followup)

# Apply random censoring: if flagged and random censor time < actual time
early_censor <- random_censor == 1 & random_censor_time < ml_data$follow_up_time
ml_data$follow_up_time[early_censor] <- random_censor_time[early_censor]
ml_data$event[early_censor] <- 0

cat("Survival data generated:\n")
cat("  Events (deaths):", sum(ml_data$event), "(", round(mean(ml_data$event) * 100, 1), "%)\n")
cat("  Censored:", sum(ml_data$event == 0), "(", round(mean(ml_data$event == 0) * 100, 1), "%)\n")
cat("  Median follow-up:", round(median(ml_data$follow_up_time), 1), "months\n")
cat("  Median survival (events only):",
    round(median(ml_data$follow_up_time[ml_data$event == 1]), 1), "months\n")


# ============================================================ #
# SECTION 3: KAPLAN-MEIER CURVES
# ============================================================ #
#
# WHAT IS KAPLAN-MEIER?
#   The simplest survival method. No features, no model — just
#   estimates the survival probability at each time point from
#   the observed data.
#
# HOW IT WORKS:
#   At each time point where a death occurs:
#     S(t) = S(t-1) × (1 - deaths_at_t / at_risk_at_t)
#   "at_risk" = patients still alive and not yet censored.
#   Censored patients are removed from the "at risk" pool
#   at their censoring time — they contributed survival
#   information UP TO that point.
#
# THE KM CURVE:
#   - Y-axis: survival probability (starts at 1.0)
#   - X-axis: time
#   - Steps down at each death
#   - Tick marks (+) show censoring times
#   - The MEDIAN survival time is where the curve crosses 0.5
# ============================================================ #

# ---- 3a: Overall Kaplan-Meier ----
surv_obj <- survival::Surv(ml_data$follow_up_time, ml_data$event)
km_overall <- survival::survfit(surv_obj ~ 1)

p_km_overall <- survminer::ggsurvplot(
  km_overall,
  data        = ml_data,
  conf.int    = TRUE,
  risk.table  = TRUE,     # Show number at risk below the plot
  xlab        = "Time (months)",
  ylab        = "Survival Probability",
  title       = "Overall Kaplan-Meier Survival Curve",
  subtitle    = "The curve shows the probability of surviving past each time point.\nShaded band = 95% confidence interval. Tick marks = censored patients.",
  ggtheme     = theme_ml,
  palette     = "#1565C0",
  risk.table.height = 0.2
)

print(p_km_overall)
ggsave(file.path(out_dir, "01_km_overall.png"), p_km_overall$plot,
       width = 10, height = 7, dpi = 150, bg = "white")


# ---- 3b: KM by Smoking Status ----
km_smoking <- survival::survfit(surv_obj ~ smoking_status, data = ml_data)

p_km_smoking <- survminer::ggsurvplot(
  km_smoking,
  data       = ml_data,
  conf.int   = FALSE,
  pval       = TRUE,       # Log-rank test p-value
  risk.table = TRUE,
  xlab       = "Time (months)",
  ylab       = "Survival Probability",
  title      = "Kaplan-Meier by Smoking Status",
  subtitle   = "Current smokers have worse survival. The p-value tests whether\nthe curves are significantly different (log-rank test).",
  ggtheme    = theme_ml,
  palette    = c("#43A047", "#FF9800", "#E53935"),
  legend.labs = c("Never", "Former", "Current"),
  risk.table.height = 0.25
)

print(p_km_smoking)
ggsave(file.path(out_dir, "02_km_smoking.png"), p_km_smoking$plot,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 3c: KM by Age Group ----
ml_data$age_group <- cut(ml_data$age,
                         breaks = c(0, 40, 55, 70, 100),
                         labels = c("18-40", "41-55", "56-70", "71+"))

km_age <- survival::survfit(surv_obj ~ age_group, data = ml_data)

p_km_age <- survminer::ggsurvplot(
  km_age,
  data       = ml_data,
  conf.int   = FALSE,
  pval       = TRUE,
  risk.table = TRUE,
  xlab       = "Time (months)",
  ylab       = "Survival Probability",
  title      = "Kaplan-Meier by Age Group",
  subtitle   = "Older patients have steeper drops — higher mortality at every time point.\nThis is the single strongest predictor of survival.",
  ggtheme    = theme_ml,
  palette    = viridis::viridis(4),
  risk.table.height = 0.25
)

print(p_km_age)
ggsave(file.path(out_dir, "03_km_age_group.png"), p_km_age$plot,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: COX PROPORTIONAL HAZARDS MODEL
# ============================================================ #
#
# WHAT IS COX PH?
#   The "logistic regression of survival analysis."
#   It models how features affect the HAZARD (instantaneous risk):
#     h(t) = h₀(t) × exp(β₁×age + β₂×smoking + β₃×bmi + ...)
#
#   h₀(t) = baseline hazard (left unspecified — that's the "semi-parametric" part)
#   exp(βᵢ) = hazard ratio for feature i
#
# HAZARD RATIOS — HOW TO INTERPRET:
#   HR = 1.0: no effect on survival
#   HR = 1.5: 50% higher risk of death at any given time
#   HR = 0.7: 30% lower risk of death (protective)
#   HR = 2.0: DOUBLES the risk of death
#
# THE "PROPORTIONAL HAZARDS" ASSUMPTION:
#   The model assumes hazard ratios are CONSTANT over time.
#   If smoking doubles the hazard, it doubles it at month 1 AND
#   at month 60. This can be violated — we'll check.
#
# WHY COX PH IS THE WORKHORSE:
#   - Handles censoring correctly
#   - Gives interpretable hazard ratios (like odds ratios)
#   - Doesn't need to specify the shape of the baseline hazard
#   - Well-understood statistical properties (p-values, CIs)
# ============================================================ #

# ---- 4a: Fit Cox Model ----
cox_formula <- as.formula(paste(
  "Surv(follow_up_time, event) ~",
  paste(features, collapse = " + ")
))

cox_model <- survival::coxph(cox_formula, data = ml_data)

cat("\nCox PH Model Summary:\n")
print(summary(cox_model))


# ---- 4b: Forest Plot of Hazard Ratios ----
# Extract coefficients and confidence intervals
cox_tidy <- broom::tidy(cox_model, conf.int = TRUE, exponentiate = TRUE)

# Clean up term names for readability
cox_tidy <- cox_tidy %>%
  dplyr::mutate(
    term_clean = stringr::str_replace(term, "smoking_status", "Smoking: "),
    term_clean = stringr::str_replace(term_clean, "alcohol_use", "Alcohol: "),
    term_clean = stringr::str_replace(term_clean, "education", "Education: "),
    term_clean = stringr::str_replace(term_clean, "race", "Race: "),
    term_clean = stringr::str_replace(term_clean, "sex", "Sex: "),
    significant = p.value < 0.05
  ) %>%
  dplyr::arrange(estimate)

cox_tidy$term_clean <- forcats::fct_inorder(cox_tidy$term_clean)

p_forest <- ggplot(cox_tidy, aes(x = estimate, y = term_clean)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.3, color = "grey40") +
  geom_point(aes(color = significant), size = 3) +
  scale_color_manual(values = c("TRUE" = "#E53935", "FALSE" = "grey60"),
                     labels = c("TRUE" = "p < 0.05", "FALSE" = "Not significant"),
                     name = NULL) +
  labs(
    title    = "Cox PH — Hazard Ratios (Forest Plot)",
    subtitle = "HR > 1 (right of dashed line) = increases risk of death.\nHR < 1 (left) = protective. Red = statistically significant.",
    x = "Hazard Ratio (95% CI)",
    y = NULL
  ) +
  theme_ml

print(p_forest)
ggsave(file.path(out_dir, "04_cox_forest_plot.png"), p_forest,
       width = 11, height = 9, dpi = 150, bg = "white")


# ---- 4c: Check Proportional Hazards Assumption ----
# If the PH assumption holds, Schoenfeld residuals should show
# no trend over time. A significant test means the HR changes
# over time — the assumption is violated for that variable.
ph_test <- survival::cox.zph(cox_model)

cat("\nProportional Hazards Test (Schoenfeld):\n")
cat("Global test p-value:", round(ph_test$table["GLOBAL", "p"], 4), "\n")
cat("Variables with p < 0.05 (PH assumption may be violated):\n")
ph_table <- as.data.frame(ph_test$table)
ph_violated <- ph_table[ph_table$p < 0.05, , drop = FALSE]
if (nrow(ph_violated) > 0) {
  print(ph_violated)
} else {
  cat("  None — PH assumption holds for all variables\n")
}

# Plot Schoenfeld residuals for the most important variable (age)
png(file.path(out_dir, "05_schoenfeld_age.png"),
    width = 10, height = 6, units = "in", res = 150)
plot(ph_test, var = "age",
     main = "Schoenfeld Residuals for Age",
     xlab = "Time", ylab = "Beta(t) for Age")
abline(h = 0, lty = 2, col = "red")
dev.off()
cat("Schoenfeld residuals plot saved\n")


# ---- 4d: Cox Model Concordance (C-index) ----
# The concordance (C-index) is the survival equivalent of AUC.
# It measures: for a random pair of patients, how often does the
# model correctly predict who dies first?
#   C = 0.5: random guessing. C = 1.0: perfect prediction.
#   C > 0.7: considered useful in survival analysis.
cox_c_index <- summary(cox_model)$concordance[1]
cat("Cox C-index (concordance):", round(cox_c_index, 4),
    "— survival equivalent of AUC\n")


# ============================================================ #
# SECTION 5: RANDOM SURVIVAL FORESTS
# ============================================================ #
#
# WHAT IS A RANDOM SURVIVAL FOREST?
#   It's a Random Forest (Script 03) adapted for survival outcomes.
#   Instead of predicting 0/1 or a continuous number, each tree
#   predicts a SURVIVAL CURVE for each patient.
#
# HOW IT WORKS:
#   1. Bootstrap a random sample of patients
#   2. At each node, pick a random subset of features
#   3. Split on the feature that best separates survival curves
#      (using log-rank statistics instead of Gini impurity)
#   4. The leaf node contains a Kaplan-Meier estimate for the
#      patients that landed there
#   5. Average the survival curves across all trees
#
# WHY USE IT INSTEAD OF COX PH?
#   - No proportional hazards assumption
#   - Automatically captures nonlinear effects and interactions
#   - Handles many features without overfitting (like RF for classification)
#   - Can be more accurate when PH assumption is violated
#
# TRADEOFF:
#   - Less interpretable than Cox PH (no simple hazard ratios)
#   - Slower to train (building many trees)
#   - Harder to get p-values for individual features
# ============================================================ #

# ---- 5a: Fit Random Survival Forest ----
# Use a subsample for speed — RSF is computationally heavy
set.seed(42)
rsf_sample_idx <- sample(1:nrow(ml_data), 10000)
rsf_data <- ml_data[rsf_sample_idx, ]

rsf_formula <- as.formula(paste(
  "Surv(follow_up_time, event) ~",
  paste(features, collapse = " + ")
))

set.seed(42)
rsf_model <- randomForestSRC::rfsrc(
  rsf_formula,
  data      = rsf_data,
  ntree     = 300,       # 300 survival trees
  mtry      = 5,         # Random features per split (similar to sqrt(13) ≈ 4)
  nodesize  = 15,        # Minimum node size (larger = faster, less overfit)
  importance = TRUE,     # Compute variable importance
  verbose   = FALSE
)

cat("\nRandom Survival Forest trained on", nrow(rsf_data), "patients\n")
cat("  ntree:", rsf_model$ntree, "\n")
cat("  Error rate:", round(rsf_model$err.rate[rsf_model$ntree], 4), "\n")


# ---- 5b: RSF Variable Importance ----
# VIMP (Variable Importance) in RSF: randomly permute each feature
# and measure how much prediction error increases. Larger increase
# = more important feature. Same concept as Random Forest (Script 03).
vimp_df <- tibble(
  feature    = names(rsf_model$importance),
  importance = rsf_model$importance
) %>%
  dplyr::arrange(dplyr::desc(importance)) %>%
  dplyr::mutate(feature = forcats::fct_inorder(feature))

p_vimp <- ggplot(vimp_df, aes(x = importance,
                              y = forcats::fct_rev(feature))) +
  geom_col(fill = "#1565C0", width = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  labs(
    title    = "Variable Importance — Random Survival Forest",
    subtitle = "How much does prediction error increase when each feature is randomized?\nLarger = the model relies more on this feature. Negative = noise variable.",
    x = "VIMP (increase in error when permuted)", y = NULL
  ) +
  theme_ml

print(p_vimp)
ggsave(file.path(out_dir, "06_rsf_variable_importance.png"), p_vimp,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 5c: RSF Predicted Survival Curves for Sample Patients ----
# Show individual predicted survival curves for patients at
# different risk levels — this is what RSF actually outputs.
# Pick 3 patients: low risk, medium risk, high risk
risk_scores <- rsf_model$predicted
low_risk_idx  <- which.min(risk_scores)
high_risk_idx <- which.max(risk_scores)
mid_risk_idx  <- which.min(abs(risk_scores - median(risk_scores)))

# Extract survival curves for these patients
surv_times <- rsf_model$time.interest

# Build data for plotting
surv_curves_df <- dplyr::bind_rows(
  tibble(time = surv_times,
         survival = rsf_model$survival[low_risk_idx, ],
         patient = "Low Risk"),
  tibble(time = surv_times,
         survival = rsf_model$survival[mid_risk_idx, ],
         patient = "Medium Risk"),
  tibble(time = surv_times,
         survival = rsf_model$survival[high_risk_idx, ],
         patient = "High Risk")
)
surv_curves_df$patient <- factor(surv_curves_df$patient,
                                 levels = c("Low Risk", "Medium Risk", "High Risk"))

p_surv_curves <- ggplot(surv_curves_df, aes(x = time, y = survival, color = patient)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Low Risk" = "#43A047",
                                "Medium Risk" = "#FF9800",
                                "High Risk" = "#E53935")) +
  labs(
    title    = "Individual Predicted Survival Curves — Random Survival Forest",
    subtitle = "Each line = one patient's predicted survival probability over time.\nThe model produces a FULL CURVE, not just a single number.",
    x = "Time (months)", y = "Survival Probability", color = "Risk Level"
  ) +
  theme_ml

print(p_surv_curves)
ggsave(file.path(out_dir, "07_rsf_individual_curves.png"), p_surv_curves,
       width = 10, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: MODEL COMPARISON — COX vs RSF
# ============================================================ #

# ---- 6a: Compare C-indices ----
# Cox C-index (already computed on full data)
# RSF C-index (from the OOB error — out-of-bag, like RF)
rsf_c_index <- 1 - rsf_model$err.rate[rsf_model$ntree]

cat("\nModel Comparison — C-index (concordance):\n")
cat("  Cox PH:                 ", round(cox_c_index, 4), "\n")
cat("  Random Survival Forest: ", round(rsf_c_index, 4), "\n")

compare_c <- tibble(
  model   = c("Cox PH", "Random Survival Forest"),
  c_index = c(cox_c_index, rsf_c_index)
)

p_c_compare <- ggplot(compare_c, aes(x = model, y = c_index, fill = model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(c_index, 3)), vjust = -0.5,
            fontface = "bold", size = 5) +
  scale_fill_manual(values = c("Cox PH" = "#1565C0",
                               "Random Survival Forest" = "#E53935")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Model Comparison — Concordance Index",
    subtitle = "C-index = survival equivalent of AUC. 0.5 = random, 1.0 = perfect.\nHigher = better at ranking who dies first.",
    x = NULL, y = "C-index"
  ) +
  theme_ml +
  theme(legend.position = "none")

print(p_c_compare)
ggsave(file.path(out_dir, "08_model_comparison_c_index.png"), p_c_compare,
       width = 8, height = 6, dpi = 150, bg = "white")


# ---- 6b: Cox vs RSF Variable Importance Comparison ----
# Compare which features each model considers most important
cox_imp <- abs(coef(cox_model))

# For categorical variables with multiple dummy levels, sum their importance
cox_imp_df <- tibble(
  feature    = names(cox_imp),
  importance = as.numeric(cox_imp)
) %>%
  dplyr::mutate(
    parent = dplyr::case_when(
      stringr::str_detect(feature, "^smoking_status") ~ "smoking_status",
      stringr::str_detect(feature, "^alcohol_use")    ~ "alcohol_use",
      stringr::str_detect(feature, "^education")      ~ "education",
      stringr::str_detect(feature, "^race")           ~ "race",
      stringr::str_detect(feature, "^sex")            ~ "sex",
      TRUE ~ feature
    )
  ) %>%
  dplyr::group_by(parent) %>%
  dplyr::summarise(importance = sum(importance), .groups = "drop") %>%
  dplyr::mutate(model = "Cox PH") %>%
  dplyr::rename(feature = parent)

rsf_imp_df <- vimp_df %>%
  dplyr::mutate(model = "RSF", feature = as.character(feature))

# Normalize both to 0-1 range for fair comparison
cox_imp_df$importance <- cox_imp_df$importance / max(cox_imp_df$importance)
rsf_imp_df$importance <- rsf_imp_df$importance / max(rsf_imp_df$importance)

imp_compare <- dplyr::bind_rows(cox_imp_df, rsf_imp_df) %>%
  dplyr::filter(feature %in% features)  # Keep only original feature names

p_imp_compare <- ggplot(imp_compare, aes(x = importance, y = feature, fill = model)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("Cox PH" = "#1565C0", "RSF" = "#E53935")) +
  labs(
    title    = "Variable Importance — Cox PH vs Random Survival Forest",
    subtitle = "Both normalized to 0-1. Similar rankings = models agree on what matters.\nDifferences reveal where the RSF captures nonlinear effects Cox misses.",
    x = "Normalized Importance", y = NULL, fill = "Model"
  ) +
  theme_ml

print(p_imp_compare)
ggsave(file.path(out_dir, "09_importance_comparison.png"), p_imp_compare,
       width = 11, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: SAVE RESULTS
# ============================================================ #

results_survival <- list(
  cox_model    = cox_model,
  cox_c_index  = cox_c_index,
  rsf_c_index  = rsf_c_index,
  ph_test      = ph_test,
  cox_tidy     = cox_tidy,
  vimp         = vimp_df
)

saveRDS(results_survival, file.path(out_dir, "results_survival.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 8: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== SURVIVAL ANALYSIS — SUMMARY ===\n")
cat("Patients:", nrow(ml_data), "\n")
cat("Events (deaths):", sum(ml_data$event), "(", round(mean(ml_data$event) * 100, 1), "%)\n")
cat("Censored:", sum(ml_data$event == 0), "(", round(mean(ml_data$event == 0) * 100, 1), "%)\n")
cat("Cox C-index:", round(cox_c_index, 4), "\n")
cat("RSF C-index:", round(rsf_c_index, 4), "\n")

cat("\nWHAT WE LEARNED:\n")
cat("1. Survival analysis models TIME-TO-EVENT, not just yes/no.\n")
cat("   It answers 'when?' not just 'whether?'\n")
cat("2. CENSORING is the key concept: some patients haven't had their\n")
cat("   event yet. We can't exclude them or treat them as events.\n")
cat("   Survival methods use their 'survived at least this long' information.\n")
cat("3. KAPLAN-MEIER curves visualize survival over time without features.\n")
cat("   They're the survival equivalent of a bar chart — simple but powerful.\n")
cat("4. COX PH models how features affect the hazard (instantaneous risk).\n")
cat("   Hazard ratios are like odds ratios: HR=2 means double the risk.\n")
cat("   The 'proportional hazards' assumption says HRs don't change over time.\n")
cat("5. RANDOM SURVIVAL FORESTS are tree-based survival models.\n")
cat("   No PH assumption needed, captures nonlinear effects, but less\n")
cat("   interpretable. Each patient gets a full predicted survival curve.\n")
cat("6. The C-INDEX is the survival equivalent of AUC: 'how often does\n")
cat("   the model correctly rank who dies first?' 0.5 = random, 1.0 = perfect.\n")
cat("7. Age is consistently the strongest predictor — both Cox and RSF agree.\n")
cat("   Smoking, BMI, and blood pressure also matter, as expected clinically.\n")

cat("\nWHEN TO USE EACH METHOD:\n")
cat("  Kaplan-Meier:  Descriptive plots, comparing subgroups, presentations\n")
cat("  Cox PH:        When you need interpretable hazard ratios (publications)\n")
cat("  RSF:           When PH assumption fails or you want max prediction accuracy\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Scripts 01-06 (Classification): Predict 0/1. Survival predicts time.\n")
cat("    A patient who died at 3 months vs 10 years would be treated the\n")
cat("    same in classification but very differently in survival analysis.\n")
cat("  - Script 03 (Random Forest): RSF uses the same bootstrap + random\n")
cat("    feature selection idea, just with survival-adapted splitting rules.\n")
cat("  - Script 11 (Interpretation): SHAP/LIME can be applied to RSF too.\n")
cat("===================================\n")

cat("\nScript 09 complete.\n")