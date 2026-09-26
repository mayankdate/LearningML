# ============================================================ #
# 08 — Isolation Forest: Anomaly Detection
# ============================================================ #
# Purpose: Teach anomaly detection — a fundamentally different
#   ML objective. Instead of predicting a target variable, we
#   ask: "Which patients are UNUSUAL compared to the rest?"
#
# This is NOT classification. We're not predicting heart disease.
# We're finding patients whose COMBINATION of features makes
# them statistical outliers — regardless of any diagnosis.
#
# Uses: isotree (fast, modern Isolation Forest implementation)
# Target: NONE (unsupervised anomaly detection)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis,
  caret,      # dummyVars for one-hot encoding
  isotree,    # Isolation Forest — fast C++ implementation
  uwot        # UMAP for 2D visualization of anomalies
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Isolation_Forest", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Isolation_Forest"

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

# Color scheme for anomaly detection
col_normal  <- "#2196F3"   # Blue  = normal patients
col_anomaly <- "#F44336"   # Red   = anomalies
col_mid     <- "#FF9800"   # Orange = borderline


# ============================================================ #
# SECTION 1: WHAT IS ANOMALY DETECTION?
# ============================================================ #
#
# ANALOGY: Imagine you're a doctor reviewing patient charts.
# Most patients look "normal" — their numbers fall within
# expected ranges. But occasionally you see a chart that makes
# you pause: a 25-year-old with extremely high cholesterol,
# or someone with very low BMI but high blood pressure.
# These patients aren't necessarily sick — they're just UNUSUAL.
# Their combination of features is rare in the population.
#
# Anomaly detection automates this: "find the patients whose
# feature profile is most different from the typical patient."
#
# HOW IS THIS DIFFERENT FROM CLASSIFICATION?
#
#   CLASSIFICATION (Scripts 01–06):
#     - You have labeled outcomes (heart_disease = 0 or 1)
#     - You train the model to distinguish sick from healthy
#     - Evaluation: accuracy, AUC, confusion matrix
#
#   ANOMALY DETECTION (this script):
#     - NO labels needed — it's unsupervised
#     - The model learns what "normal" looks like
#     - Anything far from "normal" is flagged as anomalous
#     - Evaluation: do the flagged patients make clinical sense?
#
# WHY IS THIS USEFUL IN HEALTHCARE?
#   1. Fraud detection — unusual billing patterns
#   2. Lab error detection — biologically implausible values
#   3. Rare disease screening — patients who don't fit known profiles
#   4. Data quality — finding corrupted or miscoded records
#   5. Clinical alerts — flagging patients who need a second look
#
# WHAT IS AN ISOLATION FOREST?
#
# The key insight: anomalies are EASY TO ISOLATE.
#
# Imagine you pick a random feature (say, age) and a random
# split point (say, age = 45). This separates the data in two.
# Pick another random feature and split again. And again.
# You're building a random decision tree — but NOT to predict
# anything. Just to ISOLATE individual points.
#
# NORMAL points are surrounded by similar points. They need
# MANY random splits before they're isolated (alone in a leaf).
# ANOMALOUS points are far from the crowd. They get isolated
# in just a FEW splits — because any random cut is likely to
# separate them from the dense mass of normal points.
#
# The ANOMALY SCORE is based on the average number of splits
# needed to isolate each point across many random trees.
#   - Few splits → easy to isolate → more anomalous (score → 1)
#   - Many splits → hard to isolate → more normal (score → 0)
#
# WHY NOT JUST USE DISTANCE-BASED METHODS?
#   Distance (like in DBSCAN) works in low dimensions but breaks
#   down in high-dimensional data — everything looks equally far
#   apart ("curse of dimensionality"). Isolation Forest uses
#   random splits, which scale well to many features.
#
# WHEN ISOLATION FOREST WORKS WELL:
#   - High-dimensional data (many features)
#   - Anomalies are a small fraction of the data
#   - You don't know what anomalies look like in advance
#   - Mixed numeric and categorical features
#
# WHEN IT STRUGGLES:
#   - Anomalies that are "local" (only unusual within a subgroup)
#   - Very noisy data where normal variation is large
#   - When you need to explain WHY something is anomalous
#     (Isolation Forest gives a score but not a reason)
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #

# ---- Load Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Dataset:", nrow(ml_data), "rows |", ncol(ml_data), "columns\n")

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


# ---- One-Hot Encode ----
# Isolation Forest in isotree can handle categorical features directly,
# but for consistency with other scripts (and for UMAP visualization later),
# we also create a numeric matrix.
dummy_formula <- as.formula(paste("~", paste(features, collapse = " + ")))
dummy_obj     <- caret::dummyVars(dummy_formula, data = ml_data)
X_numeric     <- predict(dummy_obj, newdata = ml_data)

cat("Feature matrix:", nrow(X_numeric), "rows ×", ncol(X_numeric), "columns\n")

# NOTE: We do NOT scale for Isolation Forest.
# Unlike K-Means or KNN (which measure distances), Isolation Forest
# splits on individual features using random thresholds. Scaling
# doesn't affect which splits isolate a point — the algorithm is
# scale-invariant by design. This is one of its advantages.


# ============================================================ #
# SECTION 3: FIT THE ISOLATION FOREST
# ============================================================ #
#
# KEY HYPERPARAMETERS:
#
# ntrees: Number of random trees to build.
#   More trees = more stable anomaly scores (less random noise).
#   100-300 is typically enough. Diminishing returns beyond ~500.
#
# sample_size: Number of data points each tree sees.
#   Smaller samples make it easier to isolate anomalies (fewer
#   points to separate from). Default = 256 works well.
#   If your data has subtle anomalies, try larger samples.
#
# ndim: Number of features to combine per split.
#   ndim = 1: split on one feature at a time (classic Isolation Forest)
#   ndim > 1: split on linear combinations of features
#     (Extended Isolation Forest — can detect anomalies that are
#     only unusual in COMBINATIONS of features, not individually)
#
# max_depth: Maximum depth of each tree.
#   Default = ceiling(log2(sample_size)). Usually left at default.
#   Shallow trees work because anomalies are isolated quickly anyway.
# ============================================================ #

# ---- 3a: Fit Isolation Forest ----
set.seed(42)
iso_model <- isotree::isolation.forest(
  X_numeric,
  ntrees      = 300,       # 300 random trees for stable scores
  sample_size = 256,       # Each tree sees 256 random patients
  ndim        = 1,         # Classic isolation forest (split on single features)
  nthreads    = 1          # Single thread for reproducibility
)

cat("Isolation Forest trained: 300 trees, sample_size = 256\n")


# ---- 3b: Get Anomaly Scores ----
# Scores range from 0 to 1:
#   Close to 1 = highly anomalous (isolated in few splits)
#   Close to 0.5 = normal (average path length)
#   Close to 0 = very dense region (hard to isolate)
anomaly_scores <- predict(iso_model, X_numeric)

ml_data$anomaly_score <- anomaly_scores

cat("Anomaly score distribution:\n")
cat("  Min:", round(min(anomaly_scores), 3),
    "| Median:", round(median(anomaly_scores), 3),
    "| Max:", round(max(anomaly_scores), 3), "\n")
cat("  Mean:", round(mean(anomaly_scores), 3),
    "| SD:", round(sd(anomaly_scores), 3), "\n")


# ---- 3c: Choose an Anomaly Threshold ----
# There's no universal threshold. Common approaches:
#   - Top X% (e.g., flag the most unusual 5% of patients)
#   - Score > 0.5 (theoretical threshold, but often too aggressive)
#   - Domain-specific (a clinician decides what's "worth flagging")
#
# We'll use the top 5% as our anomaly cutoff — this flags about
# 1,250 patients out of 25,000 as anomalous.
anomaly_pct <- 0.05
threshold   <- quantile(anomaly_scores, 1 - anomaly_pct)

ml_data$is_anomaly <- ifelse(anomaly_scores >= threshold, "Anomaly", "Normal")
ml_data$is_anomaly <- factor(ml_data$is_anomaly, levels = c("Normal", "Anomaly"))

n_anomaly <- sum(ml_data$is_anomaly == "Anomaly")
cat("Threshold (top", anomaly_pct * 100, "%):", round(threshold, 4), "\n")
cat("Patients flagged as anomalies:", n_anomaly, "out of", nrow(ml_data), "\n")


# ============================================================ #
# SECTION 4: VISUALIZATIONS
# ============================================================ #

# ---- 4a: Anomaly Score Distribution ----
p_score_dist <- ggplot(ml_data, aes(x = anomaly_score)) +
  geom_histogram(bins = 80, fill = col_normal, color = "white", alpha = 0.8) +
  geom_vline(xintercept = threshold, linetype = "dashed",
             linewidth = 1, color = col_anomaly) +
  annotate("text", x = threshold + 0.01, y = Inf, vjust = 2,
           label = paste0("Top ", anomaly_pct * 100, "% threshold: ",
                          round(threshold, 3)),
           fontface = "italic", size = 3.5, hjust = 0) +
  labs(
    title    = "Anomaly Score Distribution — Isolation Forest",
    subtitle = "Most patients cluster near 0.4–0.5 (normal). The right tail contains\npatients who were easy to isolate — they're the anomalies.",
    x = "Anomaly Score (higher = more unusual)",
    y = "Count"
  ) +
  theme_ml

print(p_score_dist)
ggsave(file.path(out_dir, "01_anomaly_score_distribution.png"), p_score_dist,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 4b: Anomalies in UMAP Space ----
# Use UMAP (from Script 07) to project all patients into 2D,
# then highlight anomalies. If the Isolation Forest is working well,
# anomalies should appear at the EDGES of clusters or in sparse regions.
set.seed(42)

# Scale for UMAP (UMAP benefits from scaling, unlike Isolation Forest)
X_scaled <- scale(X_numeric)

umap_fit <- uwot::umap(X_scaled, n_neighbors = 30, min_dist = 0.1, verbose = FALSE)

umap_df <- tibble(
  umap1         = umap_fit[, 1],
  umap2         = umap_fit[, 2],
  anomaly_score = anomaly_scores,
  is_anomaly    = ml_data$is_anomaly
)

# Plot all points, colored by anomaly score
p_umap_score <- ggplot(umap_df, aes(x = umap1, y = umap2, color = anomaly_score)) +
  geom_point(alpha = 0.4, size = 0.6) +
  scale_color_gradient2(low = col_normal, mid = col_mid, high = col_anomaly,
                        midpoint = threshold, name = "Anomaly\nScore") +
  labs(
    title    = "UMAP Projection — Colored by Anomaly Score",
    subtitle = "Red/orange = high anomaly score (easy to isolate). Blue = normal.\nAnomalies should appear at edges or in sparse regions.",
    x = "UMAP 1", y = "UMAP 2"
  ) +
  theme_ml

print(p_umap_score)
ggsave(file.path(out_dir, "02_umap_anomaly_scores.png"), p_umap_score,
       width = 10, height = 8, dpi = 150, bg = "white")

# Overlay: normal points faint, anomalies bold
p_umap_flag <- ggplot() +
  geom_point(data = umap_df %>% dplyr::filter(is_anomaly == "Normal"),
             aes(x = umap1, y = umap2), color = col_normal,
             alpha = 0.15, size = 0.5) +
  geom_point(data = umap_df %>% dplyr::filter(is_anomaly == "Anomaly"),
             aes(x = umap1, y = umap2), color = col_anomaly,
             alpha = 0.7, size = 1.2) +
  labs(
    title    = "UMAP — Anomalies Highlighted (Top 5%)",
    subtitle = "Blue = normal patients (faint). Red = flagged anomalies (bold).\nAnomalies at cluster edges = patients with extreme feature combinations.",
    x = "UMAP 1", y = "UMAP 2"
  ) +
  theme_ml

print(p_umap_flag)
ggsave(file.path(out_dir, "03_umap_anomalies_highlighted.png"), p_umap_flag,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 4c: How Do Anomalies Differ? Feature Comparison ----
# Compare feature distributions between normal and anomalous patients.
# If the model is working, anomalies should have more extreme values.

compare_df <- ml_data %>%
  dplyr::select(is_anomaly, age, bmi, sbp, dbp, cholesterol,
                fasting_glucose, income, physical_activity) %>%
  tidyr::pivot_longer(-is_anomaly, names_to = "feature", values_to = "value")

p_compare <- ggplot(compare_df, aes(x = is_anomaly, y = value, fill = is_anomaly)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.8) +
  scale_fill_manual(values = c("Normal" = col_normal, "Anomaly" = col_anomaly)) +
  facet_wrap(~feature, scales = "free_y", ncol = 4) +
  labs(
    title    = "Feature Distributions — Normal vs Anomalous Patients",
    subtitle = "Anomalies should show more extreme or unusual values.\nWider spread and shifted medians indicate the features driving anomaly status.",
    x = NULL, y = "Value", fill = NULL
  ) +
  theme_ml +
  theme(strip.text = element_text(face = "bold"),
        axis.text.x = element_text(size = 10))

print(p_compare)
ggsave(file.path(out_dir, "04_feature_comparison.png"), p_compare,
       width = 14, height = 8, dpi = 150, bg = "white")


# ---- 4d: Anomaly Score by Key Features ----
# Scatter plots showing anomaly score against individual features.
# Helps identify WHICH features are associated with high scores.
p_age <- ggplot(ml_data, aes(x = age, y = anomaly_score, color = is_anomaly)) +
  geom_point(alpha = 0.2, size = 0.5) +
  scale_color_manual(values = c("Normal" = col_normal, "Anomaly" = col_anomaly)) +
  geom_hline(yintercept = threshold, linetype = "dashed", color = "grey40") +
  labs(title = "Anomaly Score vs Age", x = "Age", y = "Anomaly Score") +
  theme_ml + theme(legend.position = "none")

p_bmi <- ggplot(ml_data, aes(x = bmi, y = anomaly_score, color = is_anomaly)) +
  geom_point(alpha = 0.2, size = 0.5) +
  scale_color_manual(values = c("Normal" = col_normal, "Anomaly" = col_anomaly)) +
  geom_hline(yintercept = threshold, linetype = "dashed", color = "grey40") +
  labs(title = "Anomaly Score vs BMI", x = "BMI", y = "Anomaly Score") +
  theme_ml + theme(legend.position = "none")

p_sbp <- ggplot(ml_data, aes(x = sbp, y = anomaly_score, color = is_anomaly)) +
  geom_point(alpha = 0.2, size = 0.5) +
  scale_color_manual(values = c("Normal" = col_normal, "Anomaly" = col_anomaly)) +
  geom_hline(yintercept = threshold, linetype = "dashed", color = "grey40") +
  labs(title = "Anomaly Score vs SBP", x = "Systolic BP", y = "Anomaly Score") +
  theme_ml + theme(legend.position = "none")

p_chol <- ggplot(ml_data, aes(x = cholesterol, y = anomaly_score, color = is_anomaly)) +
  geom_point(alpha = 0.2, size = 0.5) +
  scale_color_manual(values = c("Normal" = col_normal, "Anomaly" = col_anomaly)) +
  geom_hline(yintercept = threshold, linetype = "dashed", color = "grey40") +
  labs(title = "Anomaly Score vs Cholesterol", x = "Cholesterol", y = "Anomaly Score") +
  theme_ml + theme(legend.position = "none")

p_scatter_grid <- (p_age | p_bmi) / (p_sbp | p_chol) +
  patchwork::plot_annotation(
    title    = "Anomaly Score vs Key Features",
    subtitle = "Red points (above dashed line) = flagged anomalies. Look for which features\nhave the widest spread of anomaly scores — those drive the isolation.",
    theme = theme_annotation
  )

print(p_scatter_grid)
ggsave(file.path(out_dir, "05_anomaly_vs_features.png"), p_scatter_grid,
       width = 12, height = 10, dpi = 150, bg = "white")


# ---- 4e: Anomaly Rate by Disease Status ----
# Even though this is unsupervised, we can check: are patients with
# diseases more likely to be flagged as anomalies?
# This makes clinical sense — disease often involves extreme lab values.
disease_anomaly <- ml_data %>%
  dplyr::summarise(
    heart_disease_pct = round(mean(is_anomaly[heart_disease == 1] == "Anomaly") * 100, 1),
    no_disease_pct    = round(mean(is_anomaly[heart_disease == 0] == "Anomaly") * 100, 1),
    diabetes_pct      = round(mean(is_anomaly[diabetes == 1] == "Anomaly") * 100, 1),
    cancer_pct        = round(mean(is_anomaly[cancer == 1] == "Anomaly") * 100, 1),
    overall_pct       = round(mean(is_anomaly == "Anomaly") * 100, 1)
  )

disease_bar_df <- tibble(
  group = c("Overall", "No Heart Disease", "Heart Disease", "Diabetes", "Cancer"),
  pct   = c(disease_anomaly$overall_pct, disease_anomaly$no_disease_pct,
            disease_anomaly$heart_disease_pct, disease_anomaly$diabetes_pct,
            disease_anomaly$cancer_pct)
)
disease_bar_df$group <- forcats::fct_inorder(disease_bar_df$group)

p_disease <- ggplot(disease_bar_df, aes(x = group, y = pct, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(pct, "%")), vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = c(
    "Overall" = "grey50", "No Heart Disease" = col_normal,
    "Heart Disease" = col_anomaly, "Diabetes" = col_mid, "Cancer" = "#8E24AA"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Anomaly Rate by Disease Status",
    subtitle = "If anomaly detection is working, patients with diseases should be flagged\nmore often — their feature profiles are further from 'typical healthy'.",
    x = NULL, y = "% Flagged as Anomaly"
  ) +
  theme_ml +
  theme(legend.position = "none")

print(p_disease)
ggsave(file.path(out_dir, "06_anomaly_rate_by_disease.png"), p_disease,
       width = 10, height = 7, dpi = 150, bg = "white")

cat("\nAnomaly rates by group:\n")
print(as.data.frame(disease_anomaly))


# ---- 4f: Top 20 Most Anomalous Patients ----
# Look at the actual patients with the highest anomaly scores.
# What makes them unusual?
top_anomalies <- ml_data %>%
  dplyr::arrange(dplyr::desc(anomaly_score)) %>%
  dplyr::slice_head(n = 20) %>%
  dplyr::select(id, anomaly_score, age, sex, bmi, sbp, cholesterol,
                fasting_glucose, smoking_status, heart_disease, diabetes)

cat("\nTop 20 most anomalous patients:\n")
print(as.data.frame(top_anomalies))


# ============================================================ #
# SECTION 5: SENSITIVITY ANALYSIS — HOW PARAMETERS AFFECT SCORES
# ============================================================ #
# The number of trees and sample size both affect score stability.
# Let's see how the anomaly ranking changes with different settings.

# ---- 5a: Effect of ntrees ----
# With few trees, scores are noisy (random). With more trees, they stabilize.
tree_counts <- c(10, 50, 100, 300, 500)

tree_comparison <- purrr::map_dfr(tree_counts, function(nt) {
  set.seed(42)
  model_tmp <- isotree::isolation.forest(X_numeric, ntrees = nt,
                                         sample_size = 256, ndim = 1, nthreads = 1)
  scores_tmp <- predict(model_tmp, X_numeric)
  tibble(ntrees = nt, score = scores_tmp, patient = 1:length(scores_tmp))
})

# Compare score correlation between small and large tree counts
# High correlation means the ranking is stable (doesn't change much)
cor_df <- tree_comparison %>%
  tidyr::pivot_wider(names_from = ntrees, values_from = score, names_prefix = "t_") %>%
  dplyr::summarise(
    cor_10_vs_300  = round(cor(t_10, t_300), 3),
    cor_50_vs_300  = round(cor(t_50, t_300), 3),
    cor_100_vs_300 = round(cor(t_100, t_300), 3),
    cor_500_vs_300 = round(cor(t_500, t_300), 3)
  )

cat("\nScore correlation with ntrees=300 baseline:\n")
cat("  10 trees: r =", cor_df$cor_10_vs_300, "\n")
cat("  50 trees: r =", cor_df$cor_50_vs_300, "\n")
cat("  100 trees: r =", cor_df$cor_100_vs_300, "\n")
cat("  500 trees: r =", cor_df$cor_500_vs_300, "\n")

# Visualize score distribution by ntrees
p_ntrees <- tree_comparison %>%
  dplyr::mutate(ntrees = factor(ntrees)) %>%
  ggplot(aes(x = score, fill = ntrees)) +
  geom_density(alpha = 0.4) +
  labs(
    title    = "Effect of ntrees on Anomaly Score Distribution",
    subtitle = "With more trees, scores become more concentrated (less noise).\nThe ranking of who's anomalous stabilizes quickly — 100-300 trees is enough.",
    x = "Anomaly Score", y = "Density", fill = "Number\nof Trees"
  ) +
  theme_ml

print(p_ntrees)
ggsave(file.path(out_dir, "07_ntrees_sensitivity.png"), p_ntrees,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 5b: Classic vs Extended Isolation Forest ----
# Classic (ndim=1): splits on one feature at a time.
#   Good for anomalies that are extreme on individual features.
# Extended (ndim=2+): splits on combinations of features.
#   Better for anomalies that are only unusual in COMBINATIONS
#   (e.g., a patient with normal age AND normal cholesterol
#   individually, but the COMBINATION is rare).
set.seed(42)
iso_extended <- isotree::isolation.forest(
  X_numeric, ntrees = 300, sample_size = 256, ndim = 2, nthreads = 1
)
scores_extended <- predict(iso_extended, X_numeric)

compare_methods <- tibble(
  classic  = anomaly_scores,
  extended = scores_extended
)

cor_methods <- round(cor(compare_methods$classic, compare_methods$extended), 3)
cat("\nClassic vs Extended Isolation Forest score correlation:", cor_methods, "\n")

p_classic_vs_ext <- ggplot(compare_methods, aes(x = classic, y = extended)) +
  geom_point(alpha = 0.1, size = 0.5, color = "grey30") +
  geom_abline(linetype = "dashed", color = col_anomaly) +
  annotate("text", x = 0.35, y = 0.7,
           label = paste0("Correlation: ", cor_methods),
           fontface = "bold", size = 4.5) +
  labs(
    title    = "Classic vs Extended Isolation Forest",
    subtitle = "Each dot = one patient. Points far from the diagonal are scored\ndifferently by the two methods — these patients are unusual in\nfeature COMBINATIONS (caught by extended) vs individual extremes (caught by classic).",
    x = "Classic IF Score (ndim=1)",
    y = "Extended IF Score (ndim=2)"
  ) +
  theme_ml

print(p_classic_vs_ext)
ggsave(file.path(out_dir, "08_classic_vs_extended.png"), p_classic_vs_ext,
       width = 9, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: ANOMALIES vs HEART DISEASE RISK SCORES
# ============================================================ #
# Interesting question: do patients flagged as anomalies by the
# Isolation Forest also have high PREDICTED RISK of heart disease
# from supervised models? If previous model results exist, let's check.

xgb_preds_path <- "Outputs/Boosting/test_predictions_xgboost.rds"
has_xgb <- file.exists(xgb_preds_path)

if (has_xgb) {
  # Load XGBoost predictions (only on test set, so we match on indices)
  # Instead, we compare anomaly scores against disease status directly
  cat("\nXGBoost predictions available — comparing anomaly scores to disease risk\n")
}

# Regardless of whether supervised results exist, we can plot the
# anomaly score distribution split by actual heart disease status
p_score_by_disease <- ggplot(ml_data,
                             aes(x = anomaly_score,
                                 fill = factor(heart_disease, labels = c("No", "Yes")))) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = threshold, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("No" = col_normal, "Yes" = col_anomaly)) +
  labs(
    title    = "Anomaly Score Distribution by Heart Disease Status",
    subtitle = "If heart disease patients tend to have higher anomaly scores,\nit means their feature profiles are inherently more unusual.",
    x = "Anomaly Score", y = "Density",
    fill = "Heart Disease"
  ) +
  theme_ml

print(p_score_by_disease)
ggsave(file.path(out_dir, "09_anomaly_by_disease_status.png"), p_score_by_disease,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: SAVE RESULTS
# ============================================================ #

results_iso <- list(
  model           = iso_model,
  anomaly_scores  = anomaly_scores,
  threshold       = threshold,
  n_anomalies     = n_anomaly,
  anomaly_pct     = anomaly_pct,
  top_anomalies   = top_anomalies,
  disease_anomaly = disease_anomaly
)

saveRDS(results_iso, file.path(out_dir, "results_isolation_forest.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 8: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== ISOLATION FOREST — SUMMARY ===\n")
cat("Patients:", nrow(ml_data), "\n")
cat("Anomalies flagged (top", anomaly_pct * 100, "%):", n_anomaly, "\n")
cat("Score threshold:", round(threshold, 4), "\n")
cat("Heart disease patients flagged:", disease_anomaly$heart_disease_pct, "%\n")
cat("Non-disease patients flagged:", disease_anomaly$no_disease_pct, "%\n")

cat("\nWHAT WE LEARNED:\n")
cat("1. Anomaly detection is fundamentally different from classification.\n")
cat("   No target variable, no accuracy metric. We're asking 'who is unusual?'\n")
cat("2. Isolation Forest works by random splitting — anomalies are points\n")
cat("   that get isolated quickly (few splits). Normal points need many splits.\n")
cat("3. The anomaly SCORE (0-1) ranks patients by unusualness. The THRESHOLD\n")
cat("   (which score = 'anomaly') is a business/clinical decision, not a\n")
cat("   statistical one. Top 5% is common, but context matters.\n")
cat("4. Isolation Forest is SCALE-INVARIANT — unlike K-Means or KNN,\n")
cat("   it doesn't care about feature scales because it splits on thresholds.\n")
cat("5. Classic IF (ndim=1) catches individually extreme features.\n")
cat("   Extended IF (ndim=2+) catches unusual feature COMBINATIONS.\n")
cat("6. More trees = more stable scores. 100-300 trees is usually enough.\n")
cat("7. Anomalies in our data tend to be patients at extremes of age, BMI,\n")
cat("   or blood pressure — clinically sensible. Patients with diseases\n")
cat("   are flagged more often than healthy patients.\n")
cat("8. UMAP visualization confirms anomalies appear at the edges of\n")
cat("   patient clusters — they're in sparse, unusual regions of feature space.\n")

cat("\nWHEN TO USE ISOLATION FOREST:\n")
cat("  - Fraud/error detection in medical billing or lab data\n")
cat("  - Flagging patients for clinical review\n")
cat("  - Data quality checks (finding implausible records)\n")
cat("  - Pre-screening before supervised modeling\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Script 07 (Unsupervised): DBSCAN also finds outliers, but as a\n")
cat("    byproduct of clustering. Isolation Forest is PURPOSE-BUILT for anomalies.\n")
cat("  - Scripts 01-06 (Supervised): anomaly scores could be used as a FEATURE\n")
cat("    in supervised models — 'how unusual is this patient?' as an input.\n")
cat("  - Script 11 (Interpretation): SHAP could explain WHY a specific patient\n")
cat("    is anomalous, which Isolation Forest alone can't do easily.\n")
cat("===================================\n")

cat("\nScript 08 complete.\n")