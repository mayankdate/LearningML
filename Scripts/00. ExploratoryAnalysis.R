# ============================================================ #
# 02 — Exploratory Data Analysis (EDA)
# ============================================================ #
# Purpose: Explore the synthetic clinical dataset before modeling.
#   Understand distributions, relationships, and patterns.
#
# Input:  ml_teaching_data.rds (from 01_data_generation.R)
# Output: Visualizations + summary tables
# ============================================================ #

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, readr,
  ggplot2, patchwork, viridis, scales, GGally,
  data.table, Hmisc, table1, kableExtra,
  corrplot
)

rm(list = ls()) #Clear environment
if (!is.null(dev.list())) dev.off() #Clear plot/graphics
gc() #Garbage Collection - Free memory


setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/EDA", showWarnings = FALSE, recursive = TRUE)

# ---- Load Data ----

ml_data <- readRDS("Data/ml_teaching_data.rds")

cat("Dataset:", nrow(ml_data), "rows x", ncol(ml_data), "columns\n")
str(ml_data, give.attr = FALSE)


# ---- Global Plot Theme ----
# Consistent, clean look across all plots
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    strip.text    = element_text(face = "bold"),
    legend.position = "bottom"
  )

# Separate theme for patchwork plot_annotation() — must be a raw theme() object
theme_annotation <- theme(
  plot.title    = element_text(face = "bold", size = 14),
  plot.subtitle = element_text(color = "grey40", size = 11)
)

# Color palettes we'll reuse
pal_binary  <- c("No" = "#2196F3", "Yes" = "#F44336")   # Blue = no, Red = yes
pal_sex     <- c("Female" = "#E91E63", "Male" = "#1565C0")


# ============================================================ #
# SECTION 1: UNIVARIATE DISTRIBUTIONS
# ============================================================ #
# What does each variable look like on its own?
# ============================================================ #


# ---- 1A: Continuous Variable Distributions ----
# Histograms for all continuous variables in a grid

continuous_vars <- c("age", "income", "bmi", "sbp", "dbp",
                     "cholesterol", "fasting_glucose", "physical_activity",
                     "healthcare_cost", "medication_count", "hospitalizations")

# Build individual histogram plots, then combine with patchwork
hist_list <- purrr::map(continuous_vars, function(var) {
  ggplot(ml_data, aes(x = .data[[var]])) +
    geom_histogram(fill = "#5C6BC0", color = "white", bins = 40, alpha = 0.85) +
    labs(title = var, x = NULL, y = NULL) +
    theme_ml +
    theme(plot.title = element_text(size = 11))
})

# Arrange in a grid
p_hists <- wrap_plots(hist_list, ncol = 4) +
  plot_annotation(
    title    = "Distributions of Continuous Variables",
    subtitle = "Checking for skewness, outliers, and range",
    theme    = theme_annotation
  )

print(p_hists)

ggsave("Outputs/EDA/01_continuous_distributions.png", p_hists,
       width = 16, height = 12, dpi = 150, bg = "white")


# ---- 1B: Categorical Variable Distributions ----
# Bar charts for factors

cat_vars <- c("sex", "race", "education", "smoking_status", "alcohol_use")

bar_list <- purrr::map(cat_vars, function(var) {
  ml_data %>%
    dplyr::count(.data[[var]]) %>%
    ggplot(aes(x = reorder(.data[[var]], -n), y = n)) +
    geom_col(fill = "#26A69A", alpha = 0.85) +
    geom_text(aes(label = scales::comma(n)), vjust = -0.3, size = 3) +
    labs(title = var, x = NULL, y = NULL) +
    theme_ml +
    theme(
      plot.title = element_text(size = 11),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
})

p_bars <- wrap_plots(bar_list, ncol = 3) +
  plot_annotation(
    title    = "Distributions of Categorical Variables",
    subtitle = "Sample composition by demographic and behavioral groups",
    theme    = theme_annotation
  )

print(p_bars)

ggsave("Outputs/EDA/02_categorical_distributions.png", p_bars,
       width = 14, height = 8, dpi = 150, bg = "white")


# ---- 1C: Outcome Prevalence ----
# How common are our prediction targets?

outcome_summary <- tibble(
  Outcome    = c("Diabetes", "Heart Disease", "Cancer", "Mortality"),
  Count      = c(sum(ml_data$diabetes), sum(ml_data$heart_disease),
                 sum(ml_data$cancer), sum(ml_data$mortality)),
  Prevalence = round(Count / nrow(ml_data) * 100, 1)
)

p_outcomes <- outcome_summary %>%
  ggplot(aes(x = reorder(Outcome, -Prevalence), y = Prevalence)) +
  geom_col(fill = "#EF5350", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = paste0(Prevalence, "%")), vjust = -0.3, size = 4, fontface = "bold") +
  labs(
    title    = "Prevalence of Binary Outcomes",
    subtitle = "These are the classification targets we'll predict",
    x = NULL, y = "Prevalence (%)"
  ) +
  theme_ml

print(p_outcomes)

ggsave("Outputs/EDA/03_outcome_prevalence.png", p_outcomes,
       width = 8, height = 5, dpi = 150, bg = "white")

cat("Outcome prevalences:\n")
print(outcome_summary)


# ============================================================ #
# SECTION 2: BIVARIATE RELATIONSHIPS
# ============================================================ #
# How do variables relate to each other and to outcomes?
# ============================================================ #


# ---- 2A: Correlation Heatmap (Continuous Variables) ----
# Pearson correlations among numeric variables

numeric_cols <- ml_data %>%
  dplyr::select(where(is.numeric), -id) %>%
  names()

cor_matrix <- ml_data %>%
  dplyr::select(all_of(numeric_cols)) %>%
  cor(use = "pairwise.complete.obs")

# Save correlation plot to file (corrplot doesn't return a ggplot)
png("Outputs/EDA/04_correlation_heatmap.png", width = 1200, height = 1000, res = 150)
corrplot::corrplot(
  cor_matrix,
  method  = "color",
  type    = "lower",
  order   = "hclust",                     # Cluster similar variables together
  tl.cex  = 0.75,
  tl.col  = "black",
  addCoef.col = "black",
  number.cex  = 0.6,
  col = colorRampPalette(c("#1565C0", "white", "#C62828"))(200),
  title = "Correlation Matrix — Numeric Variables",
  mar   = c(0, 0, 2, 0)
)
dev.off()
cat("Correlation heatmap saved\n")


# ---- 2B: Key Clinical Relationships ----
# Show the relationships we BUILT INTO the data
# These are the patterns ML models should be able to learn

# BMI vs Diabetes
p_bmi_diab <- ml_data %>%
  dplyr::mutate(diabetes_label = ifelse(diabetes == 1, "Diabetic", "Non-Diabetic")) %>%
  ggplot(aes(x = diabetes_label, y = bmi, fill = diabetes_label)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = c("Non-Diabetic" = "#2196F3", "Diabetic" = "#F44336")) +
  labs(title = "BMI by Diabetes Status",
       subtitle = "Higher BMI → higher diabetes risk (built into the data)",
       x = NULL, y = "BMI") +
  theme_ml + theme(legend.position = "none")

# Age vs Heart Disease
p_age_hd <- ml_data %>%
  dplyr::mutate(hd_label = ifelse(heart_disease == 1, "Heart Disease", "No Heart Disease")) %>%
  ggplot(aes(x = hd_label, y = age, fill = hd_label)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = c("No Heart Disease" = "#2196F3", "Heart Disease" = "#F44336")) +
  labs(title = "Age by Heart Disease Status",
       subtitle = "Older age → higher heart disease risk",
       x = NULL, y = "Age") +
  theme_ml + theme(legend.position = "none")

# Smoking vs Cancer
p_smoke_cancer <- ml_data %>%
  dplyr::group_by(smoking_status) %>%
  dplyr::summarise(cancer_pct = mean(cancer) * 100, .groups = "drop") %>%
  ggplot(aes(x = smoking_status, y = cancer_pct, fill = smoking_status)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = paste0(round(cancer_pct, 1), "%")), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Never" = "#4CAF50", "Former" = "#FF9800", "Current" = "#F44336")) +
  labs(title = "Cancer Prevalence by Smoking Status",
       subtitle = "Smoking is a major cancer risk factor",
       x = NULL, y = "Cancer Prevalence (%)") +
  theme_ml + theme(legend.position = "none")

# Disease Count vs Healthcare Cost
p_disease_cost <- ml_data %>%
  dplyr::mutate(disease_count = factor(diabetes + heart_disease + cancer)) %>%
  ggplot(aes(x = disease_count, y = healthcare_cost, fill = disease_count)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.1) +
  scale_y_log10(labels = scales::dollar) +              # Log scale for cost
  scale_fill_viridis_d(option = "plasma", end = 0.85) +
  labs(title = "Healthcare Cost by Disease Burden",
       subtitle = "More diseases → exponentially higher cost (note log scale)",
       x = "Number of Diseases", y = "Annual Healthcare Cost (log scale)") +
  theme_ml + theme(legend.position = "none")

# Combine into a 2x2 panel
p_relationships <- (p_bmi_diab | p_age_hd) / (p_smoke_cancer | p_disease_cost) +
  plot_annotation(
    title = "Key Clinical Relationships in the Dataset",
    subtitle = "These are the patterns our ML models should learn to detect",
    theme = theme_annotation
  )

print(p_relationships)

ggsave("Outputs/EDA/05_key_relationships.png", p_relationships,
       width = 14, height = 10, dpi = 150, bg = "white")


# ---- 2C: Blood Pressure vs Age, Colored by Heart Disease ----
# Scatterplot showing continuous-continuous relationship + outcome overlay

p_bp_age <- ml_data %>%
  dplyr::mutate(heart_disease_label = factor(heart_disease, labels = c("No", "Yes"))) %>%
  ggplot(aes(x = age, y = sbp, color = heart_disease_label)) +
  geom_point(alpha = 0.15, size = 0.8) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = pal_binary, name = "Heart Disease") +
  labs(
    title    = "Systolic Blood Pressure vs Age",
    subtitle = "Both BP and heart disease increase with age — they share risk factors",
    x = "Age", y = "Systolic BP (mmHg)"
  ) +
  theme_ml

print(p_bp_age)

ggsave("Outputs/EDA/06_bp_vs_age_by_hd.png", p_bp_age,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 2D: Income vs Healthcare Cost by Education ----
# Shows the SES gradient in health outcomes

p_income_cost <- ml_data %>%
  ggplot(aes(x = income, y = healthcare_cost, color = education)) +
  geom_point(alpha = 0.1, size = 0.5) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
  scale_x_log10(labels = scales::dollar) +
  scale_y_log10(labels = scales::dollar) +
  scale_color_viridis_d(option = "viridis", end = 0.9) +
  labs(
    title    = "Income vs Healthcare Cost by Education",
    subtitle = "Higher income/education → slightly lower healthcare costs",
    x = "Annual Income (log)", y = "Healthcare Cost (log)",
    color = "Education"
  ) +
  theme_ml

print(p_income_cost)

ggsave("Outputs/EDA/07_income_vs_cost_by_edu.png", p_income_cost,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 3: STRATIFIED SUMMARIES
# ============================================================ #
# Aggregate views of how outcomes differ across groups
# ============================================================ #


# ---- 3A: Disease Prevalence by Age Group ----

p_prev_age <- ml_data %>%
  dplyr::mutate(age_group = cut(age,
                                breaks = c(18, 30, 40, 50, 60, 70, 80, 95),
                                labels = c("18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"),
                                right = FALSE
  )) %>%
  tidyr::pivot_longer(
    cols = c(diabetes, heart_disease, cancer, mortality),
    names_to = "outcome", values_to = "value"
  ) %>%
  dplyr::group_by(age_group, outcome) %>%
  dplyr::summarise(prevalence = mean(value) * 100, .groups = "drop") %>%
  dplyr::mutate(
    outcome = factor(outcome,
                     levels = c("diabetes", "heart_disease", "cancer", "mortality"),
                     labels = c("Diabetes", "Heart Disease", "Cancer", "Mortality")
    )
  ) %>%
  ggplot(aes(x = age_group, y = prevalence, color = outcome, group = outcome)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c(
    "Diabetes" = "#FF9800", "Heart Disease" = "#F44336",
    "Cancer" = "#9C27B0", "Mortality" = "#212121"
  )) +
  labs(
    title    = "Disease Prevalence by Age Group",
    subtitle = "All outcomes increase with age — age is the dominant risk factor",
    x = "Age Group", y = "Prevalence (%)", color = "Outcome"
  ) +
  theme_ml

print(p_prev_age)

ggsave("Outputs/EDA/08_prevalence_by_age_group.png", p_prev_age,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 3B: Outcome Rates by Sex ----

p_sex_outcomes <- ml_data %>%
  tidyr::pivot_longer(
    cols = c(diabetes, heart_disease, cancer, mortality),
    names_to = "outcome", values_to = "value"
  ) %>%
  dplyr::group_by(sex, outcome) %>%
  dplyr::summarise(prevalence = mean(value) * 100, .groups = "drop") %>%
  dplyr::mutate(outcome = str_replace_all(outcome, "_", " ") %>% str_to_title()) %>%
  ggplot(aes(x = outcome, y = prevalence, fill = sex)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
  geom_text(
    aes(label = paste0(round(prevalence, 1), "%")),
    position = position_dodge(width = 0.7), vjust = -0.3, size = 3
  ) +
  scale_fill_manual(values = pal_sex) +
  labs(
    title = "Outcome Prevalence by Sex",
    subtitle = "Males have higher heart disease and mortality rates (built into the data)",
    x = NULL, y = "Prevalence (%)", fill = "Sex"
  ) +
  theme_ml

print(p_sex_outcomes)

ggsave("Outputs/EDA/09_outcomes_by_sex.png", p_sex_outcomes,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 3C: Healthcare Cost Distribution by Disease Combination ----

p_cost_dist <- ml_data %>%
  dplyr::mutate(
    disease_profile = case_when(
      diabetes == 1 & heart_disease == 1 ~ "Diabetes + Heart Disease",
      diabetes == 1                      ~ "Diabetes Only",
      heart_disease == 1                 ~ "Heart Disease Only",
      cancer == 1                        ~ "Cancer Only",
      TRUE                               ~ "No Major Disease"
    ),
    disease_profile = factor(disease_profile,
                             levels = c("No Major Disease", "Diabetes Only", "Heart Disease Only",
                                        "Cancer Only", "Diabetes + Heart Disease")
    )
  ) %>%
  ggplot(aes(x = healthcare_cost, fill = disease_profile)) +
  geom_density(alpha = 0.5) +
  scale_x_log10(labels = scales::dollar) +
  scale_fill_viridis_d(option = "turbo", end = 0.9) +
  labs(
    title    = "Healthcare Cost Distribution by Disease Profile",
    subtitle = "Comorbidities shift the entire cost distribution rightward",
    x = "Annual Healthcare Cost (log scale)", y = "Density",
    fill = "Disease Profile"
  ) +
  theme_ml

print(p_cost_dist)

ggsave("Outputs/EDA/10_cost_by_disease_profile.png", p_cost_dist,
       width = 12, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: SUMMARY TABLE
# ============================================================ #
# A Table 1 style summary — standard in clinical research
# ============================================================ #

# ---- 4A: Overall Summary Statistics ----

summary_stats <- ml_data %>%
  dplyr::summarise(
    N = dplyr::n(),
    age_mean_sd       = paste0(round(mean(age), 1), " (", round(sd(age), 1), ")"),
    pct_female        = round(mean(sex == "Female") * 100, 1),
    bmi_mean_sd       = paste0(round(mean(bmi), 1), " (", round(sd(bmi), 1), ")"),
    sbp_mean_sd       = paste0(round(mean(sbp), 1), " (", round(sd(sbp), 1), ")"),
    chol_mean_sd      = paste0(round(mean(cholesterol), 1), " (", round(sd(cholesterol), 1), ")"),
    income_median_iqr = paste0(
      scales::dollar(median(income)),
      " [", scales::dollar(quantile(income, 0.25)),
      " - ", scales::dollar(quantile(income, 0.75)), "]"
    ),
    diabetes_pct      = round(mean(diabetes) * 100, 1),
    heart_disease_pct = round(mean(heart_disease) * 100, 1),
    cancer_pct        = round(mean(cancer) * 100, 1),
    mortality_pct     = round(mean(mortality) * 100, 1),
    cost_median_iqr   = paste0(
      scales::dollar(median(healthcare_cost)),
      " [", scales::dollar(quantile(healthcare_cost, 0.25)),
      " - ", scales::dollar(quantile(healthcare_cost, 0.75)), "]"
    ),
    hosp_mean_sd      = paste0(round(mean(hospitalizations), 2),
                               " (", round(sd(hospitalizations), 2), ")")
  )

# Print as a readable format
cat("\n--- Dataset Summary (Table 1 Style) ---\n")
cat("N:", summary_stats$N, "\n")
cat("Age, mean (SD):", summary_stats$age_mean_sd, "\n")
cat("Female (%):", summary_stats$pct_female, "\n")
cat("BMI, mean (SD):", summary_stats$bmi_mean_sd, "\n")
cat("SBP, mean (SD):", summary_stats$sbp_mean_sd, "\n")
cat("Cholesterol, mean (SD):", summary_stats$chol_mean_sd, "\n")
cat("Income, median [IQR]:", summary_stats$income_median_iqr, "\n")
cat("Diabetes (%):", summary_stats$diabetes_pct, "\n")
cat("Heart Disease (%):", summary_stats$heart_disease_pct, "\n")
cat("Cancer (%):", summary_stats$cancer_pct, "\n")
cat("Mortality (%):", summary_stats$mortality_pct, "\n")
cat("Healthcare Cost, median [IQR]:", summary_stats$cost_median_iqr, "\n")
cat("Hospitalizations, mean (SD):", summary_stats$hosp_mean_sd, "\n")


# ============================================================ #
# SECTION 5: EDA TAKEAWAYS
# ============================================================ #
# Print key findings to console — what we learned before modeling
# ============================================================ #

cat("\n")
cat("=== EDA KEY TAKEAWAYS ===\n")
cat("1. Age is the dominant risk factor for all outcomes\n")
cat("2. BMI clearly separates diabetics from non-diabetics\n")
cat("3. Smoking shows dose-response: Never < Former < Current for heart disease & cancer\n")
cat("4. Healthcare cost is right-skewed and log-normal — we should log-transform for regression\n")
cat("5. Disease outcomes are moderately imbalanced (10-25% prevalence) — may need to consider class weights\n")
cat("6. Comorbidities (diabetes + heart disease) have compounding cost effects\n")
cat("7. Males have consistently higher disease prevalence and mortality\n")
cat("8. Income and education show expected SES gradients\n")
cat("=========================\n")