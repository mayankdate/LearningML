# ============================================================ #
# 01 — Synthetic Clinical Dataset Generation
# ============================================================ #
# Purpose: Generate a realistic population health dataset with
#   meaningful clinical relationships for ML teaching.
#
# Key design: Variables are NOT independent. We build them in
#   layers so that downstream variables depend on upstream ones,
#   mimicking real epidemiological structure.
#
# Output: ml_teaching_data.rds (25,000 rows)
# ============================================================ #

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, readr, writexl,
  purrr, data.table, lubridate
)

rm(list = ls()) #Clear environment
if (!is.null(dev.list())) dev.off() #Clear plot/graphics
gc() #Garbage Collection - Free memory

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Data", showWarnings = FALSE, recursive = TRUE)

set.seed(42)  # Reproducibility
n <- 25000

# ---- Demographics ----
# These are the "root" variables — everything else flows from them

demographics <- tibble(
  id = 1:n,
  
  # Age: right-skewed distribution reflecting adult population (18-95)
  age = pmin(95, pmax(18, round(rnorm(n, mean = 52, sd = 18)))),
  
  # Sex: roughly 51% female, 49% male (US census proportions)
  sex = sample(c("Female", "Male"), n, replace = TRUE, prob = c(0.51, 0.49)),
  
  # Race/ethnicity: approximate US proportions
  race = sample(
    c("White", "Black", "Hispanic", "Asian", "Other"),
    n, replace = TRUE,
    prob = c(0.58, 0.13, 0.19, 0.06, 0.04)
  ),
  
  # Education: ordered factor, distribution shifts with age
  # Younger cohorts have more college education
  education = {
    age_vec <- pmin(95, pmax(18, round(rnorm(n, mean = 52, sd = 18))))
    # We'll overwrite with the actual age below; placeholder logic here
    sample(
      c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate"),
      n, replace = TRUE,
      prob = c(0.10, 0.25, 0.30, 0.22, 0.13)
    )
  },
  
  # Income: log-normal distribution, correlated with education later
  # Placeholder — we'll refine after education is set
  income_base = exp(rnorm(n, mean = 10.5, sd = 0.7))
)

# Adjust income by education level
# Higher education → higher income (with noise)
education_income_multiplier <- c(
  "Less than HS" = 0.55,
  "High School"  = 0.75,
  "Some College"  = 0.90,
  "Bachelor's"   = 1.25,
  "Graduate"     = 1.60
)

demographics <- demographics %>%
  dplyr::mutate(
    income = income_base * education_income_multiplier[education],
    income = round(pmax(8000, pmin(300000, income))),  # Clip to realistic range
    income_base = NULL  # Drop temporary column
  )

cat("Demographics generated:", nrow(demographics), "rows\n")
cat("Age range:", min(demographics$age), "-", max(demographics$age), "\n")
cat("Income range: $", min(demographics$income), "- $", max(demographics$income), "\n")


# ---- Behavioral / Lifestyle Variables ----
# These depend on demographics — e.g., smoking rates differ by age, sex, education

behavioral <- demographics %>%
  dplyr::mutate(
    
    # Smoking status: higher rates in lower education, middle age
    # Uses a latent propensity score approach
    smoke_propensity = -1.5 +
      0.01 * age +                                     # Slight increase with age
      -0.3 * (sex == "Female") +                       # Lower in females
      0.5 * (education == "Less than HS") +            # Higher in low education
      0.3 * (education == "High School") +
      -0.3 * (education == "Graduate") +
      rnorm(n, 0, 0.8),                               # Random noise
    
    smoking_status = case_when(
      smoke_propensity > 1.0  ~ "Current",
      smoke_propensity > 0.0  ~ "Former",
      TRUE                    ~ "Never"
    ),
    
    # Alcohol use: correlated with age and sex
    alcohol_propensity = -0.5 +
      -0.005 * age +
      0.3 * (sex == "Male") +
      0.2 * (education %in% c("Bachelor's", "Graduate")) +
      rnorm(n, 0, 0.7),
    
    alcohol_use = case_when(
      alcohol_propensity > 0.8  ~ "Heavy",
      alcohol_propensity > 0.0  ~ "Moderate",
      alcohol_propensity > -0.5 ~ "Light",
      TRUE                      ~ "None"
    ),
    
    # Physical activity (hours/week): higher in younger, higher education
    activity_mean = pmax(0.5,
                         5.0 - 0.03 * age +
                           0.5 * (sex == "Male") +
                           0.8 * (education %in% c("Bachelor's", "Graduate")) +
                           -1.0 * (education == "Less than HS")
    ),
    physical_activity = round(pmax(0, rgamma(n, shape = 2, rate = 2 / activity_mean)), 1)
    
  ) %>%
  dplyr::select(id, smoking_status, alcohol_use, physical_activity)

cat("Behavioral variables generated\n")
cat("Smoking:", table(behavioral$smoking_status), "\n")


# ---- Clinical Measurements ----
# BMI, blood pressure, cholesterol — depend on demographics + behavior

clinical <- demographics %>%
  dplyr::left_join(behavioral, by = "id") %>%
  dplyr::mutate(
    
    # BMI: depends on age, sex, activity, smoking, income
    # Mean BMI ~27 (US population average), with realistic spread
    bmi = round(
      27.0 +
        0.05 * (age - 50) +                            # Slight increase with age
        -0.8 * (sex == "Male") +                        # Males slightly lower avg
        -0.3 * physical_activity +                      # Activity lowers BMI
        1.5 * (smoking_status == "Former") +            # Former smokers gain weight
        -0.5 * (smoking_status == "Current") +          # Current smokers slightly lower
        -0.0001 * (income - 50000) +                    # Small inverse income effect
        rnorm(n, 0, 4.5),                               # Individual variation
      1
    ),
    bmi = pmax(15, pmin(55, bmi)),                      # Clip to physiological range
    
    # Systolic blood pressure: increases with age, BMI, smoking
    sbp = round(
      110 +
        0.4 * (age - 50) +                              # Strong age effect
        0.5 * (bmi - 27) +                              # BMI effect
        5.0 * (smoking_status == "Current") +            # Smoking raises BP
        3.0 * (alcohol_use == "Heavy") +                 # Heavy drinking raises BP
        rnorm(n, 0, 12)                                  # Individual variation
    ),
    sbp = pmax(85, pmin(220, sbp)),
    
    # Diastolic blood pressure: correlated with systolic
    dbp = round(
      60 + 0.4 * (sbp - 120) + rnorm(n, 0, 6)
    ),
    dbp = pmax(50, pmin(130, dbp)),
    
    # Total cholesterol: depends on age, BMI, diet (proxied by activity/income)
    cholesterol = round(
      190 +
        0.3 * (age - 50) +
        1.2 * (bmi - 27) +
        -2.0 * physical_activity +
        10 * (smoking_status == "Current") +
        rnorm(n, 0, 30)
    ),
    cholesterol = pmax(100, pmin(400, cholesterol)),
    
    # Fasting glucose: depends on BMI, age, activity
    fasting_glucose = round(
      90 +
        0.15 * (age - 50) +
        0.8 * (bmi - 27) +
        -1.5 * physical_activity +
        rnorm(n, 0, 15)
    ),
    fasting_glucose = pmax(60, pmin(350, fasting_glucose))
    
  ) %>%
  dplyr::select(id, bmi, sbp, dbp, cholesterol, fasting_glucose)

cat("Clinical measurements generated\n")
cat("BMI range:", min(clinical$bmi), "-", max(clinical$bmi), "\n")
cat("SBP range:", min(clinical$sbp), "-", max(clinical$sbp), "\n")


# ---- Medical History / Disease Outcomes ----
# These are the KEY teaching targets. Each disease has known risk factors
# built into its probability via logistic regression structure.

# Join everything we have so far
full_data <- demographics %>%
  dplyr::left_join(behavioral, by = "id") %>%
  dplyr::left_join(clinical, by = "id")

# Helper: convert linear predictor to probability via logistic function
logistic <- function(x) 1 / (1 + exp(-x))

diseases <- full_data %>%
  dplyr::mutate(
    
    # ---- Diabetes (Type 2) ----
    # Risk factors: age, BMI, physical inactivity, family hx (simulated),
    #   race (higher in Black, Hispanic populations — real epi pattern)
    diabetes_lp = -5.5 +
      0.04 * age +
      0.12 * (bmi - 25) +
      -0.15 * physical_activity +
      0.3 * (race == "Black") +
      0.2 * (race == "Hispanic") +
      0.01 * (fasting_glucose - 90) +
      0.3 * (smoking_status == "Current"),
    diabetes_prob = logistic(diabetes_lp),
    diabetes = rbinom(n, 1, diabetes_prob),
    
    # ---- Heart Disease ----
    # Risk factors: age, sex (male), smoking, BP, cholesterol, BMI, diabetes
    heart_disease_lp = -7.0 +
      0.06 * age +
      0.6 * (sex == "Male") +
      0.5 * (smoking_status == "Current") +
      0.3 * (smoking_status == "Former") +
      0.02 * (sbp - 120) +
      0.01 * (cholesterol - 200) +
      0.05 * (bmi - 25) +
      0.8 * diabetes +                                   # Diabetes is a major risk factor
      rnorm(n, 0, 0.3),
    heart_disease_prob = logistic(heart_disease_lp),
    heart_disease = rbinom(n, 1, heart_disease_prob),
    
    # ---- Cancer (any) ----
    # Risk factors: age (strongest), smoking, alcohol, BMI
    cancer_lp = -6.5 +
      0.05 * age +
      0.7 * (smoking_status == "Current") +
      0.3 * (smoking_status == "Former") +
      0.4 * (alcohol_use == "Heavy") +
      0.03 * (bmi - 25) +
      rnorm(n, 0, 0.3),
    cancer_prob = logistic(cancer_lp),
    cancer = rbinom(n, 1, cancer_prob),
    
    # ---- Medication Use ----
    # Driven by having diseases + age
    medication_count = rpois(n, lambda = pmax(0.1,
                                              0.5 +
                                                0.03 * age +
                                                1.5 * diabetes +
                                                2.0 * heart_disease +
                                                1.0 * cancer +
                                                0.01 * pmax(0, sbp - 140) +    # Extra meds if hypertensive
                                                0.005 * pmax(0, cholesterol - 240)
    )),
    on_medication = as.integer(medication_count > 0)
    
  ) %>%
  dplyr::select(id, diabetes, heart_disease, cancer, medication_count, on_medication)

cat("Disease outcomes generated\n")
cat("Diabetes prevalence:", round(mean(diseases$diabetes) * 100, 1), "%\n")
cat("Heart disease prevalence:", round(mean(diseases$heart_disease) * 100, 1), "%\n")
cat("Cancer prevalence:", round(mean(diseases$cancer) * 100, 1), "%\n")


# ---- Mortality, Hospitalizations, Healthcare Cost ----
# These are downstream outcomes that depend on everything above

outcomes <- full_data %>%
  dplyr::left_join(diseases, by = "id") %>%
  dplyr::mutate(
    
    # ---- Mortality (5-year) ----
    # Strong age effect, diseases, smoking
    mortality_lp = -6.0 +
      0.07 * age +
      0.5 * (sex == "Male") +
      0.8 * heart_disease +
      0.6 * cancer +
      0.4 * diabetes +
      0.5 * (smoking_status == "Current") +
      0.02 * (bmi - 25)^2 / 10 +                        # U-shaped BMI effect
      -0.00001 * income,                                 # Small SES protective effect
    mortality_prob = logistic(mortality_lp),
    mortality = rbinom(n, 1, mortality_prob),
    
    # ---- Hospitalizations (annual count) ----
    # Poisson distributed, driven by disease burden and age
    hosp_rate = pmax(0.05,
                     0.2 +
                       0.015 * age +
                       0.8 * heart_disease +
                       0.6 * diabetes +
                       0.5 * cancer +
                       0.1 * medication_count +
                       -0.05 * physical_activity
    ),
    hospitalizations = rpois(n, lambda = hosp_rate),
    
    # ---- Healthcare Cost (annual, USD) ----
    # Log-normal base, multiplied by disease burden
    cost_base = exp(
      8.0 +                                              # ~$3000 baseline
        0.02 * age +
        1.0 * heart_disease +
        0.8 * diabetes +
        0.7 * cancer +
        0.15 * medication_count +
        0.5 * pmax(0, hospitalizations - 1) +
        -0.000005 * income +                             # Higher income → better preventive care
        rnorm(n, 0, 0.6)
    ),
    healthcare_cost = round(pmax(200, pmin(500000, cost_base)))
    
  ) %>%
  dplyr::select(id, mortality, hospitalizations, healthcare_cost)

cat("Outcomes generated\n")
cat("Mortality rate:", round(mean(outcomes$mortality) * 100, 1), "%\n")
cat("Mean hospitalizations:", round(mean(outcomes$hospitalizations), 2), "\n")
cat("Median healthcare cost: $", median(outcomes$healthcare_cost), "\n")


# ---- Assemble Final Dataset ----

ml_data <- demographics %>%
  dplyr::left_join(behavioral, by = "id") %>%
  dplyr::left_join(clinical, by = "id") %>%
  dplyr::left_join(diseases, by = "id") %>%
  dplyr::left_join(outcomes, by = "id")

# Convert character columns to factors with meaningful ordering
ml_data <- ml_data %>%
  dplyr::mutate(
    sex       = factor(sex),
    race      = factor(race),
    education = factor(education,
                       levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate"),
                       ordered = TRUE
    ),
    smoking_status = factor(smoking_status,
                            levels = c("Never", "Former", "Current"),
                            ordered = TRUE
    ),
    alcohol_use = factor(alcohol_use,
                         levels = c("None", "Light", "Moderate", "Heavy"),
                         ordered = TRUE
    )
  )


# ---- Export ----

saveRDS(ml_data, "Data/ml_teaching_data.rds")
# writexl::write_xlsx(ml_data, "Data/ml_teaching_data.xlsx")  # Optional Excel copy

cat("\nFinal dataset: ", nrow(ml_data), " rows x ", ncol(ml_data), " columns\n")
cat("Saved to: ml_teaching_data.rds\n")


# ---- Validation ----

cat("\n--- Quick Validation ---\n")
cat("Columns:", paste(names(ml_data), collapse = ", "), "\n")
cat("Missing values:", sum(is.na(ml_data)), "\n")

# Check that relationships exist in the right direction
cat("\n--- Sanity Checks (should show expected directions) ---\n")

# BMI should be higher in diabetics
cat("Mean BMI — Diabetic:", round(mean(ml_data$bmi[ml_data$diabetes == 1]), 1),
    "| Non-diabetic:", round(mean(ml_data$bmi[ml_data$diabetes == 0]), 1), "\n")

# Smokers should have more heart disease
hd_by_smoke <- ml_data %>%
  dplyr::group_by(smoking_status) %>%
  dplyr::summarise(hd_pct = round(mean(heart_disease) * 100, 1), .groups = "drop")
cat("Heart disease % by smoking:", paste(hd_by_smoke$smoking_status, "=", hd_by_smoke$hd_pct, "%", collapse = " | "), "\n")

# Age should predict mortality
cat("Mean age — Deceased:", round(mean(ml_data$age[ml_data$mortality == 1]), 1),
    "| Alive:", round(mean(ml_data$age[ml_data$mortality == 0]), 1), "\n")

# Healthcare cost should be higher with more diseases
ml_data %>%
  dplyr::mutate(disease_count = diabetes + heart_disease + cancer) %>%
  dplyr::group_by(disease_count) %>%
  dplyr::summarise(
    median_cost = scales::dollar(median(healthcare_cost)),
    n = dplyr::n(),
    .groups = "drop"
  ) %>%
  print()

cat("\nData generation complete.\n")