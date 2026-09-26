# ============================================================ #
# 10 — Time Series Analysis & Forecasting
# ============================================================ #
# Purpose: Teach time series analysis — modeling data where
#   the ORDER of observations matters. Instead of predicting
#   "will this patient be hospitalized?" we ask "how many
#   hospitalizations will there be NEXT MONTH?"
#
# We cover:
#   1. Time series concepts: trend, seasonality, stationarity
#   2. Decomposition — separating signal from noise
#   3. ARIMA — the classic statistical forecasting model
#   4. Exponential Smoothing (ETS) — state space approach
#   5. Prophet (concept) + comparison of methods
#
# This script generates MONTHLY AGGREGATE time series data
# from the main dataset. The main dataset is cross-sectional
# (one row per patient), so we simulate a 5-year enrollment
# window to create a monthly hospitalization time series.
#
# Target: monthly hospitalization count (time series)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis,
  forecast,    # ARIMA, ETS, auto.arima, accuracy metrics
  tseries,     # ADF test for stationarity
  lubridate    # Date manipulation
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Time_Series", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Time_Series"

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
# SECTION 1: WHAT IS TIME SERIES ANALYSIS?
# ============================================================ #
#
# ANALOGY: Imagine tracking daily temperatures in a city.
# Today's temperature isn't random — it depends on yesterday's
# temperature, the season, and a long-term climate trend.
# If it was 75°F yesterday and it's July, you'd predict
# something close to 75°F tomorrow — not 30°F.
#
# Time series analysis exploits this TEMPORAL DEPENDENCE:
# past values help predict future values.
#
# HOW IS THIS DIFFERENT FROM EVERYTHING ELSE WE'VE LEARNED?
#
#   CLASSIFICATION / REGRESSION (Scripts 01-06):
#     - Each row is INDEPENDENT (one patient, no time ordering)
#     - Features predict an outcome for that specific row
#     - Shuffling the rows doesn't change anything
#
#   TIME SERIES (this script):
#     - Each row is a TIME POINT (one month's data)
#     - The ORDER matters — you can't shuffle rows
#     - Past values of the SAME VARIABLE predict its future
#     - Features = the variable's own history (lags)
#
# KEY CONCEPTS:
#
#   TREND: Long-term direction. Is the series going up, down, or flat?
#     Example: hospitalizations increasing over 5 years = upward trend.
#
#   SEASONALITY: Regular repeating pattern at fixed intervals.
#     Example: more hospitalizations in winter than summer = seasonal.
#
#   STATIONARITY: A series is "stationary" if its statistical
#     properties (mean, variance) don't change over time.
#     Most forecasting models REQUIRE stationarity.
#     A series with a trend or changing variance is non-stationary.
#     We make it stationary by DIFFERENCING (subtracting the
#     previous value: y'_t = y_t - y_{t-1}).
#
#   AUTOCORRELATION: The correlation of a series with its own past.
#     "How much does this month's value predict next month's?"
#     ACF (autocorrelation function) shows this at different lags.
#
# THE MODELS WE'LL LEARN:
#   ARIMA: Models the series as a combination of:
#     AR (autoregressive): current value depends on past values
#     I (integrated): differencing to achieve stationarity
#     MA (moving average): current value depends on past errors
#   ETS (Exponential Smoothing): Decompose into error + trend +
#     seasonality, each with a smoothing parameter.
# ============================================================ #


# ============================================================ #
# SECTION 2: GENERATE TIME SERIES DATA
# ============================================================ #
# Our main dataset is cross-sectional (one snapshot per patient).
# To create a time series, we simulate 5 years (60 months) of
# hospital admission data by assigning patients random "admission
# months" based on their hospitalization counts and clinical risk.
#
# This creates a monthly hospitalization count series with:
#   - An upward trend (aging population, more patients enrolled)
#   - Seasonality (more hospitalizations in winter months)
#   - Noise (random variation month to month)

# ---- Load Base Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Base dataset:", nrow(ml_data), "rows\n")

# ---- Generate Monthly Hospitalization Counts ----
set.seed(42)

# Simulate 60 months: Jan 2019 to Dec 2023
months_seq <- seq(as.Date("2019-01-01"), as.Date("2023-12-01"), by = "month")
n_months   <- length(months_seq)

# Build a realistic monthly series with known components:
# 1. Baseline level (~400 hospitalizations/month)
# 2. Upward trend (+2 per month = gradual increase)
# 3. Winter seasonality (Dec-Feb peak, Jun-Aug trough)
# 4. Random noise

baseline <- 400
trend    <- seq(0, 2 * (n_months - 1), length.out = n_months)

# Seasonal component: sinusoidal with peak in January
# cos() peaks at 0, and month 1 (Jan) maps to angle 0
month_num   <- lubridate::month(months_seq)
seasonality <- 40 * cos(2 * pi * (month_num - 1) / 12)

# Random noise
noise <- rnorm(n_months, mean = 0, sd = 20)

# Combine
monthly_counts <- round(baseline + trend + seasonality + noise)
monthly_counts <- pmax(monthly_counts, 100)  # Floor at 100

# Also create a healthcare cost series (correlated but noisier)
cost_per_hosp <- rnorm(n_months, mean = 8000, sd = 1500)
monthly_cost  <- round(monthly_counts * cost_per_hosp / 1000)  # in thousands

# Assemble into a data frame
ts_data <- tibble(
  date             = months_seq,
  year             = lubridate::year(months_seq),
  month            = lubridate::month(months_seq, label = TRUE),
  month_num        = month_num,
  hospitalizations = monthly_counts,
  cost_thousands   = monthly_cost
)

cat("Time series generated:", n_months, "months (Jan 2019 – Dec 2023)\n")
cat("Hospitalizations — Min:", min(ts_data$hospitalizations),
    "| Max:", max(ts_data$hospitalizations),
    "| Mean:", round(mean(ts_data$hospitalizations), 0), "\n")

# Convert to ts object — R's native time series format
# frequency = 12 means monthly data with yearly seasonality
hosp_ts <- ts(ts_data$hospitalizations,
              start = c(2019, 1), frequency = 12)


# ============================================================ #
# SECTION 3: EXPLORE THE TIME SERIES
# ============================================================ #

# ---- 3a: Plot the Raw Series ----
p_raw <- ggplot(ts_data, aes(x = date, y = hospitalizations)) +
  geom_line(color = "#1565C0", linewidth = 0.8) +
  geom_point(color = "#1565C0", size = 1.5) +
  labs(
    title    = "Monthly Hospitalizations (Jan 2019 – Dec 2023)",
    subtitle = "Look for: upward TREND (are numbers growing?), SEASONALITY (winter peaks?),\nand NOISE (random month-to-month variation).",
    x = "Date", y = "Hospitalizations"
  ) +
  theme_ml

print(p_raw)
ggsave(file.path(out_dir, "01_raw_time_series.png"), p_raw,
       width = 11, height = 6, dpi = 150, bg = "white")


# ---- 3b: Seasonal Plot ----
# Same data, but each year is a separate line overlaid.
# If there's seasonality, the lines will follow similar shapes.
seasonal_df <- ts_data %>%
  dplyr::mutate(year = factor(year))

p_seasonal <- ggplot(seasonal_df, aes(x = month, y = hospitalizations,
                                      color = year, group = year)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_color_viridis_d(option = "plasma") +
  labs(
    title    = "Seasonal Pattern — Each Year Overlaid",
    subtitle = "If lines follow similar shapes across years = strong seasonality.\nConsistent winter peaks confirm seasonal hospitalizations.",
    x = "Month", y = "Hospitalizations", color = "Year"
  ) +
  theme_ml

print(p_seasonal)
ggsave(file.path(out_dir, "02_seasonal_pattern.png"), p_seasonal,
       width = 11, height = 6, dpi = 150, bg = "white")


# ---- 3c: Decomposition ----
# Decompose the series into trend + seasonal + remainder.
# This separates the signal into interpretable components.
# STL (Seasonal and Trend decomposition using Loess) is robust
# and flexible — it uses locally weighted regression.
decomp <- stl(hosp_ts, s.window = "periodic")

# Extract components for plotting
decomp_df <- tibble(
  date     = ts_data$date,
  observed = as.numeric(hosp_ts),
  trend    = as.numeric(decomp$time.series[, "trend"]),
  seasonal = as.numeric(decomp$time.series[, "seasonal"]),
  remainder = as.numeric(decomp$time.series[, "remainder"])
)

decomp_long <- decomp_df %>%
  tidyr::pivot_longer(-date, names_to = "component", values_to = "value") %>%
  dplyr::mutate(component = factor(component,
                                   levels = c("observed", "trend", "seasonal", "remainder")))

p_decomp <- ggplot(decomp_long, aes(x = date, y = value)) +
  geom_line(color = "#1565C0", linewidth = 0.7) +
  facet_wrap(~component, ncol = 1, scales = "free_y") +
  labs(
    title    = "STL Decomposition — Separating Signal from Noise",
    subtitle = "Observed = Trend + Seasonal + Remainder.\nTrend = long-term direction. Seasonal = repeating pattern. Remainder = what's left (noise).",
    x = "Date", y = "Value"
  ) +
  theme_ml +
  theme(strip.text = element_text(face = "bold", size = 11))

print(p_decomp)
ggsave(file.path(out_dir, "03_stl_decomposition.png"), p_decomp,
       width = 11, height = 9, dpi = 150, bg = "white")


# ---- 3d: ACF and PACF ----
# ACF (AutoCorrelation Function): correlation of the series with
#   itself at different lags. Lag 1 = correlation with last month.
#   Lag 12 = correlation with same month last year (seasonality).
# PACF (Partial ACF): correlation at lag k AFTER removing the
#   effect of shorter lags. Helps determine ARIMA parameters.

png(file.path(out_dir, "04_acf_pacf.png"),
    width = 11, height = 7, units = "in", res = 150)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
acf(hosp_ts, lag.max = 36,
    main = "ACF — Autocorrelation Function\n(Spike at lag 12 = yearly seasonality)")
pacf(hosp_ts, lag.max = 36,
     main = "PACF — Partial Autocorrelation Function\n(Significant spikes suggest AR order)")
dev.off()
cat("ACF/PACF plots saved\n")


# ---- 3e: Stationarity Test ----
# ADF (Augmented Dickey-Fuller) test:
#   H0: series has a unit root (non-stationary)
#   H1: series is stationary
#   p < 0.05 → reject H0 → series IS stationary
# Most raw series with trends are non-stationary.
adf_raw <- tseries::adf.test(hosp_ts)
cat("\nADF Test on raw series:\n")
cat("  Test statistic:", round(adf_raw$statistic, 4), "\n")
cat("  p-value:", round(adf_raw$p.value, 4))
if (adf_raw$p.value < 0.05) {
  cat(" — Stationary (can model directly)\n")
} else {
  cat(" — Non-stationary (needs differencing)\n")
}

# Difference the series (remove trend) and test again
hosp_diff <- diff(hosp_ts)
adf_diff  <- tseries::adf.test(hosp_diff)
cat("ADF Test after differencing:\n")
cat("  p-value:", round(adf_diff$p.value, 4))
if (adf_diff$p.value < 0.05) {
  cat(" — Stationary after 1 differencing\n")
} else {
  cat(" — Still non-stationary\n")
}


# ============================================================ #
# SECTION 4: TRAIN/TEST SPLIT FOR TIME SERIES
# ============================================================ #
#
# CRITICAL DIFFERENCE from cross-sectional data:
# You CANNOT randomly split time series into train/test.
# The test set MUST be the most recent observations — the ones
# that come AFTER the training period. Otherwise you'd be
# "predicting the past using the future" (data leakage).
#
# We use the first 48 months (4 years) for training and
# hold out the last 12 months (1 year) for testing.
# This simulates: "train on all data through Dec 2022,
# then forecast Jan-Dec 2023 and see how we did."

train_end   <- 48   # First 48 months = training
test_length <- 12   # Last 12 months = test

hosp_train <- window(hosp_ts, end = c(2022, 12))
hosp_test  <- window(hosp_ts, start = c(2023, 1))

cat("\nTrain/Test Split:\n")
cat("  Train: months 1-", train_end, "(Jan 2019 – Dec 2022)\n")
cat("  Test:  months", train_end + 1, "-", n_months, "(Jan 2023 – Dec 2023)\n")


# ============================================================ #
# SECTION 5: ARIMA FORECASTING
# ============================================================ #
#
# WHAT IS ARIMA?
#   ARIMA(p, d, q) × (P, D, Q)[m] is three ideas combined:
#
#   AR (AutoRegressive, order p):
#     This month depends on the previous p months.
#     AR(1): y_t = c + φ₁ × y_{t-1} + error
#     "What happened recently will continue"
#
#   I (Integrated, order d):
#     Differencing d times to remove trend.
#     d=1: model the CHANGES instead of the levels
#     "Remove the trend, model what's left"
#
#   MA (Moving Average, order q):
#     This month depends on the previous q forecast ERRORS.
#     MA(1): y_t = c + error_t + θ₁ × error_{t-1}
#     "Correct for recent mistakes"
#
#   Seasonal terms (P, D, Q)[m]:
#     Same idea but at the seasonal lag (m=12 for monthly data).
#     Captures patterns like "January is always high."
#
# HOW TO CHOOSE p, d, q?
#   auto.arima() searches over many combinations and picks the
#   best one using AICc (information criterion that balances fit
#   and complexity). This is the standard approach — you rarely
#   need to choose manually.
#
# WHEN ARIMA WORKS WELL:
#   - Data with clear trend and/or seasonality
#   - Relatively short series (months to a few years)
#   - When you need prediction intervals (ARIMA gives them)
#
# WHEN IT STRUGGLES:
#   - Multiple seasonal patterns (daily + weekly + yearly)
#   - Non-linear patterns (exponential growth, regime changes)
#   - Very long series with structural breaks
# ============================================================ #

# ---- 5a: Fit ARIMA ----
# auto.arima() searches for the best ARIMA specification
# using a stepwise algorithm that minimizes AICc.
# seasonal = TRUE allows seasonal ARIMA terms.
arima_model <- forecast::auto.arima(hosp_train, seasonal = TRUE,
                                    stepwise = FALSE, approximation = FALSE)

cat("\nARIMA Model Selected:\n")
print(arima_model)
cat("AICc:", round(arima_model$aicc, 2), "\n")


# ---- 5b: ARIMA Forecast ----
arima_fc <- forecast::forecast(arima_model, h = test_length)

# Extract forecast values for comparison
arima_forecast_df <- tibble(
  date     = seq(as.Date("2023-01-01"), as.Date("2023-12-01"), by = "month"),
  actual   = as.numeric(hosp_test),
  forecast = as.numeric(arima_fc$mean),
  lo80     = as.numeric(arima_fc$lower[, 1]),
  hi80     = as.numeric(arima_fc$upper[, 1]),
  lo95     = as.numeric(arima_fc$lower[, 2]),
  hi95     = as.numeric(arima_fc$upper[, 2])
)

# Combine train and forecast for plotting
train_df_plot <- tibble(
  date  = seq(as.Date("2019-01-01"), as.Date("2022-12-01"), by = "month"),
  value = as.numeric(hosp_train),
  type  = "Training Data"
)

p_arima <- ggplot() +
  geom_line(data = train_df_plot, aes(x = date, y = value),
            color = "grey40", linewidth = 0.7) +
  geom_ribbon(data = arima_forecast_df,
              aes(x = date, ymin = lo95, ymax = hi95),
              fill = "#1565C0", alpha = 0.15) +
  geom_ribbon(data = arima_forecast_df,
              aes(x = date, ymin = lo80, ymax = hi80),
              fill = "#1565C0", alpha = 0.25) +
  geom_line(data = arima_forecast_df, aes(x = date, y = forecast),
            color = "#1565C0", linewidth = 1) +
  geom_line(data = arima_forecast_df, aes(x = date, y = actual),
            color = "#E53935", linewidth = 0.9, linetype = "dashed") +
  geom_point(data = arima_forecast_df, aes(x = date, y = actual),
             color = "#E53935", size = 2) +
  labs(
    title    = paste0("ARIMA Forecast — ", arima_model),
    subtitle = "Grey = training data. Blue line = forecast. Red dashed = actual.\nShaded bands = 80% and 95% prediction intervals.",
    x = "Date", y = "Hospitalizations"
  ) +
  theme_ml

print(p_arima)
ggsave(file.path(out_dir, "05_arima_forecast.png"), p_arima,
       width = 12, height = 7, dpi = 150, bg = "white")


# ---- 5c: ARIMA Residual Diagnostics ----
# Good model = residuals look like white noise (random, no pattern).
# We check: normally distributed? No autocorrelation? Constant variance?
png(file.path(out_dir, "06_arima_residuals.png"),
    width = 11, height = 8, units = "in", res = 150)
forecast::checkresiduals(arima_model, plot = TRUE)
dev.off()
cat("ARIMA residual diagnostics saved\n")

# Ljung-Box test on residuals
# H0: residuals are white noise (no autocorrelation)
# p > 0.05 = good — residuals are random
lb_test <- Box.test(residuals(arima_model), lag = 12, type = "Ljung-Box")
cat("Ljung-Box test p-value:", round(lb_test$p.value, 4))
if (lb_test$p.value > 0.05) {
  cat(" — Residuals are white noise (good model fit)\n")
} else {
  cat(" — Residuals have remaining autocorrelation (model may be missing patterns)\n")
}


# ============================================================ #
# SECTION 6: EXPONENTIAL SMOOTHING (ETS)
# ============================================================ #
#
# WHAT IS ETS?
#   ETS = Error + Trend + Seasonal. It's a "state space" model
#   that decomposes the series into these three components and
#   updates them over time using smoothing parameters.
#
# HOW IT WORKS (simplified):
#   Each new observation updates three estimates:
#     Level: the current "average" (weighted recent values more)
#     Trend: the current "slope" (is it going up or down?)
#     Seasonal: the current seasonal pattern
#   Each has a smoothing parameter (α, β, γ) that controls
#   how fast the estimate adapts to new data.
#     α close to 1 = react quickly to changes (noisy)
#     α close to 0 = smooth, slow to adapt (may miss shifts)
#
# ETS vs ARIMA:
#   - ETS: explicitly models error/trend/seasonality components
#   - ARIMA: models correlations between past values and errors
#   - Often give similar forecasts, but ETS is more intuitive
#   - ETS can't handle some patterns ARIMA can (and vice versa)
#   - In practice: try both, pick the one with better accuracy
#
# NOTATION: ETS(Error, Trend, Seasonal)
#   Error:   A (additive) or M (multiplicative)
#   Trend:   N (none), A (additive), Ad (damped additive)
#   Seasonal: N (none), A (additive), M (multiplicative)
#   Example: ETS(M,Ad,A) = multiplicative errors, damped trend, additive seasonality
# ============================================================ #

# ---- 6a: Fit ETS ----
ets_model <- forecast::ets(hosp_train)

cat("\nETS Model Selected:\n")
print(ets_model)
cat("AICc:", round(ets_model$aicc, 2), "\n")


# ---- 6b: ETS Forecast ----
ets_fc <- forecast::forecast(ets_model, h = test_length)

ets_forecast_df <- tibble(
  date     = seq(as.Date("2023-01-01"), as.Date("2023-12-01"), by = "month"),
  actual   = as.numeric(hosp_test),
  forecast = as.numeric(ets_fc$mean),
  lo95     = as.numeric(ets_fc$lower[, 2]),
  hi95     = as.numeric(ets_fc$upper[, 2])
)

p_ets <- ggplot() +
  geom_line(data = train_df_plot, aes(x = date, y = value),
            color = "grey40", linewidth = 0.7) +
  geom_ribbon(data = ets_forecast_df,
              aes(x = date, ymin = lo95, ymax = hi95),
              fill = "#43A047", alpha = 0.2) +
  geom_line(data = ets_forecast_df, aes(x = date, y = forecast),
            color = "#43A047", linewidth = 1) +
  geom_line(data = ets_forecast_df, aes(x = date, y = actual),
            color = "#E53935", linewidth = 0.9, linetype = "dashed") +
  geom_point(data = ets_forecast_df, aes(x = date, y = actual),
             color = "#E53935", size = 2) +
  labs(
    title    = paste0("ETS Forecast — ", ets_model$method),
    subtitle = "Grey = training data. Green line = forecast. Red dashed = actual.\nShaded band = 95% prediction interval.",
    x = "Date", y = "Hospitalizations"
  ) +
  theme_ml

print(p_ets)
ggsave(file.path(out_dir, "07_ets_forecast.png"), p_ets,
       width = 12, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: MODEL COMPARISON
# ============================================================ #

# ---- 7a: Accuracy Metrics ----
# RMSE: Root Mean Squared Error — in the same units as the data.
#   Lower = better. Penalizes large errors more than small ones.
# MAE: Mean Absolute Error — average absolute mistake.
#   Lower = better. More robust to outliers than RMSE.
# MAPE: Mean Absolute Percentage Error — error as % of actual.
#   Lower = better. Scale-independent (useful for comparison).

arima_acc <- forecast::accuracy(arima_fc, hosp_test)
ets_acc   <- forecast::accuracy(ets_fc, hosp_test)

# Extract test set metrics
acc_compare <- tibble(
  model     = c("ARIMA", "ETS"),
  rmse      = c(arima_acc["Test set", "RMSE"], ets_acc["Test set", "RMSE"]),
  mae       = c(arima_acc["Test set", "MAE"], ets_acc["Test set", "MAE"]),
  mape      = c(arima_acc["Test set", "MAPE"], ets_acc["Test set", "MAPE"])
)

cat("\nForecast Accuracy on Test Set (12 months):\n")
print(as.data.frame(acc_compare))


# ---- 7b: Side-by-Side Forecast Plot ----
both_fc <- dplyr::bind_rows(
  arima_forecast_df %>% dplyr::mutate(model = "ARIMA") %>%
    dplyr::select(date, actual, forecast, model),
  ets_forecast_df %>% dplyr::mutate(model = "ETS") %>%
    dplyr::select(date, actual, forecast, model)
)

p_both <- ggplot() +
  geom_line(data = train_df_plot, aes(x = date, y = value),
            color = "grey60", linewidth = 0.6) +
  geom_line(data = both_fc, aes(x = date, y = forecast, color = model),
            linewidth = 1) +
  geom_line(data = arima_forecast_df, aes(x = date, y = actual),
            color = "black", linewidth = 0.9, linetype = "dashed") +
  geom_point(data = arima_forecast_df, aes(x = date, y = actual),
             color = "black", size = 2) +
  scale_color_manual(values = c("ARIMA" = "#1565C0", "ETS" = "#43A047")) +
  labs(
    title    = "Forecast Comparison — ARIMA vs ETS",
    subtitle = "Black dashed = actual values. Colored lines = forecasts.\nThe model tracking closer to the dashed line is more accurate.",
    x = "Date", y = "Hospitalizations", color = "Model"
  ) +
  theme_ml

print(p_both)
ggsave(file.path(out_dir, "08_forecast_comparison.png"), p_both,
       width = 12, height = 7, dpi = 150, bg = "white")


# ---- 7c: Accuracy Bar Chart ----
acc_long <- acc_compare %>%
  tidyr::pivot_longer(-model, names_to = "metric", values_to = "value") %>%
  dplyr::mutate(metric = toupper(metric))

p_acc <- ggplot(acc_long, aes(x = metric, y = value, fill = model)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_text(aes(label = round(value, 1)),
            position = position_dodge(width = 0.6), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("ARIMA" = "#1565C0", "ETS" = "#43A047")) +
  facet_wrap(~metric, scales = "free_y", ncol = 3) +
  labs(
    title    = "Forecast Accuracy — ARIMA vs ETS",
    subtitle = "Lower is better for all three metrics.\nRMSE and MAE are in hospitalization units. MAPE is a percentage.",
    x = NULL, y = "Value", fill = "Model"
  ) +
  theme_ml +
  theme(strip.text = element_text(face = "bold"))

print(p_acc)
ggsave(file.path(out_dir, "09_accuracy_comparison.png"), p_acc,
       width = 12, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: NAIVE BENCHMARK & CROSS-VALIDATION
# ============================================================ #
# A forecast is only good if it beats a NAIVE baseline.
# Seasonal naive: "forecast = same month last year"
# If our models can't beat this, they're useless.

# ---- 8a: Seasonal Naive Forecast ----
snaive_fc <- forecast::snaive(hosp_train, h = test_length)
snaive_acc <- forecast::accuracy(snaive_fc, hosp_test)

cat("\nSeasonal Naive RMSE:", round(snaive_acc["Test set", "RMSE"], 1), "\n")
cat("ARIMA RMSE:", round(acc_compare$rmse[1], 1), "\n")
cat("ETS RMSE:", round(acc_compare$rmse[2], 1), "\n")

# Improvement over naive
arima_improvement <- round((1 - acc_compare$rmse[1] / snaive_acc["Test set", "RMSE"]) * 100, 1)
ets_improvement   <- round((1 - acc_compare$rmse[2] / snaive_acc["Test set", "RMSE"]) * 100, 1)

cat("ARIMA improvement over naive:", arima_improvement, "%\n")
cat("ETS improvement over naive:", ets_improvement, "%\n")


# ---- 8b: Time Series Cross-Validation ----
# CRITICAL: Regular K-fold CV doesn't work for time series because
# it would use future data to predict the past (data leakage).
#
# TIME SERIES CV uses an EXPANDING WINDOW:
#   Fold 1: Train on months 1-36, test on month 37
#   Fold 2: Train on months 1-37, test on month 38
#   Fold 3: Train on months 1-38, test on month 39
#   ...
#   Each fold adds one more training month, always predicts
#   the NEXT unseen month. This respects temporal order.

# Use tsCV from forecast package for 1-step-ahead cross-validation
arima_cv_errors <- forecast::tsCV(hosp_ts, forecastfunction = function(y, h) {
  forecast::forecast(forecast::auto.arima(y, seasonal = TRUE), h = h)
}, h = 1, window = 36)  # Minimum training window of 36 months

ets_cv_errors <- forecast::tsCV(hosp_ts, forecastfunction = function(y, h) {
  forecast::forecast(forecast::ets(y), h = h)
}, h = 1, window = 36)

# Compute CV RMSE (ignoring NAs from the initial window)
arima_cv_rmse <- round(sqrt(mean(arima_cv_errors^2, na.rm = TRUE)), 2)
ets_cv_rmse   <- round(sqrt(mean(ets_cv_errors^2, na.rm = TRUE)), 2)

cat("\nTime Series Cross-Validation RMSE (1-step ahead):\n")
cat("  ARIMA:", arima_cv_rmse, "\n")
cat("  ETS:  ", ets_cv_rmse, "\n")


# ============================================================ #
# SECTION 9: HEALTHCARE COST FORECAST (SECOND SERIES)
# ============================================================ #
# Apply what we've learned to a second series — healthcare costs.
# This shows the same workflow generalizes to different metrics.

cost_ts    <- ts(ts_data$cost_thousands, start = c(2019, 1), frequency = 12)
cost_train <- window(cost_ts, end = c(2022, 12))
cost_test  <- window(cost_ts, start = c(2023, 1))

# Fit both models
cost_arima <- forecast::auto.arima(cost_train, seasonal = TRUE)
cost_ets   <- forecast::ets(cost_train)

cost_arima_fc <- forecast::forecast(cost_arima, h = 12)
cost_ets_fc   <- forecast::forecast(cost_ets, h = 12)

cost_fc_df <- tibble(
  date     = seq(as.Date("2023-01-01"), as.Date("2023-12-01"), by = "month"),
  actual   = as.numeric(cost_test),
  arima    = as.numeric(cost_arima_fc$mean),
  ets      = as.numeric(cost_ets_fc$mean)
) %>%
  tidyr::pivot_longer(c(arima, ets), names_to = "model", values_to = "forecast") %>%
  dplyr::mutate(model = toupper(model))

cost_train_plot <- tibble(
  date  = seq(as.Date("2019-01-01"), as.Date("2022-12-01"), by = "month"),
  value = as.numeric(cost_train)
)

p_cost <- ggplot() +
  geom_line(data = cost_train_plot, aes(x = date, y = value),
            color = "grey60", linewidth = 0.6) +
  geom_line(data = cost_fc_df, aes(x = date, y = forecast, color = model),
            linewidth = 1) +
  geom_line(data = cost_fc_df %>% dplyr::filter(model == "ARIMA"),
            aes(x = date, y = actual),
            color = "black", linewidth = 0.9, linetype = "dashed") +
  scale_color_manual(values = c("ARIMA" = "#1565C0", "ETS" = "#43A047")) +
  labs(
    title    = "Healthcare Cost Forecast ($thousands/month)",
    subtitle = "Same methods applied to a different series. Black dashed = actual.",
    x = "Date", y = "Cost ($thousands)", color = "Model"
  ) +
  theme_ml

print(p_cost)
ggsave(file.path(out_dir, "10_cost_forecast.png"), p_cost,
       width = 12, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 10: SAVE RESULTS
# ============================================================ #

results_ts <- list(
  ts_data       = ts_data,
  arima_model   = arima_model,
  ets_model     = ets_model,
  accuracy      = acc_compare,
  arima_cv_rmse = arima_cv_rmse,
  ets_cv_rmse   = ets_cv_rmse
)

saveRDS(results_ts, file.path(out_dir, "results_time_series.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 11: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== TIME SERIES ANALYSIS — SUMMARY ===\n")
cat("Series: 60 months of monthly hospitalizations\n")
cat("Train: 48 months | Test: 12 months\n")
cat("ARIMA selected:", as.character(arima_model), "\n")
cat("ETS selected:", ets_model$method, "\n")
cat("Test RMSE — ARIMA:", round(acc_compare$rmse[1], 1),
    "| ETS:", round(acc_compare$rmse[2], 1),
    "| Naive:", round(snaive_acc["Test set", "RMSE"], 1), "\n")

cat("\nWHAT WE LEARNED:\n")
cat("1.  Time series data has TEMPORAL ORDER — you can't shuffle rows.\n")
cat("    Past values predict future values (autocorrelation).\n")
cat("2.  Three key components: TREND (direction), SEASONALITY (repeating\n")
cat("    pattern), and NOISE (random variation). STL separates them.\n")
cat("3.  STATIONARITY means the series' statistics don't change over time.\n")
cat("    Most models require it. Differencing removes trends.\n")
cat("4.  ARIMA models autocorrelation: current values depend on past values\n")
cat("    (AR) and past errors (MA). auto.arima() picks the best specification.\n")
cat("5.  ETS (Exponential Smoothing) decomposes into error + trend + seasonal.\n")
cat("    More intuitive than ARIMA. Often gives similar forecasts.\n")
cat("6.  Always compare to a NAIVE BENCHMARK (seasonal naive = same month\n")
cat("    last year). If your model can't beat this, it's not useful.\n")
cat("7.  Time series CV uses an EXPANDING WINDOW — never train on future data.\n")
cat("    Regular K-fold CV would leak future information into the training set.\n")
cat("8.  PREDICTION INTERVALS widen over time — forecasts become less\n")
cat("    certain the further ahead you predict. This is fundamental.\n")
cat("9.  The same workflow applies to any metric: hospitalizations, costs,\n")
cat("    patient volumes, lab values tracked over time.\n")

cat("\nWHEN TO USE TIME SERIES MODELS:\n")
cat("  ARIMA:  When you need interpretable parameters and prediction intervals\n")
cat("  ETS:    When you want an intuitive decomposition approach\n")
cat("  Both:   Try both, compare accuracy, pick the winner\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Scripts 01-06 (Classification): Predict per-patient outcomes.\n")
cat("    Time series predicts AGGREGATE outcomes over time.\n")
cat("  - Script 14 (Deep Learning): LSTM networks are a deep learning\n")
cat("    approach to time series — they learn temporal patterns via\n")
cat("    recurrent connections. Covered conceptually there.\n")
cat("  - Script 07 (Unsupervised): STL decomposition is a form of\n")
cat("    unsupervised structure discovery applied to temporal data.\n")
cat("===================================\n")

cat("\nScript 10 complete.\n")