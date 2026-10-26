# Learning Machine Learning with Claude

Domains: LearningResource, Software
Project Type: Project
pin: Yes

# ML TEACHING PROJECT — FULL SPECIFICATION

## GOAL

Build a **complete machine learning and modern AI teaching system in R** — one standalone script per model family, all running on the same synthetic clinical dataset. The system teaches me ML through **code, visualization, and plain-language explanation**. I learn by running each script top-to-bottom, seeing results, and reading the embedded explanations.

The project now covers the **full AI landscape**: from classical tabular ML through deep learning, NLP, generative AI, and agentic systems. Scripts 00–15 use R. Scripts 16–20 (NLP, Generative AI, Embeddings, Agents) use Python Jupyter notebooks (`.ipynb`) — these are the only exceptions, because the modern AI ecosystem lives in Python. Each notebook is self-contained and fully annotated.

---

## MY BACKGROUND

- **Intermediate-advanced R user**: comfortable with tidyverse, survey/health data, joins, grouping, functional programming.
- **I understand basic regression models - linear, logistic, multivariate** (coefficients, odds ratios, log-odds). That's my bridge into ML.
- **I know NOTHING about machine learning** beyond that. Every model needs first-principles teaching — what it does, how it learns, when it works, when it fails. No assumed ML knowledge.
- **Visual and tactile learner**: I learn by seeing models run on real data. Plots with annotated interpretations teach me more than paragraphs of theory.
- **Python preference**: I use Jupyter notebooks (`.ipynb`), not `.py` scripts. All packages are installed into a **venv kernel**, never global Python. I prefer a **5–6 cell structure** (not one massive block, not 50 tiny ones) — see Python notebook conventions below.

---

## DATASET

A **synthetic population health dataset** with 25,000 rows, saved at `Data/ml_teaching_data.rds`. Generated with `set.seed(42)` using layered causal structure:

**Variables:**

- **Demographics:** age, sex, race, education, income
- **Behavioral:** smoking_status, alcohol_use, physical_activity
- **Clinical:** bmi, sbp, dbp, cholesterol, fasting_glucose
- **Diseases:** diabetes, heart_disease, cancer
- **Medical:** medication_count, on_medication
- **Outcomes:** mortality, hospitalizations, healthcare_cost
- **ID:** id

**Key facts about the data:**

- **Heart disease prevalence is ~6%** — this is our primary prediction target. The low prevalence affects threshold selection, pruning, and class weighting decisions. Don't ignore this.
- Age is the strongest predictor for all outcomes
- BMI drives diabetes risk; smoking drives cancer and heart disease
- Diseases compound healthcare cost exponentially
- Relationships were built in via logistic/linear equations with noise, so ML models will find real signal
- Structure of the data:

```
ml_data <- readRDS("Data/ml_teaching_data.rds")
> str(ml_data)
tibble [25,000 × 22] (S3: tbl_df/tbl/data.frame)
 $ id               : int [1:25000] 1 2 3 4 5 6 7 8 9 10 ...
 $ age              : num [1:25000] 77 42 59 63 59 50 79 50 88 51 ...
 $ sex              : Factor w/ 2 levels "Female","Male": 1 1 1 1 1 1 1 2 2 1 ...
 $ race             : Factor w/ 5 levels "Asian","Black",..: 5 5 5 4 3 2 1 5 5 2 ...
 $ education        : Ord.factor w/ 5 levels "Less than HS"<..: 2 4 3 3 5 2 4 3 3 1 ...
 $ income           : num [1:25000] 25419 51110 23975 42732 68122 ...
 $ smoking_status   : Ord.factor w/ 3 levels "Never"<"Former"<..: 3 1 1 1 1 1 1 1 2 2 ...
 $ alcohol_use      : Ord.factor w/ 4 levels "None"<"Light"<..: 1 1 1 3 1 1 3 1 2 1 ...
 $ physical_activity: num [1:25000] 1.2 1.1 1 2.3 5 6.7 6.8 5.2 1.3 4.5 ...
 $ bmi              : num [1:25000] 21.1 29.7 21 21.5 25.7 26.5 34.1 29.4 41.7 35.7 ...
 $ sbp              : num [1:25000] 106 112 100 130 112 111 111 104 123 137 ...
 $ dbp              : num [1:25000] 55 50 58 63 54 53 53 56 50 62 ...
 $ cholesterol      : num [1:25000] 222 160 151 199 165 165 231 190 254 197 ...
 $ fasting_glucose  : num [1:25000] 87 74 81 77 60 91 88 92 73 85 ...
 $ diabetes         : int [1:25000] 0 0 0 0 0 0 0 0 0 1 ...
 $ heart_disease    : int [1:25000] 0 0 0 0 0 0 0 0 1 0 ...
 $ cancer           : int [1:25000] 1 0 0 0 0 0 0 0 0 0 ...
 $ medication_count : int [1:25000] 5 0 5 1 3 4 0 0 3 9 ...
 $ on_medication    : int [1:25000] 1 0 1 1 1 1 0 0 1 1 ...
 $ mortality        : int [1:25000] 0 0 0 0 0 0 0 0 1 0 ...
 $ hospitalizations : int [1:25000] 2 2 0 2 0 0 0 1 0 2 ...
 $ healthcare_cost  : num [1:25000] 75029 4240 11480 8730 17741 ...

```

**NOTE ON DATASET EXTENSIONS:** Some advanced scripts (survival analysis, time series, deep learning concepts) will need to generate their own supplementary synthetic data within the script. This is fine — the script explains what it's generating and why. The core `ml_teaching_data.rds` remains the anchor dataset.

## PRIMARY TARGET

All classification scripts predict **heart_disease** (binary 0/1) as the running example.

**Feature set for heart disease prediction:**

```r
target <- "heart_disease"

features <- c(
  "age", "sex", "race", "education", "income",
  "smoking_status", "alcohol_use", "physical_activity",
  "bmi", "sbp", "dbp", "cholesterol", "fasting_glucose"
)
```

---

## SCRIPT PLAN

Each script is **fully standalone** — loads raw data, does its own prep, trains, evaluates, visualizes, explains. No shared pipeline. No dependency on other scripts except the data file (and optionally loading previous model results for ROC comparison).

The syllabus is organized in 6 phases:

### PHASE 1 — Foundations & Data

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 00 | `00. CreateDummyData.R` | R | — | Synthetic data with causal layers |
| 00 | `00. ExploratoryAnalysis.R` | R | — | Distributions, correlations, clinical relationships |

### PHASE 2 — Core Supervised Learning

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 01 | `01. Logistic Regression.R` | R | Logistic Regression | Baseline model, evaluation framework (ROC, confusion matrix, CV) |
| 02 | `02. Decision Trees.R` | R | Decision Trees | Yes/no splits, Gini impurity, pruning, overfitting visual |
| 03 | `03. Random Forest.R` | R | Random Forest | Bagging, feature randomization, OOB error, variance reduction |
| 04 | `04. GradientBoosting_XGBoost.R` | R | GBM + XGBoost | Sequential correction, learning rate, early stopping |
| 05 | `05. KNN_SVM.R` | R | KNN + SVM | Distance-based, margin-based, why scaling matters |
| 06 | `06. Neural Network.R` | R | Neural Networks | Layers, activations, backpropagation intuition |

### PHASE 3 — Unsupervised & Specialized Methods

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 07 | `07. Unsupervised.R` | R | K-means, DBSCAN, Hierarchical + PCA, t-SNE, UMAP | No target variable, finding structure, dimensionality reduction |
| 08 | `08. Isolation Forest.R` | R | Isolation Forest | Anomaly detection — different objective than classification |
| 09 | `09. Survival ML.R` | R | Cox PH, Random Survival Forests, Survival GBM | Time-to-event outcomes (mortality as survival), censoring, hazard ratios. Generates supplementary time-to-event data within script. |
| 10 | `10. Time Series.R` | R | ARIMA, state space, LSTM concepts | Forecasting hospitalizations over time. Generates monthly aggregate time series from the main dataset. |

### PHASE 4 — Advanced & Applied Methods

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 11 | `11. Model Interpretation.R` | R | SHAP, LIME, PDP, Calibration, Decision Curves | Model-agnostic interpretation. Trains its own XGBoost internally, then dissects it. |
| 12 | `12. Ensemble_SuperLearner.R` | R | Super Learner, Stacking, AutoML concepts | Combining models optimally. The "best of everything" approach. |
| 13 | `13. Causal ML.R` | R | Causal Forests, Double ML, TMLE concepts | Treatment effects, not just prediction. "Does smoking CAUSE heart disease, or just predict it?" |
| 14 | `14. Deep Learning Concepts.R` | R | CNN, RNN, Transformer intuition | Architecture tour — what each does, when to use, visual explanations. Uses simple demonstrations, not full training. |
| 15 | `15. Model Comparison.R` | R | — | Compare all models side-by-side: performance table, ROC overlay, strengths/weaknesses |

### PHASE 5 — Natural Language Processing & Generative AI *(Python)*

> **Why Python?** The modern NLP and generative AI ecosystem (Hugging Face, LangChain, OpenAI SDK, sentence-transformers) is Python-native. These tools either don't exist in R or are thin wrappers. Scripts 16–19 are Python (.py) files — fully standalone, heavily annotated, run top-to-bottom in any Python 3.10+ environment with the listed packages.

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 16 | `16. NLP_Health_Text.ipynb` | Python | Tokenization, TF-IDF, BERT embeddings, text classification | What NLP is, why text needs different preprocessing than tabular data. Generates synthetic clinical notes from our dataset patients and classifies them. |
| 17 | `17. Embeddings_and_Semantic_Search.ipynb` | Python | Sentence embeddings, cosine similarity, vector search, RAG concepts | What an embedding is (turning words into numbers that capture meaning). Builds a simple semantic search engine over synthetic clinical notes. |
| 18 | `18. Generative_AI_Concepts.ipynb` | Python | GPT API, prompt engineering, temperature, tokens, fine-tuning concepts | How LLMs generate text. What tokens are. What prompt engineering is and why it matters. Uses the OpenAI API to demonstrate live generation with the clinical use case. |
| 19 | `19. Synthetic_Data_Generation.ipynb` | Python | GANs (concept), CTGAN, Gaussian copula, SDV library | How to generate realistic synthetic tabular data. Why this matters for healthcare privacy. Generates a new synthetic version of our dataset using learned distributions. GPU-accelerated on RTX 4060. |

### PHASE 6 — Agentic AI *(Python)*

> **What is Agentic AI?** Agentic AI is not a machine learning algorithm — it's a **system design paradigm**. An agent combines a language model, tools, memory, and a reasoning loop. The model decides which tools to call, observes the results, updates its plan, and repeats. This is closer to software architecture than statistics. Script 20 teaches the concepts and demonstrates a working agent that can analyze our clinical dataset.

| # | Script | Language | Model Family | Key Teaching Points |
| --- | --- | --- | --- | --- |
| 20 | `20. Agentic_AI.ipynb` | Python | ReAct agents, tool use, planning loops, LangChain/OpenAI Agents | What an agent is, how the reasoning loop works, tool definitions, memory types (in-context vs persistent). Demonstrates an agent that can query the dataset, run analyses, and explain results. |

---

## WHERE THIS CURRICULUM FITS IN THE AI WORLD

```
Artificial Intelligence
│
├── Machine Learning  ← PHASES 1–4 (Scripts 00–15, R)
│   ├── Supervised learning      (Scripts 01–06)
│   ├── Unsupervised learning    (Script 07)
│   ├── Anomaly detection        (Script 08)
│   ├── Survival ML              (Script 09)
│   ├── Time series              (Script 10)
│   ├── Model interpretation     (Script 11)
│   ├── Ensemble / AutoML        (Script 12)
│   └── Causal ML                (Script 13)
│
├── Deep Learning
│   ├── CNN / RNN / Transformers (Script 14 — concepts in R)
│   └── Full training            (covered conceptually; production use via APIs)
│
├── Natural Language Processing  ← PHASE 5 (Scripts 16–17, Python)
│   ├── Text classification
│   ├── Embeddings
│   └── Semantic search / RAG
│
├── Generative AI                ← PHASE 5 (Scripts 18–19, Python)
│   ├── LLMs (GPT)
│   ├── Prompt engineering
│   └── Synthetic data generation
│
└── Agent Systems                ← PHASE 6 (Script 20, Python)
    ├── Tool use
    ├── Planning loops
    └── LLM orchestration
```

**Coverage after this curriculum:**

| Area | Coverage |
| --- | --- |
| Machine learning fundamentals | 95% |
| Tabular data ML | 100% |
| Clinical ML methods | 90% |
| Deep learning (concepts) | 70% |
| NLP fundamentals | 80% |
| Embeddings & semantic search | 75% |
| Generative AI (concepts + API use) | 70% |
| Synthetic data generation | 80% |
| Agentic AI | 65% |

---

## NOTES ON ADVANCED SCRIPTS

**Scripts 09–10 (Survival, Time Series):** These need different outcome structures than our main dataset. Each script will generate its own supplementary synthetic data at the top, explain what it's creating and why, then proceed with the same teach-train-evaluate-visualize pattern. The main `ml_teaching_data.rds` is still loaded and used as the base for generating the extended data.

**Script 11 (Model Interpretation):** Trains its OWN XGBoost model internally (no dependency on script 04), then applies SHAP, LIME, partial dependence, calibration curves, and decision curve analysis to it. Teaches HOW to interpret any black-box model.

**Script 14 (Deep Learning Concepts):** Our tabular dataset isn't ideal for CNNs or RNNs. This script is more of an **architecture tour** — explaining what each architecture does, when you'd use it, with small toy demonstrations and visual diagrams. It focuses on intuition and "when would I use this?" rather than full model training.

**Script 13 (Causal ML):** This is conceptually different from prediction. The script explicitly teaches the distinction: "ML predicts outcomes; Causal ML estimates what CAUSES outcomes." Uses our dataset to estimate the causal effect of smoking on heart disease vs just the predictive association.

**Script 16 (NLP):** Generates a synthetic clinical notes dataset from our existing patient records — e.g. "Patient is a 67-year-old male smoker with elevated cholesterol presenting with chest pain." Applies tokenization, TF-IDF, then BERT embeddings to classify notes as high/low risk. Teaches why text needs fundamentally different preprocessing from tabular data.

**Script 17 (Embeddings):** Teaches what an embedding actually is: a vector of numbers that encodes meaning, so that "myocardial infarction" and "heart attack" are close together in vector space. Builds a simple semantic search engine over the synthetic clinical notes. Introduces RAG (Retrieval-Augmented Generation) as a concept — how agents use semantic search to find relevant context before generating a response.

**Script 18 (Generative AI):** Requires an OpenAI API key (or Anthropic key — the script explains how to use either). Does NOT train any model. Instead, it teaches: what tokens are, how autoregressive generation works, what temperature does, and how to write effective prompts for clinical tasks. Demonstrates prompt engineering patterns (zero-shot, few-shot, chain-of-thought) using our clinical dataset as context.

**Script 19 (Synthetic Data):** Uses the `sdv` Python library (CTGAN or Gaussian copula) to learn the statistical distribution of our dataset and generate a new synthetic version that preserves relationships but contains no real records. Teaches why this matters for health data privacy (HIPAA, research sharing). Compares the generated data to the original using statistical tests.

**Script 20 (Agentic AI):** Demonstrates a working ReAct-pattern agent (Reason → Act → Observe → Repeat). The agent has three tools: (1) run a SQL query on the clinical dataset, (2) call a Python function to train a quick logistic regression, (3) generate a text summary. A human-readable trace shows every step: what the model decided, what tool it called, what it received back, and how it updated its plan. Teaches that agents are LLMs + tools + memory + loops — not a new kind of model.

---

## STANDALONE TEACHING PRINCIPLE — CRITICAL

**Every script must work as a complete, self-contained learning resource.** A reader should be able to pick up ANY script — without having read any other script — and fully understand everything in it.

This means:

### 1. Re-explain foundational ML concepts in every script where they appear

Do NOT say "as we learned in the Decision Trees script..." or "recall from the logistic regression script that..."

Instead, re-explain the concept fresh each time. For example:
- **Train/test split**: Explain WHY we hold out data (to simulate new patients) in EVERY script, not just the first one.
- **Overfitting**: Define it from scratch whenever it's relevant: "The model memorized the training data instead of learning general patterns."
- **ROC curve / AUC**: Briefly explain what ROC measures and what AUC values mean in every script that shows one.
- **Confusion matrix**: Explain TP/FP/TN/FN and what precision/recall mean each time.
- **Cross-validation**: Explain the fold concept each time it's used.
- **Feature importance**: Explain what it means conceptually before showing the model-specific implementation.

### 2. Each explanation should feel native to that script

Don't copy-paste the same paragraph into every script. Tailor the explanation to the model being taught. For example:
- In the Random Forest script, explain overfitting in the context of "one tree overfits, but 500 averaged trees don't."
- In the KNN script, explain scaling in the context of "distance measurement treats all features equally, so larger-scaled features dominate."
- In the Neural Network script, explain train/test in the context of "the network will memorize if trained too long — we need unseen data to know when to stop."

### 3. Comparison to other models is welcome, but should be self-explanatory

It's fine (and encouraged) to say things like: "Unlike logistic regression, which draws a straight decision boundary, a Random Forest creates complex nonlinear boundaries by averaging many trees."

But the reader should understand this comparison even if they've never seen the logistic regression script. Frame comparisons so the OTHER model is briefly described in the comparison itself.

### 4. ROC overlays with previous models are optional visual bonuses

The pattern of loading previous model results (`has_logistic <- file.exists(...)`) is a nice-to-have visual comparison. But the script must be fully interpretable even if those files don't exist. Always wrap these in `if (has_logistic) { ... }` blocks and never rely on them for the narrative.

---

## SCRIPT STRUCTURE — R SCRIPTS (00–15)

```
SECTION 1: WHAT IS [MODEL]?
  → Plain-language analogy (doctor, voting, etc.)
  → How the algorithm works step by step
  → When it works well / when it struggles
  → How it differs from other approaches (briefly, self-contained)

SECTION 2: DATA PREPARATION
  → Load data, define features/target
  → Train/test split (explain WHY — don't assume prior knowledge)
  → Preprocessing specific to THIS model (explain WHY this model needs it)

SECTION 3+: FIT THE MODEL
  → Train the model, show key output
  → Explain hyperparameters in plain language

EVALUATION SECTIONS:
  → ROC curve (explain what ROC/AUC means, overlay previous models if available)
  → Confusion matrix (explain TP/FP/TN/FN, cells labeled with plain-English meaning)
  → Predicted probability distribution
  → Overfitting check (explain the concept, show train vs test)
  → Cross-validation (explain the fold concept, show results)

INTERPRETATION SECTIONS:
  → Feature importance (explain the concept, then model-specific method)
  → Partial dependence / marginal effects where applicable
  → Model-specific visualizations (tree plots, boundaries, etc.)

SAVE RESULTS:
  → results_[model].rds + test_predictions_[model].rds

SUMMARY & TAKEAWAYS:
  → What we learned (numbered list)
  → How this model compares to others (self-contained description)
  → What kinds of problems this model is best for
```

## SCRIPT STRUCTURE — PYTHON SCRIPTS (16–20)

```
SECTION 1: WHAT IS THIS TECHNIQUE?
  → Plain-language analogy
  → How it works conceptually (before any code)
  → Where it fits in the AI landscape
  → Why Python for this (not R) — briefly explain ecosystem reason

SECTION 2: SETUP & DATA
  → Package imports with one-line comment per package explaining what it does
  → Load or generate dataset (explain any generation clearly)
  → Show data structure with print statements

SECTION 3+: IMPLEMENT THE TECHNIQUE
  → Step-by-step implementation
  → Every non-obvious line gets a comment explaining WHY, not just WHAT
  → Explain all hyperparameters in plain language

EVALUATION / DEMONSTRATION:
  → Show outputs in a way that teaches what to look for
  → Visualize where possible (matplotlib/seaborn)
  → Annotated interpretations in comments alongside plots

SUMMARY & TAKEAWAYS:
  → What we learned (numbered list as comments)
  → How this connects to the R scripts (tabular ML → NLP → GenAI → Agents)
  → What kinds of real-world problems this technique addresses
```

---

## CONVENTIONS — R SCRIPTS (FOLLOW EXACTLY)

### Script Header

```r
# ============================================================ #
# XX — Script Title
# ============================================================ #
# Purpose: ...
# Target: heart_disease (binary: 0/1)
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()
```

### Packages & Working Directory

```r
# ---- Packages ----
pacman::p_load(...)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/XX_Model_Name", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/XX_Model_Name"
```

### Global Plot Theme

```r
theme_ml <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "grey40", size = 11),
    legend.position = "bottom"
  )
```

### For patchwork `plot_annotation()` — MUST use a raw theme() object:

```r
# plot_annotation() throws warnings if you pass theme_ml (which is theme_minimal + theme).
# Always use a separate raw theme() for annotations:
theme_annotation <- theme(
  plot.title    = element_text(face = "bold", size = 14),
  plot.subtitle = element_text(color = "grey40", size = 11)
)
```

### Data Loading

```r
ml_data <- readRDS("Data/ml_teaching_data.rds")
```

### Train/Test Split — same everywhere for comparability

```r
set.seed(42)
train_index <- caret::createDataPartition(ml_data[[target]], p = 0.75, list = FALSE)
train_df <- ml_data[train_index[, 1], ]
test_df  <- ml_data[-train_index[, 1], ]
```

### Factor Setup — same everywhere

```r
train_df <- train_df %>%
  dplyr::mutate(
    sex            = factor(sex),
    race           = factor(race),
    education      = factor(education,
      levels = c("Less than HS", "High School", "Some College", "Bachelor's", "Graduate")),
    smoking_status = factor(smoking_status, levels = c("Never", "Former", "Current")),
    alcohol_use    = factor(alcohol_use, levels = c("None", "Light", "Moderate", "Heavy"))
  )
# Apply identical mutation to test_df
```

### ROC Comparison — load previous models if available (OPTIONAL visual bonus)

```r
# Pattern: check if previous results exist, load them, overlay on ROC
# IMPORTANT: Script must be fully interpretable even if these files don't exist
logistic_results_path <- "Outputs/Logistic_Regression/results_logistic.rds"
has_logistic <- file.exists(logistic_results_path)
if (has_logistic) {
  logistic_results <- readRDS(logistic_results_path)
  logistic_test_probs <- readRDS("Outputs/Logistic_Regression/test_predictions_logistic.rds")
  # ... overlay on ROC plot
}
```

### Results Save Format — standardized for model comparison script

```r
results_[model] <- tibble(
  model          = "Model Name",
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
saveRDS(results_[model], file.path(out_dir, "results_[model].rds"))
saveRDS(test_probs, file.path(out_dir, "test_predictions_[model].rds"))
```

### Threshold Selection

```r
# Use Youden's J from pROC:
optimal_coords <- pROC::coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
optimal_threshold <- optimal_coords$threshold
```

### Confusion Matrix — always label cells with plain-English meaning

```r
# Every confusion matrix plot should label cells:
# "True Negative (correctly ruled out)"
# "False Positive (false alarm)"
# "False Negative (MISSED case)"
# "True Positive (correctly caught)"
```

### Plot Saving

```r
ggsave(file.path(out_dir, "XX_plot_name.png"), plot_object,
       width = 10, height = 7, dpi = 150, bg = "white")
```

### Namespace Masking — always namespace these functions

```r
dplyr::select(), dplyr::filter(), dplyr::mutate(),
dplyr::summarise(), dplyr::rename(), dplyr::n()
```

---

## CONVENTIONS — PYTHON NOTEBOOKS (16–20)

### Format: Jupyter Notebook (`.ipynb`), not `.py` scripts

All Python scripts in this project are Jupyter notebooks. They run in VS Code with a **venv kernel** (never global Python). The kernel is named `.venv` and is selected at the top-right of the notebook in VS Code.

### Notebook Cell Structure — 5–6 cells per notebook

Each notebook uses **5–6 cells**, not one massive block and not 50 tiny ones. Cells are split at natural "restart points" — places where you'd hate to re-run everything above just because you tweaked something below. This is the direct equivalent of `# ---- Section ----` headers in R.

| Cell | Contents | Equivalent in R |
| --- | --- | --- |
| **Cell 1** | `pip install` commands (commented out after first run) | `install.packages()` — run once |
| **Cell 2** | All imports + constants + config + output directory setup | Top of R script: library(), constants, out_dir |
| **Cell 3** | Data loading + preprocessing | `# ---- Data Prep ----` section |
| **Cell 4** | Main work — model training, API calls, analysis | `# ---- Fit Model ----` sections |
| **Cell 5** | Outputs, plots, saving results | `# ---- Visualize ----` + `ggsave()` sections |
| **Cell 6** | (Optional) Summary printout / "What We Learned" | End-of-script summary comments |

**Why this structure:** If your model finishes training in Cell 4 and you want to tweak a plot in Cell 5, you re-run only Cell 5. You never re-train. If an import breaks, you know immediately from Cell 2 without running anything slow. This is the same logic as sectioning an R script — just enforced by cell boundaries instead of comment headers.

### Cell 1 — pip installs (always commented after first run)

```python
# ---- Install packages (run once per project, then comment out) ----
# !pip install pandas numpy matplotlib seaborn scikit-learn pyreadr
# !pip install sentence-transformers                  # Script 17
# !pip install openai                                 # Scripts 18, 20
# !pip install sdv                                    # Script 19
# !pip install langchain langchain-openai             # Script 20
```

### Cell 2 — Notebook Header + Imports + Config

```python
# ============================================================
# XX — Script Title
# ============================================================
# Purpose: ...
# Kernel: .venv (Python 3.10+)
# GPU: RTX 4060 — CUDA acceleration enabled where available
# ============================================================

import pandas as pd          # DataFrames — like R's tibble
import numpy as np           # Numerical arrays — like R's vectors
import matplotlib.pyplot as plt  # Plotting — like R's base graphics
import seaborn as sns        # Statistical plots — like R's ggplot2
import os

# GPU detection — uses CUDA (NVIDIA) if available, falls back to CPU
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")  # Should print 'cuda' on your 4060

# Output directory
OUT_DIR = "Outputs/Script_Name"
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent color palette (matches R scripts)
BLUE  = "#2196F3"   # Negative / no disease
RED   = "#F44336"   # Positive / disease
GREEN = "#4CAF50"   # Correct prediction
```

### Data Loading

```python
# The core dataset is loaded from CSV (exported from R with write.csv)
# OR pyreadr can read .rds directly: pyreadr.read_r("Data/ml_teaching_data.rds")[None]
ml_data = pd.read_csv("Data/ml_teaching_data.csv")
```

### API Key Handling (Scripts 18, 20)

```python
# API keys are stored in a plain text file in the project folder.
# File: openai_key.txt — one line, just the key, nothing else.
# IMPORTANT: Add openai_key.txt to .gitignore so it never gets uploaded to GitHub.

def load_openai_key(path="openai_key.txt") -> str:
    """Load OpenAI API key from a text file in the project folder."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file not found: {path}\n"
            "Create openai_key.txt in your project folder and paste your API key in it."
        )

OPENAI_API_KEY = load_openai_key()
print("API key loaded")
```

**Setup (one time only):**
1. Create a file called `openai_key.txt` in your project root folder
2. Paste your API key as a single line — nothing else in the file
3. If using GitHub: add `openai_key.txt` to your `.gitignore` file so it never gets accidentally uploaded


---

## TEACHING APPROACH — DETAILED

### Each script is a STANDALONE learning resource

The reader may pick up any script without having read any other. Every foundational concept must be re-explained from scratch wherever it appears, tailored to the specific model being taught. Never say "as we covered earlier" or assume knowledge from another script. See the **STANDALONE TEACHING PRINCIPLE** section above for detailed rules.

### Explanations in comments must:

- Start with **simple analogy** before any technical explanation
- Explain **WHAT** the model does (conceptually), **HOW** it learns (mechanism), **WHEN** it works well, **WHEN** it fails
- Be **granular** — don't say "we scale the features," say "we center (subtract mean) and scale (divide by SD) so that age (18-95) and income (8000-300000) are on the same scale — otherwise income would dominate just because its numbers are bigger"
- Explain **WHY** each preprocessing step is needed for THIS specific model — "trees don't need scaling because they split on thresholds; KNN does because it measures distance"
- Re-explain foundational ML concepts (train/test, overfitting, ROC, CV, etc.) each time they appear, framed in the context of the current model

### Plots must:

- Have **informative subtitles** that tell me what to look for — not just "ROC Curve" but "Closer to top-left corner = better. Diagonal = random guessing."
- Include **annotations** on key regions where helpful (overfitting zones, optimal points, reference lines with labels)
- Use **consistent color coding**: Blue (#2196F3) = negative/no disease, Red (#F44336) = positive/disease, Green (#4CAF50) = correct, Red = incorrect
- Compare to previous models visually where files are available (overlay ROC curves), but the plot must be interpretable standalone

### Console output must:

- Be **essential validation only** — no decorative `cat("\n")` or banner boxes
- Combine related info: `cat("Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")`
- Include **plain-language interpretation** after numbers: `cat("AUC: 0.82 — Good discrimination\n")`

### Each script should end with:

- Summary of key metrics
- Self-contained comparison to other model types (briefly describe the other models in the comparison — don't assume the reader knows them)
- "WHAT WE LEARNED" — numbered list of key takeaways
- What kinds of problems this model is best suited for

## R CODE STYLE

### Commenting

- `# ---- Section Name ----` for major sections (shows in RStudio outline, always close with trailing characters)
- Comments explain **intent, method, interpretation** — not just what the code does
- No decorative comment boxes with `===` except the script header
- No `cat("\n")` alone, no decorative `cat("========")` banners

### Code clarity

- Explicit namespacing: `dplyr::filter()`, `dplyr::select()`, etc.
- Small logical steps with intermediate objects (not long pipes)
- Complete, runnable scripts — no placeholders

## PYTHON CODE STYLE

### Structure
- All Python work is in **Jupyter notebooks (`.ipynb`)**, run in VS Code with a `.venv` kernel
- **5–6 cells per notebook** — see the Python Notebook Conventions section above for the exact cell breakdown
- Cell 1 is always pip installs (commented out after first run); Cell 2 is always imports + config

### Commenting
- `# ---- Section Name ----` for major sections within a cell (mirrors R convention)
- Comments explain intent and interpretation, not just what the line does
- Each non-obvious step gets a "WHY:" comment — `# WHY: we normalize embeddings so cosine similarity works correctly`
- R-to-Python translations noted where helpful: `# like dplyr::filter() in R`

### Code clarity
- No magic numbers — assign constants with explanatory names: `N_SIMILAR = 5  # how many similar notes to retrieve`
- Small logical steps with intermediate variables (not deeply nested one-liners)
- Every import explained with a one-line comment
- Complete, runnable notebooks — no placeholders

---

## FOLDER STRUCTURE

```
Project Root/
├── Data/
│   ├── ml_teaching_data.rds         ← R format (main dataset)
│   └── ml_teaching_data.csv         ← CSV export for Python scripts
├── Scripts/
│   ├── 00. CreateDummyData.R
│   ├── 00. ExploratoryAnalysis.R
│   ├── 01. Logistic Regression.R
│   ├── 02. Decision Trees.R
│   ├── 03. Random Forest.R
│   ├── 04. GradientBoosting_XGBoost.R
│   ├── 05. KNN_SVM.R
│   ├── 06. Neural Network.R
│   ├── 07. Unsupervised.R
│   ├── 08. Isolation Forest.R
│   ├── 09. Survival ML.R
│   ├── 10. Time Series.R
│   ├── 11. Model Interpretation.R
│   ├── 12. Ensemble_SuperLearner.R
│   ├── 13. Causal ML.R
│   ├── 14. Deep Learning Concepts.R
│   ├── 15. Model Comparison.R
│   ├── 16. NLP_Health_Text.ipynb
│   ├── 17. Embeddings_and_Semantic_Search.ipynb
│   ├── 18. Generative_AI_Concepts.ipynb
│   ├── 19. Synthetic_Data_Generation.ipynb
│   └── 20. Agentic_AI.ipynb
└── Outputs/
    ├── EDA/
    ├── Logistic_Regression/
    ├── Decision_Trees/
    ├── Random_Forest/
    ├── Boosting/
    ├── KNN_SVM/
    ├── Neural_Network/
    ├── Unsupervised/
    ├── Isolation_Forest/
    ├── Survival_ML/
    ├── Time_Series/
    ├── Model_Interpretation/
    ├── Ensemble_SuperLearner/
    ├── Causal_ML/
    ├── Deep_Learning_Concepts/
    ├── Model_Comparison/
    ├── NLP_Health_Text/
    ├── Embeddings_Semantic_Search/
    ├── Generative_AI/
    ├── Synthetic_Data_Generation/
    └── Agentic_AI/
```

---

## KNOWN PITFALLS (ALREADY DEBUGGED — DON'T REPEAT)

### R Pitfalls

1. **`plot_annotation(theme = theme_ml)` throws warnings.** `plot_annotation()` requires a raw `theme()` object. Use `theme_annotation` (defined above), never `theme_ml`.
2. **Color palette key mismatch.** If a factor has levels `"No"/"Yes"`, the palette must use those same strings as names: `c("No" = "#2196F3", "Yes" = "#F44336")`. Not `"0"/"1"`.
3. **Prediction grid length mismatch.** When building grids for marginal effect plots (varying one feature while holding others constant), build each grid as its own fresh `tibble()`. Don't `mutate()` a new-length vector into an existing data frame.
4. **1-SE rule over-prunes with rare outcomes.** Heart disease is ~6%. The 1-SE rule for `rpart` prunes to a stump (zero splits). Use **minimum CV error** instead, with a fallback: if min CV error still selects 0 splits, pick the best cp among trees that DO have splits.
5. **`rpart` variable importance can be NULL.** If the pruned tree is very simple, `tree$variable.importance` may be empty. Always check `length()` and fall back to the full tree's importance.
6. **`geom_errorbar` with separate `data` inherits parent `aes`.** Add `inherit.aes = FALSE` when a geom layer uses its own `data` argument and the parent `aes()` references columns that don't exist in the new data.
7. **`tibble()` + `mutate()` column self-reference.** Don't `fct_reorder(feature, importance)` inside the same `mutate()` where `feature` was just created via `names(...)`. Create the tibble first, then `mutate()` in a separate step.

### Python / Notebook Pitfalls

8. **Hugging Face model downloads require internet.** Scripts 16–17 download models from Hugging Face on first run. This can take several minutes. The notebook should warn the user in a comment and cache to a local directory. After first run, models load from cache instantly.
9. **OpenAI API key not set.** Scripts 18 and 20 require `OPENAI_API_KEY` in a `.env` file. Always check `if not api_key: raise ValueError(...)` in Cell 2 before any API call, so it fails fast and clearly.
10. **CTGAN on GPU (RTX 4060):** CTGAN detects CUDA automatically when PyTorch is installed with CUDA support. On the 4060, training on 25,000 rows takes ~2–3 minutes (vs 10–20 on CPU). No special code needed — just ensure `torch` is installed from the CUDA wheel: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
11. **`sentence-transformers` conflicts with old `transformers` versions.** Always install into the venv fresh. The Cell 1 pip install block should install both together: `!pip install sentence-transformers transformers` so versions resolve together.
12. **LangChain API changes frequently.** Script 20 notes which version it was written for. Cell 1 uses pinned versions: `!pip install langchain==0.3.0 langchain-openai==0.2.0`
13. **Notebook cell order matters.** Always run cells top-to-bottom on a fresh kernel before considering a notebook "done." Variables from a previous run can linger and hide bugs. Use "Restart Kernel and Run All" as a final check.
14. **`openai_key.txt` should never be committed to git.** Add `openai_key.txt` to `.gitignore`. The key file contains a single line — just the API key, nothing else. The load_openai_key() function in Cell 2 handles reading it and gives a clear error message if the file is missing.

---

## REMAINING SCRIPTS
source("Scripts/19. Synthetic_Data_Generation.ipynb")
source("Scripts/20. Agentic_AI.ipynb")