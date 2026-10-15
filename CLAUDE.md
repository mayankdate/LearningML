# LearningML: contributor and AI coding guidance

## Purpose

LearningML is a personal R learning library covering 20 machine-learning
topics, from classical supervised models through deep learning, NLP, embeddings,
and agentic AI. Each script is a self-contained tutorial: it generates synthetic
data, trains models, produces evaluation outputs, and is heavily commented for
learning.

## Repository layout

```
LearningML/
├── Scripts/
│   ├── 00. CreateDummyData.R
│   ├── 00. ExploratoryAnalysis.R
│   ├── 00. RunAllScripts.R          ← sources 01–15 in order
│   ├── 01. Logistic Regression.R
│   ├── 02. Decision Trees.R
│   ├── 03. Random Forest.R
│   ├── 04. GradientBoosting_XGBoost.R
│   ├── 05. KNN_SVM.R
│   ├── 06. Neural_Network.R
│   ├── 07. Unsupervised.R
│   ├── 08. Isolation_Forest.R
│   ├── 09. Survival_ML.R
│   ├── 10. Time_Series.R
│   ├── 11. Model_Interpretation.R
│   ├── 12. Ensemble_SuperLearner.R
│   ├── 13. Causal_ML.R
│   ├── 14. Deep_Learning_Concepts.R
│   ├── 15. Model_Comparison.R
│   ├── 16. NLP_Health_Text.ipynb
│   ├── 17. Embeddings_and_Semantic_Search.ipynb
│   ├── 18. Generative_AI_Concepts.ipynb
│   ├── 19. Synthetic_Data_Generation.ipynb
│   └── 20. Agentic_AI.ipynb
└── LearningML.Rproj
```

Scripts 01–15 are R; 16–20 are Python notebooks. Open the project in
RStudio using `LearningML.Rproj`.

## Key conventions

- **Self-contained scripts.** Each R script generates its own synthetic data
  (or loads from `Scripts/00. CreateDummyData.R`) and produces its own output.
  There is no shared mutable state across scripts.
- **Package management via `pacman`.** All scripts use `pacman::p_load(...)`
  at the top. To add a new package, use `pacman::p_load(new_package)` — never
  a bare `library()` call at the file top.
- **Comments are load-bearing.** These scripts are a learning resource.
  Comments explain *why* and *how* — do not remove or reduce them when making
  changes. Add comments when adding new code.
- **No shared output directory.** Scripts write any plots or outputs inline
  (RStudio Plots pane) or to a path relative to the script's working directory.
  Do not introduce a shared `Output/` folder without the owner's approval.
- **Run order:** `00. RunAllScripts.R` sources 01–15 sequentially. It does not
  run 16–20 (Python notebooks) — open those in Jupyter.

## Commands

```r
# Run all R scripts (01–15)
source("Scripts/00. RunAllScripts.R")

# Run a single script
source("Scripts/03. Random Forest.R")

# Install all R packages
source("Scripts/00. CreateDummyData.R")  # will auto-install via pacman
```

## Change guidelines

- When adding a new topic, add a new numbered script (`21. <Topic>.R` or
  `.ipynb`). Do not fold new content into an existing script — keep each
  topic separate.
- When modifying a script, preserve all inline comments and explanations.
  The teaching purpose matters as much as the code.
- Do not refactor scripts to share helper functions; the self-contained
  structure is intentional.
