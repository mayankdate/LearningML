# LearningML

A personal R + Python learning library covering 20 machine-learning topics, from logistic regression through agentic AI. Each script is a self-contained tutorial: it generates synthetic data, trains a model, and produces evaluation outputs — with extensive comments explaining the concepts.

## Scripts

| # | File | Topic |
|---|------|-------|
| 00 | `CreateDummyData.R` | Synthetic data generation |
| 00 | `ExploratoryAnalysis.R` | EDA |
| 01 | `Logistic Regression.R` | Logistic regression |
| 02 | `Decision Trees.R` | Decision trees |
| 03 | `Random Forest.R` | Random forests |
| 04 | `GradientBoosting_XGBoost.R` | Gradient boosting / XGBoost |
| 05 | `KNN_SVM.R` | k-NN and SVM |
| 06 | `Neural_Network.R` | Neural networks |
| 07 | `Unsupervised.R` | Clustering / PCA |
| 08 | `Isolation_Forest.R` | Anomaly detection |
| 09 | `Survival_ML.R` | Survival analysis |
| 10 | `Time_Series.R` | Time series |
| 11 | `Model_Interpretation.R` | SHAP, LIME, feature importance |
| 12 | `Ensemble_SuperLearner.R` | Ensembles / SuperLearner |
| 13 | `Causal_ML.R` | Causal ML |
| 14 | `Deep_Learning_Concepts.R` | Deep learning concepts in R |
| 15 | `Model_Comparison.R` | Model comparison |
| 16 | `NLP_Health_Text.ipynb` | NLP on health text (Python) |
| 17 | `Embeddings_and_Semantic_Search.ipynb` | Embeddings + semantic search |
| 18 | `Generative_AI_Concepts.ipynb` | Generative AI concepts |
| 19 | `Synthetic_Data_Generation.ipynb` | Synthetic data (Python) |
| 20 | `Agentic_AI.ipynb` | Agentic AI |

## Running

Open `LearningML.Rproj` in RStudio, then:

```r
# Run all R scripts (01–15)
source("Scripts/00. RunAllScripts.R")

# Or run a single script
source("Scripts/03. Random Forest.R")
```

For Python notebooks (16–20), set up the environment once with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Then open a notebook in VS Code and pick `.venv` as the kernel. The `!pip install` lines inside
the notebooks are no longer needed; `pyproject.toml` lists every package they use. PyTorch
runs on the CPU by default; `pyproject.toml` explains the three-line switch to the GPU.

## Requirements

- R 4.0+ with `pacman` (`install.packages("pacman")`)
- Each script auto-installs its own R packages via `pacman::p_load()`
- [uv](https://docs.astral.sh/uv/) for notebooks 16–20 (installs Python 3.14 and the packages in `uv.lock`)
