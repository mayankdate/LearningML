# ============================================================ #
# RUN ALL — ML Teaching Scripts
# ============================================================ #
# Sources every script in sequence.
# Comment out any line to skip that script.
# ============================================================ #

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))

# ---- Phase 1: Foundations ----
source("Scripts/00. CreateDummyData.R")
source("Scripts/00. ExploratoryAnalysis.R")

# ---- Phase 2: Core Supervised Learning ----
source("Scripts/01. Logistic Regression.R")
source("Scripts/02. Decision Trees.R")
source("Scripts/03. Random Forest.R")
source("Scripts/04. GradientBoosting_XGBoost.R")
source("Scripts/05. KNN_SVM.R")
source("Scripts/06. Neural_Network.R")

# ---- Phase 3: Unsupervised & Specialized ----
source("Scripts/07. Unsupervised.R")
source("Scripts/08. Isolation_Forest.R")
source("Scripts/09. Survival_ML.R")
source("Scripts/10. Time_Series.R")

# ---- Phase 4: Advanced & Applied ----
source("Scripts/11. Model_Interpretation.R")
source("Scripts/12. Ensemble_SuperLearner.R")
source("Scripts/13. Causal_ML.R")
source("Scripts/14. Deep_Learning_Concepts.R")
source("Scripts/15. Model_Comparison.R")

# ---- Phase 5: NLP & Generative AI (Python — run separately) ----
# These are Jupyter notebooks (.ipynb) — open and run manually in JupyterLab.
# Scripts/16. NLP_Health_Text.ipynb
# Scripts/17. Embeddings_and_Semantic_Search.ipynb
# Scripts/18. Generative_AI_Concepts.ipynb
# Scripts/19. Synthetic_Data_Generation.ipynb

# ---- Phase 6: Agentic AI (Python — run separately) ----
# Scripts/20. Agentic_AI.ipynb

cat("\n===========================\n")
cat("All R scripts complete.\n")
cat("===========================\n")