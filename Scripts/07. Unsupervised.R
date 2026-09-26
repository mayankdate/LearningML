# ============================================================ #
# 07 — Unsupervised Learning: Clustering & Dimensionality Reduction
# ============================================================ #
# Purpose: Teach unsupervised ML — finding structure in data
#   WITHOUT a target variable. No "right answer" to predict.
#   Instead, we ask: "Are there natural groups of patients?"
#   and "Can we visualize 25 features in 2 dimensions?"
#
# We cover:
#   CLUSTERING (grouping similar patients):
#     1. K-Means — partition into K groups by distance
#     2. Hierarchical — build a tree of nested clusters
#     3. DBSCAN — density-based, finds arbitrary shapes
#
#   DIMENSIONALITY REDUCTION (compressing features):
#     4. PCA — linear projection, preserves variance
#     5. t-SNE — nonlinear, preserves local neighborhoods
#     6. UMAP — nonlinear, preserves both local and global structure
#
# Target: NONE. This is unsupervised — no outcome variable.
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis,
  caret,        # dummyVars for one-hot encoding
  cluster,      # K-means and silhouette analysis
  factoextra,   # Beautiful cluster visualizations (wraps ggplot2)
  dbscan,       # DBSCAN density-based clustering
  dendextend,   # Enhanced dendrograms for hierarchical clustering
  Rtsne,        # t-SNE dimensionality reduction
  uwot          # UMAP dimensionality reduction (fast C++ implementation)
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Unsupervised", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Unsupervised"

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

# Cluster color palette — distinct colors for up to 8 clusters
cluster_colors <- c(
  "1" = "#E53935", "2" = "#1E88E5", "3" = "#43A047",
  "4" = "#FB8C00", "5" = "#8E24AA", "6" = "#00ACC1",
  "7" = "#6D4C41", "8" = "#546E7A"
)


# ============================================================ #
# SECTION 1: WHAT IS UNSUPERVISED LEARNING?
# ============================================================ #
#
# ANALOGY: Imagine you're a doctor who's never studied medicine.
# You're given charts for 25,000 patients — age, BMI, blood
# pressure, cholesterol, etc. — but NO diagnoses. Nobody tells
# you who is sick or healthy. You just have the measurements.
#
# You'd start noticing patterns:
#   "These 3,000 patients all have high BMI, high glucose,
#    and high blood pressure — they seem similar to each other
#    and different from the rest."
# That's CLUSTERING — finding natural groups.
#
# You might also think:
#   "I have 13 measurements per patient. Can I boil them down
#    to 2 or 3 summary numbers that capture the key differences?"
# That's DIMENSIONALITY REDUCTION — compressing many features
# into fewer dimensions while preserving the important structure.
#
# HOW IS THIS DIFFERENT FROM SUPERVISED LEARNING?
#
#   SUPERVISED (Scripts 01–06):
#     - You have a TARGET variable (heart_disease = 0 or 1)
#     - The model learns to PREDICT the target from features
#     - Success = accuracy, AUC, F1 score (how well it predicts)
#
#   UNSUPERVISED (this script):
#     - There is NO target variable
#     - The model finds STRUCTURE in the features themselves
#     - Success = do the groups make clinical sense? Do they
#       reveal patterns a human wouldn't easily spot?
#     - There's no single "right answer" — interpretation matters
#
# WHY IS UNSUPERVISED LEARNING USEFUL?
#   1. Patient segmentation — group patients by risk profile
#   2. Exploratory analysis — discover subgroups you didn't know about
#   3. Feature engineering — PCA components as inputs to supervised models
#   4. Anomaly detection — patients who don't fit any cluster (Script 08)
#   5. Visualization — see 13-dimensional data in 2D
# ============================================================ #


# ============================================================ #
# SECTION 2: DATA PREPARATION
# ============================================================ #

# ---- Load Data ----
ml_data <- readRDS("Data/ml_teaching_data.rds")
cat("Dataset:", nrow(ml_data), "rows |", ncol(ml_data), "columns\n")

# Use only the feature columns — NOT the target or outcome variables.
# Unsupervised learning discovers structure in the FEATURES THEMSELVES.
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


# ---- One-Hot Encode and Scale ----
# WHY: Clustering algorithms measure DISTANCE between patients.
# Distance only works on numbers, so we convert factors to 0/1 dummies.
# Then we SCALE all features (mean=0, SD=1) so that income (8,000–300,000)
# doesn't dominate age (18–95) just because its numbers are bigger.
# Without scaling, distance is meaningless — a 1-unit change in income
# would swamp a 1-unit change in BMI.
dummy_formula <- as.formula(paste("~", paste(features, collapse = " + ")))
dummy_obj     <- caret::dummyVars(dummy_formula, data = ml_data)
X_raw         <- predict(dummy_obj, newdata = ml_data)

# Scale: center (subtract mean) and divide by SD
X_scaled <- scale(X_raw)

cat("Scaled feature matrix:", nrow(X_scaled), "rows ×", ncol(X_scaled), "columns\n")

# For clustering on 25k rows, some methods are slow.
# We'll use a random subsample for slow methods (hierarchical, t-SNE)
# and the full dataset for fast methods (K-means, PCA, UMAP).
set.seed(42)
sample_size <- 5000
sample_idx  <- sample(1:nrow(X_scaled), sample_size)
X_sample    <- X_scaled[sample_idx, ]

cat("Subsample for slow methods:", nrow(X_sample), "rows\n")


# ============================================================ #
# PART A: CLUSTERING
# ============================================================ #
# Clustering = grouping similar patients together.
# "Similar" means close in feature space (after scaling).
# Different algorithms define "close" differently.
# ============================================================ #


# ============================================================ #
# SECTION 3: K-MEANS CLUSTERING
# ============================================================ #
#
# ANALOGY: Imagine dropping K pins on a map. Each person walks
# to the nearest pin. Then each pin moves to the center of its
# crowd. People re-walk to the nearest (possibly different) pin.
# Repeat until nobody switches. The pins are "centroids" and
# the crowds are "clusters."
#
# HOW IT WORKS:
#   1. Pick K random points as initial cluster centers
#   2. Assign each patient to the NEAREST center (by Euclidean distance)
#   3. Recompute each center as the MEAN of its assigned patients
#   4. Repeat steps 2-3 until assignments stop changing
#
# KEY DECISION: How many clusters (K)?
#   Too few = lumps together different groups
#   Too many = splits natural groups artificially
#   We use two methods to find the "right" K:
#     - Elbow method: plot total within-cluster variance vs K
#     - Silhouette method: measures how well each point fits its cluster
#
# WHEN K-MEANS WORKS WELL:
#   - Clusters are roughly spherical and similar size
#   - You have a rough idea of how many groups to expect
#   - Large datasets (fast: O(n) per iteration)
#
# WHEN IT STRUGGLES:
#   - Non-spherical clusters (crescent, ring shapes)
#   - Clusters of very different sizes or densities
#   - Must specify K in advance (chicken-and-egg problem)
# ============================================================ #

# ---- 3a: Finding the Right K — Elbow Method ----
# Run K-means for K = 2 through 10. For each K, record the total
# within-cluster sum of squares (WSS). WSS measures how spread out
# points are within their clusters — lower = tighter clusters.
# As K increases, WSS always decreases (more clusters = tighter fit).
# The "elbow" is where adding another cluster stops helping much.

set.seed(42)
wss_values <- sapply(2:10, function(k) {
  km <- kmeans(X_scaled, centers = k, nstart = 25, iter.max = 50)
  km$tot.withinss
})

elbow_df <- tibble(k = 2:10, wss = wss_values)

p_elbow <- ggplot(elbow_df, aes(x = k, y = wss)) +
  geom_line(linewidth = 1, color = "#1565C0") +
  geom_point(size = 3, color = "#1565C0") +
  scale_x_continuous(breaks = 2:10) +
  labs(
    title    = "Elbow Method — How Many Clusters?",
    subtitle = "Look for the 'bend' where adding more clusters stops reducing WSS much.\nThe elbow suggests a natural number of groups.",
    x = "Number of Clusters (K)",
    y = "Total Within-Cluster Sum of Squares"
  ) +
  theme_ml

print(p_elbow)
ggsave(file.path(out_dir, "01_elbow_method.png"), p_elbow,
       width = 9, height = 6, dpi = 150, bg = "white")

# ---- 3b: Finding the Right K — Silhouette Method ----
# Silhouette score measures how well each point fits in its cluster
# vs the next-nearest cluster. Ranges from -1 (wrong cluster) to
# +1 (perfectly placed). Average silhouette across all points
# gives an overall score — higher is better.

set.seed(42)
sil_sample_idx <- sample(1:nrow(X_scaled), 3000)
sil_X <- X_scaled[sil_sample_idx, ]
sil_dist <- dist(sil_X)

sil_values <- sapply(2:10, function(k) {
  km <- kmeans(sil_X, centers = k, nstart = 25, iter.max = 50)
  ss <- cluster::silhouette(km$cluster, sil_dist)
  mean(ss[, 3])
})

sil_df <- tibble(k = 2:10, silhouette = sil_values)
best_k <- sil_df$k[which.max(sil_df$silhouette)]

p_sil <- ggplot(sil_df, aes(x = k, y = silhouette)) +
  geom_line(linewidth = 1, color = "#E53935") +
  geom_point(size = 3, color = "#E53935") +
  geom_point(data = sil_df %>% dplyr::filter(k == best_k),
             size = 5, color = "#E53935", shape = 18) +
  annotate("text", x = best_k + 0.3, y = max(sil_values) - 0.005,
           label = paste0("Best K = ", best_k), fontface = "bold", size = 4) +
  scale_x_continuous(breaks = 2:10) +
  labs(
    title    = "Silhouette Method — How Many Clusters?",
    subtitle = "Higher = points fit their clusters well. Peak suggests the most natural grouping.",
    x = "Number of Clusters (K)",
    y = "Average Silhouette Score"
  ) +
  theme_ml

print(p_sil)
ggsave(file.path(out_dir, "02_silhouette_method.png"), p_sil,
       width = 9, height = 6, dpi = 150, bg = "white")

cat("Elbow plot saved. Silhouette optimal K:", best_k, "\n")


# ---- 3c: Fit K-Means with Chosen K ----
# nstart = 25: run K-means 25 times with different random starts
# and keep the best result. K-means can converge to different
# solutions depending on initial centers — multiple starts help.
set.seed(42)
km_fit <- kmeans(X_scaled, centers = best_k, nstart = 25, iter.max = 100)

cat("K-Means with K =", best_k, "clusters:\n")
cat("  Cluster sizes:", paste(km_fit$size, collapse = ", "), "\n")
cat("  Total WSS:", round(km_fit$tot.withinss, 0), "\n")
cat("  Between-SS / Total-SS:", round(km_fit$betweenss / km_fit$totss * 100, 1),
    "% — higher = more separation between clusters\n")

# Store cluster assignments on the full dataset
ml_data$cluster_kmeans <- factor(km_fit$cluster)


# ---- 3d: Profile the Clusters ----
# What kind of patients ended up in each cluster?
# Compute mean of key features per cluster to interpret the groups.
cluster_profiles <- ml_data %>%
  dplyr::group_by(cluster_kmeans) %>%
  dplyr::summarise(
    n             = dplyr::n(),
    age_mean      = round(mean(age), 1),
    bmi_mean      = round(mean(bmi), 1),
    sbp_mean      = round(mean(sbp), 1),
    cholesterol_mean = round(mean(cholesterol), 1),
    glucose_mean  = round(mean(fasting_glucose), 1),
    income_mean   = round(mean(income), 0),
    pct_smoker    = round(mean(smoking_status == "Current") * 100, 1),
    pct_diabetes  = round(mean(diabetes) * 100, 1),
    pct_heart_dis = round(mean(heart_disease) * 100, 1),
    .groups = "drop"
  )

cat("\nCluster profiles (K-Means):\n")
print(as.data.frame(cluster_profiles))

# ---- 3e: Visualize Cluster Profiles ----
# Heatmap showing standardized means per cluster — makes it easy
# to see which clusters are high/low on each feature.
profile_long <- cluster_profiles %>%
  dplyr::select(cluster_kmeans, age_mean, bmi_mean, sbp_mean,
                cholesterol_mean, glucose_mean, income_mean, pct_smoker) %>%
  tidyr::pivot_longer(-cluster_kmeans, names_to = "feature", values_to = "value") %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(z_value = (value - mean(value)) / sd(value)) %>%
  dplyr::ungroup()

# Clean up feature labels
profile_long$feature <- stringr::str_replace(profile_long$feature, "_mean$", "")
profile_long$feature <- stringr::str_replace(profile_long$feature, "pct_", "% ")

p_profile <- ggplot(profile_long, aes(x = feature, y = cluster_kmeans, fill = z_value)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = round(value, 1)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#1565C0", mid = "white", high = "#E53935",
                       midpoint = 0, name = "Standardized\nvalue") +
  labs(
    title    = "K-Means Cluster Profiles",
    subtitle = "Blue = below average for that feature. Red = above average.\nNumbers show actual values. Look for clusters that are consistently high/low.",
    x = NULL, y = "Cluster"
  ) +
  theme_ml +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1))

print(p_profile)
ggsave(file.path(out_dir, "03_cluster_profiles.png"), p_profile,
       width = 10, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: HIERARCHICAL CLUSTERING
# ============================================================ #
#
# ANALOGY: Imagine sorting photos of animals. Start with each
# photo in its own pile. Find the two most similar photos and
# merge them into one pile. Now find the next two most similar
# piles and merge them. Keep going until everything is one pile.
# The order you merged in creates a TREE (called a dendrogram).
# Cut the tree at any height to get clusters.
#
# HOW IT WORKS:
#   1. Start with N clusters (each patient is their own cluster)
#   2. Compute distance between all pairs of clusters
#   3. Merge the two closest clusters
#   4. Repeat until there's one cluster
#   5. Cut the tree at a chosen height to get K clusters
#
# LINKAGE METHODS (how to define "distance" between clusters):
#   - Complete: distance = max distance between any two members
#   - Average: distance = average of all pairwise distances
#   - Ward's: merge the pair that increases total variance least
#     (Ward's tends to produce compact, equal-sized clusters)
#
# WHEN HIERARCHICAL WORKS WELL:
#   - You want to SEE the nesting structure (dendrogram)
#   - Small to medium datasets (distance matrix is O(n²) memory)
#   - You're not sure how many clusters there are
#
# WHEN IT STRUGGLES:
#   - Large datasets — distance matrix for 25,000 rows is huge
#     (that's why we use a subsample here)
#   - Once a merge is made, it can't be undone (greedy algorithm)
# ============================================================ #

# ---- 4a: Build Dendrogram (on subsample) ----
# Computing a distance matrix for 25k rows would need ~2.3 GB of RAM.
# We use the 5,000-row subsample — still shows the structure clearly.
dist_matrix <- dist(X_sample, method = "euclidean")
hc_fit      <- hclust(dist_matrix, method = "ward.D2")

# ---- 4b: Visualize Dendrogram ----
# The height of each merge shows how different the clusters were.
# Low merges = very similar patients. High merges = very different groups.
# Cutting at a given height produces a specific number of clusters.
dend <- as.dendrogram(hc_fit)
dend <- dendextend::color_branches(dend, k = best_k)

png(file.path(out_dir, "04_dendrogram.png"),
    width = 12, height = 7, units = "in", res = 150)
plot(dend,
     main = "Hierarchical Clustering Dendrogram (Ward's Method)",
     sub  = paste0("5,000-patient subsample. Colors show ", best_k,
                   " clusters. Height = dissimilarity at merge."),
     ylab = "Height (dissimilarity)",
     leaflab = "none")  # Don't print 5000 leaf labels
abline(h = max(hc_fit$height) * 0.5, lty = 2, col = "grey40")
dev.off()

cat("Dendrogram saved\n")

# ---- 4c: Cut the Tree ----
hc_clusters <- cutree(hc_fit, k = best_k)
cat("Hierarchical cluster sizes (subsample):",
    paste(table(hc_clusters), collapse = ", "), "\n")


# ============================================================ #
# SECTION 5: DBSCAN — DENSITY-BASED CLUSTERING
# ============================================================ #
#
# ANALOGY: Imagine you're at a party. Clusters are groups of
# people standing close together in conversation. If you can
# walk from person to person in a group without taking a big
# step, they're all in the same cluster. People standing alone
# far from any group are "noise" — not assigned to any cluster.
#
# HOW IT WORKS:
#   1. For each point, count neighbors within radius eps
#   2. If a point has ≥ minPts neighbors, it's a "core point"
#   3. Core points near each other form the same cluster
#   4. Non-core points near a core point join that cluster
#   5. Points not near ANY core point → labeled as NOISE (-1)
#
# KEY HYPERPARAMETERS:
#   eps: the radius to search for neighbors
#     Too small = everything is noise (no clusters found)
#     Too large = everything in one cluster
#   minPts: minimum neighbors to be a core point
#     Typically 5-10 for most datasets
#
# WHY DBSCAN IS DIFFERENT:
#   - K-Means: you specify K clusters. Every point gets assigned.
#   - DBSCAN: you don't specify K. It DISCOVERS how many clusters
#     exist and labels outliers as noise. It can find non-spherical
#     shapes (crescents, rings) that K-Means can't.
#
# WHEN DBSCAN WORKS WELL:
#   - Clusters of different shapes and sizes
#   - You want automatic outlier detection
#   - You don't know how many clusters exist
#
# WHEN IT STRUGGLES:
#   - Clusters of very different densities (one eps can't fit all)
#   - High-dimensional data (distance becomes less meaningful —
#     called the "curse of dimensionality")
# ============================================================ #

# ---- 5a: Find Good eps Using k-NN Distance Plot ----
# For each point, compute distance to its 5th nearest neighbor.
# Sort these distances. The "elbow" suggests a good eps value.
# Points in dense clusters have small kNN distances; outliers have large ones.
knn_dists <- dbscan::kNNdist(X_sample, k = 5)
knn_sorted <- sort(knn_dists)

knn_df <- tibble(index = seq_along(knn_sorted), distance = knn_sorted)

p_knn <- ggplot(knn_df, aes(x = index, y = distance)) +
  geom_line(color = "#1565C0", linewidth = 0.5) +
  labs(
    title    = "k-NN Distance Plot — Finding eps for DBSCAN",
    subtitle = "Each point's distance to its 5th nearest neighbor, sorted.\nThe 'elbow' (sharp bend) suggests a good eps value.",
    x = "Points (sorted by distance)", y = "Distance to 5th Nearest Neighbor"
  ) +
  theme_ml

print(p_knn)
ggsave(file.path(out_dir, "05_knn_distance_plot.png"), p_knn,
       width = 9, height = 6, dpi = 150, bg = "white")

# ---- 5b: Fit DBSCAN ----
# Choose eps based on the kNN plot elbow. We'll try a reasonable value
# and let DBSCAN discover clusters. If it finds 0 clusters (all noise)
# or 1 giant cluster, we adjust eps.
# For high-dimensional scaled data, eps typically ranges from 3 to 8.
eps_try <- quantile(knn_sorted, 0.95)  # Use 95th percentile as a heuristic
cat("DBSCAN eps (95th pct of kNN distances):", round(eps_try, 2), "\n")

db_fit <- dbscan::dbscan(X_sample, eps = eps_try, minPts = 5)

n_dbscan_clusters <- max(db_fit$cluster)
n_noise <- sum(db_fit$cluster == 0)

cat("DBSCAN results:\n")
cat("  Clusters found:", n_dbscan_clusters, "\n")
cat("  Noise points:", n_noise, "(", round(n_noise / nrow(X_sample) * 100, 1), "%)\n")
cat("  Cluster sizes:", paste(table(db_fit$cluster[db_fit$cluster > 0]), collapse = ", "), "\n")

# If DBSCAN found meaningful clusters, label the subsample
ml_data_sample <- ml_data[sample_idx, ]
ml_data_sample$cluster_dbscan <- factor(db_fit$cluster)


# ============================================================ #
# PART B: DIMENSIONALITY REDUCTION
# ============================================================ #
# We have 25+ features (after dummy coding). How do we VISUALIZE
# patients in 2D? We need algorithms that compress many dimensions
# into 2, preserving the important structure.
# ============================================================ #


# ============================================================ #
# SECTION 6: PCA — PRINCIPAL COMPONENT ANALYSIS
# ============================================================ #
#
# ANALOGY: Imagine photographing a 3D statue. Each photo is a
# 2D projection — it captures some detail but loses some depth.
# PCA finds the "best camera angle" — the 2D projection that
# preserves the most variance (spread) in the data.
#
# HOW IT WORKS:
#   1. Find the direction of maximum variance → PC1
#      (the single axis that captures the most spread)
#   2. Find the direction of maximum remaining variance,
#      PERPENDICULAR to PC1 → PC2
#   3. Continue for as many components as you want
#   Each PC is a LINEAR COMBINATION of original features.
#   Example: PC1 = 0.4×age + 0.3×sbp + 0.2×bmi + ...
#
# WHAT THE OUTPUT MEANS:
#   - "% variance explained": how much information each PC captures
#   - "Loadings": which original features contribute most to each PC
#   - PC1 and PC2 together might explain 30-50% of total variance —
#     not perfect, but enough to see major groupings
#
# WHY PCA IS USEFUL:
#   - Reduce 25 features to 2 for visualization
#   - Identify which features co-vary (move together)
#   - Create uncorrelated inputs for other models
#   - Fast — just linear algebra (eigenvalue decomposition)
#
# LIMITATION:
#   PCA is LINEAR. If the real structure is curved or nonlinear
#   (like a Swiss roll), PCA will miss it. That's where t-SNE
#   and UMAP come in.
# ============================================================ #

# ---- 6a: Run PCA ----
pca_fit <- prcomp(X_scaled, center = FALSE, scale. = FALSE)
# center/scale already done — X_scaled is already centered and scaled

# Variance explained by each component
var_explained <- tibble(
  pc      = paste0("PC", 1:length(pca_fit$sdev)),
  var_pct = (pca_fit$sdev^2 / sum(pca_fit$sdev^2)) * 100,
  cum_pct = cumsum((pca_fit$sdev^2 / sum(pca_fit$sdev^2)) * 100)
)

cat("\nPCA — Variance explained:\n")
print(head(var_explained, 10))
cat("First 2 PCs explain:", round(var_explained$cum_pct[2], 1), "% of variance\n")
cat("First 5 PCs explain:", round(var_explained$cum_pct[5], 1), "% of variance\n")

# ---- 6b: Scree Plot ----
p_scree <- ggplot(var_explained[1:15, ],
                  aes(x = forcats::fct_inorder(pc), y = var_pct)) +
  geom_col(fill = "#1565C0", width = 0.7) +
  geom_line(aes(x = 1:15, y = var_pct), color = "#E53935", linewidth = 1, group = 1) +
  geom_point(aes(x = 1:15, y = var_pct), color = "#E53935", size = 2) +
  geom_text(aes(label = paste0(round(var_pct, 1), "%")),
            vjust = -0.5, size = 3) +
  labs(
    title    = "PCA Scree Plot — Variance Explained per Component",
    subtitle = "Each bar = one PC's share of total variance. The 'elbow' shows\nwhere additional PCs stop adding much information.",
    x = "Principal Component", y = "Variance Explained (%)"
  ) +
  theme_ml +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_scree)
ggsave(file.path(out_dir, "06_pca_scree_plot.png"), p_scree,
       width = 10, height = 6, dpi = 150, bg = "white")

# ---- 6c: PCA Biplot — Patients in PC1 vs PC2 Space ----
# Color by K-means cluster to see if clusters separate in PCA space.
pca_scores <- tibble(
  pc1     = pca_fit$x[, 1],
  pc2     = pca_fit$x[, 2],
  cluster = ml_data$cluster_kmeans
)

p_pca <- ggplot(pca_scores, aes(x = pc1, y = pc2, color = cluster)) +
  geom_point(alpha = 0.3, size = 0.8) +
  scale_color_manual(values = cluster_colors) +
  labs(
    title    = "PCA — Patients in PC1 vs PC2 Space",
    subtitle = paste0("Colors = K-Means clusters. First 2 PCs explain ",
                      round(var_explained$cum_pct[2], 1),
                      "% of variance.\nClusters that separate here differ on the features PCA captured."),
    x = paste0("PC1 (", round(var_explained$var_pct[1], 1), "% variance)"),
    y = paste0("PC2 (", round(var_explained$var_pct[2], 1), "% variance)"),
    color = "Cluster"
  ) +
  theme_ml

print(p_pca)
ggsave(file.path(out_dir, "07_pca_clusters.png"), p_pca,
       width = 10, height = 8, dpi = 150, bg = "white")

# ---- 6d: PCA Loadings — What Do the Components Mean? ----
# Loadings tell us which original features drive each PC.
# Large positive loading = feature pushes scores UP on that PC.
# Large negative loading = feature pushes scores DOWN.
loadings_df <- tibble(
  feature = rownames(pca_fit$rotation),
  pc1     = pca_fit$rotation[, 1],
  pc2     = pca_fit$rotation[, 2]
)

# Show top features for PC1
loadings_pc1 <- loadings_df %>%
  dplyr::arrange(dplyr::desc(abs(pc1))) %>%
  dplyr::slice_head(n = 10) %>%
  dplyr::mutate(feature = forcats::fct_reorder(feature, abs(pc1)))

p_loadings <- ggplot(loadings_pc1, aes(x = pc1, y = feature, fill = pc1 > 0)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("TRUE" = "#E53935", "FALSE" = "#1565C0"),
                    labels = c("TRUE" = "Positive", "FALSE" = "Negative"),
                    name = "Direction") +
  labs(
    title    = "PCA Loadings — What Drives PC1?",
    subtitle = "Features with largest absolute loadings contribute most to PC1.\nRed = higher values push PC1 score up. Blue = push it down.",
    x = "Loading on PC1", y = NULL
  ) +
  theme_ml

print(p_loadings)
ggsave(file.path(out_dir, "08_pca_loadings.png"), p_loadings,
       width = 10, height = 6, dpi = 150, bg = "white")


# ---- 6e: PCA Colored by Heart Disease ----
# Even though this is unsupervised (we didn't tell the algorithm about
# heart disease), we can check: do patients with heart disease cluster
# together in PCA space? If yes, the features carry signal about disease.
pca_disease <- tibble(
  pc1            = pca_fit$x[, 1],
  pc2            = pca_fit$x[, 2],
  heart_disease  = factor(ml_data$heart_disease, levels = c(0, 1),
                          labels = c("No", "Yes"))
)

p_pca_disease <- ggplot(pca_disease, aes(x = pc1, y = pc2, color = heart_disease)) +
  geom_point(alpha = 0.3, size = 0.8) +
  scale_color_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  labs(
    title    = "PCA — Colored by Heart Disease (Unsupervised Found This)",
    subtitle = "We never told PCA about heart disease. If cases separate,\nthe features naturally carry signal about disease status.",
    x = paste0("PC1 (", round(var_explained$var_pct[1], 1), "% variance)"),
    y = paste0("PC2 (", round(var_explained$var_pct[2], 1), "% variance)"),
    color = "Heart Disease"
  ) +
  theme_ml

print(p_pca_disease)
ggsave(file.path(out_dir, "09_pca_heart_disease.png"), p_pca_disease,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: t-SNE — t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING
# ============================================================ #
#
# ANALOGY: Imagine you have a ball of tangled yarn in 3D. You
# want to lay it flat on a table (2D) so that threads that were
# close together in 3D stay close on the table. t-SNE does this —
# it preserves LOCAL neighborhoods. Points that were neighbors in
# high-dimensional space will be neighbors in 2D.
#
# HOW IT WORKS (simplified):
#   1. In high-D: compute probability that point A would "pick" B
#      as a neighbor (based on distance, using a Gaussian distribution)
#   2. In 2D: start with random positions. Compute the same
#      probabilities (using a t-distribution — fatter tails)
#   3. Move points in 2D until the 2D probabilities match the
#      high-D probabilities as closely as possible (minimize KL divergence)
#
# KEY PARAMETER:
#   perplexity: roughly "how many neighbors to consider"
#     Low (5-15): focuses on very local structure
#     High (30-50): captures more global structure
#     30 is a good default for most datasets
#
# WHY t-SNE IS DIFFERENT FROM PCA:
#   - PCA: LINEAR. Finds the best flat projection.
#   - t-SNE: NONLINEAR. Can unroll curved structures.
#   - PCA preserves GLOBAL variance. t-SNE preserves LOCAL neighborhoods.
#   - PCA is deterministic. t-SNE depends on random initialization.
#
# IMPORTANT CAVEATS:
#   - Distances BETWEEN clusters in t-SNE don't mean much —
#     only distances WITHIN clusters are reliable.
#   - Cluster sizes in t-SNE plots don't reflect real group sizes.
#   - t-SNE is SLOW — O(n²). We use the subsample.
#   - Results change with perplexity and random seed.
# ============================================================ #

# ---- 7a: Run t-SNE ----
# Rtsne expects no duplicate rows — check_duplicates = FALSE
# skips this check (we have scaled continuous data, duplicates unlikely)
set.seed(42)
tsne_fit <- Rtsne::Rtsne(
  X_sample,
  dims           = 2,
  perplexity     = 30,
  verbose        = FALSE,
  max_iter       = 1000,
  check_duplicates = FALSE
)

tsne_df <- tibble(
  tsne1   = tsne_fit$Y[, 1],
  tsne2   = tsne_fit$Y[, 2],
  cluster = ml_data$cluster_kmeans[sample_idx]
)

p_tsne <- ggplot(tsne_df, aes(x = tsne1, y = tsne2, color = cluster)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = cluster_colors) +
  labs(
    title    = "t-SNE — 2D Projection of Patient Features",
    subtitle = "Colors = K-Means clusters. t-SNE preserves LOCAL neighborhoods.\nSeparated groups = genuinely different patient profiles. BUT: distances\nbetween groups and cluster sizes are not reliable — only within-cluster structure is.",
    x = "t-SNE 1", y = "t-SNE 2", color = "Cluster"
  ) +
  theme_ml

print(p_tsne)
ggsave(file.path(out_dir, "10_tsne_clusters.png"), p_tsne,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 7b: t-SNE Colored by Heart Disease ----
tsne_disease <- tsne_df %>%
  dplyr::mutate(
    heart_disease = factor(ml_data$heart_disease[sample_idx],
                           levels = c(0, 1), labels = c("No", "Yes"))
  )

p_tsne_disease <- ggplot(tsne_disease, aes(x = tsne1, y = tsne2, color = heart_disease)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  labs(
    title    = "t-SNE — Colored by Heart Disease",
    subtitle = "Do heart disease patients cluster together in t-SNE space?\nIf yes, the nonlinear structure of features captures disease signal.",
    x = "t-SNE 1", y = "t-SNE 2", color = "Heart Disease"
  ) +
  theme_ml

print(p_tsne_disease)
ggsave(file.path(out_dir, "11_tsne_heart_disease.png"), p_tsne_disease,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 8: UMAP — UNIFORM MANIFOLD APPROXIMATION AND PROJECTION
# ============================================================ #
#
# UMAP is the modern alternative to t-SNE. Same goal (nonlinear
# 2D embedding), but with important advantages:
#
# UMAP vs t-SNE:
#   - MUCH faster (especially on large data)
#   - Better preserves GLOBAL structure (cluster distances are
#     more meaningful than in t-SNE)
#   - More stable across runs (less sensitive to random init)
#   - Has a proper mathematical framework (based on topology)
#   - Can handle the full 25k dataset, not just a subsample
#
# KEY PARAMETERS:
#   n_neighbors: similar to perplexity in t-SNE
#     Small (5-15) = focus on very local structure
#     Large (30-100) = more global structure preserved
#   min_dist: how tightly points can be packed
#     Small (0.01) = tight dense clusters
#     Large (0.5) = more spread out
#
# WHEN TO USE WHICH:
#   PCA: first look, linear patterns, fast, interpretable
#   t-SNE: when you need to see local cluster structure
#   UMAP: default choice for nonlinear visualization (best balance)
# ============================================================ #

# ---- 8a: Run UMAP (full dataset — UMAP is fast enough) ----
set.seed(42)
umap_fit <- uwot::umap(
  X_scaled,
  n_neighbors = 30,
  min_dist    = 0.1,
  n_components = 2,
  verbose     = FALSE
)

umap_df <- tibble(
  umap1   = umap_fit[, 1],
  umap2   = umap_fit[, 2],
  cluster = ml_data$cluster_kmeans
)

p_umap <- ggplot(umap_df, aes(x = umap1, y = umap2, color = cluster)) +
  geom_point(alpha = 0.3, size = 0.6) +
  scale_color_manual(values = cluster_colors) +
  labs(
    title    = "UMAP — 2D Projection (Full Dataset)",
    subtitle = "Colors = K-Means clusters. UMAP preserves both local AND global structure.\nUnlike t-SNE, distances between clusters are more meaningful here.",
    x = "UMAP 1", y = "UMAP 2", color = "Cluster"
  ) +
  theme_ml

print(p_umap)
ggsave(file.path(out_dir, "12_umap_clusters.png"), p_umap,
       width = 10, height = 8, dpi = 150, bg = "white")

# ---- 8b: UMAP Colored by Heart Disease ----
umap_disease <- umap_df %>%
  dplyr::mutate(
    heart_disease = factor(ml_data$heart_disease,
                           levels = c(0, 1), labels = c("No", "Yes"))
  )

p_umap_disease <- ggplot(umap_disease, aes(x = umap1, y = umap2, color = heart_disease)) +
  geom_point(alpha = 0.3, size = 0.6) +
  scale_color_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  labs(
    title    = "UMAP — Colored by Heart Disease",
    subtitle = "Unsupervised projection, colored by disease status after the fact.\nClustering of red points would suggest features naturally separate disease.",
    x = "UMAP 1", y = "UMAP 2", color = "Heart Disease"
  ) +
  theme_ml

print(p_umap_disease)
ggsave(file.path(out_dir, "13_umap_heart_disease.png"), p_umap_disease,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 8c: UMAP Colored by Age ----
# Show a continuous feature to see if UMAP captures gradients
umap_age <- umap_df %>%
  dplyr::mutate(age = ml_data$age)

p_umap_age <- ggplot(umap_age, aes(x = umap1, y = umap2, color = age)) +
  geom_point(alpha = 0.3, size = 0.6) +
  scale_color_viridis_c(option = "magma") +
  labs(
    title    = "UMAP — Colored by Age",
    subtitle = "Smooth gradients indicate UMAP preserved the age structure.\nAbrupt boundaries would suggest age drives cluster separation.",
    x = "UMAP 1", y = "UMAP 2", color = "Age"
  ) +
  theme_ml

print(p_umap_age)
ggsave(file.path(out_dir, "14_umap_age_gradient.png"), p_umap_age,
       width = 10, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 9: COMPARISON — ALL THREE PROJECTIONS SIDE BY SIDE
# ============================================================ #
# PCA vs t-SNE vs UMAP on the same subsample, same clusters.

pca_sub <- tibble(
  dim1 = pca_fit$x[sample_idx, 1], dim2 = pca_fit$x[sample_idx, 2],
  cluster = ml_data$cluster_kmeans[sample_idx], method = "PCA"
)
tsne_sub <- tibble(
  dim1 = tsne_fit$Y[, 1], dim2 = tsne_fit$Y[, 2],
  cluster = ml_data$cluster_kmeans[sample_idx], method = "t-SNE"
)
umap_sub <- tibble(
  dim1 = umap_fit[sample_idx, 1], dim2 = umap_fit[sample_idx, 2],
  cluster = ml_data$cluster_kmeans[sample_idx], method = "UMAP"
)

compare_df <- dplyr::bind_rows(pca_sub, tsne_sub, umap_sub)
compare_df$method <- factor(compare_df$method, levels = c("PCA", "t-SNE", "UMAP"))

p_compare <- ggplot(compare_df, aes(x = dim1, y = dim2, color = cluster)) +
  geom_point(alpha = 0.4, size = 0.8) +
  scale_color_manual(values = cluster_colors) +
  facet_wrap(~method, scales = "free") +
  labs(
    title    = "Three Projection Methods Compared",
    subtitle = "Same patients, same K-Means clusters, different algorithms.\nPCA = linear. t-SNE = local nonlinear. UMAP = global + local nonlinear.",
    color = "Cluster"
  ) +
  theme_ml +
  theme(strip.text = element_text(face = "bold", size = 12))

print(p_compare)
ggsave(file.path(out_dir, "15_projection_comparison.png"), p_compare,
       width = 15, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 10: SAVE RESULTS
# ============================================================ #

# Save cluster assignments and PCA/UMAP coordinates for later use
unsupervised_results <- list(
  kmeans_fit      = km_fit,
  cluster_profiles = cluster_profiles,
  pca_fit         = pca_fit,
  var_explained   = var_explained,
  umap_coords     = umap_fit,
  best_k          = best_k
)

saveRDS(unsupervised_results, file.path(out_dir, "results_unsupervised.rds"))

cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 11: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== UNSUPERVISED LEARNING — SUMMARY ===\n")
cat("K-Means: K =", best_k, "clusters | WSS =", round(km_fit$tot.withinss, 0), "\n")
cat("PCA: First 2 PCs explain", round(var_explained$cum_pct[2], 1), "% of variance\n")
cat("DBSCAN: found", n_dbscan_clusters, "clusters +", n_noise, "noise points\n")

cat("\nWHAT WE LEARNED:\n")
cat("1. Unsupervised learning finds structure WITHOUT a target variable.\n")
cat("   We're asking 'what groups exist?' not 'can we predict disease?'\n")
cat("2. K-Means partitions patients into K spherical groups by distance.\n")
cat("   You choose K; the elbow and silhouette methods help decide.\n")
cat("3. Hierarchical clustering builds a tree (dendrogram) showing\n")
cat("   which patients merge first. You can cut at any level.\n")
cat("4. DBSCAN discovers clusters by density — no need to choose K.\n")
cat("   It also identifies outliers as noise. Great when clusters\n")
cat("   have non-spherical shapes.\n")
cat("5. PCA is a LINEAR dimensionality reduction — it finds the\n")
cat("   best flat projection. Fast and interpretable (via loadings)\n")
cat("   but misses curved structure.\n")
cat("6. t-SNE is NONLINEAR — it preserves local neighborhoods.\n")
cat("   Great for visualization but distances between clusters are\n")
cat("   unreliable, and it's slow on large data.\n")
cat("7. UMAP is the modern default — nonlinear, fast, preserves both\n")
cat("   local and global structure. Best overall for visualization.\n")
cat("8. SCALING is critical for ALL these methods. Without it,\n")
cat("   income (thousands) would dominate BMI (20-50) in distance\n")
cat("   calculations, making clusters and projections meaningless.\n")
cat("9. Coloring unsupervised projections by heart disease AFTER\n")
cat("   the fact reveals whether features naturally carry signal\n")
cat("   about disease — useful validation for supervised models.\n")

cat("\nWHEN TO USE EACH METHOD:\n")
cat("  K-Means:       Quick segmentation, when K is roughly known\n")
cat("  Hierarchical:  When you want to see nesting structure (small data)\n")
cat("  DBSCAN:        When clusters have odd shapes or you need outlier detection\n")
cat("  PCA:           First-pass dimensionality reduction, feature engineering\n")
cat("  t-SNE:         Detailed local structure visualization\n")
cat("  UMAP:          Default nonlinear visualization (fast, reliable)\n")

cat("\nHOW THIS CONNECTS TO SUPERVISED LEARNING:\n")
cat("  - Cluster labels can be FEATURES in supervised models\n")
cat("  - PCA components can replace correlated inputs\n")
cat("  - UMAP can reveal why a supervised model struggles in certain regions\n")
cat("  - Anomaly detection (Script 08) extends these ideas to find outliers\n")
cat("===================================\n")

cat("\nScript 07 complete.\n")