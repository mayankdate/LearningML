# ============================================================ #
# 14 — Deep Learning Concepts: An Architecture Tour
# ============================================================ #
# Purpose: Teach the INTUITION behind deep learning architectures.
#   Our tabular dataset isn't ideal for CNNs or RNNs — those shine
#   on images, text, and sequences. So this script is a conceptual
#   tour with small toy demonstrations, not full production training.
#
# We cover:
#   1. What "deep" means (and why depth matters)
#   2. CNNs — Convolutional Neural Networks (images)
#   3. RNNs / LSTMs — Recurrent Neural Networks (sequences)
#   4. Transformers / Attention — the architecture behind GPT, BERT
#   5. When to use what — a decision framework
#
# This script uses VISUALIZATIONS and small numerical examples
# to build intuition. No GPU needed, no heavy packages.
# ============================================================ #

rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()

# ---- Packages ----
pacman::p_load(
  dplyr, tidyr, stringr, forcats, tibble, purrr,
  ggplot2, patchwork, scales, viridis
)

setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
dir.create("Outputs/Deep_Learning_Concepts", showWarnings = FALSE, recursive = TRUE)
out_dir <- "Outputs/Deep_Learning_Concepts"

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
# SECTION 1: WHAT DOES "DEEP" MEAN?
# ============================================================ #
#
# Script 06 taught a neural network with ONE hidden layer.
# "Deep learning" means MANY hidden layers (2, 5, 50, 100+).
#
# WHY DOES DEPTH MATTER?
#
# ANALOGY — RECOGNIZING A FACE:
#   Layer 1: Detect edges (lines, curves)
#   Layer 2: Combine edges into simple shapes (circles, angles)
#   Layer 3: Combine shapes into parts (eyes, noses, mouths)
#   Layer 4: Combine parts into faces
#   Layer 5: Recognize WHOSE face it is
#
#   Each layer builds on the previous one, creating increasingly
#   ABSTRACT representations. The first layer sees pixels. The
#   last layer understands concepts like "this is a cat."
#
# SHALLOW vs DEEP:
#   A shallow network (1 hidden layer, like Script 06) can theoretically
#   approximate any function — but it might need exponentially many neurons.
#   A deep network can represent the same function with fewer total
#   parameters by building hierarchical features.
#
# THE KEY ARCHITECTURES:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  Architecture │ Best For        │ Key Idea              │
#   │───────────────│─────────────────│───────────────────────│
#   │  Feedforward  │ Tabular data    │ Layers of weighted    │
#   │  (Script 06)  │                 │ sums + activations    │
#   │───────────────│─────────────────│───────────────────────│
#   │  CNN          │ Images, spatial │ Sliding filters that  │
#   │               │ data            │ detect local patterns │
#   │───────────────│─────────────────│───────────────────────│
#   │  RNN / LSTM   │ Sequences, time │ Memory of previous    │
#   │               │ series, text    │ inputs in the sequence│
#   │───────────────│─────────────────│───────────────────────│
#   │  Transformer  │ Text, language, │ Attention: every word │
#   │               │ everything now  │ looks at every other  │
#   │               │                 │ word simultaneously   │
#   └─────────────────────────────────────────────────────────┘
#
# We'll demonstrate each with toy examples and visualizations.
# ============================================================ #


# ============================================================ #
# SECTION 2: ACTIVATION FUNCTIONS — THE NONLINEAR BUILDING BLOCKS
# ============================================================ #
# Every deep learning architecture uses activation functions.
# Without them, stacking layers is pointless (multiple linear
# layers collapse into one linear equation).

# ---- 2a: Visualize Common Activations ----
x_vals <- seq(-5, 5, length.out = 200)

activation_df <- dplyr::bind_rows(
  tibble(x = x_vals, y = 1 / (1 + exp(-x_vals)),    activation = "Sigmoid"),
  tibble(x = x_vals, y = tanh(x_vals),               activation = "Tanh"),
  tibble(x = x_vals, y = pmax(0, x_vals),             activation = "ReLU"),
  tibble(x = x_vals, y = ifelse(x_vals > 0, x_vals, 0.01 * x_vals),
         activation = "Leaky ReLU")
)

activation_df$activation <- factor(activation_df$activation,
                                   levels = c("Sigmoid", "Tanh", "ReLU", "Leaky ReLU"))

p_activations <- ggplot(activation_df, aes(x = x, y = y, color = activation)) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0, color = "grey60", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "grey60", linewidth = 0.3) +
  facet_wrap(~activation, scales = "free_y") +
  scale_color_manual(values = c("#E53935", "#1565C0", "#43A047", "#FF9800")) +
  labs(
    title    = "Activation Functions — The Nonlinear Building Blocks",
    subtitle = "Without activations, stacking layers is pointless (linear × linear = linear).\nSigmoid: squashes to 0-1 (used in output). ReLU: fast, used in hidden layers.\nTanh: like sigmoid but centered at 0. Leaky ReLU: fixes 'dying neurons' in ReLU.",
    x = "Input", y = "Output"
  ) +
  theme_ml +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold", size = 12))

print(p_activations)
ggsave(file.path(out_dir, "01_activation_functions.png"), p_activations,
       width = 12, height = 8, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 3: CNNs — CONVOLUTIONAL NEURAL NETWORKS
# ============================================================ #
#
# WHAT ARE CNNs FOR?
#   Images. Spatial data. Anything with LOCAL STRUCTURE where
#   nearby pixels/cells are more related than distant ones.
#
# THE KEY IDEA — CONVOLUTION (SLIDING FILTERS):
#   Instead of connecting every pixel to every neuron (which would
#   need millions of weights for a 1000×1000 image), a CNN slides
#   a small FILTER (e.g., 3×3 pixels) across the image.
#
#   The filter detects a LOCAL PATTERN — like a vertical edge,
#   a horizontal line, or a color gradient. The same filter is
#   applied at every position, so a vertical edge detector works
#   whether the edge is top-left or bottom-right.
#   This is called WEIGHT SHARING — massive parameter reduction.
#
# HOW A CNN PROCESSES AN IMAGE:
#
#   Input Image (e.g., 224×224 pixels, 3 color channels)
#     ↓
#   Conv Layer 1: Detect edges (32 different 3×3 filters)
#     → 32 "feature maps" showing where each edge pattern appears
#     ↓
#   Pooling: Shrink each map by taking max of 2×2 blocks
#     → Smaller maps, but keeps the important features
#     ↓
#   Conv Layer 2: Detect shapes (64 filters on the edge maps)
#     → Combinations of edges = shapes (circles, corners)
#     ↓
#   More pooling + conv layers...
#     ↓
#   Flatten: Turn 2D maps into a 1D vector
#     ↓
#   Dense layers: Like our feedforward network (Script 06)
#     ↓
#   Output: "This is a cat" (classification)
#
# WHY CNNs DON'T WORK FOR TABULAR DATA:
#   Tabular data has no spatial structure. "Age" being next to
#   "sex" in the spreadsheet is arbitrary — they're not spatially
#   related. A 3×3 filter sliding across a patient's features
#   would combine adjacent columns, which means nothing.
#   Trees and boosting work much better for tables.

# ---- 3a: Demonstrate Convolution on a Toy Image ----
# Create a simple 8×8 "image" with a vertical edge
set.seed(42)
toy_image <- matrix(0, nrow = 8, ncol = 8)
toy_image[, 1:4] <- 0.2 + rnorm(32, sd = 0.05)   # Left half: dark
toy_image[, 5:8] <- 0.8 + rnorm(32, sd = 0.05)   # Right half: bright
toy_image <- pmin(pmax(toy_image, 0), 1)

# Vertical edge detection filter
edge_filter <- matrix(c(-1, 0, 1,
                        -1, 0, 1,
                        -1, 0, 1), nrow = 3, ncol = 3, byrow = TRUE)

# Apply convolution manually
conv_result <- matrix(0, nrow = 6, ncol = 6)  # Output is smaller (8-3+1 = 6)
for (i in 1:6) {
  for (j in 1:6) {
    patch <- toy_image[i:(i+2), j:(j+2)]
    conv_result[i, j] <- sum(patch * edge_filter)
  }
}

# Visualize input, filter, and result
img_df <- expand.grid(row = 1:8, col = 1:8)
img_df$value <- as.vector(toy_image)

conv_df <- expand.grid(row = 1:6, col = 1:6)
conv_df$value <- as.vector(conv_result)

filter_df <- expand.grid(row = 1:3, col = 1:3)
filter_df$value <- as.vector(edge_filter)

p_input <- ggplot(img_df, aes(x = col, y = -row, fill = value)) +
  geom_tile(color = "grey80") +
  scale_fill_gradient(low = "black", high = "white") +
  coord_equal() +
  labs(title = "Input Image (8×8)", subtitle = "Dark left, bright right = vertical edge") +
  theme_ml + theme(legend.position = "none", axis.text = element_blank(),
                   axis.title = element_blank(), panel.grid = element_blank())

p_filter <- ggplot(filter_df, aes(x = col, y = -row, fill = value)) +
  geom_tile(color = "grey80") +
  geom_text(aes(label = value), size = 6, fontface = "bold") +
  scale_fill_gradient2(low = "#E53935", mid = "white", high = "#1565C0", midpoint = 0) +
  coord_equal() +
  labs(title = "Edge Filter (3×3)", subtitle = "[-1, 0, 1] detects vertical edges") +
  theme_ml + theme(legend.position = "none", axis.text = element_blank(),
                   axis.title = element_blank(), panel.grid = element_blank())

p_conv <- ggplot(conv_df, aes(x = col, y = -row, fill = value)) +
  geom_tile(color = "grey80") +
  scale_fill_gradient2(low = "#1565C0", mid = "white", high = "#E53935", midpoint = 0) +
  coord_equal() +
  labs(title = "Convolution Output (6×6)", subtitle = "Bright stripe = detected edge location") +
  theme_ml + theme(legend.position = "none", axis.text = element_blank(),
                   axis.title = element_blank(), panel.grid = element_blank())

p_cnn_demo <- p_input | p_filter | p_conv
p_cnn_demo <- p_cnn_demo + patchwork::plot_annotation(
  title = "CNN Convolution — How a Filter Detects an Edge",
  subtitle = "The 3×3 filter slides across the input. At each position, it multiplies\nelement-wise and sums. High output = the filter's pattern was found.",
  theme = theme_annotation
)

print(p_cnn_demo)
ggsave(file.path(out_dir, "02_cnn_convolution_demo.png"), p_cnn_demo,
       width = 14, height = 6, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 4: RNNs — RECURRENT NEURAL NETWORKS
# ============================================================ #
#
# WHAT ARE RNNs FOR?
#   Sequences. Text. Time series. Anything where ORDER matters
#   and the current input depends on what came before.
#
# THE KEY IDEA — MEMORY:
#   A feedforward network processes each input independently.
#   An RNN has a HIDDEN STATE that carries information from
#   previous inputs. At each time step:
#     new_state = f(current_input, previous_state)
#   The hidden state is a "memory" of everything the network
#   has seen so far in the sequence.
#
# THE PROBLEM — VANISHING GRADIENTS:
#   In theory, the hidden state carries all past information.
#   In practice, information from early in the sequence FADES
#   as it passes through many time steps. This is called the
#   "vanishing gradient problem" — the error signal for early
#   inputs shrinks exponentially as it's backpropagated through time.
#
# THE SOLUTION — LSTM (Long Short-Term Memory):
#   LSTMs add GATES that control information flow:
#     Forget gate: "How much of the old memory should I keep?"
#     Input gate: "How much of the new input should I store?"
#     Output gate: "How much of the memory should I output now?"
#   These gates are learned — the network decides what to remember
#   and what to forget. This solves the vanishing gradient problem.
#
# EXAMPLE — PROCESSING A SENTENCE:
#   Input: "The patient with a history of smoking was diagnosed with..."
#   Step 1: "The" → state captures "starting a sentence"
#   Step 2: "patient" → state now knows "about a patient"
#   Step 3: "with" → state expecting a description
#   ...
#   Step 8: "smoking" → CRITICAL — state stores "patient smokes"
#   ...
#   Step 12: "diagnosed with..." → use the stored smoking fact to predict
#
#   The LSTM remembers "smoking" from step 8 even when predicting
#   at step 12, because the forget gate kept that information alive.

# ---- 4a: Visualize RNN Hidden State Over a Sequence ----
# Simulate how a hidden state evolves as it processes a sequence
# of health measurements (e.g., monthly blood pressure readings)
set.seed(42)

# Simulated monthly SBP readings for one patient
n_steps <- 24
sbp_sequence <- 120 + cumsum(rnorm(n_steps, mean = 0.5, sd = 3))

# Simulate RNN hidden state (simplified: exponential moving average)
# In a real RNN, the hidden state is a learned transformation.
# Here we show the concept: the state "remembers" past inputs.
alpha <- 0.3  # How much weight to give new input vs memory
hidden_state <- numeric(n_steps)
hidden_state[1] <- sbp_sequence[1]
for (t in 2:n_steps) {
  hidden_state[t] <- alpha * sbp_sequence[t] + (1 - alpha) * hidden_state[t - 1]
}

rnn_demo_df <- tibble(
  step  = 1:n_steps,
  month = paste0("M", 1:n_steps),
  input = sbp_sequence,
  state = hidden_state
) %>%
  tidyr::pivot_longer(c(input, state), names_to = "type", values_to = "value") %>%
  dplyr::mutate(type = ifelse(type == "input", "Raw Input (SBP)",
                              "Hidden State (smoothed memory)"))

p_rnn <- ggplot(rnn_demo_df, aes(x = step, y = value, color = type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.5) +
  scale_color_manual(values = c("Raw Input (SBP)" = "#E53935",
                                "Hidden State (smoothed memory)" = "#1565C0")) +
  labs(
    title    = "RNN Concept — Hidden State as Sequence Memory",
    subtitle = "Red = raw monthly SBP readings (noisy). Blue = hidden state (smoothed memory).\nThe hidden state carries a summary of all past inputs. At each step:\nnew_state = α × current_input + (1-α) × previous_state.",
    x = "Time Step (Month)", y = "Value", color = NULL
  ) +
  theme_ml

print(p_rnn)
ggsave(file.path(out_dir, "03_rnn_hidden_state.png"), p_rnn,
       width = 11, height = 7, dpi = 150, bg = "white")


# ---- 4b: LSTM Gates Concept Diagram ----
# Visualize how LSTM gates control information flow
gate_demo <- tibble(
  step = rep(1:12, 3),
  gate = rep(c("Forget Gate", "Input Gate", "Output Gate"), each = 12),
  value = c(
    # Forget gate: high = keep memory, low = forget
    c(0.9, 0.9, 0.8, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.5),
    # Input gate: high when new important info arrives
    c(0.2, 0.8, 0.3, 0.9, 0.2, 0.1, 0.7, 0.1, 0.1, 0.8, 0.3, 0.2),
    # Output gate: high when memory should influence prediction
    c(0.3, 0.4, 0.5, 0.3, 0.2, 0.3, 0.6, 0.4, 0.3, 0.5, 0.8, 0.9)
  )
)

words <- c("The", "patient", "was", "diagnosed", "with", "diabetes",
           "and", "has", "a", "history", "of", "smoking")

p_gates <- ggplot(gate_demo, aes(x = step, y = value, fill = gate)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("Forget Gate" = "#E53935",
                               "Input Gate" = "#43A047",
                               "Output Gate" = "#1565C0")) +
  scale_x_continuous(breaks = 1:12, labels = words) +
  labs(
    title    = "LSTM Gates — Controlling Memory Flow",
    subtitle = "Forget gate (red): keep or discard old memory. Input gate (green): store new info.\nOutput gate (blue): use memory for current prediction.\nHigh values at 'diabetes' and 'smoking' = the LSTM stores these critical facts.",
    x = "Input Token", y = "Gate Activation (0-1)", fill = NULL
  ) +
  theme_ml +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"))

print(p_gates)
ggsave(file.path(out_dir, "04_lstm_gates.png"), p_gates,
       width = 12, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 5: TRANSFORMERS & ATTENTION
# ============================================================ #
#
# WHAT ARE TRANSFORMERS?
#   The architecture behind GPT, BERT, ChatGPT, and most modern AI.
#   They replaced RNNs for most NLP tasks starting in 2017.
#
# THE KEY IDEA — SELF-ATTENTION:
#   RNNs process sequences one word at a time (sequential).
#   Transformers process ALL words SIMULTANEOUSLY and let each
#   word "attend to" every other word.
#
#   "The patient was admitted because their blood pressure was high"
#
#   When processing "high", the transformer asks:
#   "Which other words should I pay attention to?"
#   Answer: "blood" (0.4), "pressure" (0.3), "patient" (0.15)...
#   It learns these ATTENTION WEIGHTS from data.
#
# WHY THIS IS BETTER THAN RNNs:
#   1. PARALLEL: All words processed at once (fast on GPUs)
#   2. LONG-RANGE: Word 1 can directly attend to word 100
#      (RNNs must pass info through 99 intermediate steps)
#   3. INTERPRETABLE: Attention weights show what the model "focuses on"
#
# THE TRANSFORMER ARCHITECTURE:
#   Input tokens → Embedding (words → numbers)
#     → Self-attention (every token looks at every other)
#     → Feed-forward (process each token individually)
#     → Repeat N times (stack layers)
#     → Output
#
#   GPT (decoder-only): predicts NEXT word. Used for generation.
#   BERT (encoder-only): understands CONTEXT. Used for classification.
#
# WHY TRANSFORMERS DOMINATE:
#   - Scale well with data and compute (more data = better)
#   - Transfer learning: pre-train on billions of words,
#     fine-tune on your small dataset
#   - Same architecture works for text, images (Vision Transformer),
#     audio, protein sequences, and more

# ---- 5a: Attention Heatmap — Toy Example ----
# Simulate attention weights for a clinical sentence
sentence <- c("Patient", "with", "diabetes", "and", "smoking",
              "has", "elevated", "blood", "pressure")
n_words <- length(sentence)

# Simulate an attention pattern: "elevated" attends to "blood" and "pressure"
# "diabetes" and "smoking" attend to "Patient"
set.seed(42)
attention_matrix <- matrix(runif(n_words^2, 0.01, 0.08), nrow = n_words, ncol = n_words)

# Make some attention weights strong (clinically meaningful connections)
attention_matrix[7, 8] <- 0.35  # "elevated" → "blood"
attention_matrix[7, 9] <- 0.30  # "elevated" → "pressure"
attention_matrix[3, 1] <- 0.25  # "diabetes" → "Patient"
attention_matrix[5, 1] <- 0.25  # "smoking" → "Patient"
attention_matrix[8, 9] <- 0.35  # "blood" → "pressure"
attention_matrix[9, 7] <- 0.30  # "pressure" → "elevated"
attention_matrix[6, 3] <- 0.20  # "has" → "diabetes"
attention_matrix[6, 5] <- 0.20  # "has" → "smoking"

# Normalize rows to sum to 1 (softmax output)
attention_matrix <- attention_matrix / rowSums(attention_matrix)

attn_df <- expand.grid(query = sentence, key = sentence)
attn_df$weight <- as.vector(attention_matrix)
attn_df$query <- factor(attn_df$query, levels = sentence)
attn_df$key   <- factor(attn_df$key, levels = rev(sentence))

p_attention <- ggplot(attn_df, aes(x = query, y = key, fill = weight)) +
  geom_tile(color = "white") +
  geom_text(aes(label = ifelse(weight > 0.12, round(weight, 2), "")),
            size = 3, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#E53935", name = "Attention\nWeight") +
  labs(
    title    = "Self-Attention Heatmap — What Each Word 'Looks At'",
    subtitle = "Each row = one word asking 'which other words should I pay attention to?'\nBright red = high attention. 'elevated' strongly attends to 'blood' and 'pressure'.\nThis is how Transformers (GPT, BERT) understand context.",
    x = "Query (the word asking)", y = "Key (the word being attended to)"
  ) +
  theme_ml +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
        axis.text.y = element_text(size = 11, face = "bold"))

print(p_attention)
ggsave(file.path(out_dir, "05_attention_heatmap.png"), p_attention,
       width = 10, height = 9, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 6: THE DEEP LEARNING LANDSCAPE — WHEN TO USE WHAT
# ============================================================ #

# ---- 6a: Architecture Decision Framework ----
framework_df <- tibble(
  data_type     = c("Tabular\n(spreadsheets)", "Images\n(X-rays, photos)",
                    "Sequences\n(text, time series)", "Text\n(clinical notes)",
                    "Multi-modal\n(text + images)"),
  best_model    = c("XGBoost/RF\n(NOT deep learning)", "CNN\n(ResNet, EfficientNet)",
                    "LSTM/GRU\nor Transformer", "Transformer\n(BERT, GPT)",
                    "Vision Transformer\n+ LLM"),
  why           = c("Trees handle tabular\nbetter than neural nets",
                    "Convolutions detect\nlocal spatial patterns",
                    "Memory captures\ntemporal dependencies",
                    "Attention understands\nlong-range context",
                    "Different architectures\nfor different modalities"),
  deep_learning = c("Rarely", "Always", "Usually", "Always", "Always"),
  x_pos = 1:5,
  y_pos = rep(1, 5)
)

p_framework <- ggplot(framework_df, aes(x = x_pos, y = 1)) +
  geom_tile(aes(fill = deep_learning), color = "white",
            width = 0.9, height = 0.8) +
  geom_text(aes(label = paste0(data_type, "\n\n", best_model)),
            size = 3.2, fontface = "bold", lineheight = 0.9) +
  scale_fill_manual(values = c("Rarely" = "#BBDEFB", "Usually" = "#64B5F6",
                               "Always" = "#1565C0"),
                    name = "Use Deep Learning?") +
  scale_x_continuous(limits = c(0.4, 5.6)) +
  labs(
    title    = "When to Use Deep Learning — A Decision Framework",
    subtitle = "Deep learning shines on images, text, and sequences. For tabular data\n(what we've been doing in Scripts 01-13), tree-based methods usually win."
  ) +
  theme_ml +
  theme(axis.text = element_blank(), axis.title = element_blank(),
        panel.grid = element_blank())

print(p_framework)
ggsave(file.path(out_dir, "06_architecture_decision.png"), p_framework,
       width = 14, height = 6, dpi = 150, bg = "white")


# ---- 6b: Performance vs Interpretability Tradeoff ----
tradeoff_df <- tibble(
  model = c("Linear/Logistic\nRegression", "Decision\nTree", "Random\nForest",
            "XGBoost", "Neural\nNetwork", "Deep CNN/\nTransformer"),
  complexity     = c(1, 2, 3, 4, 5, 6),
  performance    = c(0.55, 0.60, 0.78, 0.82, 0.80, 0.95),
  interpretability = c(0.95, 0.90, 0.70, 0.55, 0.30, 0.10),
  category = c("Simple", "Simple", "Ensemble", "Ensemble",
               "Neural", "Neural")
)

p_tradeoff <- ggplot(tradeoff_df, aes(x = interpretability, y = performance)) +
  geom_point(aes(color = category), size = 6) +
  ggrepel::geom_text_repel(aes(label = model), size = 3.5, fontface = "bold",
                           max.overlaps = 10) +
  scale_color_manual(values = c("Simple" = "#43A047", "Ensemble" = "#1565C0",
                                "Neural" = "#E53935")) +
  scale_x_reverse() +
  labs(
    title    = "The Fundamental Tradeoff — Performance vs Interpretability",
    subtitle = "Moving right → more interpretable but less powerful.\nMoving up → better predictions but harder to explain.\nDeep learning lives in the top-left: powerful but opaque.",
    x = "Interpretability (← less | more →)",
    y = "Typical Performance on Complex Tasks",
    color = "Model Type"
  ) +
  theme_ml

print(p_tradeoff)
ggsave(file.path(out_dir, "07_performance_vs_interpretability.png"), p_tradeoff,
       width = 10, height = 8, dpi = 150, bg = "white")


# ---- 6c: Model Parameters Scale ----
# Show the exponential growth in model size over time
scale_df <- tibble(
  model = c("Logistic Reg\n(Script 01)", "XGBoost\n(Script 04)",
            "Our nnet\n(Script 06)", "ResNet-50\n(2015)",
            "BERT\n(2018)", "GPT-3\n(2020)", "GPT-4\n(2023)"),
  params = c(15, 5000, 500, 2.5e7, 1.1e8, 1.75e11, 1.8e12),
  year = c(NA, NA, NA, 2015, 2018, 2020, 2023),
  type = c("This Project", "This Project", "This Project",
           "Published", "Published", "Published", "Published")
)

scale_df$model <- forcats::fct_inorder(scale_df$model)

p_scale <- ggplot(scale_df, aes(x = model, y = params, fill = type)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::label_number(scale_cut = scales::cut_short_scale())(params)),
            vjust = -0.5, size = 3.5, fontface = "bold") +
  scale_y_log10(labels = scales::label_number(scale_cut = scales::cut_short_scale())) +
  scale_fill_manual(values = c("This Project" = "#1565C0", "Published" = "#E53935")) +
  labs(
    title    = "Model Scale — From 15 Parameters to 1.8 Trillion",
    subtitle = "Log scale! Each step up is 10× more parameters.\nOur scripts use tiny models. Production deep learning uses billions.\nGPT-4 has ~120 million times more parameters than our XGBoost.",
    x = NULL, y = "Number of Parameters (log scale)", fill = NULL
  ) +
  theme_ml +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 10))

print(p_scale)
ggsave(file.path(out_dir, "08_model_scale.png"), p_scale,
       width = 12, height = 7, dpi = 150, bg = "white")


# ============================================================ #
# SECTION 7: SAVE RESULTS
# ============================================================ #

# This script is primarily educational — no model to save.
# Save the key visualizations and a summary.
results_dl <- list(
  toy_image     = toy_image,
  edge_filter   = edge_filter,
  conv_result   = conv_result,
  attention_matrix = attention_matrix,
  model_scale   = scale_df
)

saveRDS(results_dl, file.path(out_dir, "results_deep_learning_concepts.rds"))
cat("\nResults saved to:", out_dir, "\n")


# ============================================================ #
# SECTION 8: SUMMARY & WHAT WE LEARNED
# ============================================================ #

cat("\n")
cat("=== DEEP LEARNING CONCEPTS — SUMMARY ===\n")

cat("\nWHAT WE LEARNED:\n")
cat("1.  'Deep' means many layers. Each layer builds increasingly\n")
cat("    abstract representations (edges → shapes → objects → concepts).\n")
cat("2.  ACTIVATION FUNCTIONS add nonlinearity. Without them, deep\n")
cat("    networks collapse to a single linear equation. ReLU is the\n")
cat("    most common in hidden layers; sigmoid for binary output.\n")
cat("3.  CNNs use SLIDING FILTERS to detect local spatial patterns.\n")
cat("    Weight sharing means the same edge detector works everywhere\n")
cat("    in the image. Great for images, useless for tabular data.\n")
cat("4.  RNNs process SEQUENCES by maintaining a hidden state — a\n")
cat("    memory of everything seen so far. But they struggle with\n")
cat("    long-range dependencies (vanishing gradients).\n")
cat("5.  LSTMs solve the vanishing gradient problem with GATES that\n")
cat("    learn what to remember (forget gate), what to store (input\n")
cat("    gate), and what to output (output gate).\n")
cat("6.  TRANSFORMERS replaced RNNs by using SELF-ATTENTION: every\n")
cat("    word looks at every other word simultaneously. This enables\n")
cat("    parallel processing and long-range understanding.\n")
cat("7.  GPT, BERT, ChatGPT are all Transformers. They're pre-trained\n")
cat("    on billions of words and can be fine-tuned for specific tasks.\n")
cat("8.  For TABULAR data (what we've been doing), tree-based models\n")
cat("    (XGBoost, Random Forest) usually BEAT deep learning.\n")
cat("    Deep learning shines on images, text, audio, and sequences.\n")
cat("9.  There's a fundamental tradeoff: more powerful models are\n")
cat("    harder to interpret. Deep learning lives at the high-power,\n")
cat("    low-interpretability end of the spectrum.\n")
cat("10. Modern AI models have TRILLIONS of parameters. Our scripts\n")
cat("    use models with hundreds to thousands. Same concepts,\n")
cat("    vastly different scale.\n")

cat("\nWHEN TO USE DEEP LEARNING:\n")
cat("  Images:       CNN (ResNet, EfficientNet)\n")
cat("  Text/NLP:     Transformer (BERT, GPT) — see Scripts 16-18\n")
cat("  Time series:  LSTM or Transformer (for long sequences)\n")
cat("  Tabular data: USUALLY NOT — use XGBoost/RF instead\n")
cat("  Multi-modal:  Vision Transformer + LLM combinations\n")

cat("\nHOW THIS CONNECTS TO OTHER SCRIPTS:\n")
cat("  - Script 06 (Neural Net): Feedforward = the simplest 'deep'\n")
cat("    architecture. One hidden layer. This script shows what\n")
cat("    happens when you go deeper and change the architecture.\n")
cat("  - Script 10 (Time Series): ARIMA/ETS are statistical approaches.\n")
cat("    LSTMs are the deep learning alternative for time series.\n")
cat("  - Scripts 16-18 (Python): NLP, Embeddings, Generative AI\n")
cat("    all use Transformers — the architecture taught here.\n")
cat("===================================\n")

cat("\nScript 14 complete.\n")
