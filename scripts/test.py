import tokenizer
import embeddings
import positional
import attention
import normalization
import training

from parameters import *


# -----------------------------
# Create a small training example
# -----------------------------

text = "Hello World!"

tokens = tokenizer.text_to_tokens(text)

# We need one target token for this test.
input_tokens = tokens[:-1]
target = tokens[-1]


# -----------------------------
# Forward pass
# -----------------------------

token_embeddings = embeddings.tokens_to_embeddings(input_tokens)

position_aware_embeddings = positional.get_position_aware_embeddings(
    token_embeddings
)

queries, keys, values = attention.get_qkv(position_aware_embeddings)

scores = attention.calculate_scores(queries, keys)

masked_scores = attention.apply_casual_mask(scores)

attention_weights = attention.apply_softmax_to_masked(masked_scores)

attention_outputs = attention.apply_weighted_value_sums(
    attention_weights,
    values
)

residual_attention = normalization.add_residual_connection(
    position_aware_embeddings,
    attention_outputs
)

first_layernorm_input = residual_attention

normalized_attention_vectors = normalization.apply_layernorm_to_vectors(
    residual_attention
)

normalized_attention = normalized_attention_vectors[-1]


# -----------------------------
# Feed-forward pass
# -----------------------------

linear_1_output = normalization.linear_layer(
    normalized_attention,
    weights_1,
    biases_1
)

relu_output = normalization.relu(
    linear_1_output
)

linear_2_output = normalization.linear_layer(
    relu_output,
    weights_2,
    biases_2
)

final_layernorm_input = [
    normalized_attention[i] + linear_2_output[i]
    for i in range(len(normalized_attention))
]

output_vector = normalization.layerNorm(
    final_layernorm_input
)


# -----------------------------
# Output layer
# -----------------------------

logits = normalization.linear_layer(
    output_vector,
    output_weights,
    output_biases
)

predictions = attention.apply_softmax(logits)


# -----------------------------
# Backpropagation
# -----------------------------

results = training.backpropagate(
    predictions,
    target,
    output_vector,
    first_layernorm_input[-1],
    final_layernorm_input,
    relu_output,
    linear_1_output,
    normalized_attention,
    position_aware_embeddings,
    attention_weights,
    queries,
    keys,
    values,
    input_tokens
)


# -----------------------------
# Unpack gradients
# -----------------------------

(
    output_weight_gradients,
    output_bias_gradients,
    weights_1_gradients,
    biases_1_gradients,
    weights_2_gradients,
    biases_2_gradients,
    W_Q_gradient,
    W_K_gradient,
    W_V_gradient,
    embedding_gradients,
    positional_gradients
) = results

learning_rate = 0.001

training.update_parameters(
    output_weight_gradients,
    output_bias_gradients,
    weights_1_gradients,
    biases_1_gradients,
    weights_2_gradients,
    biases_2_gradients,
    W_Q_gradient,
    W_K_gradient,
    W_V_gradient,
    embedding_gradients,
    positional_gradients,
    learning_rate
)


# # -----------------------------
# # Print gradient shapes
# # -----------------------------

# print("Gradient shapes:")

# print(
#     "output_weight_gradients:",
#     len(output_weight_gradients),
#     "x",
#     len(output_weight_gradients[0])
# )

# print(
#     "output_bias_gradients:",
#     len(output_bias_gradients)
# )

# print(
#     "weights_1_gradients:",
#     len(weights_1_gradients),
#     "x",
#     len(weights_1_gradients[0])
# )

# print(
#     "biases_1_gradients:",
#     len(biases_1_gradients)
# )

# print(
#     "weights_2_gradients:",
#     len(weights_2_gradients),
#     "x",
#     len(weights_2_gradients[0])
# )

# print(
#     "biases_2_gradients:",
#     len(biases_2_gradients)
# )

# print(
#     "W_Q_gradient:",
#     len(W_Q_gradient),
#     "x",
#     len(W_Q_gradient[0])
# )

# print(
#     "W_K_gradient:",
#     len(W_K_gradient),
#     "x",
#     len(W_K_gradient[0])
# )

# print(
#     "W_V_gradient:",
#     len(W_V_gradient),
#     "x",
#     len(W_V_gradient[0])
# )

# print(
#     "embedding_gradients:",
#     len(embedding_gradients),
#     "x",
#     len(embedding_gradients[0])
# )

# print(
#     "positional_gradients:",
#     len(positional_gradients),
#     "x",
#     len(positional_gradients[0])
# )