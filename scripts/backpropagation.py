import math
import parameters

def create_training_example(tokens: list[int], start: int, context_length: int) -> tuple[list[int], list[int]]:
    input_tokens = tokens[start:start + context_length]
    target_tokens = tokens[start + 1:start + context_length + 1]

    return input_tokens, target_tokens

def calculate_loss(predictions: list[list[float]], targets: list[int]) -> float:
    losses = []

    for i, prediction in enumerate(predictions):
        correct_probability = prediction[targets[i]]
        loss = -math.log(correct_probability)
        losses.append(loss)

    return sum(losses) / len(losses)


def calculate_logits_gradient(predictions: list[float], target: int) -> list[float]:
    gradient = predictions.copy()

    gradient[target] -= 1

    return gradient

def calculate_output_weight_gradients(output_vector: list[float], logits_gradient: list[float]) -> list[list[float]]:
    weight_gradient = []

    for i in range(len(logits_gradient)):
        row = []

        for j in range(len(output_vector)):
            row.append(logits_gradient[i] * output_vector[j])

        weight_gradient.append(row)

    return weight_gradient

def calculate_output_bias_gradients(logits_gradient: list[float]) -> list[float]:
    bias_gradient = logits_gradient.copy()

    return bias_gradient

def calculate_output_vector_gradient(logits_gradient: list[float], output_weights: list[list[float]]) -> list[float]:
    vector_gradient = []

    for j in range(len(output_weights[0])):
        total = 0

        for i in range(len(logits_gradient)):
            total += logits_gradient[i] * output_weights[i][j]

        vector_gradient.append(total)

    return vector_gradient

def calculate_layernorm_gradient(gradient: list[float], input_vector: list[float]) -> list[float]:
    mean = sum(input_vector) / len(input_vector)


    sqr_differences = [(num-mean)**2 for num in input_vector]
    variance = sum(sqr_differences) / len(input_vector)
    epsilon = 0.000001

    normalized = [(input_vector[i] - mean) / math.sqrt(variance+epsilon) for i in range(len(input_vector))]

    gradient_sum = sum(gradient)

    gradient_normalized_sum = sum([gradient[i]*normalized[i] for i in range(len(gradient))])

    result = []
    n = len(input_vector)
    for i in range(len(gradient)):
        value = 1 / n
        value *= 1 / math.sqrt(variance + epsilon)

        value *= (
            n * gradient[i]
            - gradient_sum
            - normalized[i] * gradient_normalized_sum
        )

        result.append(value)

    return result

def calculate_residual_gradient(gradient: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    first_gradient = []
    second_gradient = []

    for row in gradient:
        first_gradient.append(row.copy())
        second_gradient.append(row.copy())

    return first_gradient, second_gradient

def calculate_linear_layer_gradients(gradient: list[float], input_vector: list[float], weights: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    weight_gradients = []
    bias_gradient = []

    for i in range(len(gradient)):
        row = []
        for j in range(len(input_vector)):
            row.append(gradient[i] * input_vector[j])

        weight_gradients.append(row)
        bias_gradient.append(gradient[i])

    input_gradients = []

    for j in range(len(input_vector)):
        total = 0

        for i in range(len(gradient)):
            total += gradient[i] * weights[i][j]

        input_gradients.append(total)

    return weight_gradients, bias_gradient, input_gradients

def calculate_relu_gradient(gradient: list[float], input_vector: list[float]) -> list[float]:
    output_gradient = []

    for i in range(len(input_vector)):
        if input_vector[i] > 0:
            output_gradient.append(gradient[i])
        else:
            output_gradient.append(0)

    return output_gradient

def calculate_weighted_value_sums_gradient(
    gradient: list[list[float]],
    attention_weights: list[list[float]],
    values: list[list[float]]) -> tuple[
    list[list[float]],  # attention weights gradient
    list[list[float]]   # values gradient
]:
    attention_weights_gradients = []

    for i in range(len(attention_weights)):
        row = []
        for k in range(len(values)):
            total = 0
            for j in range(len(values[k])):
                total += gradient[i][j] * values[k][j]

            row.append(total)

        attention_weights_gradients.append(row)

    values_gradients = []

    for k in range(len(values)):
        row = []
        for j in range(len(values[k])):
            total = 0
            for i in range(len(attention_weights)):
                total += gradient[i][j] * attention_weights[i][k]

            row.append(total)

        values_gradients.append(row)

    return attention_weights_gradients, values_gradients

def calculate_softmax_gradient(gradient: list[float], softmax_output: list[float]) -> list[float]:
    total = 0

    for i in range(len(gradient)):
        total += gradient[i] * softmax_output[i]

    output = []

    for i in range(len(gradient)):
        value = softmax_output[i] * (gradient[i] - total)
        output.append(value)

    return output

def calculate_softmax_gradients(gradient: list[list[float]], softmax_outputs: list[list[float]]) -> list[list[float]]:
    output = []
    for i in range(len(gradient)):
        output.append(calculate_softmax_gradient(gradient[i], softmax_outputs[i]))

    return output

def calculate_causal_mask_gradient(gradient: list[list[float]]) -> list[list[float]]:
    output = [[0 for _ in range(len(gradient[0]))] for _ in range(len(gradient))]

    for i in range(len(gradient)):
        for j in range(len(gradient[i])):
            if j <= i:
                output[i][j] = gradient[i][j]
            else:
                output[i][j] = 0

    return output

def calculate_scores_gradient(
    gradient: list[list[float]],
    queries: list[list[float]],
    keys: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    queries_gradients = [
        [0.0 for _ in range(len(queries[i]))]
        for i in range(len(queries))
    ]

    keys_gradients = [
        [0.0 for _ in range(len(keys[j]))]
        for j in range(len(keys))
    ]

    for i in range(len(queries)):
        for j in range(len(keys)):
            for k in range(len(queries[i])):
                queries_gradients[i][k] += (
                    gradient[i][j] * keys[j][k] / parameters.ATTENTION_SQRT
                )

                keys_gradients[j][k] += (
                    gradient[i][j] * queries[i][k] / parameters.ATTENTION_SQRT
                )

    return queries_gradients, keys_gradients

def calculate_qkv_gradient(
    queries_gradient: list[list[float]],
    keys_gradient: list[list[float]],
    values_gradient: list[list[float]],
    position_aware_embeddings: list[list[float]]
) -> list[list[float]]:

    result = []

    for i in range(len(position_aware_embeddings)):
        _, _, query_input_gradient = calculate_linear_layer_gradients(
            queries_gradient[i],
            position_aware_embeddings[i],
            parameters.w_Q
        )

        _, _, key_input_gradient = calculate_linear_layer_gradients(
            keys_gradient[i],
            position_aware_embeddings[i],
            parameters.w_K
        )

        _, _, value_input_gradient = calculate_linear_layer_gradients(
            values_gradient[i],
            position_aware_embeddings[i],
            parameters.w_V
        )

        embedding_gradient = []

        for k in range(len(position_aware_embeddings[i])):
            embedding_gradient.append(
                query_input_gradient[k]
                + key_input_gradient[k]
                + value_input_gradient[k]
            )

        result.append(embedding_gradient)

    return result

def calculate_qkv_parameter_gradients(
    queries_gradient: list[list[float]],
    keys_gradient: list[list[float]],
    values_gradient: list[list[float]],
    position_aware_embeddings: list[list[float]]
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]]
]:
    W_Q_gradient = [
        [0.0 for _ in range(len(position_aware_embeddings[0]))]
        for _ in range(len(queries_gradient[0]))
    ]

    W_K_gradient = [
        [0.0 for _ in range(len(position_aware_embeddings[0]))]
        for _ in range(len(keys_gradient[0]))
    ]

    W_V_gradient = [
        [0.0 for _ in range(len(position_aware_embeddings[0]))]
        for _ in range(len(values_gradient[0]))
    ]

    for i in range(len(position_aware_embeddings)):
        for r in range(len(queries_gradient[i])):
            for c in range(len(position_aware_embeddings[i])):
                W_Q_gradient[r][c] += queries_gradient[i][r] * position_aware_embeddings[i][c]
        
        for r in range(len(keys_gradient[i])):
            for c in range(len(position_aware_embeddings[i])):
                W_K_gradient[r][c] += keys_gradient[i][r] * position_aware_embeddings[i][c]
        
        for r in range(len(values_gradient[i])):
            for c in range(len(position_aware_embeddings[i])):
                W_V_gradient[r][c] += values_gradient[i][r] * position_aware_embeddings[i][c]

    return W_Q_gradient, W_K_gradient, W_V_gradient

def calculate_embedding_gradients(
    position_aware_embeddings_gradient: list[list[float]],
    tokens: list[int]
) -> tuple[list[list[float]], list[list[float]]]:
    embedding_gradients = [
        [0.0 for _ in range(32)]
        for _ in range(59)
    ]
    # Both are currently hardcoded - Reminder: replace these with variables/use table lengths
    positional_gradients = [
        [0.0 for _ in range(32)]
        for _ in range(128)
    ]

    for i in range(len(position_aware_embeddings_gradient)):
        token_id = tokens[i]

        for j in range(len(position_aware_embeddings_gradient[i])):
            gradient = position_aware_embeddings_gradient[i][j]

            embedding_gradients[token_id][j] += gradient
            positional_gradients[i][j] += gradient

    return embedding_gradients, positional_gradients



def backpropagate(
    predictions: list[float],
    target: int,
    output_vector: list[float],
    first_layernorm_input: list[float],
    final_layernorm_input: list[float],
    relu_output: list[float],
    linear_1_output: list[float],
    normalized_attention: list[float],
    position_aware_embeddings: list[list[float]],
    attention_weights: list[list[float]],
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
    tokens: list[int]
):
    # -----------------------------
    # Output layer
    # -----------------------------

    logits_gradient = calculate_logits_gradient(
        predictions,
        target
    )

    output_weight_gradients = calculate_output_weight_gradients(
        output_vector,
        logits_gradient
    )

    output_bias_gradients = calculate_output_bias_gradients(
        logits_gradient
    )

    output_vector_gradient = calculate_output_vector_gradient(
        logits_gradient,
        parameters.output_weights
    )


    # -----------------------------
    # Final layer normalization
    # -----------------------------

    final_layernorm_gradient = calculate_layernorm_gradient(
        output_vector_gradient,
        final_layernorm_input
    )


    # -----------------------------
    # Feed-forward residual
    # -----------------------------

    normalized_attention_gradient_from_residual, feed_forward_output_gradient = calculate_residual_gradient(
        [final_layernorm_gradient]
    )

    normalized_attention_gradient_from_residual = normalized_attention_gradient_from_residual[0]
    feed_forward_output_gradient = feed_forward_output_gradient[0]


    # -----------------------------
    # Second linear layer
    # -----------------------------

    weights_2_gradients, biases_2_gradients, relu_gradient = calculate_linear_layer_gradients(
        feed_forward_output_gradient,
        relu_output,
        parameters.weights_2
    )


    # -----------------------------
    # ReLU
    # -----------------------------

    linear_1_gradient = calculate_relu_gradient(
        relu_gradient,
        linear_1_output
    )


    # -----------------------------
    # First linear layer
    # -----------------------------

    weights_1_gradients, biases_1_gradients, normalized_attention_gradient_from_ffn = calculate_linear_layer_gradients(
        linear_1_gradient,
        normalized_attention,
        parameters.weights_1
    )


    # -----------------------------
    # Combine FFN residual gradients
    # -----------------------------

    normalized_attention_gradient = [
        a + b
        for a, b in zip(
            normalized_attention_gradient_from_residual,
            normalized_attention_gradient_from_ffn
        )
    ]


    # -----------------------------
    # First layer normalization
    # -----------------------------

    residual_attention_gradient = calculate_layernorm_gradient(
        normalized_attention_gradient,
        first_layernorm_input
    )


    # -----------------------------
    # Expand final-token gradient
    # to the entire sequence
    # -----------------------------

    full_residual_attention_gradient = [
        [0.0 for _ in range(len(residual_attention_gradient))]
        for _ in range(len(position_aware_embeddings))
    ]

    last_position = len(position_aware_embeddings) - 1

    for j in range(len(residual_attention_gradient)):
        full_residual_attention_gradient[last_position][j] = residual_attention_gradient[j]


    # -----------------------------
    # Attention residual
    # -----------------------------

    position_aware_embeddings_gradient, attention_output_gradient = calculate_residual_gradient(
        full_residual_attention_gradient
    )


    # -----------------------------
    # Weighted value sums
    # -----------------------------

    attention_weights_gradient, values_gradient = calculate_weighted_value_sums_gradient(
        attention_output_gradient,
        attention_weights,
        values
    )


    # -----------------------------
    # Softmax
    # -----------------------------

    scores_after_softmax_gradient = calculate_softmax_gradients(
        attention_weights_gradient,
        attention_weights
    )


    # -----------------------------
    # Causal mask
    # -----------------------------

    scores_gradient = calculate_causal_mask_gradient(
        scores_after_softmax_gradient
    )


    # -----------------------------
    # Attention scores
    # -----------------------------

    queries_gradient, keys_gradient = calculate_scores_gradient(
        scores_gradient,
        queries,
        keys
    )


    # -----------------------------
    # Q/K/V input gradients
    # -----------------------------

    position_aware_embeddings_gradient_from_qkv = calculate_qkv_gradient(
        queries_gradient,
        keys_gradient,
        values_gradient,
        position_aware_embeddings
    )


    # -----------------------------
    # Combine gradients flowing
    # into position-aware embeddings
    # -----------------------------

    position_aware_embeddings_gradient = [
        [
            a + b
            for a, b in zip(
                position_aware_embeddings_gradient[i],
                position_aware_embeddings_gradient_from_qkv[i]
            )
        ]
        for i in range(len(position_aware_embeddings))
    ]


    # -----------------------------
    # Q/K/V parameter gradients
    # -----------------------------

    W_Q_gradient, W_K_gradient, W_V_gradient = calculate_qkv_parameter_gradients(
        queries_gradient,
        keys_gradient,
        values_gradient,
        position_aware_embeddings
    )


    # -----------------------------
    # Embedding gradients
    # -----------------------------

    embedding_gradients, positional_gradients = calculate_embedding_gradients(
        position_aware_embeddings_gradient,
        tokens
    )


    # -----------------------------
    # Return all parameter gradients
    # -----------------------------

    return (
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
    )

def update_parameters(
    output_weight_gradients: list[list[float]],
    output_bias_gradients: list[float],
    weights_1_gradients: list[list[float]],
    biases_1_gradients: list[float],
    weights_2_gradients: list[list[float]],
    biases_2_gradients: list[float],
    W_Q_gradient: list[list[float]],
    W_K_gradient: list[list[float]],
    W_V_gradient: list[list[float]],
    embedding_gradients: list[list[float]],
    positional_gradients: list[list[float]],
    learning_rate: float
) -> None:

    # Output weights
    for i in range(len(parameters.output_weights)):
        for j in range(len(parameters.output_weights[i])):
            parameters.output_weights[i][j] -= learning_rate * output_weight_gradients[i][j]

    # Output biases
    for i in range(len(parameters.output_biases)):
        parameters.output_biases[i] -= learning_rate * output_bias_gradients[i]

    # First feed-forward layer weights
    for i in range(len(parameters.weights_1)):
        for j in range(len(parameters.weights_1[i])):
            parameters.weights_1[i][j] -= learning_rate * weights_1_gradients[i][j]

    # First feed-forward layer biases
    for i in range(len(parameters.biases_1)):
        parameters.biases_1[i] -= learning_rate * biases_1_gradients[i]

    # Second feed-forward layer weights
    for i in range(len(parameters.weights_2)):
        for j in range(len(parameters.weights_2[i])):
            parameters.weights_2[i][j] -= learning_rate * weights_2_gradients[i][j]

    # Second feed-forward layer biases
    for i in range(len(parameters.biases_2)):
        parameters.biases_2[i] -= learning_rate * biases_2_gradients[i]

    # Query weights
    for i in range(len(parameters.w_Q)):
        for j in range(len(parameters.w_Q[i])):
            parameters.w_Q[i][j] -= learning_rate * W_Q_gradient[i][j]

    # Key weights
    for i in range(len(parameters.w_K)):
        for j in range(len(parameters.w_K[i])):
            parameters.w_K[i][j] -= learning_rate * W_K_gradient[i][j]

    # Value weights
    for i in range(len(parameters.w_V)):
        for j in range(len(parameters.w_V[i])):
            parameters.w_V[i][j] -= learning_rate * W_V_gradient[i][j]

    # Token embeddings
    for i in range(len(parameters.embedding_table)):
        for j in range(len(parameters.embedding_table[i])):
            parameters.embedding_table[i][j] -= learning_rate * embedding_gradients[i][j]

    # Positional embeddings
    for i in range(len(parameters.positional_table)):
        for j in range(len(parameters.positional_table[i])):
            parameters.positional_table[i][j] -= learning_rate * positional_gradients[i][j]