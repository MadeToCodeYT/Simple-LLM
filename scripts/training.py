import math
from parameters import *

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
                    gradient[i][j] * keys[j][k] / ATTENTION_SQRT
                )

                keys_gradients[j][k] += (
                    gradient[i][j] * queries[i][k] / ATTENTION_SQRT
                )

    return queries_gradients, keys_gradients


def backpropagate(
    predictions: list[float], 
    target: int, 
    output_vector: list[float], 
    first_layernorm_input: list[float],
    final_layernorm_input: list[float], 
    relu_output: list[float],
    linear_1_output: list[float],
    normalized_attention: list[float]
):
    logits_gradient = calculate_logits_gradient(predictions, target)

    output_weight_gradients = calculate_output_weight_gradients(output_vector, logits_gradient)

    output_bias_gradients = calculate_output_bias_gradients(logits_gradient)

    output_vector_gradient = calculate_output_vector_gradient(logits_gradient, output_weights)

    final_layernorm_gradient = calculate_layernorm_gradient(output_vector_gradient, final_layernorm_input)

    normalized_attention_vector, feed_forward_output_gradient = calculate_residual_gradient(final_layernorm_gradient)

    weights_2_gradients, biases_2_gradients, relu_gradient = calculate_linear_layer_gradients(feed_forward_output_gradient, relu_output, weights_2)

    linear_1_gradient = calculate_relu_gradient(relu_gradient, linear_1_output)

    weights_1_gradients, biases_1_gradients, normalized_attention_gradient_from_ffn = calculate_linear_layer_gradients(
        linear_1_gradient, normalized_attention, weights_1
    )

    normalized_attention_gradient = [a + b for a, b in zip(normalized_attention_vector, normalized_attention_gradient_from_ffn)]

    residual_attention_gradient = calculate_layernorm_gradient(normalized_attention_gradient, first_layernorm_input)

    position_embeddings_gradient, attention_output_gradient = calculate_residual_gradient(residual_attention_gradient)