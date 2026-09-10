import math

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

def calculate_residual_gradient(gradient: list[list[float]]) -> list[list[float]]:
    result = []

    for row in gradient:
        result.append(row.copy())

    return result

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
