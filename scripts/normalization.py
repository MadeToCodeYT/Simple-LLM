from random import uniform

def generate_random_vector(length: int) -> list[float]:
    vector = []
    for _ in range(length):
        vector.append(uniform(-1, 1))

    return vector

def layerNorm(layer: list[float]) -> list[float]:
    mean = sum(layer) / len(layer)

    # Calculate variance
    sqr_differences = [(num-mean)**2 for num in layer]
    variance = sum(sqr_differences) / len(layer)

    vector = []
    epsilon = 0.000001
    for num in layer:
        vector.append((num - mean)/((variance+epsilon)**0.5))

    return vector

def apply_layernorm_to_vectors(vectors: list[list[float]]) -> list[list[float]]:
    result = []
    for vector in vectors:
        result.append(layerNorm(vector))

    return result

def add_residual_connection(position_aware_embeddings: list[list[float]], attention_outputs: list[list[float]]) -> list[list[float]]:
    residual = []
    for i in range(len(position_aware_embeddings)):
        row = []
        for j in range(len(position_aware_embeddings[i])):
            row.append(position_aware_embeddings[i][j] + attention_outputs[i][j])

        residual.append(row)

    return residual

# Weights -> 128 vectors containing 32 weights
def linear_layer(input: list[float], weights: list[list[float]], biases: list[float]) -> list[float]:
    result = []

    for i, weight in enumerate(weights):
        total = 0
        for j in range(len(weight)):
            total += input[j] * weight[j]
        result.append(total + biases[i])

    return result

def relu(input: list[float]) -> list[float]:
    output = []
    for val in input:
        output.append(max(0, val))

    return output

def apply_feed_forward_to_vectors(tokens: list[list[float]], weights_1: list[list[float]], bias_1: list[float], weights_2: list[list[float]], bias_2: list[float]) -> list[list[float]]:
    applied = []

    for token in tokens:
        hidden = linear_layer(token, weights_1, bias_1)

        activated = relu(hidden)

        output = linear_layer(activated, weights_2, bias_2)

        applied.append(output)

    return applied

def add_feed_forward_residual(vectors: list[list[float]], feed_forward_outputs: list[list[float]]) -> list[list[float]]:
    residual = []
    for i in range(len(vectors)):
        row = []
        for j in range(len(vectors[i])):
            row.append(vectors[i][j] + feed_forward_outputs[i][j])

        residual.append(row)

    return residual