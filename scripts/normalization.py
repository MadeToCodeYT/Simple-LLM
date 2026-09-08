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