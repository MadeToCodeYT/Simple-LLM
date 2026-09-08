from random import uniform
import math

def generate_random_vector(length: int) -> list[float]:
    vector = []
    for _ in range(length):
        vector.append(uniform(-1, 1))

    return vector

def calc_prod(vector: list[float], matrix: list[list[float]]) -> list[float]:
    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix columns must match vector length.")

    product = []
    
    # Calculate the dot product for each row
    for row in matrix:
        row_total = 0
        for i in range(len(vector)):
            row_total += row[i] * vector[i]
        product.append(row_total)
        
    return product

EMBEDDING_SIZE = 32
ATTENTION_SIZE = 32
ATTENTION_SQRT = ATTENTION_SIZE**0.5

w_Q = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_K = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_V = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]

def get_qkv(embeddings: list[list[float]]) -> dict[str, list[list[float]]]:
    queries = []
    keys = []
    values = []

    for embedding in embeddings:
        queries.append(calc_prod(embedding, w_Q))
        keys.append(calc_prod(embedding, w_K))
        values.append(calc_prod(embedding, w_V))

    return {
        "queries": queries,
        "keys": keys,
        "values": values,
    }

def attention_score(query: list[float], key: list[float]) -> float:
    score = 0

    for i in range(len(query)):
        score += query[i]*key[i]

    return score

def calculate_scores(queries: list[list[float]], keys: list[list[float]]) -> list[list[float]]:
    score_table = []

    for query in queries:
        row = []

        for key in keys:
            score = attention_score(query, key) / ATTENTION_SQRT
            row.append(score)

        score_table.append(row)

    return score_table

def apply_casual_mask(score_table: list[list[float]]):
    """
    Make each token only be able to see itself and the tokens before it
    """
    
    masked_table = [[float("-inf") for _ in range(len(score_table[0]))] for _ in range(len(score_table))]
    
    for i in range(len(score_table)):
        for j in range(len(score_table[0])):
            if j > i:
                break
            masked_table[i][j] = score_table[i][j]

    return masked_table

def apply_softmax(row: list[float]) -> list[float]:
    """
    Modifies each value so that:
        - Every value between `0` and `1`
        - The values add up to **1**
        - `-inf` becomes **0**
        - Larger scores get larger probabilities
    """

    exponentials = [math.exp(val) for val in row]
    total = sum(exponentials)
    probabilities = [exp/total for exp in exponentials]

    return probabilities

def apply_softmax_to_masked(masked_score_table: list[list[float]]) -> list[list[float]]:
    result = []

    for row in masked_score_table:
        result.append(apply_softmax(row))

    return result

def apply_weighted_value_sums(attention_weights: list[list[float]], values: list[list[float]]) -> list[list[float]]:
    result = []

    for row in attention_weights:
        vector = [0.0] * len(values[0])

        for i, weight in enumerate(row):
            for j, value in enumerate(values[i]):
                vector[j] += weight * value

        result.append(vector)

    return result