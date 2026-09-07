from random import uniform

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

W_Q = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
W_K = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
W_V = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]

def get_qkv(embeddings: list[list[float]]) -> dict[str, list[list[float]]]:
    queries = []
    keys = []
    values = []

    for embedding in embeddings:
        queries.append(calc_prod(embedding, W_Q))
        keys.append(calc_prod(embedding, W_K))
        values.append(calc_prod(embedding, W_V))

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

def calculate_scores(queries: list[list[float]], keys: list[list[float]]):
    score_table = []

    for query in queries:
        row = []

        for key in keys:
            score = attention_score(query, key) / ATTENTION_SQRT
            row.append(score)

        score_table.append(row)

    return score_table