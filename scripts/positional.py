from random import uniform

MAX_CONTEXT_LENGTH = 128
EMBEDDING_SIZE = 32

positional_table = []
for position in range(MAX_CONTEXT_LENGTH):
    vector = []
    for _ in range(EMBEDDING_SIZE):
        vector.append(uniform(-1, 1))
    
    positional_table.append(vector)

def get_position_aware_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    if len(embeddings) > MAX_CONTEXT_LENGTH:
        raise ValueError(f"Number of embeddings must be <= {MAX_CONTEXT_LENGTH}.")

    result = []
    for pos, embedding in enumerate(embeddings):
        vector = []
        for idx, value in enumerate(embedding):
            vector.append(value + positional_table[pos][idx])

        result.append(vector)

    return result