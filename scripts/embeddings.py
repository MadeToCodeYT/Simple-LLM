import parameters

def tokens_to_embeddings(tokens: list[int]) -> list[list[float]]:
    embeddings = []

    for token in tokens:
        embeddings.append(parameters.embedding_table[token])

    return embeddings