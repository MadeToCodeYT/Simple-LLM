import parameters

def get_position_aware_embeddings(embeddings: list[list[float]]) -> list[list[float]]:

    if len(embeddings) > parameters.MAX_CONTEXT_LENGTH:
        raise ValueError(
            f"Number of embeddings must be <= {parameters.MAX_CONTEXT_LENGTH}."
        )

    result = []

    for pos, embedding in enumerate(embeddings):
        vector = []

        for idx, value in enumerate(embedding):
            vector.append(
                value + parameters.positional_table[pos][idx]
            )

        result.append(vector)

    return result