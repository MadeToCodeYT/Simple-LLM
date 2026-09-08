import scripts.tokenizer as tokenizer
from random import uniform

EMBEDDING_SIZE = 32

embedding_table = []
for token in tokenizer.vocabulary.values():
    vector = []
    for _ in range(EMBEDDING_SIZE):
        vector.append(uniform(-1, 1))
    
    embedding_table.append(vector)

def tokens_to_embeddings(tokens: list[int]) -> list[list[float]]:
    embeddings = []
    for token in tokens:
        embeddings.append(embedding_table[token])

    return embeddings