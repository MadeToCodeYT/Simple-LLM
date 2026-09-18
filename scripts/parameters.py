from random import uniform
import tokenizer

EMBEDDING_SIZE = 32
CONTEXT_LENGTH = 56
MAX_CONTEXT_LENGTH = 128
ATTENTION_SIZE = 32
ATTENTION_SQRT = ATTENTION_SIZE**0.5
VOCABULARY_LENGTH = len(tokenizer.vocabulary)

def generate_random_vector(length: int) -> list[float]:
    vector = []
    for _ in range(length):
        vector.append(uniform(-1, 1))

    return vector

w_Q = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_K = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_V = [generate_random_vector(EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]

weights_1 = [generate_random_vector(32) for _ in range(128)]
biases_1 = generate_random_vector(128)
weights_2 = [generate_random_vector(128) for _ in range(32)]
biases_2 = generate_random_vector(32)
output_weights = [generate_random_vector(32) for _ in range(VOCABULARY_LENGTH)]
output_biases = generate_random_vector(VOCABULARY_LENGTH)

embedding_table = [
    generate_random_vector(EMBEDDING_SIZE)
    for _ in range(VOCABULARY_LENGTH)
]

positional_table = [
    generate_random_vector(EMBEDDING_SIZE)
    for _ in range(MAX_CONTEXT_LENGTH)
]