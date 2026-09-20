from random import uniform
import tokenizer

EMBEDDING_SIZE = 32
CONTEXT_LENGTH=160
MAX_CONTEXT_LENGTH=192
ATTENTION_SIZE = 32
ATTENTION_SQRT = ATTENTION_SIZE**0.5
VOCABULARY_LENGTH = len(tokenizer.vocabulary)

def generate_random_vector(length: int, fan_in: int) -> list[float]:
    limit = 1 / (fan_in ** 0.5)

    vector = []
    for _ in range(length):
        vector.append(uniform(-limit, limit))

    return vector

w_Q = [generate_random_vector(EMBEDDING_SIZE, EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_K = [generate_random_vector(EMBEDDING_SIZE, EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]
w_V = [generate_random_vector(EMBEDDING_SIZE, EMBEDDING_SIZE) for _ in range(EMBEDDING_SIZE)]

weights_1 = [generate_random_vector(32, 32) for _ in range(128)]
biases_1 = generate_random_vector(128, 32)
weights_2 = [generate_random_vector(128, 128) for _ in range(32)]
biases_2 = generate_random_vector(32, 128)
output_weights = [generate_random_vector(32, 32) for _ in range(VOCABULARY_LENGTH)]
output_biases = generate_random_vector(VOCABULARY_LENGTH, 32)

embedding_table = [
    generate_random_vector(EMBEDDING_SIZE, EMBEDDING_SIZE)
    for _ in range(VOCABULARY_LENGTH)
]

positional_table = [
    generate_random_vector(EMBEDDING_SIZE, EMBEDDING_SIZE)
    for _ in range(MAX_CONTEXT_LENGTH)
]