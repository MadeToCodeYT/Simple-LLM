from random import uniform

EMBEDDING_SIZE = 32
MAX_CONTEXT_LENGTH = 128
ATTENTION_SIZE = 32
ATTENTION_SQRT = ATTENTION_SIZE**0.5

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
output_weights = [generate_random_vector(32) for _ in range(59)]
output_biases = generate_random_vector(59)