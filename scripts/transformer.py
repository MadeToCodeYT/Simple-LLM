import tokenizer
import embeddings
import positional
import normalization
import attention

weights_1 = [attention.generate_random_vector(32) for _ in range(128)]
biases_1 = attention.generate_random_vector(128)
weights_2 = [attention.generate_random_vector(128) for _ in range(32)]
biases_2 = attention.generate_random_vector(32)
output_weights = [attention.generate_random_vector(32) for _ in range(59)]
output_biases = attention.generate_random_vector(59)

def transformer_block(position_aware_embeddings: list[list[float]]) -> list[list[float]]:
    queries, keys, values = attention.get_qkv(position_aware_embeddings)

    scores = attention.calculate_scores(queries, keys)

    modified_scores = attention.apply_softmax_to_masked(
        attention.apply_casual_mask(scores)
    )

    weighted_value_sums = attention.apply_weighted_value_sums(
        modified_scores, values
    )

    residual_attention = normalization.add_residual_connection(
        position_aware_embeddings, weighted_value_sums
    )

    normalized_attention = normalization.apply_layernorm_to_vectors(
        residual_attention
    )

    feed_forward_output = normalization.apply_feed_forward_to_vectors(
        normalized_attention,
        weights_1,
        biases_1,
        weights_2,
        biases_2
    )

    residual_feed_forward = normalization.add_feed_forward_residual(
        normalized_attention, feed_forward_output
    )

    output = normalization.apply_layernorm_to_vectors(
        residual_feed_forward
    )

    return output

def generate_logits(vector: list[float]) -> list[float]:
    return normalization.linear_layer(
        vector,
        output_weights,
        output_biases
    )

def generate_probabilities(logits: list[float]) -> list[float]:
    return attention.apply_softmax(logits)

def select_token(probabilities: list[float]) -> int:
    # Should introduct sampling at some point but for now chooses the highest token's probability
    # [ 0.3, 0.2, 0.5 ]
    #              ^

    index = 0
    for i in range(len(probabilities)):
        if probabilities[index] < probabilities[i]:
            index = i

    return index

def generate_next_token(tokens: list[int]) -> int:
    embeds = embeddings.tokens_to_embeddings(tokens)

    positional_info = positional.get_position_aware_embeddings(embeds)

    output_vectors = transformer_block(positional_info)

    logits = generate_logits(output_vectors[-1])

    probabilities = generate_probabilities(logits)

    chosen_token = select_token(probabilities)

    return chosen_token

def generate_text(tokens: list[int], num_tokens: int) -> list[int]:
    generated_tokens = tokens.copy()

    for _ in range(num_tokens):
        next_token = generate_next_token(generated_tokens)
        generated_tokens.append(next_token)

    return generated_tokens

def generate(prompt: str, num_tokens: int) -> str:
    tokens = tokenizer.text_to_tokens(prompt)

    generated_tokens = generate_text(tokens, num_tokens)

    return tokenizer.tokens_to_text(generated_tokens)