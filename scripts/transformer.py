import normalization
import attention

weights_1 = [attention.generate_random_vector(32) for _ in range(128)]
biases_1 = attention.generate_random_vector(128)
weights_2 = [attention.generate_random_vector(128) for _ in range(32)]
biases_2 = attention.generate_random_vector(32)

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