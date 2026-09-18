import tokenizer
import embeddings
import positional
import normalization
import attention
import parameters
import random

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
        parameters.weights_1,
        parameters.biases_1,
        parameters.weights_2,
        parameters.biases_2
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
        parameters.output_weights,
        parameters.output_biases
    )

def generate_probabilities(logits: list[float]) -> list[float]:
    return attention.apply_softmax(logits)

def select_token(probabilities: list[float]) -> int:
    # Introduces sampling based on probabilities
    # Uses random.choices with weights as probabilities

    indices = list(range(len(probabilities)))
    selected_index = random.choices(indices, weights=probabilities, k=1)[0]

    return selected_index

def generate_next_token(tokens: list[int]) -> int:
    tokens = tokens[-parameters.CONTEXT_LENGTH:]
    
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
        if tokenizer.reverse_vocabulary[next_token] == "<END>":
            break

        generated_tokens.append(next_token)

    return generated_tokens

def generate(prompt: str, num_tokens: int) -> str:
    tokens = tokenizer.text_to_tokens(prompt)

    generated_tokens = generate_text(tokens, num_tokens)

    new_tokens = generated_tokens[len(tokens):]

    return tokenizer.tokens_to_text(new_tokens)