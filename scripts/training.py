import tokenizer
import embeddings
import positional
import attention
import normalization
import backpropagation
import model
import parameters

import copy
import random
import os

def train_step(tokens: list[int], start: int, context_length: int, learning_rate: float) -> float:
    input_tokens, target_tokens = backpropagation.create_training_example(
        tokens,
        start,
        context_length
    )

    token_embeddings = embeddings.tokens_to_embeddings(input_tokens)

    position_aware_embeddings = positional.get_position_aware_embeddings(
        token_embeddings
    )

    queries, keys, values = attention.get_qkv(position_aware_embeddings)
    
    scores = attention.calculate_scores(queries, keys)

    masked_scores = attention.apply_casual_mask(scores)

    attention_weights = attention.apply_softmax_to_masked(masked_scores)

    attention_outputs = attention.apply_weighted_value_sums(
        attention_weights,
        values
    )

    residual_attention = normalization.add_residual_connection(
        position_aware_embeddings,
        attention_outputs
    )

    first_layernorm_input = residual_attention

    normalized_attention_vectors = normalization.apply_layernorm_to_vectors(
        residual_attention
    )

    normalized_attention = normalized_attention_vectors[-1]

    linear_1_output = normalization.linear_layer(
        normalized_attention,
        parameters.weights_1,
        parameters.biases_1
    )

    relu_output = normalization.relu(
        linear_1_output
    )

    linear_2_output = normalization.linear_layer(
        relu_output,
        parameters.weights_2,
        parameters.biases_2
    )

    final_layernorm_input = [
        normalized_attention[i] + linear_2_output[i]
        for i in range(len(normalized_attention))
    ]

    output_vector = normalization.layerNorm(
        final_layernorm_input
    )

    logits = normalization.linear_layer(
        output_vector,
        parameters.output_weights,
        parameters.output_biases
    )

    predictions = attention.apply_softmax(logits)

    loss = backpropagation.calculate_loss(
        [predictions],
        [target_tokens[-1]]
    )

    gradients = backpropagation.backpropagate(
        predictions,
        target_tokens[-1],
        output_vector,
        first_layernorm_input[-1],
        final_layernorm_input,
        relu_output,
        linear_1_output,
        normalized_attention,
        position_aware_embeddings,
        attention_weights,
        queries,
        keys,
        values,
        input_tokens
    )

    (
        output_weight_gradients,
        output_bias_gradients,
        weights_1_gradients,
        biases_1_gradients,
        weights_2_gradients,
        biases_2_gradients,
        W_Q_gradient,
        W_K_gradient,
        W_V_gradient,
        embedding_gradients,
        positional_gradients
    ) = gradients

    backpropagation.update_parameters(
        output_weight_gradients,
        output_bias_gradients,
        weights_1_gradients,
        biases_1_gradients,
        weights_2_gradients,
        biases_2_gradients,
        W_Q_gradient,
        W_K_gradient,
        W_V_gradient,
        embedding_gradients,
        positional_gradients,
        learning_rate
    )

    return loss

def snapshot_parameters() -> list:
    return copy.deepcopy([
        parameters.w_Q,
        parameters.w_K,
        parameters.w_V,
        parameters.weights_1,
        parameters.biases_1,
        parameters.weights_2,
        parameters.biases_2,
        parameters.output_weights,
        parameters.output_biases,
        parameters.embedding_table,
        parameters.positional_table
    ])

def train(
    tokens: list[int],
    context_length: int,
    learning_rate: float,
    epochs: int,
    starting_epoch: int
) -> None:
    length = len(tokens) - context_length
    starts = list(range(length))

    last_completed_epoch = starting_epoch
    last_saved_parameters = snapshot_parameters()

    try:
        for epoch in range(starting_epoch+1, epochs+1):
            total = 0
            random.shuffle(starts)

            for i, start in enumerate(starts):
                loss = train_step(
                    tokens,
                    start,
                    context_length,
                    learning_rate
                )

                total += loss
                print(f"Epoch: {epoch}, Completion: {round(i/length*100, 1)}%, Loss: {loss}")

            if length != 0:
                print(f"Epoch: {epoch}, Avg. Loss: {total/length}")

            # This epoch is now fully done -> take a clean snapshot
            last_completed_epoch = epoch
            last_saved_parameters = snapshot_parameters()

        model.save_model(*last_saved_parameters)
    except KeyboardInterrupt:
        # Only ever save the last epoch that fully finished, never
        # whatever's mid-flight when the interrupt lands
        model.save_checkpoint(
            last_completed_epoch,
            *last_saved_parameters
        )


tokens = tokenizer.text_to_tokens(
    open("data/dataset.txt", "r").read()
)

starting_epoch = 0

if os.path.exists("model/checkpoint.json"):
    starting_epoch = model.load_checkpoint()

train(
    tokens,
    context_length=parameters.CONTEXT_LENGTH,
    learning_rate=0.001,
    epochs=300,
    starting_epoch=starting_epoch
)
