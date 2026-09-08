import math

def create_training_example(tokens: list[int], start: int, context_length: int) -> tuple[list[int], list[int]]:
    input_tokens = tokens[start:start + context_length]
    target_tokens = tokens[start + 1:start + context_length + 1]

    return input_tokens, target_tokens

def calculate_loss(predictions: list[list[float]], targets: list[int]) -> float:
    losses = []

    for i, prediction in enumerate(predictions):
        correct_probability = prediction[targets[i]]
        loss = -math.log(correct_probability)
        losses.append(loss)

    return sum(losses) / len(losses)