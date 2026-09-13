import tokenizer
import training

tokens = tokenizer.text_to_tokens("Hello World!")

learning_rate = 0.001
context_length = 10

for step in range(20):
    loss = training.train_step(
        tokens,
        0,
        context_length,
        learning_rate
    )

    print(f"Step {step + 1}: Loss = {loss}")