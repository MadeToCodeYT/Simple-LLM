import tokenizer
import training

tokens = tokenizer.text_to_tokens(
    open("data/dataset.txt", "r").read()
)

small_dataset = tokens[:2]

final_loss = training.train(
    small_dataset,
    context_length=16,
    learning_rate=0.001,
    epochs=5
)

print("Final loss:", final_loss)