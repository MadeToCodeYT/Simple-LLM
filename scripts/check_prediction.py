import tokenizer
import model
import embeddings
import positional
import transformer
import parameters

model.load_model()

# A prefix taken verbatim from a training example, right before the word
# we expect it to predict next ("attracts").
prefix = "User: What is gravity?\nLarge Language Model: Gravity is the force that "

tokens = tokenizer.text_to_tokens(prefix)

# Use the same trailing-window logic generation uses
windowed = tokens[-parameters.CONTEXT_LENGTH:]

embeds = embeddings.tokens_to_embeddings(windowed)
positional_info = positional.get_position_aware_embeddings(embeds)
output_vectors = transformer.transformer_block(positional_info)

logits = transformer.generate_logits(output_vectors[-1])
probabilities = transformer.generate_probabilities(logits)

# Show the model's top 5 guesses for the very next token, with probabilities
ranked = sorted(
    range(len(probabilities)),
    key=lambda i: probabilities[i],
    reverse=True
)

print(f"Prefix: {prefix!r}\n")
print("Top 5 predicted next tokens:")
for idx in ranked[:5]:
    token_text = tokenizer.reverse_vocabulary[idx]
    print(f"  {token_text!r}  ({probabilities[idx]*100:.2f}%)")