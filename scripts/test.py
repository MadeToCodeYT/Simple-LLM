import tokenizer, embeddings, positional, attention, normalization, transformer

text = "Hi"

tokens = tokenizer.text_to_tokens(text)

embeds = embeddings.tokens_to_embeddings(tokens)

positional_info = positional.get_position_aware_embeddings(embeds)

print(len(embeds[0]))