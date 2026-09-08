import tokenizer, embeddings, positional, attention

text = "Hello World!"

tokens = tokenizer.text_to_tokens(text)

embeds = embeddings.tokens_to_embeddings(tokens)

positional_info = positional.get_position_aware_embeddings(embeds)

qkv = attention.get_qkv(embeds)
queries = qkv["queries"]
keys = qkv["keys"]
values = qkv["values"]

scores = attention.calculate_scores(queries, keys)

casual_mask = attention.apply_casual_mask(scores)

softmax = attention.apply_softmax_to_masked(casual_mask)

weighted = attention.apply_weighted_value_sums(softmax, values)

print(len(weighted)) # 12
print(len(weighted[0])) # 32