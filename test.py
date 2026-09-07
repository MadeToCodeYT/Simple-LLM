import attention
import tokenizer
import embeddings

text = "Hello World!"

tokens = tokenizer.text_to_tokens(text)

embeds = embeddings.tokens_to_embeddings(tokens)

result = attention.get_qkv(embeds)

print(len(result["queries"]) == 12)
print(len(result["queries"][0]) == 32)