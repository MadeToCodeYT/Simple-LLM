import tokenizer

with open("data/dataset.txt") as f:
    content = f.read()

pairs = [p.strip() for p in content.split("<END>") if p.strip()]
lengths = [len(tokenizer.text_to_tokens(p)) for p in pairs]
lengths.sort()

print("max:", lengths[-1])
print("median:", lengths[len(lengths)//2])
print("count over 96:", sum(1 for l in lengths if l > 96))
print("count over 96:", sum(1 for l in lengths if l > 160))