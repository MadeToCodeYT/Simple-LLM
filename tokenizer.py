# Tokenizer - Converts text into an array of numbers
# Needs to work in reverse too
# Needs to create a vocabulary by looping through dataset.txt

with open("dataset.txt", "r") as file:
    data = file.readlines()

vocabulary = {}
reverse_vocabulary = {}
count = 0

# Initialize the vocabulary
for line in data:
    for char in line:
        if vocabulary.get(char, None) is None:
            vocabulary[char] = count
            reverse_vocabulary[count] = char
            count += 1

def text_to_tokens(text: str) -> list[int]:
	tokens = []
	for char in text:
		tokens.append(vocabulary[char])

	return tokens

def tokens_to_text(tokens: list[int]) -> str:
	text = ""
	for id in tokens:
		text += reverse_vocabulary[id]

	return text

if __name__ == "__main__":
	print(f"Vocabulary Size: {len(vocabulary)}")