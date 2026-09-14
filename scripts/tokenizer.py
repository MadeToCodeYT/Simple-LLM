# Tokenizer - Converts text into an array of numbers
# Needs to work in reverse too
# Needs to create a vocabulary by looping through dataset.txt

with open("data/dataset.txt", "r") as file:
	data = list(file.read())

def count_pairs(characters: list[str]) -> dict[tuple[str, str], int]:
	pairs = {}
	for i in range(len(characters)-1):
		char = characters[i]
		
		pairs[(char, characters[i+1])] = pairs.get((char, characters[i+1]), 0) + 1

	return pairs

def get_most_common_pair(pairs: dict[tuple[str, str], int]) -> tuple[ tuple[str, str], int ]:
	highest = 0
	pair = ("", "")

	for key, value in pairs.items():
		if value > highest:
			pair = key
			highest = value

	return pair, highest

def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
	merged = []
	i = 0

	while i < len(tokens):
		if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
			merged.append(tokens[i] + tokens[i + 1])
			i += 2
		else:
			merged.append(tokens[i])
			i += 1

	return merged

def learn_bpe(
	tokens: list[str],
	num_merges: int
) -> list[tuple[str, str]]:
	result = tokens.copy()
	merges = []

	for _ in range(num_merges):
		pairs = count_pairs(result)

		most_common, count = get_most_common_pair(pairs)

		if count == 1:
			break

		result = merge_pair(result, most_common)
		merges.append(most_common)

	return merges

def add_to_vocabulary(vocab: str, count: int) -> bool:
	if vocab in vocabulary:
		return False

	vocabulary[vocab] = count
	reverse_vocabulary[count] = vocab

	return True

merges = learn_bpe(data, 500)

vocabulary = {}
reverse_vocabulary = {}
count = 0

required = list(set(data)) # Gets all unique single characters from the dataset

for token in required:
	if add_to_vocabulary(token, count): # Only increment if token is unique
		count += 1

for pair in merges:
	token = pair[0] + pair[1]
	if add_to_vocabulary(token, count):
		count += 1

def text_to_tokens(text: str) -> list[int]:
	tokens = list(text)

	for merge in merges:
		tokens = merge_pair(tokens, merge)

	token_ids = []

	for token in tokens:
		token_ids.append(vocabulary[token])

	return token_ids

def tokens_to_text(tokens: list[int]) -> str:
	text = ""

	for token in tokens:
		text += reverse_vocabulary[token]

	return text

if __name__ == "__main__":
	print(f"Length of vocabulary: {count}")

	text = "The cat sat on the mat"

	print(tokens_to_text(text_to_tokens(text)))