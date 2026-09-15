# Tokenizer - Converts text into an array of numbers
# Needs to work in reverse too
# Needs to create a vocabulary by looping through dataset.txt


SPECIAL_TOKEN = "<END>"

with open("data/dataset.txt", "r") as file:
	raw_data = file.read()


def split_special_tokens(text: str) -> list[str]:
	tokens = []

	parts = text.split(SPECIAL_TOKEN)

	for i, part in enumerate(parts):
		tokens.extend(list(part))

		if i < len(parts) - 1:
			tokens.append(SPECIAL_TOKEN)

	return tokens


data = split_special_tokens(raw_data)


# These tokens are boundaries.
# BPE is not allowed to merge anything with them.
restricted = ["\n", " ", SPECIAL_TOKEN]


def count_pairs(characters: list[str]) -> dict[tuple[str, str], int]:
	pairs = {}

	for i in range(len(characters) - 1):
		char = characters[i]
		next_char = characters[i + 1]

		# Don't merge across restricted tokens
		if char in restricted or next_char in restricted:
			continue

		pairs[(char, next_char)] = pairs.get((char, next_char), 0) + 1

	return pairs


def get_most_common_pair(
	pairs: dict[tuple[str, str], int]
) -> tuple[tuple[str, str], int]:

	highest = 0
	pair = ("", "")

	for key, value in pairs.items():
		if value > highest:
			pair = key
			highest = value

	return pair, highest


def merge_pair(
	tokens: list[str],
	pair: tuple[str, str]
) -> list[str]:

	merged = []
	i = 0

	while i < len(tokens):
		if (
			i + 1 < len(tokens)
			and tokens[i] == pair[0]
			and tokens[i + 1] == pair[1]
		):
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

		# No useful pairs remain
		if count <= 1:
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

# Add all individual characters and the special token
required = list(set(data))
for token in required:
	if add_to_vocabulary(token, count):
		count += 1

# Add all BPE tokens
for pair in merges:
	token = pair[0] + pair[1]

	if add_to_vocabulary(token, count):
		count += 1


def text_to_tokens(text: str) -> list[int]:
	# Make <END> one indivisible token
	tokens = split_special_tokens(text)

	# Apply every learned BPE merge
	for merge in merges:
		tokens = merge_pair(tokens, merge)

	# Convert tokens to IDs
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
	print(f"Vocabulary count: {count}")
	print(f"Original characters: {len(data)}")

	tokenized_data = data.copy()

	for merge in merges:
		tokenized_data = merge_pair(tokenized_data, merge)

	print(f"Tokens after BPE: {len(tokenized_data)}")

	print("\nLongest tokens:")

	longest_tokens = sorted(
		vocabulary.keys(),
		key=len,
		reverse=True
	)

	for token in longest_tokens[:20]:
		print(f"{len(token):3}  {repr(token)}")