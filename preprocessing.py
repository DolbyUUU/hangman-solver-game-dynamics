input_path = "datasets/words_250000_train.txt"
output_path = "datasets/words_250000_train_cleaned.txt"

with open(input_path, "r") as infile:
    words = infile.readlines()

cleaned_words = []

min_length = 4
max_length = 25

for word in words:
    word = word.strip()

    if len(word) < min_length:  # exclude too short words
        continue
    if len(word) > max_length:  # exclude too long words
        continue
    if len(set(word)) == 1:  # exclude too long words with only one kind of letter, e.g., "kkkkk"
        continue

    cleaned_words.append(word)

with open(output_path, "w") as outfile:
    for word in cleaned_words:
        outfile.write(word + "\n")

print(f"Cleaned data saved to {output_path}")