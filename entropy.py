import math

# Filter the words in the list based on a pattern.
def filter_candidates(word_list, pattern):

    matching_words = []
    for word in word_list:
        if len(word) != len(pattern):
            continue

        match = True
        for i in range(len(pattern)):
            if pattern[i] != '.' and pattern[i] != word[i]:
                match = False
                break

        if match:
            matching_words.append(word)

    return matching_words

# Calculate entropy for each unguessed letter based on the word list.
def calculate_entropy(words, guessed_letters):
    
    total_words = len(words)
    entropy_by_letter = {}

    # all possible letters that are unguessed
    all_letters = "abcdefghijklmnopqrstuvwxyz"
    unguessed_letters = []
    for letter in all_letters:
        if letter not in guessed_letters:
            unguessed_letters.append(letter)

    # loop through each unguessed letter to calculate entropy
    for letter in unguessed_letters:
        # count patterns for this letter
        pattern_counts = {}
        for word in words:
            # create a pattern for the word (e.g., "010101" for word "banana" with letter "a")
            pattern = ""
            for char in word:
                if char == letter:
                    pattern += "1"
                else:
                    pattern += "0"

            # count how many words match this pattern
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1

        # calculate entropy for this letter
        entropy = 0.0
        for pattern in pattern_counts:
            count = pattern_counts[pattern]
            prob = count / total_words
            if prob > 0:
                # calculate Shannon information entropy
                entropy -= prob * math.log(prob)

        # store the entropy for this letter, to be sorted later
        entropy_by_letter[letter] = entropy

    return entropy_by_letter