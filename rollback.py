def normalized_similarity(word1, word2):
    # ensure two words are of the same length
    if len(word1) != len(word2):
        raise ValueError("Words must be of the same length.")

    # count the number of different characters
    differing_characters = 0
    for char1, char2 in zip(word1, word2):
        if char1 != char2:
            differing_characters += 1

    # calculate the normalized similarity
    similarity = 1 - (differing_characters / len(word1))
    
    return similarity

def rollback_iterative(clean_word, candidate_letters, num_unknown_slots, guessed_letters,
                       training_words_set, similarity_threshold=0.95, max_iterations=3,
                       enable_two_unknowns=True):
    """
    The training set and testing set are disjoint.
    We need to ensure the word to be guessed is not in the training set or too similar to words in the training set.
    """

    attempted_letters = set()
    iteration = 0

    while iteration < max_iterations:
        for letter in candidate_letters:
            # skip letters already guessed or attempted
            if letter in guessed_letters or letter in attempted_letters:
                continue

            # replace '.' in the word with the current letter
            potential_word = clean_word.replace('.', letter)

            # Case 1: Two unknown slots
            if enable_two_unknowns and num_unknown_slots == 2:
                if be_similar_to_training_words(potential_word, training_words_set, similarity_threshold):
                    attempted_letters.add(letter)
                    continue
                else:
                    return letter

            # Case 2: One unknown slot
            elif num_unknown_slots == 1:
                if potential_word in training_words_set:
                    attempted_letters.add(letter)
                    continue
                else:
                    return letter

            # Default case: Return the letter directly
            else:
                return letter

        iteration += 1

    # if the threshold is set too loose, all letters will be invalid
    raise ValueError("No valid letters to guess after iterative rollback.")

def be_similar_to_training_words(word, training_words_set, similarity_threshold):
    # check if a word is too similar to any word in the training set.
    for train_word in training_words_set:
        if len(train_word) == len(word):  # only compare words of the same length
            similarity = normalized_similarity(word, train_word)  # Rename local variable to `similarity`
            if similarity >= similarity_threshold:  # if the similarity exceeds threshold
                return True
    return False