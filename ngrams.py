import collections
import numpy as np

def build_n_gram(word_list, max_n_of_ngrams, use_noncumulative_frequency=True):
    # store n-gramsn
    n_grams = {}
    for n in range(1, max_n_of_ngrams + 1):
        n_grams[n] = {}  # empty dictionary for each n

    # process each word in the word list
    for word in word_list:
        # add padding to the word
        word = '#' + word + '$'

        if use_noncumulative_frequency:
            # for noncumulative frequency: count unique n-grams in a word
            for n in range(1, max_n_of_ngrams + 1):
                seen_n_grams = set()  # track n-grams seen in this word
                for i in range(len(word) - n + 1):
                    n_gram = word[i:i + n]
                    if n_gram not in seen_n_grams:
                        seen_n_grams.add(n_gram)  # mark this n-gram as seen
                        if n_gram in n_grams[n]:
                            n_grams[n][n_gram] += 1
                        else:
                            n_grams[n][n_gram] = 1
        else:
            # for cumulative frequency: count all occurrences of n-grams in a word
            for n in range(1, max_n_of_ngrams + 1):
                for i in range(len(word) - n + 1):
                    n_gram = word[i:i + n]
                    if n_gram in n_grams[n]:
                        n_grams[n][n_gram] += 1
                    else:
                        n_grams[n][n_gram] = 1
    return n_grams

def build_n_gram_from_file(file_path, max_n_of_ngrams=6, use_noncumulative_frequency=True):
    # build n-grams from words in a file, for main script
    with open(file_path, 'r') as f:
        word_list = f.read().splitlines()
    return build_n_gram(word_list, max_n_of_ngrams, use_noncumulative_frequency)

def get_n_gram_prob(n_grams, word, guessed_letters, ngram_weights=[1, 2, 3, 4, 5, 6]):
    # map letters 'a' to 'z' to indices 0 to 25
    letters = 'abcdefghijklmnopqrstuvwxyz'
    letter_indices = {letter: idx for idx, letter in enumerate(letters)}

    # determine already guessed letters
    not_guessed_letters = [letter for letter in letters if letter not in guessed_letters]

    # store the probability for each letter
    next_letter_prob = [0.0] * 26

    # 1-grams
    gram_1_counts = [0.0] * 26  # counts for each letter
    total_gram_1 = 0.0
    for letter in not_guessed_letters:
        idx = letter_indices[letter]
        count = n_grams[1].get(letter, 0)
        gram_1_counts[idx] = count
        total_gram_1 += count

    # add weighted probabilities from 1-grams
    if total_gram_1 > 0:
        weight = ngram_weights[0]
        for idx in range(26):
            next_letter_prob[idx] += weight * (gram_1_counts[idx] / total_gram_1)

    # n-grams with 1 unknown position (n = 2 to 6)
    for n in range(2, 7):
        weight = ngram_weights[n - 1]
        gram_counts = [0.0] * 26  # counts for each letter
        total_counts = 0.0
        for i in range(len(word) - n + 1):
            gram = word[i:i + n]
            # find unknown slots
            unknown_indices = [j for j, c in enumerate(gram) if c == '.']
            if len(unknown_indices) == 1:
                idx_unknown = unknown_indices[0]
                # create prefixes and suffixes to avoid nested loops
                prefix = gram[:idx_unknown]
                suffix = gram[idx_unknown + 1:]
                # generate possible n-grams by replacing the unknown slots
                possible_ngrams = {
                    letter: prefix + letter + suffix for letter in not_guessed_letters
                }
                for letter, ngram in possible_ngrams.items():
                    count = n_grams.get(n, {}).get(ngram, 0)
                    if count > 0:
                        idx = letter_indices[letter]
                        gram_counts[idx] += count
                        total_counts += count
        # add weighted probabilities from n-grams with one unknown
        if total_counts > 0:
            for idx in range(26):
                next_letter_prob[idx] += weight * (gram_counts[idx] / total_counts)

    # n-grams with 2 unknown positions (n = 4 to 6)
    for n in range(4, 7):
        weight = ngram_weights[n - 1]
        gram_counts = [0.0] * 26  # counts for each letter
        total_counts = 0.0
        for i in range(len(word) - n + 1):
            gram = word[i:i + n]
            # find unknown slots
            unknown_indices = [j for j, c in enumerate(gram) if c == '.']
            if len(unknown_indices) == 2:
                idx_unknown1, idx_unknown2 = unknown_indices
                # create parts of the gram
                part1 = gram[:idx_unknown1]
                part2 = gram[idx_unknown1 + 1:idx_unknown2]
                part3 = gram[idx_unknown2 + 1:]
                # generate possible n-grams by replacing the unknown positions
                possible_ngrams = {}
                for letter1 in not_guessed_letters:
                    for letter2 in not_guessed_letters:
                        if letter1 != letter2:
                            ngram = part1 + letter1 + part2 + letter2 + part3
                            possible_ngrams[(letter1, letter2)] = ngram
                # process the possible n-grams
                for (letter1, letter2), ngram in possible_ngrams.items():
                    count = n_grams.get(n, {}).get(ngram, 0)
                    if count > 0:
                        idx1 = letter_indices[letter1]
                        idx2 = letter_indices[letter2]
                        gram_counts[idx1] += count
                        gram_counts[idx2] += count
                        total_counts += 2 * count  # count both letters
        # add weighted probabilities from n-grams with two unknowns
        if total_counts > 0:
            for idx in range(26):
                next_letter_prob[idx] += weight * (gram_counts[idx] / total_counts)

    # normalize the final probabilities
    total_prob = sum(next_letter_prob)
    if total_prob > 0:
        next_letter_prob = [prob / total_prob for prob in next_letter_prob]

    return np.array(next_letter_prob) # need to be numpy array