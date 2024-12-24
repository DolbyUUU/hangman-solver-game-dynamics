import collections
import string

def read_words(file_path):
    with open(file_path, "r") as file:
        words = file.read().splitlines()
    return words

def calculate_word_statistics(words):
    # Total number of words
    total_words = len(words)

    # Word length distribution
    word_lengths = [len(word) for word in words]
    length_distribution = collections.Counter(word_lengths)

    # Letter frequencies
    all_letters = "".join(words)
    letter_frequencies = collections.Counter(all_letters)

    # Words containing each letter at least once
    letter_in_word_count = {
        letter: sum(1 for word in words if letter in word) for letter in string.ascii_lowercase
    }

    # Most common starting and ending letters
    starting_letters = [word[0] for word in words if word]
    ending_letters = [word[-1] for word in words if word]
    starting_letter_freq = collections.Counter(starting_letters)
    ending_letter_freq = collections.Counter(ending_letters)

    # Longest and shortest words
    if words:
        shortest_word_length = min(len(word) for word in words)
        shortest_words = sorted([word for word in words if len(word) == shortest_word_length])
        longest_word = max(words, key=len)
    else:
        shortest_word_length = 0
        shortest_words = []
        longest_word = ""

    # Count words that consist of only one kind of letter
    single_letter_words = [word for word in words if len(set(word)) == 1]
    single_letter_word_count = len(single_letter_words)

    return {
        "total_words": total_words,
        "length_distribution": length_distribution,
        "letter_frequencies": letter_frequencies,
        "letter_in_word_count": letter_in_word_count,
        "starting_letter_freq": starting_letter_freq,
        "ending_letter_freq": ending_letter_freq,
        "longest_word": longest_word,
        "shortest_word_length": shortest_word_length,
        "shortest_words": shortest_words,
        "single_letter_word_count": single_letter_word_count,
        "single_letter_words": single_letter_words,
    }

def main():
    file_path = "datasets/words_250000_train.txt"
    words = read_words(file_path)

    stats = calculate_word_statistics(words)
    
    print("=== Word Statistics ===")
    print(f"Total words: {stats['total_words']}")

    print("\n=== Word Length Distribution ===")
    for length, count in sorted(stats["length_distribution"].items()):
        print(f"Length {length}: {count} words")

    print("\n=== Letter Frequencies ===")
    for letter, freq in sorted(stats["letter_frequencies"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{letter}: {freq}")

    print("\n=== Words Containing Each Letter ===")
    for letter, count in sorted(stats["letter_in_word_count"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{letter}: {count} words")

    print("\n=== Most Common Starting Letters ===")
    for letter, freq in sorted(stats["starting_letter_freq"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{letter}: {freq}")

    print("\n=== Most Common Ending Letters ===")
    for letter, freq in sorted(stats["ending_letter_freq"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{letter}: {freq}")

    print("\n=== Longest and Shortest Words ===")
    print(f"Longest word: {stats['longest_word']} (Length: {len(stats['longest_word'])})")
    print(f"Shortest words (Length: {stats['shortest_word_length']}):")
    print(", ".join(stats["shortest_words"]))

    print("\n=== Words Consisting of Only One Kind of Letter ===")
    print(f"Total words: {stats['single_letter_word_count']}")


if __name__ == "__main__":
    main()