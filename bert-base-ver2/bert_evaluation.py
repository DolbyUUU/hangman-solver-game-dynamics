import random
from collections import defaultdict
from tqdm import tqdm
from bert_testing import load_model_and_tokenizer, evaluate_single_prediction


def load_test_words(file_path):
    """
    Load the test dataset from the specified file path.
    """
    with open(file_path, "r") as f:
        words = [line.strip() for line in f.readlines()]
    return [word for word in words if len(word) >= 2]


def evaluate_model_on_subset(dataset, use_bert_large, num_samples, verbose=False):
    """
    Evaluate the model's accuracy on a randomly selected subset of the dataset.
    """
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(use_bert_large)

    # Shuffle and select `num_samples` random words from the dataset
    random.shuffle(dataset)
    subset = dataset[:num_samples]

    # Initialize counters for analysis
    total_predictions_by_mask = defaultdict(int)
    correct_predictions_by_mask = defaultdict(int)
    total_predictions_by_length = defaultdict(int)
    correct_predictions_by_length = defaultdict(int)

    # Add a progress bar with tqdm
    with tqdm(total=len(subset), desc="Evaluating", unit="word") as pbar:
        for word in subset:
            # Analyze accuracy for 1 to 4 masked letters
            for num_masked_letters in range(1, 5):
                try:
                    guessed_letter, masked_characters = evaluate_single_prediction(
                        word, num_masked_letters, tokenizer, model, device, verbose
                    )

                    # Check if the guessed letter matches any of the masked characters
                    total_predictions_by_mask[num_masked_letters] += 1
                    if guessed_letter in masked_characters:
                        correct_predictions_by_mask[num_masked_letters] += 1

                    # Analyze accuracy by word length
                    word_length = len(word)
                    total_predictions_by_length[word_length] += 1
                    if guessed_letter in masked_characters:
                        correct_predictions_by_length[word_length] += 1

                except ValueError as e:
                    # Skip cases where no valid masking is possible
                    if verbose:
                        print(f"Skipping word '{word}' with {num_masked_letters} masked letters: {e}")

            # Update the progress bar after processing each word
            pbar.update(1)

    # Calculate accuracies by masked letters
    accuracy_by_mask = {
        num_masked: correct_predictions_by_mask[num_masked] / total_predictions_by_mask[num_masked]
        if total_predictions_by_mask[num_masked] > 0 else 0.0
        for num_masked in range(1, 5)
    }

    # Calculate accuracies by word length
    accuracy_by_length = {
        length: correct_predictions_by_length[length] / total_predictions_by_length[length]
        if total_predictions_by_length[length] > 0 else 0.0
        for length in sorted(total_predictions_by_length.keys())
    }

    return accuracy_by_mask, accuracy_by_length


if __name__ == "__main__":
    # Load the test dataset
    test_file_path = "datasets/words_test_disjoint.txt"
    test_words = load_test_words(test_file_path)

    # Number of samples to evaluate
    num_samples = 10000  # Change this value as needed

    # Use BERT large
    use_bert_large = False

    # Evaluate the model
    accuracy_by_mask, accuracy_by_length = evaluate_model_on_subset(
        test_words, use_bert_large, num_samples, verbose=False
    )

    # Print results
    print("\nRate of Correct Guesses (Hit Rate) by Number of Masked Letters:")
    for num_masked, accuracy in accuracy_by_mask.items():
        print(f"  {num_masked} Masked Letters: {accuracy * 100:.2f}%")

    print("\nRate of Correct Guesses (Hit Rate) by Word Length:")
    for length, accuracy in accuracy_by_length.items():
        print(f"  Word Length {length}: {accuracy * 100:.2f}%")