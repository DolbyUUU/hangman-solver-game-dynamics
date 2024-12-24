import string
import torch
import random
from transformers import BertForMaskedLM
from torch.nn.functional import softmax
from bert_finetuning_base import CustomTokenizer


# Load model and tokenizer
def load_model_and_tokenizer(use_bert_large=False):
    """
    Load the model and tokenizer based on the specified flag.
    """
    if use_bert_large:
        model_path = ""
    else:
        model_path = "./bert-base-ver2/hangman_bert_base"
    
    model = BertForMaskedLM.from_pretrained(model_path)

    vocab_path = f"{model_path}/vocab.txt"
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f.readlines()]
    tokenizer = CustomTokenizer(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def adjust_probabilities(probabilities, guessed_correct, guessed_incorrect, tokenizer):
    """
    Adjust the probability distribution to exclude letters that have already been guessed, 
    either correctly or incorrectly.
    """
    exclude_ids = set(tokenizer.token2id[letter] for letter in guessed_correct.union(guessed_incorrect))
    for exclude_id in exclude_ids:
        probabilities[exclude_id] = 0.0
    if probabilities.sum() == 0:
        return probabilities  # Return unchanged if all probabilities are excluded
    probabilities /= probabilities.sum()
    return probabilities


def generate_mask_indices(word, num_masked_letters):
    """
    Generate valid mask indices for the word based on the Hangman rules.
    """
    from collections import Counter

    # Count occurrences of each letter in the word
    letter_counts = Counter(word)

    # Find eligible letters for masking based on their counts
    eligible_letters = [
        letter
        for letter, count in letter_counts.items()
        if (num_masked_letters == 1 and count == 1)
        or (num_masked_letters == 2 and count <= 2)
        or (num_masked_letters >= 3 and count <= 2)
    ]

    if not eligible_letters:
        raise ValueError(f"No valid letters to mask for word '{word}' with {num_masked_letters} masked letters.")

    # Select `num_masked_letters` unique letters to mask
    masked_letters = random.sample(eligible_letters, num_masked_letters)

    # Get all indices of the chosen masked letters in the word
    mask_indices = [i for i, char in enumerate(word) if char in masked_letters]

    return mask_indices


def predict_masked_character(
    word, mask_indices, tokenizer, model, device, guessed_correct, guessed_incorrect, verbose=False
):
    # Convert the word to tokens and mask the characters at mask_indices
    tokens = list(word)
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    for mask_index in mask_indices:
        tokens[mask_index + 1] = '[MASK]'  # Offset by 1 to account for [CLS]

    # Encode tokens
    input_ids = tokenizer.encode(tokens, add_special_tokens=False)
    max_length = 32
    input_ids, attention_mask = tokenizer.pad([input_ids], max_length=max_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Predict with the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Aggregate probabilities over all masked positions
    total_probs = torch.zeros(logits.size(-1), device=device)  # Initialize with zeros
    for mask_index in mask_indices:
        mask_token_logits = logits[0, mask_index + 1]  # Get logits for the masked token
        probs = softmax(mask_token_logits, dim=-1)
        total_probs += probs  # Sum probabilities across all masked positions

    # Adjust probabilities
    adjusted_probs = adjust_probabilities(total_probs, guessed_correct, guessed_incorrect, tokenizer)

    # Extract probabilities for all 26 letters (a-z)
    letter_probs = {letter: adjusted_probs[tokenizer.token2id[letter]].item() for letter in string.ascii_lowercase}

    # Sort probabilities by value in descending order
    sorted_letter_probs = sorted(letter_probs.items(), key=lambda x: x[1], reverse=True)

    # Print probabilities in sorted order
    if verbose:
        print(f"Probability Distribution (sorted by value):")
        for letter, prob in sorted_letter_probs:
            print(f"  {letter}: {prob:.4f}")

    # Find the letter with the highest adjusted probability
    best_idx = torch.argmax(adjusted_probs).item()
    best_letter = tokenizer.id2token[best_idx]

    if verbose:
        print(f"Predicted Letter: {best_letter} (Probability: {adjusted_probs[best_idx]:.4f})")

    # Return both the best letter and the full probabilities
    return best_letter, letter_probs


def evaluate_single_prediction(word, num_masked_letters, tokenizer, model, device, verbose=False):
    """
    Evaluate a single prediction for the given word and number of masked letters.
    """
    mask_indices = generate_mask_indices(word, num_masked_letters)
    guessed_correct = set(word[idx] for idx in range(len(word)) if idx not in mask_indices)
    guessed_incorrect = set()

    masked_characters = [word[idx] for idx in mask_indices]

    if verbose:
        print(f"Input Word: {word}")
        print(f"Masked Indices: {mask_indices} (Masked Characters: {masked_characters})")
        print(f"Guessed Correct (Revealed Letters): {guessed_correct}")

    # Get the best letter and probabilities
    guessed_letter, _ = predict_masked_character(
        word, mask_indices, tokenizer, model, device, guessed_correct, guessed_incorrect, verbose
    )

    return guessed_letter, masked_characters


def run_test_cases(test_cases, use_bert_large):
    """
    Run a series of test cases to validate the masked character prediction logic, 
    ensuring that the masking adheres to Hangman rules.
    """
    # Load model and tokenizer based on the flag
    model, tokenizer, device = load_model_and_tokenizer(use_bert_large)

    for i, test_case in enumerate(test_cases):
        word = test_case["word"]
        num_masked_letters = test_case["num_masked_letters"]

        # Generate mask indices based on the Hangman rules
        mask_indices = generate_mask_indices(word, num_masked_letters)

        # Automatically populate guessed_correct with unmasked letters
        guessed_correct = set(word[idx] for idx in range(len(word)) if idx not in mask_indices)
        guessed_incorrect = set()  # Start with an empty guessed_incorrect set

        # The masked characters are the ones the model is supposed to guess
        masked_characters = [word[idx] for idx in mask_indices]

        print(f"\n\nTest Case {i+1}:")
        print(f"\nInput Word: {word}")
        print(f"Masking Indices: {mask_indices} (Masked Characters: {masked_characters})")
        print(f"Guessed Correct (Revealed Letters): {guessed_correct}")

        # Predict one letter
        guessed_letter = predict_masked_character(
            word,
            mask_indices,
            tokenizer,
            model,
            device,
            guessed_correct,
            guessed_incorrect,
            verbose=True  # Enable verbose output to show probabilities
        )[0]

        # Check the prediction
        if guessed_letter in masked_characters:
            print(f"Test Passed! Guessed: {guessed_letter}, Expected: {masked_characters}\n")
        else:
            print(f"Test Failed! Guessed: {guessed_letter}, Expected: {masked_characters}\n")

def main():
    # Define the test cases
    test_cases = [
        # Words with 1 masked letter
        {"word": "a", "num_masked_letters": 1},
        {"word": "be", "num_masked_letters": 1},
        {"word": "cat", "num_masked_letters": 1},
        {"word": "programming", "num_masked_letters": 1},
        {"word": "extraordinarylongword", "num_masked_letters": 1},
        {"word": "califragilisticexpialidocious", "num_masked_letters": 1},

        # Words with 2 masked letters
        {"word": "it", "num_masked_letters": 2},
        {"word": "dog", "num_masked_letters": 2},
        {"word": "elephant", "num_masked_letters": 2},
        {"word": "characteristicallylongword", "num_masked_letters": 2},
        {"word": "pseudopseudohypoparathyroidism", "num_masked_letters": 2},

        # Words with 3 masked letters
        {"word": "ant", "num_masked_letters": 3},
        {"word": "engineering", "num_masked_letters": 3},
        {"word": "uncharacteristicallylongerword", "num_masked_letters": 3},
        {"word": "antidisestablishmentarianism", "num_masked_letters": 3},
    ]


    # Set whether to use the large BERT model
    use_bert_large = False

    # Run test cases
    run_test_cases(test_cases, use_bert_large)


if __name__ == "__main__":
    main()