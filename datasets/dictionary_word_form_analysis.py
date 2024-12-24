import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def load_training_set(file_path):
    with open(file_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def analyze_nouns(words):
    """
    Analyze nouns: 
    total count, 
    singular, 
    plural, and 
    plural nouns with trailing 's'."""
    total_nouns = 0
    singular_count = 0
    plural_count = 0
    plural_with_trailing_s_count = 0

    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ == "NOUN":
                total_nouns += 1
                if token.tag_ == "NN":  # Singular noun
                    singular_count += 1
                elif token.tag_ == "NNS":  # Plural noun
                    plural_count += 1
                    # Check if the plural noun ends with 's'
                    if word.endswith("s"):
                        plural_with_trailing_s_count += 1

    return total_nouns, singular_count, plural_count, plural_with_trailing_s_count

def analyze_verbs(words):
    """
    Analyze verbs: 
    total count, 
    base form, 
    third-person singular, 
    past/past participle, 
    present participle, and 
    third-person singular with trailing 's'.
    """
    total_verbs = 0
    base_form_count = 0
    third_person_count = 0
    third_person_with_trailing_s_count = 0
    past_participle_count = 0
    present_participle_count = 0

    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ == "VERB":
                total_verbs += 1
                if token.tag_ == "VB":  # Base form
                    base_form_count += 1
                elif token.tag_ == "VBZ":  # Third-person singular
                    third_person_count += 1
                    # Check if the third-person singular verb ends with 's'
                    if word.endswith("s"):
                        third_person_with_trailing_s_count += 1
                elif token.tag_ in ["VBD", "VBN"]:  # Past or past participle
                    past_participle_count += 1
                elif token.tag_ == "VBG":  # Present participle
                    present_participle_count += 1

    return total_verbs, base_form_count, third_person_count, third_person_with_trailing_s_count, past_participle_count, present_participle_count

# Analyze adjectives
def analyze_adjectives(words):
    """
    Analyze adjectives: 
    total count, 
    comparative, 
    and superlative.
    """
    total_adjectives = 0
    comparative_count = 0
    superlative_count = 0

    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ == "ADJ":
                total_adjectives += 1
                if token.tag_ == "JJR":  # Comparative adjective
                    comparative_count += 1
                elif token.tag_ == "JJS":  # Superlative adjective
                    superlative_count += 1

    return total_adjectives, comparative_count, superlative_count

def main():
    training_file = "words_250000_train.txt"
    training_words = load_training_set(training_file)

    print("\nAnalyzing Nouns...")
    total_nouns, singular_count, plural_count, plural_with_trailing_s_count = analyze_nouns(training_words)
    print(f"Total nouns: {total_nouns}")
    print(f"Singular nouns: {singular_count}")
    print(f"Plural nouns: {plural_count}")
    print(f"Plural nouns with trailing 's': {plural_with_trailing_s_count}")

    print("\nAnalyzing Verbs...")
    total_verbs, base_form_count, third_person_count, third_person_with_trailing_s_count, past_participle_count, present_participle_count = analyze_verbs(training_words)
    print(f"Total verbs: {total_verbs}")
    print(f"Base form verbs: {base_form_count}")
    print(f"Third-person singular verbs: {third_person_count}")
    print(f"Third-person singular verbs with trailing 's': {third_person_with_trailing_s_count}")
    print(f"Past and past participle verbs: {past_participle_count}")
    print(f"Present participle verbs: {present_participle_count}")

    print("\nAnalyzing Adjectives...")
    total_adjectives, comparative_count, superlative_count = analyze_adjectives(training_words)
    print(f"Total adjectives: {total_adjectives}")
    print(f"Comparative adjectives: {comparative_count}")
    print(f"Superlative adjectives: {superlative_count}")

if __name__ == "__main__":
    main()