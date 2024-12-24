# Game-Dynamics-Aware Hangman Solver

## Overview

This project implements a **Game-Dynamics-Aware Hangman Solver** that uses algorithms and models including **weighted n-grams**, **information entropy**, and a **fine-tuned BERT** to optimize letter-guessing during the **opening, midgame, and endgame** of the Hangman game.

## Directory and Scripts

### **Main Scripts**

- **`hangman_api_user.ipynb`**: An API-based version of the main script that initializes the Hangman solver and runs simulations.
- **`main_local.ipynb`**: A local version of the main script that initializes the Hangman solver and runs simulations.
- **`ngrams.py`**: Contains functions for building, storing, and utilizing weighted n-grams for letter probability calculations.
- **`entropy.py`**: Implements entropy-based optimization to prioritize guesses that maximize information gain.
- **`rollback.py`**: Implements the rollback strategy for refining guesses based on training word similarity.
- **`preprocessing.py`**: Filters and preprocesses the word dataset to create a cleaned version for training.

### **BERT Fine-Tuning and Evaluation**

- **`bert-base-ver1/`**: Contains scripts to fine-tune with extended pretrained BERT tokenizer, test, and evaluate BERT-base models.
- **`bert-base-ver2/`**: Contains scripts to fine-tune with custom tokenizer, test, and evaluate BERT-base models.
  - **`bert_finetuning_base.py`**: Fine-tunes a BERT-base model for Hangman-style masked character prediction with a custom tokenizer.
  - **`bert_testing.py`**: Loads the fine-tuned BERT model and predicts masked characters in words, following Hangman rules.
  - **`bert_evaluation.py`**: Evaluates the accuracy of the fine-tuned BERT model on a test dataset.
- **`./hangman_bert_base`**: Output directories containing the fine-tuned BERT models and necessary files for inference.
  - **`config.json`**: Model architecture and hyperparameters.
  - **`generation_config.json`**: Settings for text generation, if applicable.
  - **`model.safetensors`**: Fine-tuned model weights.
  - **`vocab.txt`**: Token vocabulary for the tokenizer.
  - **`log_base.log`**: Logs of fine-tuning progress.

### **Dictionary Statistics**

- **`datasets/dictionary_statistics.py `**: Calculates word statistics, including length distribution, letter frequencies, and patterns.
- **`datasets/dictionary_word_form_analysis.py`**: Analyzes parts of speech in words (nouns, verbs, adjectives).

## Datasets

- **`datasets/words_250000_train.txt`**: Training dataset of 250,000 words.
- **`datasets/words_250000_train_cleaned.txt`**: Training dataset of 250,000 words after data preprocessing.
- **`datasets/words_test_disjoint.txt`**: Disjoint test dataset for local evaluation.
