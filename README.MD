# 🎮 Game-Dynamics-Aware Hangman Solver

## 🚀 Overview

**Game-Dynamics-Aware Hangman Solver** is an advanced AI-powered solution for solving the classic Hangman game. This project integrates **Weighted N-Grams**, **Information Entropy**, and a **Fine-Tuned BERT Model** to achieve exceptional performance on disjoint test datasets. By leveraging **game-dynamics-aware strategies** and an **iterative rollback mechanism**, it offers a robust, high-accuracy guessing system for Hangman.

---

## ✨ Features

- 🧠 **Game-Dynamics-Aware Solver**:
  - Adapts dynamically to different game phases:
    - **Opening Phase**: Focuses on exploration and information gain.
    - **Midgame Phase**: Leverages statistical patterns using **Weighted N-Grams**.
    - **Endgame Phase**: Utilizes a **Fine-Tuned BERT Model** for precise contextual predictions.
  - Combines multiple strategies for maximum efficiency:
    - **Weighted N-Grams Model**: Captures letter co-occurrence patterns.
    - **Information Entropy Model**: Maximizes information gain in early guesses.
    - **Fine-Tuned BERT Model**: Predicts masked letters using contextual knowledge.

- 🔄 **Iterative Rollback Strategy**:
  - Ensures valid guesses even on disjoint training and testing datasets.
  - Reduces errors by iteratively evaluating and rolling back incorrect guesses.

- 📊 **High Success Rates**:
  - **Local tests**: 74.4% success rate on disjoint datasets.
  - **API simulations**: 65.5% success rate.

- ⚙️ **Customizable Parameters**:
  - Tune n-gram weights, entropy thresholds, and rollback similarity parameters for optimal performance.

---

## 📋 How It Works

The Hangman Solver progresses through three game phases:

| 🎯 Phase      | 🔍 Objective               | 🛠️ Methodology                                      |
|---------------|----------------------------|----------------------------------------------------|
| **Opening**   | Exploration                | Information Entropy                                |
| **Midgame**   | Statistical Exploitation   | Weighted N-Grams                                   |
| **Endgame**   | Contextual Prediction      | Fine-Tuned BERT                                    |

### 🧩 Dynamic Strategy Selection

The solver dynamically selects the appropriate strategy based on the current game state:
- ✅ Number of known letters.
- ❓ Number of unknown slots.
- ❌ Number of incorrect guesses.

### 🔄 Rollback Mechanism

To handle disjoint datasets, the **iterative rollback strategy** ensures that guesses are valid and adjusts predictions based on similarity thresholds.

> **Note**: Due to the large file size, the `model.safetensors` files for two versions of the fine-tuned BERT models are **not uploaded** in this repository. You will need to fine-tune BERT locally or provide your own pre-trained models.

---

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
