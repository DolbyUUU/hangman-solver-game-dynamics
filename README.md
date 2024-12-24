# Hangman-AI-Solver

## Overview

**Hangman-AI-Solver** is an advanced AI-powered solution for solving the classic Hangman game. This project integrates **Weighted N-Grams**, **Information Entropy**, and a **Fine-Tuned BERT Model** to achieve exceptional performance on disjoint test datasets. By leveraging **game-dynamics-aware strategies** and an **iterative rollback mechanism**, it offers a robust, high-accuracy guessing system for Hangman.

This repository contains the implementation, analysis, and testing scripts for the Hangman Solver, along with detailed explanations of the methodologies used.

---

## Features

- **Game-Dynamics-Aware Solver**:
  - Adapts dynamically to different game phases (opening, midgame, endgame).
  - Combines multiple strategies for maximum efficiency:
    - **Weighted N-Grams Model**: Captures letter co-occurrence patterns.
    - **Information Entropy Model**: Maximizes information gain in early guesses.
    - **Fine-Tuned BERT Model**: Utilizes masked language modeling for contextual predictions.

- **Iterative Rollback Strategy**:
  - Ensures valid guesses even on disjoint training and testing datasets.
  - Reduces errors by iteratively evaluating and rolling back incorrect guesses.

- **Customizable Parameters**:
  - Tune n-gram weights, entropy thresholds, and rollback similarity parameters for optimal performance.

- **High Success Rates**:
  - Achieves a **74.4% success rate** in local tests and a **65.5% success rate** in API simulations.

---

## How It Works

The Hangman Solver progresses through three game phases:

1. **Opening Phase**:
   - Focuses on exploration and information gain.
   - Uses **Information Entropy** to split the search space evenly.

2. **Midgame Phase**:
   - Leverages statistical patterns from **Weighted N-Grams** to refine guesses.

3. **Endgame Phase**:
   - Utilizes a **Fine-Tuned BERT Model** for precise contextual predictions.

### Dynamic Strategy Selection

The solver dynamically selects the appropriate strategy based on the current game state variables:
- Number of known letters.
- Number of unknown slots.
- Number of incorrect guesses.

### Rollback Mechanism

To handle disjoint datasets, the **iterative rollback strategy** ensures that guesses are valid and adjusts predictions based on similarity thresholds.
