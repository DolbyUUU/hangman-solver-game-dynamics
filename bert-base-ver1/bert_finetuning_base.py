import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import string
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, get_scheduler
import torch.optim as optim
from tqdm import tqdm

if __name__ == "__main__":
    epochs = 20

    # Initialize BERT's Tokenizer and Add Letters
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Letters to add
    letters = list(string.ascii_lowercase)

    # Add letters 'a' to 'z' to tokenizer's vocabulary
    num_added_toks = tokenizer.add_tokens(letters)
    print(f"Added {num_added_toks} tokens to the tokenizer.")

    #  Model Initialization
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    # Resize token embeddings to match the new tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    # Dataset Preparation
    class HangmanDataset(Dataset):
        def __init__(self, words, tokenizer, max_length=18):
            self.words = words
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.words)

        def __getitem__(self, idx):
            word = self.words[idx]
            input_ids, labels, attention_mask = self.mask_word(word)
            return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

        def mask_word(self, word):
            # Split the word into letters
            letters = list(word)

            # Tokenize letters individually, adding special tokens
            tokens = [tokenizer.cls_token] + letters + [tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            labels = input_ids.copy()

            # Mask random letters (excluding special tokens at positions 0 and -1)
            letter_positions = list(range(1, len(tokens) - 1))
            if len(letter_positions) <= 1:
                num_to_mask = 1
            else:
                num_to_mask = random.choice([1, 2])

            mask_indices = random.sample(letter_positions, num_to_mask)
            for idx in mask_indices:
                input_ids[idx] = tokenizer.mask_token_id  # Use BERT's mask token ID

            # Replace non-masked positions with -100 in labels
            for i in range(len(labels)):
                if i not in mask_indices:
                    labels[i] = -100

            # Pad sequences and generate attention masks
            encoding = tokenizer.prepare_for_model(
                input_ids,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )

            input_ids = torch.tensor(encoding['input_ids'])
            attention_mask = torch.tensor(encoding['attention_mask'])
            labels = torch.tensor(labels + [tokenizer.pad_token_id] * (self.max_length - len(labels))).long()
            labels = labels[:self.max_length]

            return input_ids, labels, attention_mask

    def load_words(file_path):
        with open(file_path, "r") as f:
            words = [line.strip() for line in f.readlines()]
        return [word for word in words if len(word) >= 2]

    # Load training words
    train_words = load_words("../datasets/words_250000_train.txt")

    # Expand dataset to N times and shuffle
    n_times_data = 3
    train_words = train_words * n_times_data
    random.shuffle(train_words)

    # Create training dataset
    train_dataset = HangmanDataset(train_words, tokenizer, max_length=18)

    # Load validation words
    val_words = load_words("../datasets/words_test_disjoint.txt")
    val_dataset = HangmanDataset(val_words, tokenizer, max_length=18)

    # DataLoader with Padding and Attention Mask
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Training with Validation Loop and Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    # Early Stopping Parameters
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model and tokenizer
            model.save_pretrained("./best_model_base")
            tokenizer.save_pretrained("./best_model_base")
            print(f"Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping: Patience exceeded.")
            early_stop = True

    # Save Model and Tokenizer
    model.save_pretrained("./hangman_bert_base")
    tokenizer.save_pretrained("./hangman_bert_base")