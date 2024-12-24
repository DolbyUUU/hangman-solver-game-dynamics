import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
import string
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, get_scheduler
import torch.optim as optim
from tqdm import tqdm

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token2id = {token: idx for idx, token in enumerate(vocab)}
        self.id2token = {idx: token for idx, token in enumerate(vocab)}

    def encode(self, text, add_special_tokens=True):
        tokens = list(text)
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        return [self.token2id.get(token, self.token2id['[UNK]']) for token in tokens]

    def decode(self, token_ids):
        return ''.join([self.id2token.get(idx, '[UNK]') for idx in token_ids if idx not in self.token2id.values()])

    def pad(self, sequences, max_length):
        padded_sequences = []
        attention_masks = []
        for seq in sequences:
            seq = seq[:max_length]
            attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
            seq = seq + [self.token2id['[PAD]']] * (max_length - len(seq))
            padded_sequences.append(seq)
            attention_masks.append(attention_mask)
        return torch.tensor(padded_sequences), torch.tensor(attention_masks)

# Define vocabulary
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
letters = list(string.ascii_lowercase)
vocab = special_tokens + letters
tokenizer = CustomTokenizer(vocab)

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
        char_list = list(word)
        indices = list(range(len(char_list)))  # Indices corresponding to the letters in the word
        
        # Determine how many tokens to mask (1 or 2)
        if len(indices) <= 1:
            num_to_mask = 1
        else:
            num_to_mask = random.choice([1, 2])
        
        # Randomly select indices to mask
        mask_indices = random.sample(indices, num_to_mask)
        
        # Tokenize the word with special tokens
        tokens = ['[CLS]'] + char_list + ['[SEP]']
        input_ids = self.tokenizer.encode(tokens, add_special_tokens=False)
        labels = input_ids[:]
        
        # Mask valid word tokens (letters only), skipping [CLS] and [SEP]
        for idx in mask_indices:
            token_idx = idx + 1  # Offset for [CLS]
            if 0 < token_idx < len(input_ids) - 1:  # Ensure not masking [CLS] or [SEP]
                input_ids[token_idx] = self.tokenizer.token2id['[MASK]']
        
        # Replace non-masked positions with -100 in labels
        for i in range(len(labels)):
            if i not in [idx + 1 for idx in mask_indices]:  # Offset for [CLS]
                labels[i] = -100
        
        # Pad sequences and generate attention masks
        input_ids, attention_mask = self.tokenizer.pad([input_ids], self.max_length)
        labels, _ = self.tokenizer.pad([labels], self.max_length)
        
        return input_ids.squeeze(), labels.squeeze(), attention_mask.squeeze()

def load_words(file_path):
    with open(file_path, "r") as f:
        words = [line.strip() for line in f.readlines()]
    return [word for word in words if len(word) >= 2]

# Wrap training logic in this block
if __name__ == "__main__":
    epochs = 20  # Total number of epochs, 10 is not enough, validation loss keeps decreasing

    # Load training words
    train_words = load_words("../datasets/words_250000_train.txt")

    # Expand dataset to N times and shuffle
    n_times_data = 3
    train_words = train_words * n_times_data
    random.shuffle(train_words)

    # Create training dataset
    train_dataset = HangmanDataset(train_words, tokenizer, max_length=18)

    # Load validation words (no doubling here)
    val_words = load_words("../datasets/words_test_disjoint.txt")
    val_dataset = HangmanDataset(val_words, tokenizer, max_length=18)

    # Model Initialization
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    # adjust the size of the embedding layer in the BERT model to match the size of custom tokenizer's vocabulary
    # shrink the embedding matrix from a size of 30,522×1024 (vocabulary size × hidden size) to 31×1024
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.gradient_checkpointing_enable()

    # DataLoader with Padding and Attention Mask
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # batch_size=64 for bert-base
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) # batch_size=64 for bert-base

# Training with Validation Loop and Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Early Stopping Parameters
    patience = 3  # Number of epochs to wait for improvement
    best_val_loss = float('inf')  # Initialize with infinity
    epochs_no_improve = 0  # Counter for epochs with no improvement
    early_stop = False  # Flag for stopping

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
            epochs_no_improve = 0  # Reset counter
            # Optionally save the best model here
            torch.save(model.state_dict(), "./best_model_base.pth")
            print(f"Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} consecutive epochs.")

        if epochs_no_improve >= patience:
            print("Early stopping: Patience exceeded.")
            early_stop = True

    # Save Model and Tokenizer
    model.save_pretrained("./hangman_bert_base")
    tokenizer_save_path = "./hangman_bert_base"
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    # Save tokenizer in Hugging Face format
    with open(os.path.join(tokenizer_save_path, "vocab.txt"), "w") as f:
        for token in tokenizer.vocab:
            f.write(token + "\n")