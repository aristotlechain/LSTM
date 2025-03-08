import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)


# BiLSTM model definition
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(BiLSTM_NER, self).__init__()

        # Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # BiLSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            batch_first=True, bidirectional=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Embedding Layer
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM Layer
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # Output Layer
        output = self.fc(lstm_out)
        return output


# Dataset class
class PHIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Function to preprocess text data
def preprocess_data(data):
    # Tokenize data
    tokenized_data = [nltk.word_tokenize(text) for text in data]

    # Simple PHI labeling heuristic (in a real scenario, you'd have proper annotations)
    all_tokens = []
    all_labels = []

    for tokens in tokenized_data:
        labels = []
        for token in tokens:
            # Very simple heuristic for identifying potential PHI
            if (any(c.isdigit() for c in token) or  # Contains digits (dates, IDs, etc.)
                    token[0].isupper() or  # Starts with uppercase (names, locations)
                    '@' in token or  # Email addresses
                    token.startswith('Dr.') or token.startswith('Mr.') or token.startswith('Mrs.')):  # Titles
                labels.append('PHI')
            else:
                labels.append('O')

        all_tokens.append(tokens)
        all_labels.append(labels)

    return all_tokens, all_labels


# Function to build vocabulary and prepare data
def prepare_data(all_tokens, all_labels, max_len=100):
    # Build vocabulary
    word_freq = defaultdict(int)
    for doc_tokens in all_tokens:
        for token in doc_tokens:
            word_freq[token] += 1

    # Create dictionaries
    word_dict = {word: idx + 1 for idx, word in enumerate(word_freq.keys())}
    label_dict = {"O": 0, "PHI": 1}

    # Determine maximum sequence length
    max_len = min(max_len, max(len(tokens) for tokens in all_tokens))

    # Convert to numeric form
    tokens_numeric = []
    labels_numeric = []

    for doc_tokens, doc_labels in zip(all_tokens, all_labels):
        # Truncate
        doc_tokens = doc_tokens[:max_len]
        doc_labels = doc_labels[:max_len]

        # Convert to numeric
        tokens_numeric.append([word_dict.get(token, 0) for token in doc_tokens])
        labels_numeric.append([label_dict[label] for label in doc_labels])

    # Pad sequences
    tokens_padded = pad_sequences(tokens_numeric, max_len)
    labels_padded = pad_sequences(labels_numeric, max_len, pad_value=-1)

    return tokens_padded, labels_padded, word_dict, max_len


# Padding function
def pad_sequences(sequences, max_len, pad_value=0):
    padded_seqs = []
    for seq in sequences:
        padded = seq + [pad_value] * (max_len - len(seq))
        padded_seqs.append(padded)
    return torch.tensor(padded_seqs)


# Function to train the LSTM model
def train_model(data, epochs=10):
    # Preprocess data
    all_tokens, all_labels = preprocess_data(data)

    # Prepare data
    tokens_padded, labels_padded, word_dict, max_len = prepare_data(all_tokens, all_labels)

    # Save word dictionary for later use
    with open("word_dict.json", "w") as f:
        json.dump(word_dict, f)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        tokens_padded, labels_padded, test_size=0.2, random_state=42
    )

    # Create dataloaders
    batch_size = min(32, len(X_train))
    train_dataset = PHIDataset(X_train, y_train)
    test_dataset = PHIDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    vocab_size = len(word_dict) + 1
    embedding_dim = 100
    hidden_dim = 64
    output_dim = 2  # O or PHI
    pad_idx = 0

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_NER(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_batch)

            # Reshape
            predictions = predictions.view(-1, output_dim)
            y_batch = y_batch.view(-1)

            # Loss
            loss = criterion(predictions, y_batch)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "lstm_phi_model.pth")

    return model, word_dict, max_len


# Function to load a saved model
def load_model(model_path, dict_path):
    # Load word dictionary
    with open(dict_path, "r") as f:
        word_dict = json.load(f)

    # Model parameters
    vocab_size = len(word_dict) + 1
    embedding_dim = 100
    hidden_dim = 64
    output_dim = 2
    pad_idx = 0
    max_len = 100

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_NER(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, word_dict, max_len


# Function to predict PHI in new text
def predict_phi(text, model, word_dict, max_len, device):
    model.eval()

    # Tokenize
    tokens = nltk.word_tokenize(text)[:max_len]

    # Convert to numeric
    token_indices = [word_dict.get(token, 0) for token in tokens]

    # Pad
    padded = token_indices + [0] * (max_len - len(token_indices))

    # To tensor
    input_tensor = torch.tensor([padded]).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=2)[0]

    # Extract PHI tokens
    phi_tokens = []
    for i, (token, pred) in enumerate(zip(tokens, predictions[:len(tokens)])):
        if pred.item() == 1:  # PHI
            phi_tokens.append((token, i))

    return phi_tokens