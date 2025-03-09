# ================================================================
# Software License Agreement
# Author: Sarthak Kaushik, Arsh Zala, Mohit Patel
# Email: skaushi1@lakeheadu.ca, mpate213@lakeheadu.ca, azala1@lakeheadu.ca
# Admission Number: 1270126, 1277400, 1277507
# Institution: Lakehead University
# Permission provided to supervisor: Dr. Jinan Fiaidhi
# This software is part of an academic submission and is licensed
# for educational purposes only. It may be used, modified, and
# distributed strictly for academic and educational activities.
#
# Permission is hereby granted to any student, educator, or academic
# institution to use this software, subject to the following conditions:
#
# 1. The software must be used solely for educational purposes.
# 2. Any modifications to the code must retain this license notice.
# 3. No commercial use of this software is permitted without the
#    express permission of the author.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHOR OR MIT BE LIABLE FOR ANY CLAIM, DAMAGES,
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# ================================================================


import os
import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


nltk.download('punkt', quiet=True)


DATA_PATH = "data/physionet_deid/"


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Directory not found: {DATA_PATH}")

# Here we have listed all files (handling either .txt files or specific deid files)
files = []
for extension in ['.txt', '.deid']:
    files.extend([f for f in os.listdir(DATA_PATH) if f.endswith(extension)])

# If no regular text files found then looking for specific PhysioNet deid files
if not files:
    if os.path.exists(os.path.join(DATA_PATH, "id.deid")):
        files = ["id.deid"]
        phi_file = "id-phi.phrase"
        if os.path.exists(os.path.join(DATA_PATH, phi_file)):
            print(f"Found PhysioNet deid files: {files} and {phi_file}")
            use_phi_annotations = True
        else:
            print(f"Found deid file but no PHI annotations: {files}")
            use_phi_annotations = False
    else:
        raise FileNotFoundError(f"No text files found in {DATA_PATH}")
else:
    print(f"Found text files: {files}")
    use_phi_annotations = False

# Here, we are loading the dataset with appropriate handling of annotations
data = []
phi_annotations = []

for file in files:
    with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

        # If content has multiple documents separated by blank lines
        if "\n\n" in content:
            documents = content.strip().split("\n\n")
            data.extend(documents)
        else:
            # Split by lines to create more training samples
            lines = [line for line in content.split("\n") if line.strip()]
            data.extend(lines)

    # If we have PHI annotations, load them
    if use_phi_annotations and file == "id.deid":
        phi_path = os.path.join(DATA_PATH, phi_file)
        if os.path.exists(phi_path):
            with open(phi_path, "r", encoding="utf-8", errors="replace") as f:
                phi_content = f.read()

                if "\n\n" in phi_content:
                    phi_docs = phi_content.strip().split("\n\n")
                    phi_annotations.extend(phi_docs)
                else:
                    # For single document, get each line as a potential PHI source
                    phi_lines = [line for line in phi_content.split("\n") if line.strip()]
                    phi_annotations.extend(phi_lines)

# Creating a small dataset if we have too few samples
if len(data) < 10:
    print(f"Warning: Only {len(data)} documents found. Creating synthetic variations.")
    original_data = data.copy()
    # Create variations by adding prefixes/suffixes
    for i in range(9):  # Create 9 more variations
        variations = [f"Variation {i}: {doc}" for doc in original_data]
        data.extend(variations)

        if use_phi_annotations:
            phi_annotations.extend(phi_annotations[:len(original_data)])

print(f"Total documents loaded: {len(data)}")


# Here, we are tokenizing and label data based on PHI annotations if available
def tokenize_and_label(text, phi_text=None):
    tokens = nltk.word_tokenize(text)

    if phi_text and phi_text.strip():
        # Using PHI annotations to label tokens
        phi_tokens = set(nltk.word_tokenize(phi_text))
        labels = ["PHI" if token in phi_tokens else "O" for token in tokens]
    else:
        # Default: in case there is no PHI information available
        labels = ["O"] * len(tokens)

    return tokens, labels


# here, we are processing the data
if use_phi_annotations and len(phi_annotations) == len(data):
    print("Using PHI annotations for labeling")
    tokenized_data = [tokenize_and_label(text, phi)
                      for text, phi in zip(data, phi_annotations)]
else:
    print("No PHI annotations available, using default labeling")
    tokenized_data = [tokenize_and_label(text) for text in data]

# Here, we are separating tokens and labels
all_tokens = []
all_labels = []
for tokens, labels in tokenized_data:
    all_tokens.append(tokens)
    all_labels.append(labels)

# Here we are building the vocabulary
word_freq = defaultdict(int)
for doc_tokens in all_tokens:
    for token in doc_tokens:
        word_freq[token] += 1

# Here, we are creating word and label dictionaries
word_dict = {word: idx + 1 for idx, word in enumerate(word_freq.keys())}
label_dict = {"O": 0, "PHI": 1}

print(f"Vocabulary size: {len(word_dict)}")

# Now determining max sequence length
MAX_LEN = min(100, max(len(tokens) for tokens in all_tokens))
print(f"Using maximum sequence length: {MAX_LEN}")

# Converting tokens and labels to numeric form
tokens_numeric = []
labels_numeric = []

for doc_tokens, doc_labels in zip(all_tokens, all_labels):
    # Truncating the tokens to MAX_LEN
    doc_tokens = doc_tokens[:MAX_LEN]
    doc_labels = doc_labels[:MAX_LEN]

    # Converting the tokens to numeric
    tokens_numeric.append([word_dict.get(token, 0) for token in doc_tokens])  # 0 for unknown tokens
    labels_numeric.append([label_dict[label] for label in doc_labels])


# Converting to tensors and pad
def pad_sequences(sequences, max_len, pad_value=0):
    padded_seqs = []
    for seq in sequences:

        padded = seq + [pad_value] * (max_len - len(seq))
        padded_seqs.append(padded)
    return torch.tensor(padded_seqs)


# Here, we are padding the sequences
tokens_padded = pad_sequences(tokens_numeric, MAX_LEN, pad_value=0)
labels_padded = pad_sequences(labels_numeric, MAX_LEN, pad_value=-1)  # -1 is ignore_index for loss

print(f"Padded data shape: {tokens_padded.shape}, labels shape: {labels_padded.shape}")

# Performing the train-test split with safeguard for small datasets
test_size = min(0.2, 1 / len(tokens_padded) * 2) if len(tokens_padded) > 5 else 0.1
X_train, X_test, y_train, y_test = train_test_split(
    tokens_padded, labels_padded, test_size=test_size, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")


# Dataset class
class PHIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Creating DataLoaders
batch_size = min(32, len(X_train))  # Adjust batch size for small datasets
train_dataset = PHIDataset(X_train, y_train)
test_dataset = PHIDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoaders initialized successfully!")


# Defining the BiLSTM model
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(BiLSTM_NER, self).__init__()

        # Here are the Word Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

       # Here are the BiLSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            batch_first=True, bidirectional=True)

        # Here is the Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM doubles hidden size

        # Here is the Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Here is thhe Embedding Layer
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM Layer
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # Output Layer
        output = self.fc(lstm_out)
        return output


# Added all the model parameters
VOCAB_SIZE = len(word_dict) + 1  # +1 for padding index
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = len(label_dict)
PAD_IDX = 0

# We're initializing the model here
model = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)
print(model)

# Then we're defining the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding labels
optimizer = optim.Adam(model.parameters(), lr=0.001)

# And then finally declaring the training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Training on device: {device}")

# Here, we have the Lists to store training history
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)

        # Reshaping for loss calculation
        predictions = predictions.view(-1, OUTPUT_DIM)
        y_batch = y_batch.view(-1)

        # Computing loss
        loss = criterion(predictions, y_batch)

        # Backward and optimizing
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    # Calculating average loss for the epoch
    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
    train_losses.append(avg_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Saving the  model
torch.save(model.state_dict(), "lstm_phi_model.pth")
print("Training complete and model saved!")

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Getting predictions
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=2)

        # Collecting true and predicted labels
        for i in range(y_batch.size(0)):
            for j in range(y_batch.size(1)):
                if y_batch[i, j] != -1:  # Skip padding
                    y_true.append(y_batch[i, j].item())
                    y_pred.append(predictions[i, j].item())

# Calculating the   accuracy
correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
total = len(y_true)
accuracy = correct / total if total > 0 else 0

print(f"Evaluation Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
# Get unique labels that actually appear in the test set
unique_labels = sorted(set(y_true))
# Map numeric labels back to their string representations
actual_target_names = [list(label_dict.keys())[list(label_dict.values()).index(label)] for label in unique_labels]
# Generate the report with only the labels that appear in the test data
print(classification_report(y_true, y_pred, target_names=actual_target_names))

# Visualize training progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
print("Training plot saved as 'training_loss.png'")


# Here we have a function  to predict PHI in new text
def predict_phi(text, model, word_dict, max_len, device):
    model.eval()

    # Tokenizing text
    tokens = nltk.word_tokenize(text)[:max_len]

    # Converting to numeric form
    token_indices = [word_dict.get(token, 0) for token in tokens]

    # Padding sequence
    padded = token_indices + [0] * (max_len - len(token_indices))

    # Converting to tensor
    input_tensor = torch.tensor([padded]).to(device)

    # Getting predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=2)[0]

    # Extract PHI tokens
    phi_tokens = []
    for i, (token, pred) in enumerate(zip(tokens, predictions[:len(tokens)])):
        if pred.item() == label_dict["PHI"]:
            phi_tokens.append((token, i))

    return phi_tokens


# Example usage of the code
print("\nExample PHI detection:")
if len(data) > 0:
    sample_text = data[0][:200] + "..." if len(data[0]) > 200 else data[0]
    print(f"Sample text: {sample_text}")

    detected_phi = predict_phi(sample_text, model, word_dict, MAX_LEN, device)

    if detected_phi:
        print("Detected PHI tokens:")
        for token, position in detected_phi:
            print(f"  • '{token}' at position {position}")
    else:
        print("No PHI detected in the sample text.")

print("\nLSTM integration complete! Model is ready for PHI detection tasks.")



