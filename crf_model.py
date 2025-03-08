import nltk
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np

# Download necessary NLTK data
nltk.download('punkt', quiet=True)


# Feature extraction for CRF
def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.contains_digit': any(c.isdigit() for c in word),
        'word.contains_hyphen': '-' in word,
        'word.contains_period': '.' in word,
        'word.contains_at': '@' in word,
    }

    # Previous word features
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True

    # Next word features
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True

    return features


# Convert sentence to features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# Function to preprocess data for CRF
def preprocess_data_crf(data):
    # Tokenize data
    tokenized_data = [nltk.word_tokenize(text) for text in data]

    # Simple PHI labeling heuristic
    all_tokens = []
    all_labels = []

    for tokens in tokenized_data:
        labels = []
        for token in tokens:
            if (any(c.isdigit() for c in token) or
                    token[0].isupper() or
                    '@' in token or
                    token.startswith('Dr.') or token.startswith('Mr.') or token.startswith('Mrs.')):
                labels.append('PHI')
            else:
                labels.append('O')

        all_tokens.append(tokens)
        all_labels.append(labels)

    # Extract features
    X = [sent2features(s) for s in all_tokens]
    y = all_labels

    return X, y, all_tokens


# Train CRF model
def train_crf_model(data):
    # Preprocess data
    X, y, _ = preprocess_data_crf(data)

    # Create and train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X, y)

    # Save model
    with open("crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)

    return crf


# Predict PHI with CRF
def predict_phi_crf(text, model):
    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Extract features
    features = sent2features(tokens)

    # Predict
    predictions = model.predict([features])[0]

    # Extract PHI tokens
    phi_tokens = []
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        if pred == 'PHI':
            phi_tokens.append((token, i))

    return phi_tokens