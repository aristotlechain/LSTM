import torch
import nltk
from model_definition import BiLSTM_NER  # assuming your model class is here

nltk.download('punkt')

VOCAB_SIZE = 5000  # update accordingly
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 2
PAD_IDX = 0
MAX_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)
model.load_state_dict(torch.load("lstm_phi_model.pth", map_location=device))
model.eval()

def predict_phi(text):
    tokens = nltk.word_tokenize(text)[:MAX_LEN]
    indices = [0] * MAX_LEN  # simplify or load your real word dict
    input_tensor = torch.tensor([indices]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=2)[0]

    # Simplified output
    phi_tokens = [token for token, pred in zip(tokens, predictions) if pred.item() == 1]
    return phi_tokens

if __name__ == "__main__":
    sample_text = "Patient John Doe was admitted on 02/12/2024 with ID 567890."
    phi = predict_phi(sample_text)
    print("Detected PHI:", phi)
