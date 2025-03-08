import os
import torch
import nltk
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import re

# Import model classes
from model import BiLSTM_NER, predict_phi, load_model, train_model
from crf_model import train_crf_model, predict_phi_crf

# Initialize FastAPI app
app = FastAPI(title="PHI Detection System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models
lstm_model = None
crf_model = None
word_dict = {}
MAX_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download NLTK data if needed
nltk.download('punkt', quiet=True)


# Pydantic models
class TextInput(BaseModel):
    text: str


class ComparisonResult(BaseModel):
    lstm_results: List[Dict[str, Any]]
    crf_results: List[Dict[str, Any]]
    lstm_metrics: Dict[str, float]
    crf_metrics: Dict[str, float]


# Helper function to generate performance plots
def generate_plot(lstm_metrics, crf_metrics):
    metrics = ['precision', 'recall', 'f1-score']
    lstm_values = [lstm_metrics.get(m, 0) for m in metrics]
    crf_values = [crf_metrics.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, lstm_values, width, label='LSTM')
    rects2 = ax.bar(x + width / 2, crf_values, width, label='CRF')

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: LSTM vs CRF')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert to base64 for HTML display
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_str


# Function to load both models
def initialize_models():
    global lstm_model, crf_model, word_dict, MAX_LEN

    # Try to load saved models, train new ones if not available
    try:
        lstm_model, word_dict, MAX_LEN = load_model("lstm_phi_model.pth", "word_dict.json")
        print("LSTM model loaded successfully")
    except:
        print("LSTM model not found, will be trained on first request")

    try:
        with open("crf_model.pkl", "rb") as f:
            import pickle
            crf_model = pickle.load(f)
        print("CRF model loaded successfully")
    except:
        print("CRF model not found, will be trained on first request")


# Load mock data for example and comparison
def load_sample_data():
    # This is mock data simulating medical records with PHI
    sample_texts = [
        "Patient ID: 12345 was admitted on 07/15/2023. Contact Dr. Smith at 555-123-4567.",
        "PATIENT: John Doe (MRN: 987654) DOB: 01/30/1975 ADDRESS: 123 Main St, Boston MA",
        "The patient's SSN is 123-45-6789 and their email is patient@email.com",
        "She was transferred from Memorial Hospital on March 15, 2023.",
        "Mr. Williams can be reached at (555) 987-6543 or room 302."
    ]

    return sample_texts


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# API to process text and detect PHI
@app.post("/detect-phi/")
async def detect_phi(input_data: TextInput):
    global lstm_model, crf_model, word_dict, MAX_LEN

    # Initialize models if not already loaded
    if lstm_model is None or crf_model is None:
        # Load sample data for training if needed
        sample_data = load_sample_data()

        if lstm_model is None:
            lstm_model, word_dict, MAX_LEN = train_model(sample_data)

        if crf_model is None:
            crf_model = train_crf_model(sample_data)

    text = input_data.text

    # Get LSTM predictions
    lstm_phi = predict_phi(text, lstm_model, word_dict, MAX_LEN, device)

    # Get CRF predictions
    crf_phi = predict_phi_crf(text, crf_model)

    # Format results
    lstm_results = [{"token": token, "position": pos} for token, pos in lstm_phi]
    crf_results = [{"token": token, "position": pos} for token, pos in crf_phi]

    # Generate mock metrics (in a real system, these would be calculated from validation data)
    lstm_metrics = {"precision": 0.92, "recall": 0.89, "f1-score": 0.91, "accuracy": 0.95}
    crf_metrics = {"precision": 0.85, "recall": 0.82, "f1-score": 0.83, "accuracy": 0.88}

    # Generate comparison plot
    plot_base64 = generate_plot(lstm_metrics, crf_metrics)

    return {
        "lstm_results": lstm_results,
        "crf_results": crf_results,
        "lstm_metrics": lstm_metrics,
        "crf_metrics": crf_metrics,
        "plot": plot_base64,
        "text_with_highlights": {
            "original": text,
            "lstm_highlighted": highlight_phi(text, lstm_phi),
            "crf_highlighted": highlight_phi(text, crf_phi)
        }
    }


# Helper function to highlight PHI in text
def highlight_phi(text, phi_tokens):
    # Sort tokens by position in reverse order to avoid index shifting
    phi_tokens = sorted(phi_tokens, key=lambda x: x[1], reverse=True)

    # Split text into characters for easier manipulation
    chars = list(text)

    # Mark each token for highlighting
    for token, pos in phi_tokens:
        # Find the actual position in the text
        token_len = len(token)
        start_idx = text.find(token, max(0, pos - 10), min(len(text), pos + 10))

        if start_idx >= 0:
            # Insert highlight markers
            chars.insert(start_idx + token_len, "</mark>")
            chars.insert(start_idx, "<mark>")

    return "".join(chars)


# API to get comparison data
@app.get("/comparison-data/")
async def get_comparison_data():
    # This would ideally be calculated from a test set
    # For demo purposes, we're providing simulated performance data

    # Model performance metrics
    performance = {
        "lstm": {
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.91,
            "accuracy": 0.95,
            "training_time": 45,  # seconds
            "inference_time": 0.015  # seconds per sample
        },
        "crf": {
            "precision": 0.85,
            "recall": 0.82,
            "f1_score": 0.83,
            "accuracy": 0.88,
            "training_time": 120,  # seconds
            "inference_time": 0.025  # seconds per sample
        }
    }

    # PHI category performance
    phi_categories = {
        "names": {"lstm": 0.94, "crf": 0.88},
        "dates": {"lstm": 0.96, "crf": 0.92},
        "locations": {"lstm": 0.91, "crf": 0.84},
        "ids": {"lstm": 0.93, "crf": 0.87},
        "contact_info": {"lstm": 0.89, "crf": 0.81},
        "organizations": {"lstm": 0.88, "crf": 0.79}
    }

    # Training convergence data
    epochs = list(range(1, 11))
    lstm_loss = [0.0756, 0.0003, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    crf_loss = [0.4523, 0.3214, 0.2145, 0.1652, 0.1324, 0.1125, 0.0956, 0.0854, 0.0782, 0.0734]

    return {
        "performance_metrics": performance,
        "phi_categories": phi_categories,
        "training_progress": {
            "epochs": epochs,
            "lstm_loss": lstm_loss,
            "crf_loss": crf_loss
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    initialize_models()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)