# PHI Detection Demo: LSTM vs CRF Comparison

This demo showcases the superiority of BiLSTM models over traditional CRF models for Protected Health Information (PHI) de-identification in clinical texts.

## Features

- Interactive web interface for comparing LSTM and CRF performance
- Real-time PHI detection on input text
- Visualizations of model performance metrics
- Complete Docker deployment solution

## Background

This implementation is based on research from:

> EVALUATION OF MACHINE LEARNING MODELS FOR PATIENT DATA DE-IDENTIFICATION IN CLINICAL RECORDS  
> by Yamani Kakarla  
> Dalhousie University, August 2018

The research demonstrated that deep learning approaches like LSTM outperform traditional methods like CRF for PHI detection tasks.

## Quick Start

### Using Docker Compose (Recommended)

1. Clone this repository
2. Run `docker-compose up`
3. Open `http://localhost:8000` in your browser

### Manual Setup

1. Install requirements: `pip install -r requirements.txt`
2. Run the server: `uvicorn api:app --host 0.0.0.0 --port 8000`
3. Open `http://localhost:8000` in your browser

## Demo Usage

1. Enter or use the sample clinical text
2. Click "Analyze Text"
3. View the comparison between LSTM and CRF results
4. Explore the performance metrics and visualizations

## Project Structure

- `api.py`: FastAPI backend implementation
- `model.py`: BiLSTM model implementation
- `crf_model.py`: CRF model implementation
- `templates/`: Frontend HTML templates
- `static/`: Static assets for the frontend
- `Dockerfile`: Docker configuration
- `docker-compose.yml`: Docker Compose configuration

## Key Advantages of LSTM over CRF

- Better handling of context and long-range dependencies
- No need for manual feature engineering
- Higher precision and recall across all PHI categories
- Faster inference time once trained
- More robust to unseen patterns

## License

This project is licensed under the MIT License.