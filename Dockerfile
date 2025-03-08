FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for static files and templates
RUN mkdir -p static templates

# Copy application code
COPY *.py ./
COPY static ./static/
COPY templates ./templates/

# Create placeholder images for architecture diagrams
RUN mkdir -p static/images && \
    touch static/lstm_architecture.png && \
    touch static/crf_architecture.png

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port for FastAPI
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]