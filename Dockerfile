FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your project files
COPY . .

# Download nltk resources
RUN python -c "import nltk; nltk.download('punkt')"

EXPOSE 8080

# Define your default command
CMD ["python", "inference.py"]
