# Minimal Dockerfile to run the HQNN demo
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command: run the training script with modest defaults
CMD ["python", "app/main.py", "--epochs", "5", "--batch", "32"]
