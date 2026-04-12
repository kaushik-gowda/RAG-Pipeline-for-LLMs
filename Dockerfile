FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Expose port (Hugging Face Spaces expects 7860)
EXPOSE 7860

# Pre-download models during build to speed up startup
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); \
AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2'); \
print('Models downloaded successfully')"

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "2", "app:app"]
