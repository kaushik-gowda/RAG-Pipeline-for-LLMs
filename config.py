# ============================================================
# Configuration for the RAG Pipeline
# ============================================================

# Embedding model used for both chunking (tokenizer) and encoding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Question-Answering model
QA_MODEL_NAME = "deepset/roberta-base-squad2"

# Text chunking parameters
CHUNK_SIZE = 256        # number of tokens per chunk
CHUNK_OVERLAP = 20      # overlapping tokens between consecutive chunks

# FAISS retrieval parameter
TOP_K = 3               # number of top chunks to retrieve

# Flask server
# Port 7860 is required for Hugging Face Spaces deployment
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 7860
FLASK_DEBUG = False  # Set to False for production
