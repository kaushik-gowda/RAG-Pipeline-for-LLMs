# 🧠 RAG Pipeline for LLMs

A **Retrieval-Augmented Generation** pipeline that fetches knowledge from Wikipedia, indexes it with FAISS, and answers questions using a Hugging Face QA model — reducing hallucinations by grounding answers in real-world context.

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Retriever  │────▶│   Vector Store   │────▶│   Generator    │
│ (Wikipedia) │     │ (FAISS + SentTF) │     │ (RoBERTa QA)   │
└─────────────┘     └──────────────────┘     └────────────────┘
     │                      │                        │
  Fetch &               Embed &                 Extract
  Chunk text           Similarity search         Answer
```

## Project Structure

```
├── config.py          # All tunable parameters
├── retriever.py       # Wikipedia fetching & text chunking
├── vector_store.py    # Sentence Transformer embeddings + FAISS index
├── generator.py       # Hugging Face QA model wrapper
├── rag_pipeline.py    # End-to-end pipeline orchestrator
├── main.py            # CLI interface
├── app.py             # Flask web application
├── templates/
│   └── index.html     # Web UI
└── requirements.txt   # Python dependencies
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

**Option A – Web UI (recommended):**
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000)

**Option B – Command Line:**
```bash
python main.py
```

## How It Works

| Step | Module | Description |
|------|--------|-------------|
| 1 | `retriever.py` | Fetches a full Wikipedia article and splits it into overlapping 256-token chunks |
| 2 | `vector_store.py` | Encodes chunks into embeddings with `all-mpnet-base-v2` and stores them in a FAISS L2 index |
| 3 | `rag_pipeline.py` | Encodes the user's question, retrieves the top-3 most similar chunks via FAISS |
| 4 | `generator.py` | Feeds the question + retrieved context into `deepset/roberta-base-squad2` to extract the answer |

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 256 | Tokens per chunk |
| `CHUNK_OVERLAP` | 20 | Overlapping tokens between chunks |
| `TOP_K` | 3 | Number of chunks to retrieve |
| `EMBEDDING_MODEL_NAME` | `all-mpnet-base-v2` | Sentence Transformer model |
| `QA_MODEL_NAME` | `roberta-base-squad2` | QA model |
