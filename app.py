# ============================================================
# app.py – Flask web application for the RAG Pipeline
# ============================================================
"""
A beautiful web UI for the RAG pipeline.

Usage (local):
    python app.py
    Then open http://localhost:7860 in your browser.

Production:
    gunicorn --bind 0.0.0.0:7860 --timeout 120 app:app
"""

from flask import Flask, render_template, request, jsonify

from rag_pipeline import RAGPipeline
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
pipeline = RAGPipeline()


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Ingest a Wikipedia topic into the pipeline."""
    data = request.get_json(force=True)
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"success": False, "message": "Topic cannot be empty."}), 400

    result = pipeline.ingest(topic)
    status_code = 200 if result["success"] else 404
    return jsonify(result), status_code


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Ask a question against the ingested knowledge base."""
    data = request.get_json(force=True)
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    result = pipeline.ask(question)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result), 200


@app.route("/health")
def health():
    """Health check endpoint for deployment platforms."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    print(f"Starting RAG Pipeline on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
