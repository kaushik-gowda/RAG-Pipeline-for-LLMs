# ============================================================
# generator.py – Answer generation via QA model
# ============================================================
"""
This module loads a Hugging Face question-answering model and
uses it to extract an answer from a provided context string.

Uses direct model inference with proper handling of SQuAD 2.0
style "no-answer" predictions and long-context sliding windows.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from config import QA_MODEL_NAME

# Load model & tokenizer once
_qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
_qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
_qa_model.eval()


def _get_best_span(start_logits, end_logits, input_ids, tokenizer, max_answer_len=50):
    """Find the best non-null answer span from logits.

    For SQuAD 2.0 models, index 0 (the <s>/CLS token) represents
    the "no answer" prediction.  We find the best span that does
    NOT start at index 0 and compare it to the null score.
    """
    start_logits = start_logits.squeeze()
    end_logits = end_logits.squeeze()
    input_ids = input_ids.squeeze()

    # Null score = start_logits[0] + end_logits[0]
    null_score = (start_logits[0] + end_logits[0]).item()

    # Find best non-null span
    best_score = float("-inf")
    best_start = 0
    best_end = 0

    seq_len = start_logits.size(0)
    for start in range(1, seq_len):          # skip index 0 (CLS / <s>)
        for end in range(start, min(start + max_answer_len, seq_len)):
            score = (start_logits[start] + end_logits[end]).item()
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    # Decode the best span
    answer_tokens = input_ids[best_start : best_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Confidence = how much better the span is vs. null
    # Convert to a 0-1 probability with sigmoid
    score_diff = best_score - null_score
    confidence = torch.sigmoid(torch.tensor(score_diff)).item()

    return answer, confidence, best_start, best_end


def generate_answer(question: str, context: str) -> dict:
    """Extract an answer for *question* from *context*.

    Handles long contexts by splitting into overlapping windows
    and picking the best answer across all windows.

    Parameters
    ----------
    question : str
        The user's question.
    context : str
        The concatenated retrieved chunks.

    Returns
    -------
    dict
        Keys: ``answer`` (str), ``score`` (float),
        ``start`` (int), ``end`` (int).
    """
    # Tokenize the question to find its length
    question_tokens = _qa_tokenizer.encode(question, add_special_tokens=False)
    # Reserve space for special tokens: <s> question </s></s> context </s>
    max_context_len = 512 - len(question_tokens) - 4

    # Tokenize the full context
    context_tokens = _qa_tokenizer.encode(context, add_special_tokens=False)

    # Create sliding windows over the context
    stride = max(max_context_len - 64, 64)  # 64-token overlap
    windows = []
    start = 0
    while start < len(context_tokens):
        end = min(start + max_context_len, len(context_tokens))
        window_tokens = context_tokens[start:end]
        window_text = _qa_tokenizer.decode(window_tokens, skip_special_tokens=True)
        windows.append(window_text)
        if end >= len(context_tokens):
            break
        start += stride

    # Process each window and collect the best answer
    best_answer = ""
    best_confidence = 0.0
    best_positions = (0, 0)

    for window_text in windows:
        inputs = _qa_tokenizer(
            question,
            window_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        with torch.no_grad():
            outputs = _qa_model(**inputs)

        answer, confidence, s, e = _get_best_span(
            outputs.start_logits,
            outputs.end_logits,
            inputs["input_ids"],
            _qa_tokenizer,
        )

        if confidence > best_confidence and answer:
            best_answer = answer
            best_confidence = confidence
            best_positions = (s, e)

    if not best_answer:
        best_answer = "Could not find a definitive answer in the retrieved context."
        best_confidence = 0.0

    return {
        "answer": best_answer,
        "score": round(best_confidence, 4),
        "start": best_positions[0],
        "end": best_positions[1],
    }
