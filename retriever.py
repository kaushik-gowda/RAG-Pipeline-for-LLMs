# ============================================================
# retriever.py – Knowledge retrieval & chunking
# ============================================================
"""
This module handles:
  1. Fetching content from Wikipedia for a given topic.
  2. Splitting the retrieved text into smaller, overlapping
     token-based chunks suitable for embedding and retrieval.
"""

import wikipedia
from transformers import AutoTokenizer

from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

# Load the tokenizer once at module level for reuse
_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def get_wikipedia_content(topic: str) -> str | None:
    """Fetch the full text of a Wikipedia article for *topic*.

    Uses multiple fallback strategies:
      1. Direct page lookup with auto-suggest.
      2. On PageError → search Wikipedia and try the top result.
      3. On DisambiguationError → try the first option.

    Returns
    -------
    str or None
        The page content if found, ``None`` on error.
    """
    # Strategy 1: Direct lookup
    try:
        page = wikipedia.page(topic, auto_suggest=True)
        print(f"[retriever] Found page: {page.title}")
        return page.content
    except wikipedia.exceptions.PageError:
        pass  # fall through to search
    except wikipedia.exceptions.DisambiguationError as e:
        print(
            f"[retriever] Ambiguous topic '{topic}'. "
            f"Trying first option: {e.options[0]}"
        )
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.content
        except Exception:
            pass  # fall through to search
    except Exception as e:
        print(f"[retriever] Direct lookup error: {e}")

    # Strategy 2: Search Wikipedia and try top results
    print(f"[retriever] Searching Wikipedia for '{topic}'...")
    try:
        search_results = wikipedia.search(topic, results=5)
        if not search_results:
            print(f"[retriever] No search results for '{topic}'.")
            return None

        for result_title in search_results:
            try:
                page = wikipedia.page(result_title, auto_suggest=False)
                print(f"[retriever] Found page via search: {page.title}")
                return page.content
            except (wikipedia.exceptions.PageError,
                    wikipedia.exceptions.DisambiguationError):
                continue
    except Exception as e:
        print(f"[retriever] Search error: {e}")

    print(f"[retriever] Could not retrieve any content for '{topic}'.")
    return None


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping token-based chunks.

    Parameters
    ----------
    text : str
        The full document text.
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of overlapping tokens between consecutive chunks.

    Returns
    -------
    list[str]
        A list of text chunks.
    """
    tokens = _tokenizer.tokenize(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = _tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append(chunk_text)

        if end == len(tokens):
            break
        start = end - chunk_overlap

    return chunks
