"""
Chunking utilities: split raw filing text into semantically useful pieces.
"""

from typing import List, Dict, Any


def simple_paragraph_chunker(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split text into chunks respecting paragraph boundaries.

    Why this is good:
    - LLM embeddings respect semantic boundaries
    - Prevents losing context mid-sentence
    - Works well for earnings calls & 10-Q commentary

    Args:
        text: cleaned text
        max_chars: target size per chunk
        overlap: preserve context from end of previous chunk

    Returns:
        List of dicts:
        [
            {
                "text": "...",
                "metadata": {...}
            }
        ]
    """

    # Break at empty lines or paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: List[Dict[str, Any]] = []
    current = ""

    for p in paragraphs:
        # Add paragraph to current chunk if not over max_chars
        if len(current) + len(p) + 1 <= max_chars:
            current += ("\n" if current else "") + p
        else:
            # save current chunk
            if current:
                chunks.append({"text": current, "metadata": {}})

            # carry context overlap
            if overlap > 0 and chunks:
                tail = current[-overlap:]
                current = tail + "\n" + p
            else:
                current = p

    # remaining chunk
    if current:
        chunks.append({"text": current, "metadata": {}})

    return chunks
