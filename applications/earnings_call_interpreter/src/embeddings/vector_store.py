"""
Vector store abstraction using ChromaDB and SentenceTransformers.

Responsibilities:
- Initialize a persistent Chroma collection.
- Upsert text chunks with metadata.
- Run semantic search over stored chunks.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

import numpy as np
import pandas as pd

from src.config.settings import settings

_client = chromadb.PersistentClient(path=settings.vector_db.persist_directory)
_collection = None


def get_chroma_collection(name: str = None):
    """
    Lazily initialize and return the Chroma collection.
    Uses a SentenceTransformer embedding model.
    """
    global _client, _collection

    if name is None:
        name = settings.vector_db.collection_name

    if _client is None:
        persist_dir = Path(settings.vector_db.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir))

    if _collection is None:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
        )

    return _collection


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all metadata values are plain Python types that Chroma accepts:
    str, int, float, bool, None.

    Converts:
    - numpy / pandas scalar types -> Python scalars
    - timestamps -> ISO string
    - anything weird -> str(value)
    """
    clean: Dict[str, Any] = {}

    for k, v in (meta or {}).items():
        if v is None:
            clean[k] = None
            continue

        # NumPy & pandas scalar types
        if isinstance(v, (np.generic,)):
            clean[k] = v.item()
            continue

        # pandas Timestamp / datetime64
        if isinstance(v, (pd.Timestamp,)):
            clean[k] = v.isoformat()
            continue

        # Basic Python types are fine
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
            continue

        # Lists / dicts / other objects – stringify
        clean[k] = str(v)

    return clean


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    company: str,
    filing_type: str,
    quarter: Optional[str],
):
    """
    Upsert a list of chunks into the vector store.

    Each chunk is expected to be:
        {
            "text": "...",
            "metadata": {...}   # optional
        }
    We enrich metadata with company / filing_type / quarter.
    """
    col = get_chroma_collection()

    ids = []
    texts = []
    metadatas = []

    for i, ch in enumerate(chunks):
        ids.append(f"{company}_{filing_type}_{quarter}_{i}")
        texts.append(ch["text"])

        metadata = (ch.get("metadata") or {}).copy()
        metadata.update(
            {
                "company": company,
                "filing_type": filing_type,
                "quarter": quarter,
            }
        )

        # 🔧 NEW: sanitize all metadata values
        metadatas.append(_sanitize_metadata(metadata))

    if texts:
        col.upsert(ids=ids, documents=texts, metadatas=metadatas)


def hybrid_search(
    query: str,
    company: Optional[str] = None,
    filing_type: Optional[str] = None,
    quarter: Optional[str] = None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Run a semantic search over stored chunks.

    Uses Chroma's filter syntax with $and / $eq.
    """
    col = get_chroma_collection()

    # Build filter clauses
    clauses: List[Dict[str, Any]] = []

    if company:
        clauses.append({"company": {"$eq": company}})
    if filing_type:
        clauses.append({"filing_type": {"$eq": filing_type}})
    if quarter:
        clauses.append({"quarter": {"$eq": quarter}})

    where_expr: Optional[Dict[str, Any]] = None
    if len(clauses) == 1:
        where_expr = clauses[0]
    elif len(clauses) > 1:
        where_expr = {"$and": clauses}

    res = col.query(
        query_texts=[query],
        n_results=top_k,
        where=where_expr,
    )

    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    results: List[Dict[str, Any]] = []
    for doc, meta in zip(documents, metadatas):
        results.append(
            {
                "text": doc,
                "metadata": meta,
            }
        )
    return results
