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
from src.utils.quarters import normalize_quarter  # ✅ keep quarter normalization in sync

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

    🔧 IMPORTANT:
    - Normalize company (upper-case).
    - Normalize quarter using the same logic as metrics/summary
      so that filters line up with UI & pipeline.
    """
    col = get_chroma_collection()

    # Normalize company & quarter once here
    company_norm = str(company).upper() if company else None
    quarter_norm = (
        normalize_quarter(None, None, quarter) if quarter is not None else None
    )

    ids = []
    texts = []
    metadatas = []

    for i, ch in enumerate(chunks):
        ids.append(f"{company_norm}_{filing_type}_{quarter_norm}_{i}")
        texts.append(ch["text"])

        metadata = (ch.get("metadata") or {}).copy()
        metadata.update(
            {
                "company": company_norm,
                "filing_type": filing_type,
                "quarter": quarter_norm,
            }
        )

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

    IMPORTANT:
    - Normalize company & quarter same as in upsert.
    - If no results are found with a specific quarter, fall back to
      searching for the same company/filing_type across all quarters.
    """
    col = get_chroma_collection()

    # Normalize inputs to match how we stored them
    company_norm = str(company).upper() if company else None
    quarter_norm = (
        normalize_quarter(None, None, quarter) if quarter is not None else None
    )

    def _build_where(c: Optional[str], f: Optional[str], q: Optional[str]):
        clauses: List[Dict[str, Any]] = []
        if c:
            clauses.append({"company": {"$eq": c}})
        if f:
            clauses.append({"filing_type": {"$eq": f}})
        if q:
            clauses.append({"quarter": {"$eq": q}})

        if len(clauses) == 0:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # First pass: strict filter including quarter (if provided)
    where_expr = _build_where(company_norm, filing_type, quarter_norm)

    res = col.query(
        query_texts=[query],
        n_results=top_k,
        where=where_expr,
    )

    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    # Fallback: if nothing found and we filtered by quarter,
    # try again without the quarter restriction (same company & filing_type).
    if (not documents) and quarter_norm is not None:
        where_expr_loose = _build_where(company_norm, filing_type, None)
        res = col.query(
            query_texts=[query],
            n_results=top_k,
            where=where_expr_loose,
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
