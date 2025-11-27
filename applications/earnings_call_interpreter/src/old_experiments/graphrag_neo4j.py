"""
Neo4j-based GraphRAG client.

Uses:
- Neo4j Aura (already populated by teammate)
- Gemini text-embedding-004
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
from neo4j import GraphDatabase
from google import genai

from src.config.settings import settings

# Gemini client – picks GEMINI_API_KEY from env
genai_client = genai.Client()
EMBEDDING_MODEL = "text-embedding-004"

def embed_query(query: str) -> List[float]:
    """Embed a query using Gemini text-embedding-004."""
    resp = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    vec = resp.embeddings[0].values
    v = np.array(vec, dtype="float32")
    v = v / max(np.linalg.norm(v), 1e-12)
    return v.tolist()


# Neo4j driver (Aura)
_driver = GraphDatabase.driver(
    settings.graph_db.uri,
    auth=(settings.graph_db.user, settings.graph_db.password),
)

TOP_K = 10

def graphrag_query(
    question: str,
    company_id: Optional[float] = None,
    call_period: Optional[str] = None,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    GraphRAG-style retrieval:
    1. Embed question
    2. Vector search on Statement nodes
    3. Expand to metrics/risks/segments + company/call
    """
    q_vec = embed_query(question)

    with _driver.session() as session:
        vec_res = session.run(
            """
            CALL db.index.vector.queryNodes(
                'statement_embedding_index', $top_k, $query_vec
            )
            YIELD node, score
            WHERE ($company_id IS NULL OR EXISTS {
                     MATCH (c:Company)-[:HAS_CALL]->(:Call)-[:HAS_STATEMENT]->(node)
                     WHERE c.company_id = $company_id
                  })
              AND ($call_period IS NULL OR node.call_period = $call_period)
            RETURN node.statement_id AS statement_id, score
            """,
            {
                "top_k": top_k,
                "query_vec": q_vec,
                "company_id": company_id,
                "call_period": call_period,
            },
        ).data()

        candidate_ids = [row["statement_id"] for row in vec_res]

        if not candidate_ids:
            return {
                "question": question,
                "company_id": company_id,
                "call_period": call_period,
                "statements": [],
            }

        result2 = session.run(
            """
            MATCH (c:Company)-[:HAS_CALL]->(call:Call)-[:HAS_STATEMENT]->(s:Statement)
            WHERE s.statement_id IN $statement_ids
            OPTIONAL MATCH (s)-[mm:MENTIONS_METRIC]->(m:Metric)
            OPTIONAL MATCH (s)-[rr:MENTIONS_RISK]->(r:Risk)
            OPTIONAL MATCH (s)-[sg:MENTIONS_SEGMENT]->(seg:Segment)
            RETURN
              s.statement_id AS statement_id,
              s.text         AS text,
              s.overall_sentiment AS overall_sentiment,
              c.company_id   AS company_id,
              c.name         AS company_name,
              call.call_id   AS call_id,
              call.period    AS call_period,
              collect(DISTINCT mm{.*, metric: m.name})    AS metrics,
              collect(DISTINCT rr{.*, risk: r.type})      AS risks,
              collect(DISTINCT sg{.*, segment: seg.name}) AS segments
            """,
            {"statement_ids": candidate_ids},
        ).data()

    return {
        "question": question,
        "company_id": company_id,
        "call_period": call_period,
        "statements": result2,
    }
