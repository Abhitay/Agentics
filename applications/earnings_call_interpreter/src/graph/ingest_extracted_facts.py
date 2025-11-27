"""
Build the in-memory knowledge graph + Chroma index
from the extracted parquet files.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from src.data_ingestion.chunking import simple_paragraph_chunker
from src.embeddings.vector_store import upsert_chunks
from src.graph.build_graph import get_graph, add_company_node, add_quarter_node, add_metric


# -----------------------------------------------------------------------------
# AUTO company_id → ticker mapping
# -----------------------------------------------------------------------------

def _load_company_id_to_ticker(
    universe_path: str | Path = "data/universe/tech_universe_top20_with_ciq.parquet",
) -> Dict[float, str]:
    """
    Load mapping from WRDS or CIQ parquet:
        company_id -> ticker
    No filtering — ingest every company present.
    """
    path = Path(universe_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. "
        )

    df = pd.read_parquet(path)

    # Guess id & ticker cols
    id_cols = ["ciq_company_id", "company_id", "companyid"]
    ticker_cols = ["ticker", "ticker_x", "ticker_y"]

    id_col = next((c for c in id_cols if c in df.columns), None)
    ticker_col = next((c for c in ticker_cols if c in df.columns), None)

    if id_col is None:
        raise ValueError(f"No valid company_id column found in {path}")
    if ticker_col is None:
        raise ValueError(f"No valid ticker column found in {path}")

    mapping = {}
    for _, row in df.iterrows():
        cid = row[id_col]
        ticker = row[ticker_col]
        if pd.notna(cid) and isinstance(ticker, str):
            mapping[float(cid)] = ticker.upper().strip()

    print(f"[ingest] Loaded {len(mapping)} company_id → ticker mappings.")
    return mapping


COMPANY_ID_TO_TICKER = _load_company_id_to_ticker()


# -----------------------------------------------------------------------------
# Load processed parquet files
# -----------------------------------------------------------------------------

def load_extracted_parquets(base_dir: str | Path = "data/processed") -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    return {
        "statements": pd.read_parquet(base / "statements.parquet"),
        "metrics": pd.read_parquet(base / "metrics.parquet"),
        "risks": pd.read_parquet(base / "risks.parquet"),
        "segments": pd.read_parquet(base / "segments.parquet"),
    }


def _normalize_quarter(call_year, call_quarter, call_period) -> Optional[str]:
    """
    Convert data to form like 2025Q3
    """
    if pd.notna(call_year) and pd.notna(call_quarter):
        try:
            return f"{int(call_year)}Q{str(call_quarter).lstrip('Q')}"
        except:
            pass

    if isinstance(call_period, str):
        return call_period.replace(" ", "")

    return None


# -----------------------------------------------------------------------------
# Build Graph
# -----------------------------------------------------------------------------

def build_graph_from_extracted(statements_df, metrics_df, risks_df, segments_df):
    g = get_graph()

    # ---- Statements ----
    for _, row in statements_df.iterrows():
        company_id = float(row.get("company_id")) if pd.notna(row.get("company_id")) else None
        ticker = COMPANY_ID_TO_TICKER.get(company_id)
        if not ticker:
            continue

        call_id = row.get("call_id")
        stmt_id = row.get("statement_id")
        text = row.get("text")
        quarter_str = _normalize_quarter(
            row.get("call_year"),
            row.get("call_quarter"),
            row.get("call_period"),
        )

        # Add nodes
        add_company_node(ticker, name=row.get("company_name"))
        if quarter_str:
            add_quarter_node(ticker, quarter_str)

        g.add_node(
            f"call:{call_id}",
            type="call",
            call_id=call_id,
            company=ticker,
        )
        g.add_node(
            f"statement:{stmt_id}",
            type="statement",
            text=text,
        )

        g.add_edge(f"company:{ticker}", f"call:{call_id}", type="HAS_CALL")
        g.add_edge(f"call:{call_id}", f"statement:{stmt_id}", type="HAS_STATEMENT")
        if quarter_str:
            g.add_edge(f"quarter:{ticker}:{quarter_str}", f"call:{call_id}", type="CALL_IN_QUARTER")

    # ---- Metrics ----
    for _, row in metrics_df.iterrows():
        company_id = float(row.get("company_id")) if pd.notna(row.get("company_id")) else None
        ticker = COMPANY_ID_TO_TICKER.get(company_id)
        if not ticker:
            continue

        quarter_str = _normalize_quarter(
            row.get("call_year"),
            row.get("call_quarter"),
            row.get("call_period"),
        )
        stmt_node = f"statement:{row.get('statement_id')}"

        add_metric(
            company_ticker=ticker,
            quarter=quarter_str or "UNKNOWN",
            metric_name=row.get("metric_name"),
            metric_value=row.get("metric_value") if pd.notna(row.get("metric_value")) else None,
            unit=row.get("metric_unit"),
            extra=row.to_dict(),
        )

        metric_node = f"metric:{ticker}:{quarter_str or 'UNKNOWN'}:{row.get('metric_name')}"
        if stmt_node in g:
            g.add_edge(stmt_node, metric_node, type="MENTIONS_METRIC")

    # ---- Risks ----
    for _, row in risks_df.iterrows():
        company_id = float(row.get("company_id")) if pd.notna(row.get("company_id")) else None
        ticker = COMPANY_ID_TO_TICKER.get(company_id)
        if not ticker:
            continue

        stmt_node = f"statement:{row.get('statement_id')}"
        risk_node = f"risk:{row.get('risk_type')}"
        g.add_node(risk_node, type="risk", **row.to_dict())

        if stmt_node in g:
            g.add_edge(stmt_node, risk_node, type="MENTIONS_RISK")

    # ---- Segments ----
    for _, row in segments_df.iterrows():
        company_id = float(row.get("company_id")) if pd.notna(row.get("company_id")) else None
        ticker = COMPANY_ID_TO_TICKER.get(company_id)
        if not ticker:
            continue

        stmt_node = f"statement:{row.get('statement_id')}"
        seg_node = f"segment:{row.get('segment_name')}"
        g.add_node(seg_node, type="segment", **row.to_dict())

        if stmt_node in g:
            g.add_edge(stmt_node, seg_node, type="MENTIONS_SEGMENT")


# -----------------------------------------------------------------------------
# Chroma
# -----------------------------------------------------------------------------

def index_statements_in_chroma(statements_df: pd.DataFrame):
    """
    Group statements by company+quarter, chunk, insert to Chroma
    """
    grouped = statements_df.groupby(
        ["company_id", "call_period", "call_year", "call_quarter"]
    )

    for (company_id, cp, cy, cq), group in grouped:
        ticker = COMPANY_ID_TO_TICKER.get(float(company_id))
        if not ticker:
            continue

        quarter_str = _normalize_quarter(cy, cq, cp)
        combined_text = "\n\n".join(group["text"].fillna("").tolist())

        chunks = simple_paragraph_chunker(combined_text)
        for ch in chunks:
            meta = ch.get("metadata", {}) or {}
            meta.update({"call_period": cp, "call_year": cy, "call_quarter": cq})
            ch["metadata"] = meta

        upsert_chunks(chunks, ticker, "Earnings Call", quarter_str)
        print(f"Indexed {len(chunks)} chunks → {ticker} / {quarter_str}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def build_kg_and_index_from_extracted(base_dir: str | Path = "data/processed") -> None:
    """
    Convenience entrypoint: load parquet files, populate KG, index statements in Chroma.
    """
    dfs = load_extracted_parquets(base_dir)

    statements_df = dfs["statements"]
    metrics_df = dfs["metrics"]
    risks_df = dfs["risks"]
    segments_df = dfs["segments"]

    print("Building in-memory graph...")
    build_graph_from_extracted(statements_df, metrics_df, risks_df, segments_df)
    print("Graph built.")

    print("Indexing text into Chroma...")
    index_statements_in_chroma(statements_df)
    print("DONE.")

