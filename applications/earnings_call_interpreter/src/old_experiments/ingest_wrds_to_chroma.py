#!/usr/bin/env python3
"""
Ingest WRDS-derived earnings call sections into Chroma, no Neo4j.

Assumes:
- data/corpus/tech_call_sections.parquet exists
- It has columns: companyid, company_name, call_period, call_year, call_quarter, clean_text, ...
"""

from pathlib import Path
import pandas as pd

from src.embeddings.vector_store import upsert_chunks


CALL_SECTION_PARQUET = Path("data/corpus/tech_call_sections.parquet")


def main():
    if not CALL_SECTION_PARQUET.exists():
        print(f"Missing {CALL_SECTION_PARQUET} – run data_wrds pipeline first.")
        return

    df = pd.read_parquet(CALL_SECTION_PARQUET)

    # Example: filter to one company first (e.g., Apple CIQ companyid)
    # Later you'll map companyid -> ticker, but for now assume you know it
    # aapl_df = df[df["company_name"] == "Apple Inc."].copy()

    # For now, just ingest everything with generic metadata
    for (company_name, call_period), sub in df.groupby(["company_name", "call_period"]):
        # You can later map company_name -> ticker and call_period -> "2025Q4"
        quarter_tag = call_period.replace(" ", "")  # "2025 Q4" -> "2025Q4"

        texts = sub["clean_text"].dropna().astype(str).tolist()
        if not texts:
            continue

        print(f"Ingesting {len(texts)} segments for {company_name} / {call_period}")

        # Reuse your existing upsert code
        upsert_chunks(
            chunks=texts,
            company=company_name,         # or ticker once you map it
            filing_type="Earnings Call",
            quarter=quarter_tag,
        )


if __name__ == "__main__":
    main()
