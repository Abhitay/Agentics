from pathlib import Path
from functools import lru_cache

import pandas as pd

from src.config.universe import ALLOWED_COMPANIES
from src.graph.ingest_extracted_facts import COMPANY_ID_TO_TICKER, _normalize_quarter


@lru_cache
def get_available_company_quarters(base_dir: str | Path = "data/processed"):
    """
    Read statements.parquet and return:
      - available_companies: sorted list of tickers
      - quarters_by_company: dict[ticker] -> sorted list of quarter strings (e.g. '2025Q4')
    """
    base = Path(base_dir)
    statements_path = base / "statements.parquet"
    if not statements_path.exists():
        raise FileNotFoundError(f"{statements_path} not found. Run extraction first.")

    df = pd.read_parquet(statements_path)

    # Map WRDS company_id -> ticker
    df["ticker"] = df["company_id"].map(COMPANY_ID_TO_TICKER)

    # Keep only tickers we know + configured
    df = df[df["ticker"].isin(ALLOWED_COMPANIES)]

    # Build canonical quarter string using same logic as ingest_extracted_facts
    df["quarter_str"] = df.apply(
        lambda r: _normalize_quarter(
            r.get("call_year"),
            r.get("call_quarter"),
            r.get("call_period"),
        ),
        axis=1,
    )

    df = df[df["quarter_str"].notna()]

    quarters_by_company: dict[str, list[str]] = {}
    for ticker, group in df.groupby("ticker"):
        qs = sorted(group["quarter_str"].unique())
        quarters_by_company[ticker] = qs

    available_companies = sorted(quarters_by_company.keys())
    return available_companies, quarters_by_company
