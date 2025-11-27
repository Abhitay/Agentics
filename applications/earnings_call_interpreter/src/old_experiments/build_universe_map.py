import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data/corpus/tech_call_sections.parquet"
UNIVERSE_PATH = PROJECT_ROOT / "data/universe/tech_universe_top20_with_ciq.parquet"
OUT_PATH = PROJECT_ROOT / "data/universe/mapped_universe.parquet"


def build_universe_map():
    print(f"Loading corpus from: {CORPUS_PATH}")
    corpus = pd.read_parquet(CORPUS_PATH)

    print(f"Loading top-20 universe from: {UNIVERSE_PATH}")
    universe = pd.read_parquet(UNIVERSE_PATH)

    # Normalize universe columns
    uni = universe.rename(
        columns={
            "ticker_x": "ticker",
            "company_name": "name",
            "ciq_company_id": "ciq_id",
        }
    ).copy()

    uni = uni[["ticker", "name", "ciq_id", "market_cap"]]

    # Get unique (companyid, company_name) from corpus
    companies = (
        corpus[["companyid", "company_name"]]
        .drop_duplicates()
        .rename(columns={"company_name": "name"})
    )

    # Join on name (this assumes names in WRDS ~ match CIQ)
    mapped = uni.merge(companies, on="name", how="left")

    print("\nPreview of mapped universe:")
    print(mapped[["ticker", "name", "ciq_id", "companyid"]].head(20))

    # Save for later use
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(OUT_PATH)

    print(f"\nSaved mapped universe to: {OUT_PATH}")


if __name__ == "__main__":
    build_universe_map()
