import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/universe/tech_universe_top20_with_ciq.parquet")

def main():
    df = pd.read_parquet(PARQUET_PATH)
    # Expect columns like: ['gvkey', 'ticker', 'company_name', 'ciq_company_id', ...]
    mapping = (
        df[["ticker", "ciq_company_id"]]
        .dropna(subset=["ticker", "ciq_company_id"])
        .drop_duplicates("ticker")
    )

    print("Ticker → CIQ company_id mapping:")
    for _, row in mapping.iterrows():
        print(f'"{row["ticker"].upper()}": {int(row["ciq_company_id"])},')

if __name__ == "__main__":
    main()
