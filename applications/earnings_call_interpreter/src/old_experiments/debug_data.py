# scripts/debug_data.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.rag.pipeline import (
    _load_metrics_df,
    _load_segments_df,
    _load_risks_df,
    get_company_quarter_metrics,
    get_company_quarters,
    get_previous_quarter,
)

def main():
    metrics = _load_metrics_df()
    print("Metrics shape:", metrics.shape)
    print(metrics.head(5))

    segs = _load_segments_df()
    print("\nSegments shape:", segs.shape)
    print(segs.head(5))

    risks = _load_risks_df()
    print("\nRisks shape:", risks.shape)
    print(risks.head(5))

    # Pick a random ticker from metrics
    if not metrics.empty:
        sample_ticker = metrics["ticker"].dropna().astype(str).str.upper().iloc[0]
        quarters = get_company_quarters(sample_ticker)
        print(f"\nSample ticker: {sample_ticker}")
        print("Quarters:", quarters)

        if quarters:
            q = quarters[-1]  # latest
            cm = get_company_quarter_metrics(sample_ticker, q)
            print(f"\nMetrics for {sample_ticker} {q}:")
            print(cm.head())

            prev_q = get_previous_quarter(sample_ticker, q)
            print("Prev quarter:", prev_q)

if __name__ == "__main__":
    main()
