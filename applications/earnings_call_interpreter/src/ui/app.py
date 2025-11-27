import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Make sure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

UNIVERSE_PATH = ROOT_DIR / "data/universe/mapped_universe.parquet"
STATEMENTS_PATH = ROOT_DIR / "data/processed/statements.parquet"

from src.rag.pipeline import answer_question, generate_summary, benchmark_peers
from src.utils.quarters import normalize_quarter


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Load universe for company dropdown
# --------------------------------------------------------------------
universe_df = pd.read_parquet(UNIVERSE_PATH)
universe_df = universe_df.dropna(subset=["companyid"]).copy()

# Ensure consistent column names
if "ticker" not in universe_df.columns:
    raise ValueError("Expected 'ticker' column in mapped_universe.parquet")

if "name" not in universe_df.columns:
    raise ValueError("Expected 'name' column in mapped_universe.parquet")

universe_df["label"] = universe_df["ticker"] + " – " + universe_df["name"]

# Map companyid -> ticker (float to match statements.parquet company_id)
companyid_to_ticker = {
    float(row["companyid"]): str(row["ticker"]).upper()
    for _, row in universe_df.iterrows()
}

# --------------------------------------------------------------------
# Build AVAILABLE_QUARTERS from statements.parquet
# --------------------------------------------------------------------
if STATEMENTS_PATH.exists():
    statements_df = pd.read_parquet(STATEMENTS_PATH).copy()

    # Map company_id to ticker using mapped_universe
    statements_df["ticker"] = statements_df["company_id"].map(companyid_to_ticker)

    # Normalize quarter string
    statements_df["quarter_str"] = [
        normalize_quarter(row.call_year, row.call_quarter, row.call_period)
        for row in statements_df.itertuples(index=False)
    ]

    # Drop rows without ticker or quarter
    statements_df = statements_df.dropna(subset=["ticker", "quarter_str"])

    # Build mapping: ticker -> sorted list of available quarters (latest first)
    AVAILABLE_QUARTERS = (
        statements_df.groupby("ticker")["quarter_str"]
        .apply(lambda s: sorted(set(s), reverse=True))
        .to_dict()
    )
else:
    AVAILABLE_QUARTERS = {}

# Optional: if you want to only show companies that actually have data
# universe_df = universe_df[universe_df["ticker"].isin(AVAILABLE_QUARTERS.keys())].copy()


# --------------------------------------------------------------------
# Streamlit layout
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Earnings Call RAG Analyst",
    layout="wide",
)

st.title("📈 Earnings Call RAG Analyst")


with st.sidebar:
    st.header("Settings")

    # Company dropdown
    selected_label = st.selectbox(
        "Company",
        universe_df["label"].tolist(),
        index=0,
    )

    selected_row = universe_df[universe_df["label"] == selected_label].iloc[0]
    company = selected_row["ticker"]                # used in queries
    company_id = float(selected_row["companyid"])   # if you later need it


    # Quarter dropdown driven by AVAILABLE_QUARTERS
    available_q = AVAILABLE_QUARTERS.get(company, [])

    if available_q:
        quarter = st.selectbox("Quarter", available_q, index=0)
    else:
        quarter = None
        st.selectbox("Quarter", ["No data available"], index=0, disabled=True)

    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)


tab_qna, tab_summary, tab_benchmark = st.tabs(
    ["🔍 Q&A", "📝 Executive Summary", "⚔️ Peer Benchmarking"]
)


filing_type = "Earnings Call"  # currently only supporting earnings calls
# --------------------------------------------------------------------
# Q&A tab
# --------------------------------------------------------------------
with tab_qna:
    st.subheader("Ask a question about this company's filings")

    question = st.text_area(
        "Question",
        placeholder="e.g. What did management say about revenue guidance?",
        height=120,
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif quarter is None:
            st.error("No earnings call data available for this company/period.")
        else:
            with st.spinner("Analyzing filings..."):
                answer, sources = answer_question(
                    question=question,
                    company=company,
                    filing_type=filing_type,
                    quarter=quarter,
                    temperature=temperature,
                )

            st.markdown("### Answer")
            st.write(answer)

            if sources:
                st.markdown("### Sources")
                for i, src in enumerate(sources, start=1):
                    with st.expander(f"Source {i}"):
                        st.write(src.get("text", ""))
                        meta = src.get("metadata", {})
                        if meta:
                            st.caption(str(meta))


# --------------------------------------------------------------------
# Summary tab
# --------------------------------------------------------------------
with tab_summary:
    st.subheader("Generate an executive summary")

    compare_prev = st.checkbox("Compare with previous quarter", value=True)

    if st.button("Generate summary"):
        if quarter is None:
            st.error("No earnings call data available for this company/period.")
        else:
            with st.spinner("Summarizing company performance..."):
                summary, sources = generate_summary(
                    company=company,
                    filing_type=filing_type,
                    quarter=quarter,
                    temperature=temperature,
                    compare_previous=compare_prev,
                )

            st.markdown("### Summary")
            st.markdown(summary)

            if sources:
                st.markdown("### Sources")
                for i, src in enumerate(sources, start=1):
                    with st.expander(f"Source {i}"):
                        st.write(src.get("text", ""))
                        st.caption(str(src.get("metadata", {})))



# --------------------------------------------------------------------
# Peer benchmarking tab
# --------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Peer benchmarking")

    peers_input = st.text_input(
        "Peer tickers (comma-separated)", value="MSFT, GOOGL, AMZN"
    )

    metric = st.text_input(
        "Metric to compare",
        value="Revenue growth",
        placeholder="e.g. Operating margin, EPS growth, cloud revenue",
    )

    if st.button("Run benchmarking"):
        peers = [p.strip() for p in peers_input.split(",") if p.strip()]
        if not peers or not metric.strip():
            st.warning("Please enter peers and a metric.")
        elif quarter is None:
            st.error("No earnings call data available for this company/period.")
        else:
            with st.spinner("Comparing peers..."):
                result, table = benchmark_peers(
                    base_company=company,
                    peers=peers,
                    metric=metric,
                    filing_type=filing_type,
                    quarter=quarter,
                    temperature=temperature,
                )

            st.markdown("### Benchmarking Insight")
            st.write(result)

            if table is not None:
                st.markdown("### Comparison Table")
                st.dataframe(table)
