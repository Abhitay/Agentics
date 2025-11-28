import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Optional: simple PDF generation for summaries
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# Make sure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

UNIVERSE_PATH = ROOT_DIR / "data/universe/mapped_universe.parquet"
STATEMENTS_PATH = ROOT_DIR / "data/processed/statements.parquet"

from src.rag.pipeline import (
    answer_question,
    generate_summary,
    benchmark_peers,
)
from src.utils.quarters import normalize_quarter


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _sanitize_text_for_pdf(text: str) -> str:
    """
    Replace common Unicode punctuation with simpler ASCII versions so that
    PDF generation using latin-1 encoding doesn't crash.
    """
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2022": "-",  # bullet
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _parse_summary_sections(summary_md: str):
    """
    Parse the markdown summary into:
      - snapshot heading line (## ...)
      - dict of {subheading -> markdown body}

    Assumes subheadings start with '### '.
    """
    snapshot_heading = None
    sections = {}
    current_sub = None
    buffer = []

    lines = summary_md.splitlines()
    for line in lines:
        if line.startswith("## "):
            snapshot_heading = line.strip()
            continue

        if line.startswith("### "):
            # flush previous
            if current_sub is not None:
                sections[current_sub] = "\n".join(buffer).strip()
            current_sub = line[4:].strip()
            buffer = []
        else:
            if current_sub is not None:
                buffer.append(line)

    if current_sub is not None and buffer:
        sections[current_sub] = "\n".join(buffer).strip()

    return snapshot_heading, sections


def _create_summary_pdf(company: str, quarter: str, summary_md: str, tldr: str) -> bytes:
    """
    Nicer single-page PDF:
    - Title
    - TL;DR
    - Key Numbers
    - What Changed vs Previous Quarter
    - Guidance & Outlook
    - Risks & Watchpoints

    Uses only plain ASCII so FPDF's latin-1 encoding never crashes.
    """
    if not HAS_FPDF:
        return b""

    summary_md = _sanitize_text_for_pdf(summary_md)
    tldr = _sanitize_text_for_pdf(tldr or "")

    snapshot_heading, sections = _parse_summary_sections(summary_md)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    title = f"{company} - {quarter} Earnings Snapshot"
    pdf.multi_cell(0, 10, _sanitize_text_for_pdf(title))
    pdf.ln(2)

    # TL;DR
    if tldr:
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, "TL;DR")
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, tldr)
        pdf.ln(4)

    def print_section(title_text: str, body: str):
        """Print a section with a bold heading and ASCII '-' bullets."""
        if not body:
            return
        body = _sanitize_text_for_pdf(body)

        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, title_text)
        pdf.set_font("Arial", "", 11)

        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("-", "*")):
                # Markdown bullet -> normal ASCII dash bullet
                content = stripped[1:].strip()
                pdf.multi_cell(0, 6, f"- {content}")
            else:
                pdf.multi_cell(0, 6, stripped)
        pdf.ln(3)

    # Sections in desired order
    print_section("Key numbers", sections.get("Key Numbers", ""))
    print_section(
        "What changed vs previous quarter",
        sections.get("What Changed vs Previous Quarter", ""),
    )
    print_section("Guidance & outlook", sections.get("Guidance & Outlook", ""))
    print_section("Risks & watchpoints", sections.get("Risks & Watchpoints", ""))

    # `output(dest="S")` returns a str in FPDF 1.x; make sure we return bytes.
    out = pdf.output(dest="S")
    if isinstance(out, bytes):
        return out
    return out.encode("latin-1", "replace")



def _render_summary_analytics(company: str, quarter: str, analytics: dict):
    """
    Render the overview for the summary tab:
    - sentiment bar
    - guidance chip
    - risk level chip
    - focus segments
    """
    sentiment = analytics.get("sentiment", {}) or {}
    guidance = analytics.get("guidance", {}) or {}
    risk = analytics.get("risk", {}) or {}
    focus_segments = analytics.get("focus_segments", []) or []

    sentiment_label = sentiment.get("label", "Mixed")
    pos = float(sentiment.get("positive", 0.33) or 0.33)
    neu = float(sentiment.get("neutral", 0.34) or 0.34)
    neg = float(sentiment.get("negative", 0.33) or 0.33)

    guidance_label = guidance.get("label", "None")
    risk_level = risk.get("level", "Medium")
    top_risks = risk.get("top_risks", []) or []

    st.markdown(f"### {company} · {quarter}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall feeling of the call", sentiment_label)
    with col2:
        st.metric("Did they raise or cut guidance?", guidance_label)
    with col3:
        st.metric("Risk level mentioned on the call", risk_level)

    # Sentiment bar
    sentiment_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Score": [pos, neu, neg],
        }
    ).set_index("Sentiment")
    st.markdown("**How positive or negative was management?**")
    st.bar_chart(sentiment_df)
    st.caption("Higher 'Positive' means they sounded upbeat. Higher 'Negative' means more worry or caution.")

    # Focus segments
    if focus_segments:
        st.markdown("**What they talked about most**")
        st.markdown(" · ".join(focus_segments))

    # Top risks
    if top_risks:
        st.markdown("**Main risks they highlighted**")
        st.markdown(" · ".join(top_risks))


def _render_evidence_snippets(sources):
    """
    Show short snippets as quick evidence for the Q&A answer.
    If there are no transcript chunks, at least tell the user what we used.
    """
    st.markdown("### Why this answer?")

    if not sources:
        st.caption(
            "This answer comes from structured numbers extracted from the filing. "
            "No direct transcript excerpts were available for this question."
        )
        return

    max_snippets = 3
    for i, src in enumerate(sources[:max_snippets], start=1):
        text = src.get("text", "") or ""
        meta = src.get("metadata", {}) or {}
        snippet = text.strip()
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "…"

        meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items()) if meta else ""
        st.markdown(f"**Excerpt {i}**")
        st.markdown(f"> {snippet}")
        if meta_str:
            st.caption(meta_str)


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

universe_df["ticker"] = universe_df["ticker"].astype(str).str.upper()
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

    # Normalize quarter string in the SAME way as the RAG pipeline
    statements_df["quarter_str"] = statements_df["call_period"].apply(
        lambda cp: normalize_quarter(None, None, cp)
    )

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

# Only show companies that actually have earnings call data
universe_df = universe_df[universe_df["ticker"].isin(AVAILABLE_QUARTERS.keys())].copy()


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
        placeholder="e.g. What was total revenue? Did they raise or cut guidance?",
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

            # Human-friendly evidence
            _render_evidence_snippets(sources)

            # Full raw sources (for power users)
            if sources:
                st.markdown("### All source chunks")
                for i, src in enumerate(sources, start=1):
                    with st.expander(f"Source {i}"):
                        st.write(src.get("text", ""))
                        meta = src.get("metadata", {})
                        if meta:
                            st.caption(str(meta))


# --------------------------------------------------------------------
# Summary tab – user-friendly + analytics + TL;DR + PDF
# --------------------------------------------------------------------
with tab_summary:
    st.subheader("Generate an executive summary")

    compare_prev = st.checkbox("Compare with previous quarter", value=True)

    if st.button("Generate summary"):
        if quarter is None:
            st.error("No earnings call data available for this company/period.")
        else:
            with st.spinner("Summarizing company performance in plain English..."):
                summary, sources, analytics, tldr = generate_summary(
                    company=company,
                    filing_type=filing_type,
                    quarter=quarter,
                    temperature=temperature,
                    compare_previous=compare_prev,
                )

            # Overview + sentiment/guidance/risk/focus
            _render_summary_analytics(company, quarter, analytics or {})

            # TL;DR line
            if tldr:
                st.markdown(f"**TL;DR:** {tldr}")

            # Split summary into sections so we can use expanders
            snapshot_heading, sections = _parse_summary_sections(summary or "")

            # Snapshot heading
            if snapshot_heading:
                st.markdown(snapshot_heading)
            else:
                st.markdown(f"## Snapshot – {company} {quarter}")

            # Always-visible sections
            if "Key Numbers" in sections:
                st.markdown("### 🔢 Key numbers")
                st.markdown(sections["Key Numbers"])

            if "What Changed vs Previous Quarter" in sections:
                st.markdown("### 📈 What changed vs previous quarter")
                st.markdown(sections["What Changed vs Previous Quarter"])

            # Expanders for deeper detail
            if "Segment Performance" in sections:
                with st.expander("More detail on which parts of the business did what"):
                    st.markdown(sections["Segment Performance"])

            if "Guidance & Outlook" in sections:
                with st.expander("More detail on guidance & outlook"):
                    st.markdown(sections["Guidance & Outlook"])

            if "Risks & Watchpoints" in sections:
                with st.expander("More detail on risks & things to watch"):
                    st.markdown(sections["Risks & Watchpoints"])

            # Download as PDF
            if HAS_FPDF:
                pdf_bytes = _create_summary_pdf(company, quarter, summary, tldr)
                if pdf_bytes:
                    st.download_button(
                        label="📥 Download 1-page PDF summary",
                        data=pdf_bytes,
                        file_name=f"{company}_{quarter}_summary.pdf",
                        mime="application/pdf",
                    )
            else:
                st.caption(
                    "Install `fpdf` (`pip install fpdf`) to enable PDF downloads."
                )

            # Vector sources (if you want to inspect what fed the summary)
            if sources:
                st.markdown("### Source snippets used for summary")
                for i, src in enumerate(sources, start=1):
                    with st.expander(f"Source {i}"):
                        st.write(src.get("text", ""))
                        st.caption(str(src.get("metadata", {})))


# --------------------------------------------------------------------
# Peer benchmarking tab – multiselect peers, simple language
# --------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Peer benchmarking")

    # Build list of peer labels (exclude the base company)
    all_labels = universe_df["label"].tolist()
    peer_labels = [lbl for lbl in all_labels if lbl != selected_label]

    selected_peer_labels = st.multiselect(
        "Peer companies",
        peer_labels,
        default=peer_labels[:3] if len(peer_labels) >= 3 else peer_labels,
        help="Pick a few similar companies to compare against.",
    )

    # Map labels -> tickers
    peer_tickers = []
    for lbl in selected_peer_labels:
        row = universe_df[universe_df["label"] == lbl].iloc[0]
        peer_tickers.append(str(row["ticker"]).upper())

    if st.button("Run benchmarking"):
        if not peer_tickers:
            st.warning("Please select at least one peer company.")
        elif quarter is None:
            st.error("No earnings call data available for this company/period.")
        else:
            with st.spinner("Comparing companies in simple language..."):
                result, _ = benchmark_peers(
                    base_company=company,
                    peers=peer_tickers,
                    filing_type=filing_type,
                    quarter=quarter,
                    temperature=temperature,
                )

            st.markdown("### What stands out vs peers")
            st.write(result)
