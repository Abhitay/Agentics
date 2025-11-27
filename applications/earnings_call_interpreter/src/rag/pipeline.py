"""
RAG pipeline: combines vector search + metrics/segments/risks + LLM.

Right now:
- Uses Chroma for vector search (via hybrid_search)
- Uses processed parquet files for structured metrics/facts
- Uses Gemini for generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json

import pandas as pd
from google import genai

from src.embeddings.vector_store import hybrid_search
from src.config.settings import settings
from src.utils.quarters import normalize_quarter, quarter_sort_key

# --------------------------------------------------------------------
# Gemini client
# --------------------------------------------------------------------
client = genai.Client()


def _call_llm(prompt: str, temperature: float = 0.2) -> str:
    model_name = settings.llm.model_name or "gemini-2.5-flash"
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": temperature},
    )
    return response.text


# --------------------------------------------------------------------
# Paths & lazy-loaded dataframes
# --------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT_DIR / "data" / "processed" / "metrics.parquet"
SEGMENTS_PATH = ROOT_DIR / "data" / "processed" / "segments.parquet"
RISKS_PATH = ROOT_DIR / "data" / "processed" / "risks.parquet"
UNIVERSE_PATH = ROOT_DIR / "data" / "universe" / "mapped_universe.parquet"

_metrics_df: Optional[pd.DataFrame] = None
_segments_df: Optional[pd.DataFrame] = None
_risks_df: Optional[pd.DataFrame] = None
_ticker_to_companyid: Dict[str, float] = {}
_companyid_to_ticker: Dict[float, str] = {}


def _load_universe_mapping() -> None:
    """Populate ticker <-> company_id maps from mapped_universe.parquet."""
    global _ticker_to_companyid, _companyid_to_ticker
    if _ticker_to_companyid and _companyid_to_ticker:
        return

    if not UNIVERSE_PATH.exists():
        return

    uni = pd.read_parquet(UNIVERSE_PATH).copy()
    uni = uni.dropna(subset=["companyid"])
    uni["ticker"] = uni["ticker"].astype(str).str.upper()

    _ticker_to_companyid = {
        row["ticker"]: float(row["companyid"]) for _, row in uni.iterrows()
    }
    _companyid_to_ticker = {
        float(row["companyid"]): row["ticker"] for _, row in uni.iterrows()
    }


def _get_company_id_from_ticker(ticker: str) -> Optional[float]:
    _load_universe_mapping()
    return _ticker_to_companyid.get(str(ticker).upper())


def _load_metrics_df() -> pd.DataFrame:
    """Load metrics.parquet once and cache it."""
    global _metrics_df
    if _metrics_df is not None:
        return _metrics_df

    if not METRICS_PATH.exists():
        _metrics_df = pd.DataFrame()
        return _metrics_df

    df = pd.read_parquet(METRICS_PATH).copy()
    _load_universe_mapping()

    df["ticker"] = df["company_id"].map(_companyid_to_ticker)
    df["quarter_str"] = df["call_period"].apply(
        lambda cp: normalize_quarter(None, None, cp)
    )
    _metrics_df = df
    return _metrics_df


def _load_segments_df() -> pd.DataFrame:
    """Load segments.parquet once and cache it."""
    global _segments_df
    if _segments_df is not None:
        return _segments_df

    if not SEGMENTS_PATH.exists():
        _segments_df = pd.DataFrame()
        return _segments_df

    df = pd.read_parquet(SEGMENTS_PATH).copy()
    _load_universe_mapping()
    df["ticker"] = df["company_id"].map(_companyid_to_ticker)
    df["quarter_str"] = df["call_period"].apply(
        lambda cp: normalize_quarter(None, None, cp)
    )
    _segments_df = df
    return _segments_df


def _load_risks_df() -> pd.DataFrame:
    """Load risks.parquet once and cache it."""
    global _risks_df
    if _risks_df is not None:
        return _risks_df

    if not RISKS_PATH.exists():
        _risks_df = pd.DataFrame()
        return _risks_df

    df = pd.read_parquet(RISKS_PATH).copy()
    _load_universe_mapping()
    df["ticker"] = df["company_id"].map(_companyid_to_ticker)
    df["quarter_str"] = df["call_period"].apply(
        lambda cp: normalize_quarter(None, None, cp)
    )
    _risks_df = df
    return _risks_df


# --------------------------------------------------------------------
# Metrics / segments / risks helpers
# --------------------------------------------------------------------
def get_company_quarter_metrics(company: str, quarter: str) -> pd.DataFrame:
    metrics = _load_metrics_df()
    if metrics.empty:
        return metrics

    company = str(company).upper()
    df = metrics[
        (metrics["ticker"] == company) & (metrics["quarter_str"] == quarter)
    ].copy()
    return df


def get_company_quarters(company: str) -> List[str]:
    metrics = _load_metrics_df()
    if metrics.empty:
        return []

    company = str(company).upper()
    qs = metrics.loc[metrics["ticker"] == company, "quarter_str"].dropna().unique()
    qs_sorted = sorted(qs, key=quarter_sort_key)
    return list(qs_sorted)


def get_previous_quarter(company: str, current_quarter: str) -> Optional[str]:
    qs = get_company_quarters(company)
    if not qs:
        return None

    try:
        idx = qs.index(current_quarter)
    except ValueError:
        return None

    if idx == 0:
        return None  # no previous
    return qs[idx - 1]


def get_company_quarter_segments(company: str, quarter: str) -> pd.DataFrame:
    segs = _load_segments_df()
    if segs.empty:
        return segs

    company = str(company).upper()
    df = segs[
        (segs["ticker"] == company) & (segs["quarter_str"] == quarter)
    ].copy()
    return df


def get_company_quarter_risks(company: str, quarter: str) -> pd.DataFrame:
    risks = _load_risks_df()
    if risks.empty:
        return risks

    company = str(company).upper()
    df = risks[
        (risks["ticker"] == company) & (risks["quarter_str"] == quarter)
    ].copy()
    return df


# --------------------------------------------------------------------
# Q&A – unchanged logic, still vector-only for now
# --------------------------------------------------------------------
def answer_question(
    question: str,
    company: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main Q&A entrypoint.

    For now:
    - Only uses vector search (Chroma) to fetch context.
    - No external graph database.
    """
    # 1. Retrieve from vector store
    vec_results = hybrid_search(
        query=question,
        company=company,
        filing_type=filing_type,
        quarter=quarter,
        top_k=8,
    )

    # 2. Build context string
    if vec_results:
        context = "\n\n---\n\n".join([r["text"] for r in vec_results])
    else:
        context = "NO CONTEXT AVAILABLE"

    # 3. Build prompt
    prompt = f"""
You are a senior equity research analyst.

Use ONLY the context from earnings filings below to answer the user's question.
If something is not supported by the context, say so explicitly.

Context:
{context}

Question: {question}

Answer in a concise, well-structured way and clearly state any uncertainty.
"""

    # 4. Call Gemini
    answer = _call_llm(prompt, temperature=temperature)

    # 5. Return answer + vector sources
    return answer, vec_results


# --------------------------------------------------------------------
# Executive summary – structured + metrics-aware
# --------------------------------------------------------------------
def generate_summary(
    company: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
    compare_previous: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate an executive summary of the company's earnings call.

    Uses:
    - Vector search over call statements for narrative context.
    - Structured metrics for this quarter (+ previous quarter if available).
    - Segments and risks tables for extra color.
    """
    if quarter is None:
        return "No quarter selected.", []

    # 1) Vector context from earnings call
    vec_results = hybrid_search(
        query="key highlights, management commentary, guidance, and risks from this earnings call",
        company=company,
        filing_type=filing_type,
        quarter=quarter,
        top_k=10,
    )

    context = "\n\n---\n\n".join([r["text"] for r in vec_results]) or "NO CONTEXT AVAILABLE"

    # 2) Structured metrics for current quarter
    current_metrics_df = get_company_quarter_metrics(company, quarter)
    if not current_metrics_df.empty:
        cols = [
            "metric_name",
            "metric_category",
            "metric_value",
            "metric_value_type",
            "metric_unit",
            "metric_currency",
            "metric_direction",
            "metric_is_guidance",
            "metric_period",
        ]
        current_metrics_json = current_metrics_df[cols].to_dict(orient="records")
    else:
        current_metrics_json = []

    # 3) Previous quarter metrics (if requested and available)
    prev_quarter: Optional[str] = None
    prev_metrics_json: List[Dict[str, Any]] = []
    if compare_previous:
        prev_quarter = get_previous_quarter(company, quarter)
        if prev_quarter is not None:
            prev_df = get_company_quarter_metrics(company, prev_quarter)
            if not prev_df.empty:
                cols = [
                    "metric_name",
                    "metric_category",
                    "metric_value",
                    "metric_value_type",
                    "metric_unit",
                    "metric_currency",
                    "metric_direction",
                    "metric_is_guidance",
                    "metric_period",
                ]
                prev_metrics_json = prev_df[cols].to_dict(orient="records")

    # 4) Segments & risks
    segments_df = get_company_quarter_segments(company, quarter)
    segments_json = (
        segments_df[
            [
                "segment_name",
                "segment_direction",
                "segment_is_guidance",
                "segment_certainty",
                "segment_context",
            ]
        ].to_dict(orient="records")
        if not segments_df.empty
        else []
    )

    risks_df = get_company_quarter_risks(company, quarter)
    risks_json = (
        risks_df[
            [
                "risk_type",
                "risk_sentiment",
                "risk_severity",
                "risk_certainty",
                "risk_context",
            ]
        ].to_dict(orient="records")
        if not risks_df.empty
        else []
    )

    structured = {
        "company": company,
        "quarter": quarter,
        "current_metrics": current_metrics_json,
        "previous_quarter": prev_quarter,
        "previous_metrics": prev_metrics_json,
        "segments": segments_json,
        "risks": risks_json,
    }

    structured_json_str = json.dumps(structured, indent=2)

    prompt = f"""
You are an equity research analyst writing an executive summary
for an investor who follows {company}.

You are given:
1) Structured JSON with metrics, segments, and risks for the current quarter
   and, if available, the previous quarter.
2) Text context from the earnings call.

- Use the JSON for all NUMBERS (growth rates, margins, guidance vs actual).
- Use the text context for explanations, color, and qualitative commentary.
- If previous quarter data is available, explicitly compare vs previous quarter.
- If some metrics are missing, acknowledge that briefly.

Structured data (JSON):
{structured_json_str}

Earnings call context:
{context}

Write a markdown summary with the following sections:

## Snapshot – {company} {quarter}

### Key Numbers
- Bullet points with the most important metrics (revenue, EPS, key margins, growth).

### What Changed vs Previous Quarter
- If previous quarter metrics are available, describe the main changes (growth/decline).
- If not available, say that prior-quarter data is not available.

### Segment Performance
- Call out performance of key segments (from segments JSON) and any notable trends.

### Guidance & Outlook
- Summarize forward-looking commentary and guidance (from metrics + context).

### Risks & Watchpoints
- Highlight the most important risks (from risks JSON + context).

Keep the tone concise and professional (3–6 bullets per section).
If there are contradictions between JSON and text, trust the JSON and note the discrepancy.
"""

    summary = _call_llm(prompt, temperature=temperature)
    return summary, vec_results


# --------------------------------------------------------------------
# Peer benchmarking – metrics-based
# --------------------------------------------------------------------
def _filter_metrics_for_text(df: pd.DataFrame, metric_text: str) -> pd.DataFrame:
    """
    Filter metrics rows using a fuzzy match on the user's metric text.

    We treat metrics as 'topics mentioned' rather than numeric facts.
    """
    if df.empty or not metric_text:
        return df

    metric_text = metric_text.lower()
    mask = False

    for col in ["metric_name", "metric_category", "metric_context", "metric_evidence_span"]:
        if col in df.columns:
            col_vals = df[col].fillna("").astype(str).str.lower()
            mask = mask | col_vals.str.contains(metric_text, na=False)

    filtered = df[mask].copy()
    return filtered if not filtered.empty else df  # fall back to all if nothing matched


def benchmark_peers(
    base_company: str,
    peers: List[str],
    metric: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Compare base_company vs peers based on what management said about a given topic/metric.

    Behavior:
    - For each company and quarter, pull rows from metrics.parquet (guidance/topics),
      filtered by the free-text 'metric' string if possible.
    - For each company, also retrieve top transcript snippets via hybrid_search.
    - Ask the LLM to qualitatively compare how companies talked about that topic
      (focus, confidence, tone, guidance) WITHOUT inventing exact numbers.

    Returns:
        (text_summary, table_of_metric_rows)
    """
    if quarter is None:
        return "No quarter selected.", None

    # Normalize tickers
    companies = [str(base_company).upper()] + [str(p).upper() for p in peers]
    companies = [c for c in companies if c]  # drop empties
    companies = list(dict.fromkeys(companies))  # unique, keep order

    metrics_df = _load_metrics_df()
    if metrics_df.empty:
        text_summary = (
            "metrics.parquet is empty or missing – I can't inspect what each company "
            "said about this topic yet, only generic transcript search."
        )
        return text_summary, None

    # Filter to the selected quarter + companies
    df = metrics_df[
        (metrics_df["ticker"].isin(companies)) & (metrics_df["quarter_str"] == quarter)
    ].copy()

    # If we have metric/topic rows, filter by the user's metric text
    if not df.empty:
        df_filtered = _filter_metrics_for_text(df, metric)
    else:
        df_filtered = df

    # Build a human-readable table of guidance/topics for display in UI
    if not df_filtered.empty:
        table = df_filtered[
            [
                "ticker",
                "metric_name",
                "metric_category",
                "metric_is_guidance",
                "metric_period",
                "metric_direction",
                "metric_certainty",
                "metric_value_type",
                "metric_value",
                "metric_unit",
                "metric_currency",
            ]
        ].rename(columns={"ticker": "company"})
    else:
        # No structured metric rows at all; we'll still do transcript-based compare
        table = pd.DataFrame({"company": companies})

    # For the LLM: build per-company “evidence”
    company_blobs: List[Dict[str, Any]] = []

    for c in companies:
        c_metrics = df_filtered[df_filtered["ticker"] == c].copy()

        # Turn metrics into light JSON (we don't trust metric_value to be present)
        metrics_json = []
        if not c_metrics.empty:
            metrics_json = c_metrics[
                [
                    "metric_name",
                    "metric_category",
                    "metric_is_guidance",
                    "metric_period",
                    "metric_direction",
                    "metric_certainty",
                    "metric_value_type",
                    "metric_value",
                    "metric_unit",
                    "metric_currency",
                    "metric_context",
                ]
            ].to_dict(orient="records")

        # Retrieve transcript snippets specifically about this topic
        # We reuse your hybrid_search to stay consistent with the rest of the app
        vec_results = hybrid_search(
            query=f"{metric} for this earnings call (guidance, commentary, risks)",
            company=c,
            filing_type=filing_type,
            quarter=quarter,
            top_k=4,
        )

        snippets = [r.get("text", "") for r in vec_results]

        company_blobs.append(
            {
                "company": c,
                "metrics": metrics_json,
                "snippets": snippets,
            }
        )

    structured = {
        "quarter": quarter,
        "metric_query": metric,
        "base_company": str(base_company).upper(),
        "companies": company_blobs,
    }

    structured_json_str = json.dumps(structured, indent=2)

    prompt = f"""
You are a sell-side equity research analyst.

Your task: compare how different tech companies discussed the topic/metric:

    "{metric}"

during the {quarter} earnings calls.

You are given structured evidence for each company:
- 'metrics': rows from an extracted metrics table that indicate WHAT they talked about
  (metric_name, metric_category, whether it is guidance, what period, certainty, etc.).
- 'snippets': short transcript excerpts that mention or relate to this topic.

Important:
- Many metric_value fields may be null. DO NOT invent exact numbers.
- Focus on *how* each company talked about the topic: emphasis level, confidence,
  forward-looking guidance vs historical recap, and risk tone.

Structured evidence (JSON):
{structured_json_str}

Write a concise comparison that:
- Starts with 1–2 sentences summarizing the overall picture across all companies.
- Then has a short section for each company (base company first), describing:
  - Whether they gave explicit guidance on this topic or only qualitative color.
  - How confident/hedged they sounded (use metric_certainty + snippets).
  - Any notable differences in focus vs peers (e.g., one company downplays it,
    another calls it a key growth driver, another frames it as a risk).
- Finally, add 2–3 bullets comparing the base company vs peers:
  who seems strongest/most confident, who is cautious, who gave the most detail.

Do NOT fabricate numeric values. If the data doesn't show numbers,
describe the *style and content* of the commentary instead.
"""

    text_summary = _call_llm(prompt, temperature=temperature)
    return text_summary, table
