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
# Metrics / segments / risks helpers (exported for UI)
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
# Q&A – vector + structured metrics
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

    Uses BOTH:
    - Vector search (transcript snippets) for qualitative/contextual answers.
    - Structured metrics/segments/risks for numeric facts (revenue, EPS, margins, etc.).

    If a requested number is in the structured data, the model should use that.
    If it's not available anywhere, it should explicitly say so.
    """
    if quarter is None:
        return "No quarter selected.", []

    # 1) Retrieve transcript context from vector store
    vec_results = hybrid_search(
        query=question,
        company=company,
        filing_type=filing_type,
        quarter=quarter,
        top_k=8,
    )

    if vec_results:
        context = "\n\n---\n\n".join([r.get("text", "") for r in vec_results])
    else:
        context = "NO TRANSCRIPT CONTEXT AVAILABLE"

    # 2) Structured data for this company/quarter
    current_metrics_df = get_company_quarter_metrics(company, quarter)
    if not current_metrics_df.empty:
        metric_cols = [
            "metric_name",
            "metric_category",
            "metric_value",
            "metric_value_type",
            "metric_unit",
            "metric_currency",
            "metric_direction",
            "metric_is_guidance",
            "metric_period",
            "metric_certainty",
            "metric_context",
        ]
        metric_cols = [c for c in metric_cols if c in current_metrics_df.columns]
        current_metrics_json = current_metrics_df[metric_cols].to_dict(orient="records")
    else:
        current_metrics_json = []

    segments_df = get_company_quarter_segments(company, quarter)
    if not segments_df.empty:
        seg_cols = [
            "segment_name",
            "segment_direction",
            "segment_is_guidance",
            "segment_certainty",
            "segment_context",
        ]
        seg_cols = [c for c in seg_cols if c in segments_df.columns]
        segments_json = segments_df[seg_cols].to_dict(orient="records")
    else:
        segments_json = []

    risks_df = get_company_quarter_risks(company, quarter)
    if not risks_df.empty:
        risk_cols = [
            "risk_type",
            "risk_sentiment",
            "risk_severity",
            "risk_certainty",
            "risk_context",
        ]
        risk_cols = [c for c in risk_cols if c in risks_df.columns]
        risks_json = risks_df[risk_cols].to_dict(orient="records")
    else:
        risks_json = []

    structured = {
        "company": company,
        "quarter": quarter,
        "metrics": current_metrics_json,
        "segments": segments_json,
        "risks": risks_json,
    }
    structured_json_str = json.dumps(structured, indent=2)

    prompt = f"""
You are a helpful equity research assistant answering a question about
{company} in {quarter}.

Your audience is a smart person with little or no finance background.
Explain things in simple language. Avoid heavy jargon. If you must use
a finance term (like "margin" or "guidance"), briefly explain it in
plain English.

You have TWO sources of evidence:

1) Structured JSON with extracted metrics, segments and risks.
   - Use this for all NUMERIC facts (e.g., revenue, profit, growth).
   - If the user asks for a specific number (like "What is the revenue?"),
     look for a relevant metric_name/metric_category first.
   - If multiple related metrics exist (e.g. actual vs guidance), explain clearly.

2) Transcript context from the earnings call.
   - Use this for qualitative color, commentary, drivers, and explanations.
   - If something is not clearly supported by either structured data or transcripts,
     say you don't have that information.

If you cannot find the requested numeric value in the structured data,
DO NOT invent a number. Instead, say that the exact figure is not available
in your extracted metrics, and you can only speak qualitatively if the
transcript provides hints.

Structured data (JSON):
{structured_json_str}

Transcript context:
{context}

User question:
{question}

Now provide a short, clear answer (2–6 sentences). Use plain English,
and imagine you are explaining it to a friend who owns the stock but
is not a finance professional.
"""

    answer = _call_llm(prompt, temperature=temperature)
    return answer, vec_results


# --------------------------------------------------------------------
# Analytics helper for summary (sentiment, guidance, risk, focus)
# --------------------------------------------------------------------
def _analyze_call_analytics(
    structured_json_str: str,
    context: str,
) -> Dict[str, Any]:
    """
    Ask the LLM to produce high-level analytics for the summary tab:
    - sentiment (positive/neutral/negative + label)
    - guidance stance
    - risk level and top risks
    - focus segments

    Returns a dict with sensible defaults if parsing fails.
    """
    analytics_prompt = f"""
You are an equity research assistant.

You are given structured data and earnings call context for a single
quarter of a public company. Based on this, produce a JSON object that
summarizes high-level "soft metrics" for a simple investor UI.

The user is NOT a finance expert, so when you choose labels, prefer
plain-language descriptions like "cautiously positive".

Return ONLY valid JSON, no extra commentary, with this exact shape:

{{
  "sentiment": {{
    "positive": float,   // between 0 and 1
    "neutral": float,    // between 0 and 1
    "negative": float,   // between 0 and 1
    "label": string      // e.g. "Cautiously positive"
  }},
  "guidance": {{
    "label": string      // one of ["Raised", "In-line", "Lowered", "None"]
  }},
  "risk": {{
    "level": string,     // one of ["Low", "Medium", "High"]
    "top_risks": [string, ...]  // up to 3 short risk themes
  }},
  "focus_segments": [string, ...]  // up to 3 key segments/products
}}

If you are uncertain, choose the closest label and keep the JSON valid.

Structured data:
{structured_json_str}

Earnings call context:
{context}
"""

    raw = _call_llm(analytics_prompt, temperature=0.1)

    default_analytics: Dict[str, Any] = {
        "sentiment": {
            "positive": 0.33,
            "neutral": 0.34,
            "negative": 0.33,
            "label": "Mixed",
        },
        "guidance": {"label": "None"},
        "risk": {"level": "Medium", "top_risks": []},
        "focus_segments": [],
    }

    try:
        parsed = json.loads(raw)
        # Basic sanity for sentiment numbers
        sentiment = parsed.get("sentiment", {})
        if isinstance(sentiment, dict):
            for key in ["positive", "neutral", "negative"]:
                if key in sentiment:
                    try:
                        sentiment[key] = float(sentiment[key])
                    except Exception:
                        sentiment[key] = default_analytics["sentiment"][key]
        default_analytics.update(parsed)
        return default_analytics
    except Exception:
        return default_analytics


# --------------------------------------------------------------------
# Executive summary – structured + metrics-aware + analytics
# --------------------------------------------------------------------
def generate_summary(
    company: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
    compare_previous: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate an executive summary of the company's earnings call.

    Uses:
    - Vector search over call statements for narrative context.
    - Structured metrics for this quarter (+ previous quarter if available).
    - Segments and risks tables for extra color.
    - LLM-based analytics (sentiment, guidance stance, risk, focus segments).
    """
    if quarter is None:
        return "No quarter selected.", [], {}

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
        cols = [c for c in cols if c in current_metrics_df.columns]
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
                cols = [c for c in cols if c in prev_df.columns]
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

    # 5) Analytics for UI (sentiment, guidance, risk, focus)
    analytics = _analyze_call_analytics(structured_json_str, context)

    summary_prompt = f"""
You are an equity research analyst writing a short summary
for a regular investor who follows {company}.

The reader is smart but NOT a finance expert. Use simple language,
avoid jargon, and keep each bullet easy to understand. If you use a
finance term (like "margin" or "guidance"), briefly explain it.

You are given:
1) Structured JSON with metrics, segments, and risks for the current quarter
   and, if available, the previous quarter.
2) Text context from the earnings call.

- Use the JSON for all NUMBERS (revenue, profit, growth, etc.).
- Use the text context for explanations, color, and simple storytelling.
- If previous quarter data is available, clearly say whether things are
  better, worse, or roughly the same.
- If some metrics are missing, say that the data isn't available.

Structured data (JSON):
{structured_json_str}

Earnings call context:
{context}

Write a markdown summary with the following sections:

## Snapshot – {company} {quarter}

### Key Numbers
- 3–5 bullets with the most important numbers (explain what each number means).

### What Changed vs Previous Quarter
- 2–4 bullets describing whether things got better, worse, or stayed similar.

### Segment Performance
- 2–4 bullets on which parts of the business did well or struggled
  (e.g., iPhone, services, cloud, ads).

### Guidance & Outlook
- 2–4 bullets on what management expects for the next few quarters
  (use plain language like "expecting solid growth" rather than jargon).

### Risks & Watchpoints
- 2–4 bullets on the main risks or concerns an everyday investor should watch.

Keep the whole summary fairly short (around 250–400 words).
Use simple, clear sentences and avoid long paragraphs.
"""

    summary = _call_llm(summary_prompt, temperature=temperature)
    return summary, vec_results, analytics


# --------------------------------------------------------------------
# Peer benchmarking – overall, simple language
# --------------------------------------------------------------------
def benchmark_peers(
    base_company: str,
    peers: List[str],
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Compare base_company vs peers based on overall earnings commentary and metrics.

    Behavior:
    - For each company and quarter, pull rows from metrics.parquet (all topics).
    - For each company, also retrieve top transcript snippets via hybrid_search.
    - Ask the LLM to qualitatively compare overall tone, guidance, growth, and risks
      in simple language for a non-expert.
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
            "said in a structured way yet, only generic transcript search."
        )
        return text_summary, None

    # Filter to the selected quarter + companies
    df = metrics_df[
        (metrics_df["ticker"].isin(companies)) & (metrics_df["quarter_str"] == quarter)
    ].copy()

    # Build a table for possible debugging (UI does not show it)
    if not df.empty:
        table = df[
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
        table = pd.DataFrame({"company": companies})

    company_blobs: List[Dict[str, Any]] = []

    for c in companies:
        c_metrics = df[df["ticker"] == c].copy()

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

        vec_results = hybrid_search(
            query="overall business performance, growth, margins and key risks for this earnings call",
            company=c,
            filing_type=filing_type,
            quarter=quarter,
            top_k=6,
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
        "base_company": str(base_company).upper(),
        "companies": company_blobs,
    }

    structured_json_str = json.dumps(structured, indent=2)

    prompt = f"""
You are a sell-side equity research analyst explaining things to a
curious everyday investor.

Your task: compare how {base_company} and its peers performed and what
they emphasized in their {quarter} earnings calls.

Avoid heavy finance jargon. Use simple language like:
- "growing faster/slower than peers"
- "more optimistic/less optimistic"
- "talked a lot about risks / did not talk much about risks"

You are given structured evidence for each company:
- 'metrics': rows from an extracted metrics table that indicate WHAT they talked about.
- 'snippets': short transcript excerpts that summarize performance, growth,
  profitability, and key risks.

Structured evidence (JSON):
{structured_json_str}

Write a short comparison that:
- Starts with 2–3 sentences summarizing the overall picture across all companies.
- Then has a short section for each company (base company first), describing in
  plain English:
  - Whether their business seems to be doing better, worse, or similar to peers.
  - What they are most excited about (e.g., a product line or region).
  - How worried or relaxed they sound about risks.

- Finish with 3–5 bullets clearly comparing the base company vs peers:
  - who seems in the strongest position right now,
  - who sounds most optimistic,
  - who mentioned the biggest risks.

Do NOT invent exact numbers. Focus on direction and tone only.
"""

    text_summary = _call_llm(prompt, temperature=temperature)
    return text_summary, table
