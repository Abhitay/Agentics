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
        config={
            "temperature": temperature
        },
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
# Q&A – agentic planner + execution
# --------------------------------------------------------------------
def _plan_question_tools(question: str) -> Dict[str, Any]:
    """
    Lightweight "agent" planner for Q&A.

    Given the user's question, decide:
      - Do we need vector search (transcript snippets)?
      - Do we need structured metrics?
      - Do we need segments?
      - Do we need risks?
      - If we use vector search, what query should we send?

    Returns a dict like:
    {
        "use_vector_search": true,
        "use_metrics": true,
        "use_segments": false,
        "use_risks": true,
        "vector_query": "refined query to use"
    }
    """
    planner_prompt = f"""
You are planning how to answer this question about an earnings call:

Question: "{question}"

You have four internal tools:

1) vector_search
   - Returns small chunks of the call transcript that are semantically
     related to a query string.
   - Best for: management commentary, qualitative color, explanations,
     and anything the speakers said in words.

2) metrics_lookup
   - Returns structured numerical facts (revenue, profit, growth, margins, etc.)
     extracted from the call.
   - Best for: "what is the revenue?", "what was EPS?", "how fast did X grow?",
     and other numeric questions.

3) segments_lookup
   - Returns structured info about how different parts of the business performed
     (segments/products/regions) and whether they are up, down, or flat.
   - Best for: "which segment grew the most?", "how did services do?", etc.

4) risks_lookup
   - Returns structured info about risks the company mentioned: type of risk,
     severity, sentiment, and context.
   - Best for: "what risks did they highlight?", "are they worried about macro?", etc.

Your task:
Decide WHICH of these tools are actually needed to answer the user's question,
and, if you plan to use vector_search, write a refined search query.

Return ONLY valid JSON, no extra text, with this exact shape:

{{
  "use_vector_search": true or false,
  "use_metrics": true or false,
  "use_segments": true or false,
  "use_risks": true or false,
  "vector_query": "a short search query for vector_search"
}}

Guidelines:
- If the user asks for a specific NUMBER (like revenue, EPS, margin, guidance),
  you almost always want metrics_lookup (and optionally vector_search for context).
- If the user asks about "what management said" or "tone" or "why something happened",
  you definitely want vector_search.
- If the user asks about segments (iPhone, cloud, services, regions, etc.),
  you probably want segments_lookup (+ optionally vector_search).
- If the user asks about risks, macro, uncertainty, or headwinds,
  you probably want risks_lookup (+ optionally vector_search).

If you are unsure, it's OK to set a tool to true. Just keep the JSON valid.
"""

    raw = _call_llm(planner_prompt, temperature=0.1)

    # Sensible defaults if parsing fails
    default_plan = {
        "use_vector_search": True,
        "use_metrics": True,
        "use_segments": True,
        "use_risks": True,
        "vector_query": question,
    }

    try:
        plan = json.loads(raw)
        for k, v in default_plan.items():
            plan.setdefault(k, v)
        if not isinstance(plan.get("vector_query"), str) or not plan["vector_query"].strip():
            plan["vector_query"] = question
        return plan
    except Exception:
        return default_plan


def answer_question(
    question: str,
    company: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
    use_planner: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main Q&A entrypoint with optional "agentic" planner.

    Steps:
    1) Optionally ask a small planner LLM which sources we need (fast mode skips this).
    2) Call only those data sources.
    3) Feed the combined evidence into a final answer LLM call.

    Returns:
        answer_text, vector_sources (for UI evidence display)
    """
    if quarter is None:
        return "No quarter selected.", []

    # 1) PLAN
    if use_planner:
        plan = _plan_question_tools(question)
        use_vec = bool(plan.get("use_vector_search", True))
        use_metrics = bool(plan.get("use_metrics", True))
        use_segments = bool(plan.get("use_segments", True))
        use_risks = bool(plan.get("use_risks", True))
        vec_query = plan.get("vector_query") or question
    else:
        # FAST MODE: skip planner LLM call, just use everything
        plan = {
            "use_vector_search": True,
            "use_metrics": True,
            "use_segments": True,
            "use_risks": True,
            "vector_query": question,
        }
        use_vec = use_metrics = use_segments = use_risks = True
        vec_query = question

    # 2) ACT – vector search
    vec_results: List[Dict[str, Any]] = []
    if use_vec:
        vec_results = hybrid_search(
            query=vec_query,
            company=company,
            filing_type=filing_type,
            quarter=quarter,
            top_k=4,  # reduced from 8 for speed
        )

    if vec_results:
        transcript_context = "\n\n---\n\n".join([r.get("text", "") for r in vec_results])
    elif use_vec:
        transcript_context = "NO RELEVANT TRANSCRIPT SNIPPETS FOUND FOR THIS QUESTION."
    else:
        transcript_context = "VECTOR SEARCH WAS NOT USED FOR THIS QUESTION."

    # 2b) metrics
    metrics_json: List[Dict[str, Any]] = []
    if use_metrics:
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
            metrics_json = current_metrics_df[metric_cols].to_dict(orient="records")

    # 2c) segments
    segments_json: List[Dict[str, Any]] = []
    if use_segments:
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

    # 2d) risks
    risks_json: List[Dict[str, Any]] = []
    if use_risks:
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

    # 3) OBSERVE – bundle evidence
    evidence = {
        "company": company,
        "quarter": quarter,
        "plan": plan,
        "metrics": metrics_json,
        "segments": segments_json,
        "risks": risks_json,
    }
    evidence_json_str = json.dumps(evidence, indent=2)

    # 4) ANSWER
    final_prompt = f"""
You are a helpful research assistant answering a question
about {company} in {quarter}.

Your audience is a smart person with little or no finance background.
Use simple language and avoid heavy jargon. If you must use a finance
term (like "margin" or "guidance"), briefly explain it.

I am giving you:
1) A small JSON blob that describes which data sources we used for this
   question and the structured information retrieved from them.
2) Transcript snippets from the earnings call (if vector search was used).

The planner JSON (what we decided to use + structured data):

{evidence_json_str}

Transcript context (if used):
{transcript_context}

User question:
{question}

Instructions:
- If the user asks for a specific NUMBER (revenue, EPS, growth, etc.),
  first look in the metrics section of the JSON. Use those values if present.
- Use the transcript snippets to explain *why* things happened or what
  management said in plain English.
- If a requested number is NOT in the metrics, do NOT make one up.
  Instead, say that the exact figure isn't available in your structured data,
  and answer qualitatively if the context allows.
- If a particular data source was not used (for example, segments or risks),
  don't mention it at all in the answer.

Output:
- A short, clear answer (2–6 sentences).
- Plain English, as if you are explaining it to a friend who owns the stock
  but is not a finance professional.
"""

    answer = _call_llm(final_prompt, temperature=temperature)
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


def _generate_tldr(
    structured_json_str: str,
    context: str,
    company: str,
    quarter: str,
) -> str:
    """
    Generate a one-sentence TL;DR in plain English.
    """
    prompt = f"""
You are summarizing an earnings call for {company} in {quarter}.

Using the structured data and context below, write ONE sentence
(max 35 words) that explains in plain English how the quarter went
overall for a regular investor (not a finance expert).

Avoid jargon. Do not include headings or bullet points.

Structured data:
{structured_json_str}

Earnings call context:
{context}
"""
    tldr = _call_llm(prompt, temperature=0.2).strip()
    # Make sure it's a single line
    return " ".join(tldr.split())


# --------------------------------------------------------------------
# Summary – agentic planner + execution + analytics + TL;DR
# --------------------------------------------------------------------
def _plan_summary_tools(
    company: str,
    quarter: str,
    compare_previous: bool,
) -> Dict[str, Any]:
    """
    Planner for the summary generation.

    Decide:
      - use_vector_search
      - use_metrics
      - use_segments
      - use_risks
      - use_previous_quarter
      - vector_query (what to search for in transcripts)
    """
    planner_prompt = f"""
You are planning how to build a short, plain-English summary of the
{quarter} earnings call for {company}.

You have these internal tools:

1) vector_search
   - Returns chunks of the call transcript for a query string.
   - Good for: management commentary, explanations, tone, guidance, risks.

2) metrics_lookup
   - Returns structured numerical facts: revenue, profit, growth, margins, etc.
   - Good for: "key numbers" and "what changed vs last quarter".

3) segments_lookup
   - Returns structured info about how different parts of the business did
     (segments/products/regions) and whether they are up, down, or flat.

4) risks_lookup
   - Returns structured info about risks: what type, how severe, what context.

5) previous_quarter_lookup
   - Allows comparing the current quarter vs the immediately previous one.

Your task:
Decide which of these tools are needed for a simple, investor-friendly
summary with sections like Key Numbers, What Changed, Segment Performance,
Guidance & Outlook, and Risks & Watchpoints.

Return ONLY valid JSON, no extra text, with this exact shape:

{{
  "use_vector_search": true or false,
  "use_metrics": true or false,
  "use_segments": true or false,
  "use_risks": true or false,
  "use_previous_quarter": true or false,
  "vector_query": "a short query for vector_search"
}}

Guidelines:
- You almost always want metrics_lookup for a summary (to get key numbers).
- You almost always want vector_search to get management commentary and tone.
- Use segments_lookup when segment-level color would add value.
- Use risks_lookup when the call likely has meaningful risk discussion.
- use_previous_quarter can be false if compare_previous is false or not needed.

The caller has requested compare_previous={str(compare_previous)}.
This is a strong hint for whether previous_quarter_lookup should be used.
"""

    raw = _call_llm(planner_prompt, temperature=0.1)

    default_plan = {
        "use_vector_search": True,
        "use_metrics": True,
        "use_segments": True,
        "use_risks": True,
        "use_previous_quarter": compare_previous,
        "vector_query": "overall performance, key numbers, what changed vs last quarter, segment performance, guidance, and key risks",
    }

    try:
        plan = json.loads(raw)
        for k, v in default_plan.items():
            plan.setdefault(k, v)
        if not isinstance(plan.get("vector_query"), str) or not plan["vector_query"].strip():
            plan["vector_query"] = default_plan["vector_query"]
        return plan
    except Exception:
        return default_plan


def generate_summary(
    company: str,
    filing_type: str,
    quarter: Optional[str],
    temperature: float = 0.2,
    compare_previous: bool = True,
    compare_with: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Generate an executive summary of the company's earnings call.

    Now agentic:
    - Planner decides which sources to use and how to query vector search.
    - Execution pulls only those sources.
    - We then compute analytics and TL;DR, and write a plain-English summary.

    compare_with:
        If provided, this specific quarter is used as the comparison baseline
        instead of automatically picking the immediately previous quarter.

    Returns:
        summary_markdown, vector_sources, analytics_dict, tldr_string
    """
    if quarter is None:
        return "No quarter selected.", [], {}, ""

    # 1) PLAN
    plan = _plan_summary_tools(
        company,
        quarter,
        compare_previous or bool(compare_with),
    )
    use_vec = bool(plan.get("use_vector_search", True))
    use_metrics = bool(plan.get("use_metrics", True))
    use_segments = bool(plan.get("use_segments", True))
    use_risks = bool(plan.get("use_risks", True))
    use_prev_q = bool(plan.get("use_previous_quarter", compare_previous))
    vec_query = plan.get("vector_query") or "overall performance and guidance"

    # 2) ACT – vector context
    vec_results: List[Dict[str, Any]] = []
    if use_vec:
        vec_results = hybrid_search(
            query=vec_query,
            company=company,
            filing_type=filing_type,
            quarter=quarter,
            top_k=5,  # reduced from 10
        )

    context = (
        "\n\n---\n\n".join([r["text"] for r in vec_results])
        if vec_results
        else (
            "NO TRANSCRIPT CONTEXT AVAILABLE"
            if use_vec
            else "VECTOR SEARCH WAS NOT USED FOR THIS SUMMARY."
        )
    )

    # 2b) current metrics
    if use_metrics:
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
    else:
        current_metrics_json = []

    # 2c) comparison quarter metrics (specific or previous)
    prev_quarter: Optional[str] = None
    prev_metrics_json: List[Dict[str, Any]] = []

    # If user picked a specific comparison quarter, that wins
    if compare_with:
        prev_quarter = compare_with
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

    # Otherwise fall back to automatically using the previous quarter
    elif compare_previous and use_prev_q:
        prev_quarter = get_previous_quarter(company, quarter)
        if prev_quarter:
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

    # 2d) segments
    if use_segments:
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
    else:
        segments_json = []

    # 2e) risks
    if use_risks:
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
    else:
        risks_json = []

    # 3) Bundle structured evidence
    structured = {
        "company": company,
        "quarter": quarter,
        "plan": plan,
        "current_metrics": current_metrics_json,
        "previous_quarter": prev_quarter,
        "previous_metrics": prev_metrics_json,
        "segments": segments_json,
        "risks": risks_json,
    }

    structured_json_str = json.dumps(structured, indent=2)

    # 4) Analytics & TL;DR
    analytics = _analyze_call_analytics(structured_json_str, context)
    tldr = _generate_tldr(structured_json_str, context, company, quarter)

    # 5) Final summary in plain English
    summary_prompt = f"""
You are an equity research analyst writing a short summary
for a regular investor who follows {company}.

The reader is smart but NOT a finance expert. Use simple language,
avoid jargon, and keep each bullet easy to understand. If you use a
finance term (like "margin" or "guidance"), briefly explain it.

Do NOT use markdown italics anywhere (no *word*). If you want to
highlight something, you may use bold only for short labels at the
start of a bullet, like **Total revenue:**.

You are given:
1) A JSON blob with metrics, segments, risks, and information about which
   tools were used to build this summary.
2) Text context from the earnings call (or a note if no transcript was used).

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
- If previous quarter data is missing or not used, say that briefly.

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
Mention if there are important gaps in the data (for example, if
you don't have previous-quarter metrics or certain numbers).
"""

    summary = _call_llm(summary_prompt, temperature=temperature)
    return summary, vec_results, analytics, tldr


# --------------------------------------------------------------------
# Peer benchmarking – overall, simple language (unchanged logic)
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
            query="overall business performance, growth, profitability and key risks for this earnings call",
            company=c,
            filing_type=filing_type,
            quarter=quarter,
            top_k=3,  # reduced from 6
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
