import os
import time
import uuid
from typing import List, Literal, Tuple

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
# -------------------------------------------------
# 1. Gemini client + model config
# -------------------------------------------------

# Uses GEMINI_API_KEY from your environment
client = genai.Client()

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0


# -------------------------------------------------
# 2. Pydantic schemas (same as notebook)
# -------------------------------------------------

MetricDirection = Literal["up", "down", "flat", "mixed", "unknown"]
MetricValueType = Literal["level", "change_abs", "change_pct", "ratio", "other"]
MetricCertainty = Literal["explicit", "implicit", "uncertain"]

RiskSentiment = Literal["negative", "neutral", "positive", "mixed", "unknown"]
RiskSeverity = Literal["low", "medium", "high", "unknown"]

SegmentDirection = Literal["up", "down", "flat", "mixed", "unknown"]
SegmentCertainty = Literal["explicit", "implicit", "uncertain"]

OverallSentiment = Literal["positive", "negative", "neutral", "mixed", "unknown"]


class MetricItem(BaseModel):
    name: str = Field(
        ...,
        description="Short metric name, e.g. 'revenue', 'EPS', 'operating margin', 'cloud ARR'.",
    )
    category: str | None = Field(
        default=None,
        description="Category of metric, e.g. 'financial', 'operational', 'user', 'margin', 'cashflow', 'other'.",
    )
    value: float | None = Field(
        default=None,
        description="Numeric value if explicitly mentioned; null if no clear numeric value.",
    )
    value_type: MetricValueType = Field(
        default="level",
        description="Type of value: 'level' (absolute value), 'change_abs', 'change_pct', 'ratio', or 'other'.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of the value, e.g. 'million', 'billion', 'percent', 'bps', 'users', or null.",
    )
    currency: str | None = Field(
        default=None,
        description="Currency code if relevant, e.g. 'USD', 'EUR', or null.",
    )
    direction: MetricDirection = Field(
        default="unknown",
        description="Direction of change compared to prior period: 'up', 'down', 'flat', 'mixed', or 'unknown'.",
    )
    is_guidance: bool | None = Field(
        default=None,
        description="true if clearly forward-looking guidance, false if clearly realized/historical, null if unclear.",
    )
    period: str | None = Field(
        default=None,
        description="Time period for this metric, e.g. 'Q3 2024', 'FY24', 'next quarter', 'next fiscal year', or null.",
    )
    certainty: MetricCertainty = Field(
        default="explicit",
        description="Was this metric explicitly stated ('explicit'), inferred from language ('implicit'), or uncertain?",
    )
    evidence_span: str | None = Field(
        default=None,
        description="Short snippet or phrase from the statement that supports this metric mention.",
    )
    context: str | None = Field(
        default=None,
        description="Short context for the metric, e.g. 'North America cloud segment', 'organic growth', etc.",
    )


class RiskItem(BaseModel):
    type: str = Field(
        ...,
        description="Short risk label, e.g. 'FX', 'macro', 'regulation', 'competition', 'supply chain', 'execution'.",
    )
    sentiment: RiskSentiment = Field(
        default="unknown",
        description="Sentiment associated with this risk: 'negative', 'neutral', 'positive', 'mixed', or 'unknown'.",
    )
    severity: RiskSeverity = Field(
        default="unknown",
        description="Estimated severity of the risk if mentioned: 'low', 'medium', 'high', or 'unknown'.",
    )
    certainty: SegmentCertainty = Field(
        default="explicit",
        description="Was this risk explicitly mentioned, implicitly implied, or uncertain?",
    )
    evidence_span: str | None = Field(
        default=None,
        description="Short snippet or phrase from the statement that describes this risk.",
    )
    context: str | None = Field(
        default=None,
        description="Short context for the risk, e.g. 'Europe', 'consumer hardware', 'China regulation'.",
    )


class SegmentItem(BaseModel):
    name: str = Field(
        ...,
        description="Name of the business segment, product, or region, e.g. 'cloud', 'PC', 'APAC', 'gaming'.",
    )
    direction: SegmentDirection = Field(
        default="unknown",
        description="Direction of performance for this segment: 'up', 'down', 'flat', 'mixed', or 'unknown'.",
    )
    is_guidance: bool | None = Field(
        default=None,
        description="true if the statement is about future performance for this segment, false if historical, null if unclear.",
    )
    certainty: SegmentCertainty = Field(
        default="explicit",
        description="Was this segment performance explicitly described, implicitly implied, or uncertain?",
    )
    evidence_span: str | None = Field(
        default=None,
        description="Short snippet or phrase from the statement that mentions this segment.",
    )
    context: str | None = Field(
        default=None,
        description="Short context or qualifier, e.g. 'enterprise customers', 'US market', 'SMB'.",
    )


class ExtractionResponse(BaseModel):
    metrics: List[MetricItem] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    segments: List[SegmentItem] = Field(default_factory=list)

    overall_sentiment: OverallSentiment = Field(
        default="unknown",
        description="Overall sentiment of the statement about the company's performance.",
    )


RESPONSE_SCHEMA = ExtractionResponse.model_json_schema()

SYSTEM_PROMPT = """
You are an expert financial information extraction model.
You receive a single earnings call *statement* (not the entire transcript).
Extract only the most relevant (top) few items.

Hard constraints:
- metrics: MAX 5
- risks: MAX 3
- segments: MAX 5

Never return more than these limits.
Never invent text.
Only use the exact JSON schema.
Return ONLY JSON.
"""


# -------------------------------------------------
# 3. Gemini call + prompt builder
# -------------------------------------------------


def call_gemini(prompt: str, max_retries: int = 8, base_backoff: float = 5.0) -> str:
    """
    Call Gemini with robust quota + overload handling:
    - Detect 429 RESOURCE_EXHAUSTED → sleep for RetryInfo.retryDelay (if exists)
    - Detect 503 overload → exponential backoff
    - Fails only after N hard retries
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    max_output_tokens=2048
                ),
            )
            return response.text

        except ServerError as e:
            # -----------------------------------------
            # Extract status & retryDelay if provided
            # -----------------------------------------
            err_code = None
            retry_seconds = base_backoff

            # Gemini error payload lives in e.args
            if e.args:
                err = e.args[0]
                if isinstance(err, dict) and "error" in err:
                    err = err["error"]
                    err_code = err.get("code")

                    # Check retry info from Google RPC
                    for d in err.get("details", []):
                        if "@type" in d and "RetryInfo" in d["@type"]:
                            # e.g. "22s" or "22.793s"
                            raw_delay = d.get("retryDelay", "").rstrip("s")
                            try:
                                retry_seconds = float(raw_delay)
                            except:
                                pass

                elif isinstance(err, str):
                    if "429" in err:
                        err_code = 429
                    if "503" in err:
                        err_code = 503

            # -----------------------------------------
            # Special handling
            # -----------------------------------------
            # 1) QUOTA LIMIT
            if err_code == 429:
                attempt += 1
                print(f"[429] Rate limit — sleeping {retry_seconds:.2f}s...")
                time.sleep(retry_seconds)
                continue

            # 2) SERVER OVERLOAD
            if err_code == 503:
                attempt += 1
                backoff = base_backoff * attempt
                print(f"[503] Server busy — backoff {backoff:.2f}s (attempt {attempt})")
                time.sleep(backoff)
                continue

            # -----------------------------------------
            # 3) Other server exceptions — rethrow
            # -----------------------------------------
            raise
        except Exception as e:
            # NEW: catch unexpected API issues (auth/quota/project/etc.)
            print(f"[FATAL GEMINI ERROR] {repr(e)}")
            raise

    raise RuntimeError(f"Exceeded retry attempts for extraction after {max_retries} tries.")




def build_prompt(row: pd.Series) -> str:
    return f"""
You will extract structured financial knowledge from the following earnings call statement.

Rules:
- Only extract metrics, risks, and segments that are clearly supported by the text.
- If no numeric value is explicitly mentioned, set 'value' to null but still fill 'name', 'direction', etc. if meaningful.
- Use the enums exactly as defined in the schema. Do NOT invent new values.
- For each item, include a short 'evidence_span' copied from the statement.
- If you are not sure, prefer 'unknown' for enums and null for free-text fields.

Statement metadata:
Company: {row['company_name']}
Company ID: {row['companyid']}
Call Period: {row['call_period']}
Speaker: {row['transcriptpersonname']} ({row['speakertypename']})
Segment Type: {row['transcriptcomponenttypename']}

Statement text:
\"\"\"{row['clean_text']}\"\"\"
"""


# -------------------------------------------------
# 4. Load WRDS corpus
# -------------------------------------------------


def load_corpus(path: str = "data/corpus/tech_call_sections.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["statement_id"] = df["segment_id"].astype(str)
    if "clean_text" not in df.columns:
        df["clean_text"] = df["componenttext"].fillna("").astype(str)
    return df


# -------------------------------------------------
# 5. Extract one (company_id, year, quarter) slice
# -------------------------------------------------


def extract_slice(
    call_section: pd.DataFrame,
    company_id: float,
    year: int,
    quarter: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = call_section[
        (call_section["companyid"] == company_id)
        & (call_section["call_year"] == year)
        & (call_section["call_quarter"] == quarter)
    ].copy()

    print(
        f"Running extraction for company_id={company_id}, "
        f"year={year}, quarter={quarter} -> {len(subset)} rows"
    )

    metric_rows: List[dict] = []
    risk_rows: List[dict] = []
    segment_rows: List[dict] = []
    statement_rows: List[dict] = []

    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        statement_id = row["statement_id"]
        cid = row["companyid"]
        call_id = row["transcriptid"]

        try:
            prompt = build_prompt(row)
            response_text = call_gemini(prompt)

            # Try to parse strictly; if it fails, log + fall back to empty ExtractionResponse
            try:
                data = ExtractionResponse.model_validate_json(response_text)
            except ValidationError as ve:
                print(
                    f"[VALERROR] Statement {statement_id} on Call {call_id} for Company {cid}: {ve}"
                )
                print(f"[RAW RESPONSE] {response_text[:800]}")
                data = ExtractionResponse()  # no metrics/risks/segments, but keep the statement

            # ✅ Always at least keep the statement row
            statement_rows.append(
                {
                    "statement_id": statement_id,
                    "segment_id": row["segment_id"],
                    "segment_idx": row["segment_idx"],
                    "company_id": cid,
                    "company_name": row["company_name"],
                    "call_id": call_id,
                    "call_date": row["mostimportantdateutc"],
                    "call_period": row["call_period"],
                    "call_year": row["call_year"],
                    "call_quarter": row["call_quarter"],
                    "speaker_name": row["transcriptpersonname"],
                    "speaker_role": row["speakertypename"],
                    "segment_type": row["transcriptcomponenttypename"],
                    "text": row["clean_text"],
                    "overall_sentiment": data.overall_sentiment,
                }
            )

            # Metrics
            for m in data.metrics:
                metric_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "metric_name": m.name,
                        "metric_category": m.category,
                        "metric_value": m.value,
                        "metric_value_type": m.value_type,
                        "metric_unit": m.unit,
                        "metric_currency": m.currency,
                        "metric_direction": m.direction,
                        "metric_is_guidance": m.is_guidance,
                        "metric_period": m.period,
                        "metric_certainty": m.certainty,
                        "metric_evidence_span": m.evidence_span,
                        "metric_context": m.context,
                    }
                )

            # Risks
            for r in data.risks:
                risk_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "risk_type": r.type,
                        "risk_sentiment": r.sentiment,
                        "risk_severity": r.severity,
                        "risk_certainty": r.certainty,
                        "risk_evidence_span": r.evidence_span,
                        "risk_context": r.context,
                    }
                )

            # Segments
            for s in data.segments:
                segment_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "segment_name": s.name,
                        "segment_direction": s.direction,
                        "segment_is_guidance": s.is_guidance,
                        "segment_certainty": s.certainty,
                        "segment_evidence_span": s.evidence_span,
                        "segment_context": s.context,
                    }
                )

            time.sleep(4)

        except Exception as e:
            print(
                f"[ERROR] Statement {statement_id} on Call {call_id} for Company {cid}: {repr(e)}"
            )
            continue

    statements_df = pd.DataFrame(statement_rows)
    metrics_df = pd.DataFrame(metric_rows)
    risks_df = pd.DataFrame(risk_rows)
    segments_df = pd.DataFrame(segment_rows)

    return statements_df, metrics_df, risks_df, segments_df


# -------------------------------------------------
# 6. Append to data/processed/*.parquet
# -------------------------------------------------


def append_to_processed(
    statements: pd.DataFrame,
    metrics: pd.DataFrame,
    risks: pd.DataFrame,
    segments: pd.DataFrame,
    base_dir: str = "data/processed",
) -> None:
    os.makedirs(base_dir, exist_ok=True)

    def _append(filename: str, new_df: pd.DataFrame):
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            combined = pd.concat([old, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(path)

    if not statements.empty:
        _append("statements.parquet", statements)
    if not metrics.empty:
        _append("metrics.parquet", metrics)
    if not risks.empty:
        _append("risks.parquet", risks)
    if not segments.empty:
        _append("segments.parquet", segments)


# -------------------------------------------------
# 7. main(): currently hard-coded to AAPL 2025 Q4
# -------------------------------------------------


def main():
    call_section = load_corpus()

    # 🔴 Right now: AAPL 2025 Q4 (same as your notebook)
    # You can change these later or make them CLI args.
    company_id = 24937.0
    year = 2025
    quarter = "Q3"

    stmts, metrics, risks, segments = extract_slice(
        call_section,
        company_id=company_id,
        year=year,
        quarter=quarter,
    )

    append_to_processed(stmts, metrics, risks, segments)

    print("Extraction complete. Updated data/processed/*.parquet")


if __name__ == "__main__":
    main()
