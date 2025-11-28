import os
import sys
import time
import uuid
import ast
from pathlib import Path
from typing import List, Literal, Set

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

# -------------------------------------------------
# 0. Setup paths + env
# -------------------------------------------------

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

CORPUS_PATH = ROOT_DIR / "data" / "corpus" / "tech_call_sections.parquet"
UNIVERSE_TOP20_PATH = ROOT_DIR / "data" / "universe" / "tech_universe_top20_with_ciq.parquet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

STATEMENTS_PATH = PROCESSED_DIR / "statements.parquet"
METRICS_PATH = PROCESSED_DIR / "metrics.parquet"
RISKS_PATH = PROCESSED_DIR / "risks.parquet"
SEGMENTS_PATH = PROCESSED_DIR / "segments.parquet"

CHECKPOINT_FILE = PROCESSED_DIR / "checkpoint_processed_ids.txt"

# ~14 calls/min → 60 / 4.5
PER_CALL_SLEEP_SECONDS = 4.5
# We now flush to parquet after every statement for simplicity
BATCH_SIZE = 1

# -------------------------------------------------
# 1. Gemini client + model config
# -------------------------------------------------

client = genai.Client()  # uses GEMINI_API_KEY

MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0

# -------------------------------------------------
# 2. Pydantic schemas
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
    name: str = Field(...)
    category: str | None = None
    value: float | None = None
    value_type: MetricValueType = "level"
    unit: str | None = None
    currency: str | None = None
    direction: MetricDirection = "unknown"
    is_guidance: bool | None = None
    period: str | None = None
    certainty: MetricCertainty = "explicit"
    evidence_span: str | None = None
    context: str | None = None


class RiskItem(BaseModel):
    type: str = Field(...)
    sentiment: RiskSentiment = "unknown"
    severity: RiskSeverity = "unknown"
    certainty: SegmentCertainty = "explicit"
    evidence_span: str | None = None
    context: str | None = None


class SegmentItem(BaseModel):
    name: str = Field(...)
    direction: SegmentDirection = "unknown"
    is_guidance: bool | None = None
    certainty: SegmentCertainty = "explicit"
    evidence_span: str | None = None
    context: str | None = None


class ExtractionResponse(BaseModel):
    metrics: List[MetricItem] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)
    segments: List[SegmentItem] = Field(default_factory=list)
    overall_sentiment: OverallSentiment = "unknown"


RESPONSE_SCHEMA = ExtractionResponse.model_json_schema()

SYSTEM_PROMPT = """
You are an expert financial information extraction model.
You read earnings call statements and extract metrics, guidance, risks, and business segments.

Return ONLY JSON (no prose) that exactly matches the provided JSON schema.
Respect all enum values strictly. Use null or "unknown" where appropriate.

Constraints:
- evidence_span must be a SHORT direct quote from the statement, at most 200 characters.
- context must be a SHORT paraphrase, at most 300 characters.
- Do NOT repeat the same sentence or phrase more than once.
- Do NOT include the full statement text in evidence_span or context.
"""


# -------------------------------------------------
# 3. Gemini call with rate-limit + quota handling
# -------------------------------------------------

def _extract_error_payload(exc: Exception) -> dict | None:
    """
    Try to robustly pull a JSON/dict error payload out of ClientError/ServerError.
    The google client sometimes gives:
      - a dict with 'error'
      - or a string like "429 RESOURCE_EXHAUSTED. {'error': {...}}"
    """
    if not exc.args:
        return None

    arg0 = exc.args[0]

    # Direct dict
    if isinstance(arg0, dict) and "error" in arg0:
        return arg0.get("error")

    # Try to parse dict embedded in a string
    try:
        s = str(arg0)
        first_brace = s.find("{")
        if first_brace != -1:
            maybe_dict = ast.literal_eval(s[first_brace:])
            if isinstance(maybe_dict, dict):
                # sometimes the dict itself is {"error": {...}}
                if "error" in maybe_dict and isinstance(maybe_dict["error"], dict):
                    return maybe_dict["error"]
                # or the dict IS the error payload
                return maybe_dict
    except Exception:
        pass

    return None


def _parse_quota_info(err_payload: dict) -> tuple[int | None, float | None, bool]:
    """
    Returns (error_code, retry_seconds, is_daily_quota).
    """
    err_code = err_payload.get("code")
    retry_seconds = None
    is_daily = False

    details = err_payload.get("details", []) or []
    for d in details:
        if "@type" in d and "RetryInfo" in d["@type"]:
            raw = str(d.get("retryDelay", "")).rstrip("s")
            try:
                retry_seconds = float(raw)
            except Exception:
                pass

        if "@type" in d and "QuotaFailure" in d["@type"]:
            violations = d.get("violations", []) or []
            for v in violations:
                quota_id = v.get("quotaId", "")
                # Daily limit (what you're hitting: GenerateRequestsPerDayPerProjectPerModel-FreeTier)
                if "GenerateRequestsPerDayPerProjectPerModel" in quota_id:
                    is_daily = True

    return err_code, retry_seconds, is_daily


def call_gemini(prompt: str, max_retries: int = 8, base_backoff: float = 5.0) -> str:
    """
    Call Gemini with:
    - 429 RATE (per-minute) → sleep & retry (using RetryInfo if present)
    - 429 DAILY QUOTA → raise RuntimeError("DAILY_QUOTA_EXCEEDED")
    - 503 → exponential backoff
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
                ),
            )
            return response.text

        except (ClientError, ServerError) as e:
            err_payload = _extract_error_payload(e)

            if isinstance(err_payload, dict):
                err_code, retry_seconds, is_daily = _parse_quota_info(err_payload)

                # DAILY QUOTA: stop immediately, let caller checkpoint & exit
                if is_daily:
                    msg = err_payload.get("message", "Daily quota exceeded.")
                    print(f"[DAILY QUOTA EXCEEDED] {msg}")
                    raise RuntimeError("DAILY_QUOTA_EXCEEDED")

                # PER-MINUTE 429 (rate limit)
                if err_code == 429:
                    attempt += 1
                    delay = retry_seconds or (base_backoff * attempt)
                    print(
                        f"[429] Rate limit hit. Sleeping {delay:.2f}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue

                # SERVER BUSY 503
                if err_code == 503:
                    attempt += 1
                    delay = base_backoff * attempt
                    print(
                        f"[503] Server busy. Backoff {delay:.2f}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(delay)
                    continue

            # Anything else → just rethrow
            raise

    raise RuntimeError("MAX_RETRIES_EXCEEDED")


# -------------------------------------------------
# 4. Checkpointing helpers
# -------------------------------------------------

def load_checkpoint() -> Set[str]:
    """
    Load processed statement_ids from:
    1) checkpoint file if present
    2) otherwise, reconstruct from STATEMENTS_PATH (if it exists)
    """
    processed: Set[str] = set()

    # 1) From checkpoint file
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            processed = {line.strip() for line in f if line.strip()}

    # 2) If no checkpoint but we already have statements.parquet, rebuild
    if not processed and STATEMENTS_PATH.exists():
        try:
            existing = pd.read_parquet(STATEMENTS_PATH, columns=["statement_id"])
            processed = set(existing["statement_id"].astype(str).tolist())
            # Optionally re-write checkpoint file for future fast loads
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w") as f:
                for sid in sorted(processed):
                    f.write(str(sid) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to rebuild checkpoint from {STATEMENTS_PATH}: {e}")

    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print(f"Loaded {len(processed)} processed statement_ids from checkpoint.")
    return processed


def append_checkpoint(statement_id: str):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(str(statement_id) + "\n")


# -------------------------------------------------
# 5. IO helpers (append parquet)
# -------------------------------------------------

def append_or_init(path: Path, new_df: pd.DataFrame):
    if new_df.empty:
        return
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path)


# -------------------------------------------------
# 6. Data loading
# -------------------------------------------------

def load_corpus_2025_top20() -> pd.DataFrame:
    """
    Load WRDS call sections and filter to:
      - call_year == 2025
      - companies in tech_universe_top20_with_ciq.parquet
    """
    call_section = pd.read_parquet(CORPUS_PATH).copy()
    call_section["statement_id"] = call_section["segment_id"].astype(str)
    if "clean_text" not in call_section.columns:
        call_section["clean_text"] = call_section["componenttext"].fillna("").astype(str)

    uni = pd.read_parquet(UNIVERSE_TOP20_PATH).copy()
    uni = uni.dropna(subset=["ciq_company_id"])
    top20_ids = set(uni["ciq_company_id"].astype(float).tolist())

    call_section["companyid"] = call_section["companyid"].astype(float)

    before = len(call_section)
    call_section = call_section[
        (call_section["call_year"] == 2025)
        & (call_section["companyid"].isin(top20_ids))
    ].copy()
    after = len(call_section)

    print(f"Using corpus: {CORPUS_PATH}")
    print(f"Corpus rows before filter: {before}")
    print(f"Corpus rows after  2025 + top-20 filter: {after}")

    return call_section


# -------------------------------------------------
# 7. Prompt builder
# -------------------------------------------------

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
# 8. Extraction for a group (company + quarter) with batching + checkpoint
# -------------------------------------------------

def extract_group(subset: pd.DataFrame, processed_ids: Set[str]):
    """
    subset: all rows for a (companyid, company_name, call_year, call_quarter) group
    processed_ids: global set of already-processed statement_ids (checkpoint)
    """

    statement_rows = []
    metric_rows = []
    risk_rows = []
    segment_rows = []

    def flush_batch():
        nonlocal statement_rows, metric_rows, risk_rows, segment_rows
        if statement_rows:
            append_or_init(STATEMENTS_PATH, pd.DataFrame(statement_rows))
        if metric_rows:
            append_or_init(METRICS_PATH, pd.DataFrame(metric_rows))
        if risk_rows:
            append_or_init(RISKS_PATH, pd.DataFrame(risk_rows))
        if segment_rows:
            append_or_init(SEGMENTS_PATH, pd.DataFrame(segment_rows))
        statement_rows = []
        metric_rows = []
        risk_rows = []
        segment_rows = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), leave=False):
        statement_id = str(row["statement_id"])
        cid = float(row["companyid"])
        call_id = row["transcriptid"]

        # Skip already processed
        if statement_id in processed_ids:
            continue

        text = str(row["clean_text"] or "").strip()
        if not text:
            continue

        try:
            prompt = build_prompt(row)

            # --------------------------------------------
            # SAFETY: JSON validation + salvage loop
            # --------------------------------------------
            attempts = 0
            max_attempts = 3

            while True:
                response_text = call_gemini(prompt)

                try:
                    # First, try parsing as-is
                    data = ExtractionResponse.model_validate_json(response_text)
                    break
                except Exception:
                    attempts += 1
                    print(
                        f"[WARN] Bad JSON for statement {statement_id}. "
                        f"Attempt {attempts}/{max_attempts}"
                    )
                    print("--- RAW TEXT START ---")
                    print(response_text[:2000])
                    print("--- RAW TEXT END ---")

                    # If braces are unbalanced, try trimming at last closing brace
                    if response_text.count("{") != response_text.count("}"):
                        print(
                            "[HINT] JSON looks truncated (brace mismatch). "
                            "Trying to trim at last '}'..."
                        )
                        last_brace = response_text.rfind("}")
                        if last_brace != -1:
                            trimmed = response_text[: last_brace + 1]
                            try:
                                data = ExtractionResponse.model_validate_json(trimmed)
                                print("[INFO] Successfully parsed after trimming.")
                                break
                            except Exception:
                                print("[INFO] Trimmed JSON still invalid.")

                    if attempts >= max_attempts:
                        # give up on this statement; proceed with empty extraction
                        print(
                            "[ERROR] JSON broken permanently for this statement. "
                            "Proceeding with empty ExtractionResponse()."
                        )
                        data = ExtractionResponse()  # empty/default
                        break

                    # small wait before retrying the LLM
                    time.sleep(1.5)
                    continue

            # -------------------
            # Statement row
            # -------------------
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
                    "text": text,
                    "overall_sentiment": data.overall_sentiment,
                }
            )

            # -------------------
            # Metrics
            # -------------------
            for m in data.metrics:
                metric_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "call_year": row["call_year"],
                        "call_quarter": row["call_quarter"],
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

            # -------------------
            # Risks
            # -------------------
            for r in data.risks:
                risk_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "call_year": row["call_year"],
                        "call_quarter": row["call_quarter"],
                        "risk_type": r.type,
                        "risk_sentiment": r.sentiment,
                        "risk_severity": r.severity,
                        "risk_certainty": r.certainty,
                        "risk_evidence_span": r.evidence_span,
                        "risk_context": r.context,
                    }
                )

            # -------------------
            # Segments
            # -------------------
            for s in data.segments:
                segment_rows.append(
                    {
                        "fact_id": str(uuid.uuid4()),
                        "statement_id": statement_id,
                        "company_id": cid,
                        "call_id": call_id,
                        "call_period": row["call_period"],
                        "call_year": row["call_year"],
                        "call_quarter": row["call_quarter"],
                        "segment_name": s.name,
                        "segment_direction": s.direction,
                        "segment_is_guidance": s.is_guidance,
                        "segment_certainty": s.certainty,
                        "segment_evidence_span": s.evidence_span,
                        "segment_context": s.context,
                    }
                )

            # mark checkpoint AFTER everything succeeded for this statement
            append_checkpoint(statement_id)
            processed_ids.add(statement_id)

            # flush batch (BATCH_SIZE = 1, so this is effectively per-statement)
            if len(statement_rows) >= BATCH_SIZE:
                flush_batch()

            # keep under ~15 calls/min
            time.sleep(PER_CALL_SLEEP_SECONDS)

        except RuntimeError as re:
            # Daily quota or max retries: flush what we have and bubble up
            print(
                f"[FATAL] Statement {statement_id} on Call {call_id} "
                f"for Company {cid}: {re}"
            )
            flush_batch()
            raise

        except Exception as e:
            # Non-fatal: log and continue (no checkpoint → will retry on next run)
            print(
                f"[ERROR] Statement {statement_id} on Call {call_id} "
                f"for Company {cid}: {e}"
            )
            continue

    # flush any remaining rows
    flush_batch()


# -------------------------------------------------
# 9. Main: process all 2025 quarters for top-20
# -------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    call_section = load_corpus_2025_top20()

    groups = call_section.groupby(
        ["companyid", "company_name", "call_year", "call_quarter"]
    )

    processed_ids = load_checkpoint()

    for (cid, cname, year, quarter), subset in groups:
        label = f"{cname} (id={cid}) {year}{quarter}"
        print("\n==============================")
        print(f"{cname} - companyid={cid}")
        print("==============================")
        print(f"--- Extracting {label} ---")
        print(f"Rows in group: {len(subset)}")

        if subset.empty:
            continue

        try:
            extract_group(subset, processed_ids)
        except RuntimeError as e:
            print(
                "\n[HALT] Fatal error during extraction "
                f"for {label}: {e}"
            )
            print(
                "Flushed current batch. You can rerun this script later; "
                "it will resume from the last checkpoint."
            )
            break

    print("\n✅ Rebuild complete up to last successful checkpoint.")


if __name__ == "__main__":
    main()
