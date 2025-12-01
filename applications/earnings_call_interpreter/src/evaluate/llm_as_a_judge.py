from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from src.rag.pipeline import answer_question, _call_llm


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator of answers about EARNINGS CALLS and management tone.

You will be given:
- The user's QUESTION.
- The MODEL_ANSWER produced by our system.
- Optionally: a REFERENCE_ANSWER (what an ideal answer would say).
- Optionally: KEY_FACTS (ground truth labels about tone, guidance, risks, etc.).

Your job:
1) Compare MODEL_ANSWER with the QUESTION and any provided KEY_FACTS/REFERENCE_ANSWER.
2) Score the answer on these dimensions (0.0 to 1.0 each):

- relevance: Does it directly address the user's question (especially sentiment/tone/guidance)?
- factual_correctness: Are specific claims about tone/guidance/risks consistent with KEY_FACTS/REFERENCE_ANSWER?
- groundedness: Does it avoid hallucinated details that contradict KEY_FACTS/REFERENCE_ANSWER?
- completeness: Does it mention the major sentiment themes implied by KEY_FACTS (e.g., optimism vs caution, key risks, capex caveats)?
- clarity: Is it concise, plain-English, and understandable to a non-expert investor listening to the call?

3) Compute an overall score as your best judgment (not just the mean).

If KEY_FACTS are given, you MUST treat them as ground truth.
If the answer contradicts them, penalize factual_correctness and groundedness.

Return ONLY valid JSON with this exact schema:

{
  "scores": {
    "relevance": float,
    "factual_correctness": float,
    "groundedness": float,
    "completeness": float,
    "clarity": float,
    "overall": float
  },
  "verdict": "short natural-language explanation (2-5 sentences)"
}
"""


def _call_judge_llm(prompt: str) -> Dict[str, Any]:
    """
    Uses the same client as the RAG pipeline, but with temperature=0 for stability
    """
    raw = _call_llm(prompt, temperature=0.0)
    # Robust-ish JSON extraction in case the model adds extra text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def build_judge_prompt(
    question: str,
    model_answer: str,
    reference_answer: Optional[str] = None,
    key_facts: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "question": question,
        "model_answer": model_answer,
    }
    if reference_answer is not None:
        payload["reference_answer"] = reference_answer
    if key_facts is not None:
        payload["key_facts"] = key_facts

    user_block = json.dumps(payload, indent=2)
    return JUDGE_SYSTEM_PROMPT + "\n\nUSER_INPUT:\n" + user_block


def judge_answer_for_question(
    *,
    question: str,
    company: str,           # ticker, same as your Streamlit app
    quarter: str,           # e.g. "2024Q1"
    filing_type: str = "Earnings Call",
    temperature: float = 0.2,
    reference_answer: Optional[str] = None,
    key_facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1) Calls your existing answer_question(...) exactly like Streamlit does.
    2) Feeds the result to the judge LLM.
    3) Returns both the answer and the judgment.
    """
    # ⬇️ MATCHES YOUR STREAMLIT CALL EXACTLY
    answer, sources = answer_question(
        question=question,
        company=company,
        filing_type=filing_type,
        quarter=quarter,
        temperature=temperature,
    )

    judge_prompt = build_judge_prompt(
        question=question,
        model_answer=answer,
        reference_answer=reference_answer,
        key_facts=key_facts,
    )
    judgment = _call_judge_llm(judge_prompt)

    return {
        "answer": answer,
        "sources": sources,
        "judgment": judgment,
    }
