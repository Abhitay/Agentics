# src/utils/quarters.py

from __future__ import annotations
from typing import Optional
import math
import re


def normalize_quarter(
    call_year: Optional[int],
    call_quarter: Optional[str],
    call_period: Optional[str],
) -> Optional[str]:
    """
    Turn (year, quarter, call_period) into '2025Q4' style string.
    - If year + quarter are present, use them.
    - Else fall back to call_period like '2025 Q4' -> '2025Q4'.
    """
    # Prefer explicit year + quarter if available
    if call_year is not None and not (isinstance(call_year, float) and math.isnan(call_year)) \
       and call_quarter is not None and not (isinstance(call_quarter, float) and math.isnan(call_quarter)):
        try:
            year_int = int(call_year)
            q_str = str(call_quarter).upper().strip()
            # Accept 'Q4', '4', etc.
            if q_str.startswith("Q"):
                q_str = q_str[1:]
            return f"{year_int}Q{q_str}"
        except Exception:
            pass

    # Fallback: derive from call_period string
    if isinstance(call_period, str) and call_period.strip():
        s = call_period.replace(" ", "").upper()
        # Normalize patterns like '2025Q1', 'Q1-2025', 'Q1_2025'
        # If it already looks like YYYYQn, just return
        m = re.match(r"(\d{4})Q([1-4])", s)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"

        # Try QnYYYY
        m = re.match(r"Q([1-4])(\d{4})", s)
        if m:
            return f"{m.group(2)}Q{m.group(1)}"

        # As a last resort, just strip spaces
        return s

    return None


def quarter_sort_key(q: str):
    """
    Sort key for quarter strings like '2025Q1'.
    Anything that doesn't parse cleanly gets sent to the end.
    """
    if not isinstance(q, str):
        return (9999, 9, str(q))

    s = q.replace(" ", "").upper()

    m = re.match(r"(\d{4})Q([1-4])", s)
    if m:
        year = int(m.group(1))
        qnum = int(m.group(2))
        return (year, qnum, s)

    m = re.match(r"Q([1-4])(\d{4})", s)
    if m:
        qnum = int(m.group(1))
        year = int(m.group(2))
        return (year, qnum, s)

    # Fallback: shove to end, but keep deterministic
    return (9999, 9, s)
