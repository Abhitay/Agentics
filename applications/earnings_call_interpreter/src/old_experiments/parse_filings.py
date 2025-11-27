"""
Utilities to parse earnings call transcripts and SEC-like filings.
This is a first stub; you will extend it to handle more formats/tables.
"""

from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
from bs4 import BeautifulSoup


def read_pdf(path: str | Path) -> str:
    """Extract raw text from a PDF file."""
    path = Path(path)
    text: List[str] = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def read_html(path: str | Path) -> str:
    """Extract visible text from an HTML file."""
    path = Path(path)
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")


def parse_filing(path: str | Path) -> Dict[str, Any]:
    """
    Parse a single filing file and return a standard structure.

    Returns:
        {
            "raw_text": str,
            "tables": List[...],   # placeholder for future table handling
            "metadata": {...}
        }
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = read_pdf(path)
    elif suffix in {".html", ".htm"}:
        raw_text = read_html(path)
    else:
        # assume plain text
        raw_text = path.read_text(encoding="utf-8")

    tables: List[Any] = []  # TODO: implement table extraction later

    return {
        "raw_text": raw_text,
        "tables": tables,
        "metadata": {
            "source_path": str(path),
            # later: ticker, quarter, filing_type, etc.
        },
    }
