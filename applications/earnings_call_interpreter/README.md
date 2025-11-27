# Hybrid RAG for Earnings Call Filings

Analyzing earnings calls and financial filings is slow and error-prone. This project builds a **Hybrid Retrieval-Augmented Generation (RAG)** system that combines:

- **Vector search** over unstructured text and tables.
- **Knowledge graph** over companies, metrics, quarters, and relationships.

## Features

- Natural language Q&A over earnings calls.
- Automatic executive summaries of company performance.
- Peer benchmarking and competitive analysis across industry verticals.
- Streamlit UI for interactive exploration.

## Tech Stack

- **Python**, **Streamlit**
- **Vector DB**: e.g., Chroma / Qdrant / pgvector
- **Knowledge Graph**: e.g., Neo4j / NetworkX
- **LLM + Embeddings**: OpenAI or other provider

## Structure

- \`src/\` – core code (ingestion, graph, RAG, UI)
- \`docs/\` – design docs and notes
- \`README.md\` – project overview and setup

## Quickstart

```bash
pip install -r requirements.txt
streamlit run src/ui/app.py
Configure your vector DB, graph DB, and LLM API keys in `src/config/settings.py`.
