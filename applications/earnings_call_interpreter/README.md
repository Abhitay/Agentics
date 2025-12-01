# Tech Earnings RAG — Earnings Call Interpreter

## Overview

This application builds a RAG (Retrieval-Augmented Generation) pipeline over earnings call transcripts of top tech companies.  
Users can:
- Explore structured sections of earnings calls
- Search key metrics and risk segments
- Ask LLM-powered questions over extracted facts
- Visualize insights via a Streamlit UI

## Deployment Info

- **Project Slug:** `tech-earnings-rag`
- **Main Entry File:** `src/ui/app.py`
- **Deployment URL (after ECS):** `https://[cloudfront-domain]/tech-earnings-rag`

# ⚙️ Environment Variables

Add a `.env` file locally using `.env.example` as a reference.

Example:

GEMINI_API_KEY=...
NEO4J_URI=bolt://...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...

## 📦 Project Structure

earnings_call_interpreter/
├── src/
│ ├── ui/ # Streamlit UI
│ │ └── app.py
│ ├── embeddings/
│ ├── graph/
│ ├── rag/
│ ├── utils/
│ ├── config/
│ └── scripts/
├── data/
│ ├── corpus/
│ ├── universe/
│ ├── processed/
│ └── chroma/
├── requirements.txt
├── .env.example
├── README.md
├── Dockerfile
└── .dockerignore



# 🧠 Local Setup (non-Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the streamlit app
streamlit run src/ui/app.py