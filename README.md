# Cisco Nexus 9000 Troubleshooting Assistant

A local CPU-friendly Streamlit RAG proof of concept for the Cisco Nexus 9000 Series Troubleshooting Guide. It uses MiniLM embeddings, persistent Chroma retrieval, route-aware query expansion/reranking, GPT-OSS-20B through Groq, and a local semantic cache.

## Setup

PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Put the guide at `data/source/cisco_nexus_9000_troubleshooting.pdf` or set `SOURCE_PDF` in `.env`, then add `GROQ_API_KEY` to `.env`.

## Run

Build the local index once:

```powershell
python scripts/ingest.py
streamlit run app.py
```

The app classifies cache misses as `factual` or `broad`. Both routes expand the query; broad retrieval additionally reranks candidates. The UI displays route, cache status, sources, and retrieval diagnostics.

## Tests

```powershell
python -m pytest -q
```

The PDF, `.env`, virtual environment, Chroma data, and semantic cache are excluded from Git by `.gitignore`. Commit `.env.example`, source code, tests, and documentation instead.
