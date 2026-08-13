# HH Goa 2026 — Task 2: Voice-Enabled RAG System

> **Pipeline**: Voice Input → ElevenLabs STT → Chunking / Vector Retrieval → Groq LLM → Answer

A production-grade, multilingual Retrieval-Augmented Generation (RAG) system built on the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset. Supports 12 Indic languages + English. Users can speak or type a question; the pipeline transcribes it, retrieves grounded context, and returns a verified answer.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Configuration (.env)](#configuration-env)
6. [Step 1 — Ingest & Index the Dataset](#step-1--ingest--index-the-dataset)
7. [Step 2 — Start the Backend](#step-2--start-the-backend)
8. [Step 3 — Start the Frontend](#step-3--start-the-frontend)
9. [API Reference](#api-reference)
10. [Chunking Strategies](#chunking-strategies)
11. [Running Evaluations](#running-evaluations)
12. [Environment Variables Reference](#environment-variables-reference)
13. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
hhgoa-task2/
├── backend/                   # FastAPI server + all pipeline modules
│   ├── app.py                 # Main API entrypoint
│   ├── config.py              # Central config (reads .env)
│   ├── embeddings/
│   │   └── model.py           # sentence-transformers wrapper (real + mock)
│   ├── speech/
│   │   └── elevenlabs.py      # ElevenLabs Scribe v1 STT client
│   ├── retrieval/
│   │   ├── dense.py           # Qdrant ANN search
│   │   ├── sparse.py          # BM25 (rank-bm25) sparse search
│   │   ├── hybrid.py          # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py        # Cross-encoder or cosine reranker
│   ├── generation/
│   │   ├── llm.py             # Groq REST API wrapper
│   │   └── prompts.py         # System / user prompt templates
│   ├── guardrails/
│   │   ├── safety.py          # Jailbreak / prompt-injection filter
│   │   └── grounding.py       # Multi-signal grounding checker
│   ├── cache/
│   │   └── cache_adapter.py   # Memory + Redis cache adapters
│   └── observability/
│       └── metrics.py         # Latency / grounding rate tracker
├── ingestion/                 # One-time dataset indexing scripts
│   ├── preprocess.py          # Text cleaning, dedup, lang detection
│   ├── chunking.py            # Sentence / semantic / hierarchical chunkers
│   └── build_index.py         # Orchestrates download → chunk → embed → index
├── evaluation/                # Retrieval + answer quality metrics
│   ├── retrieval_metrics.py
│   ├── answer_metrics.py
│   └── benchmark.py
├── frontend/                  # Vite + React UI
│   └── src/
│       └── components/        # Dashboard, voice recorder, results panel
├── verify_rag.py              # Quick end-to-end smoke test
├── requirements.txt           # Python dependencies
├── .env.example               # Template — copy to .env and fill in keys
└── .gitignore
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          USER                               │
│           speaks / types a question                         │
└─────────────────┬───────────────────────────────────────────┘
                  │  audio file (WAV/MP3)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  1. ElevenLabs Scribe v1  (Speech-to-Text)                  │
│     POST /v1/speech-to-text  →  transcribed text            │
└─────────────────┬───────────────────────────────────────────┘
                  │  query text
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Safety Guardrail  (jailbreak / injection detection)     │
└─────────────────┬───────────────────────────────────────────┘
                  │  safe query
                  ▼
┌──────────────────────────────────────┐
│  3. Query Embedding                  │
│     paraphrase-multilingual-MiniLM   │
└──────┬───────────────────────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐    ┌──────────────────┐
│ Dense Search │    │  Sparse Search   │
│  (Qdrant)   │    │   (BM25)         │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Reciprocal Rank Fusion  (hybrid re-scoring)             │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Reranker  (cosine similarity or CrossEncoder)           │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Groq Llama 3.3 70B  (grounded answer generation)        │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Grounding Checker  (semantic sim + word overlap + LLM)  │
└─────────────────┬───────────────────────────────────────────┘
                  ▼
              ANSWER + SOURCES + LATENCY BREAKDOWN
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm | ≥ 9 |
| Git | any |

**API keys needed (free tiers work):**

| Service | Purpose | Sign-up |
|---------|---------|---------|
| ElevenLabs | Speech-to-Text | https://elevenlabs.io |
| Groq | LLM Reader | https://console.groq.com |

---

## Setup

```bash
# 1. Clone
git clone <your-repo-url>
cd hhgoa-task2

# 2. Python virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy and fill in .env
cp .env.example .env
# Open .env and set ELEVENLABS_API_KEY and GROQ_API_KEY

# 5. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Configuration (.env)

Open `.env` and at minimum set:

```env
ELEVENLABS_API_KEY=<your_key_here>
GROQ_API_KEY=<your_key_here>
```

All other values have sensible defaults for local development. See the full [Environment Variables Reference](#environment-variables-reference) section below.

---

## Step 1 — Ingest & Index the Dataset

This step downloads the MSMARCO-XI dataset from Hugging Face, chunks each passage using your chosen strategy, generates embeddings, and writes both a Qdrant vector index and a BM25 sparse index to `data/indexes/`.

**Run once before starting the backend.**

```bash
# Default: Hindi + Tamil, 1 000 documents, semantic chunking, validation split
python -m ingestion.build_index

# Custom options
python -m ingestion.build_index \
    --langs hi ta te kn \
    --limit 5000 \
    --strategy hierarchical \
    --split validation
```

| Flag | Default | Description |
|------|---------|-------------|
| `--langs` | `hi ta` | Space-separated ISO-639-1 language codes |
| `--limit` | `1000` | Max documents to ingest (controls index size) |
| `--strategy` | `semantic` | Chunking strategy: `sentence`, `semantic`, `hierarchical` |
| `--split` | `validation` | HuggingFace dataset split: `validation` (fast) or `train` (full) |

**Supported languages**: `hi` `ta` `te` `kn` `ml` `mr` `gu` `bn` `pa` `or` `as` `en`

> First run downloads ~400 MB (validation) or ~3.7 GB (train) from Hugging Face and the embedding model (~100 MB). Subsequent runs skip if the manifest is unchanged.

---

## Step 2 — Start the Backend

```bash
# From the project root
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

**Quick smoke test** (in a separate terminal):

```bash
python verify_rag.py
```

---

## Step 3 — Start the Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser. Click the **microphone** button, speak your question, and the pipeline responds with a grounded answer and source citations.

---

## API Reference

### Health Check
```
GET /api/v1/health
```

### Text Query
```
POST /api/v1/text/query
Content-Type: application/json

{
  "query": "भारत की राजधानी क्या है?",
  "language": "hi"
}
```

### Voice Query
```
POST /api/v1/voice/query
Content-Type: multipart/form-data

file=<audio.wav>
language=hi
```

### Metrics
```
GET /api/v1/metrics
```

**Response shape (text & voice):**
```json
{
  "request_id": "req_abc12345",
  "query": "भारत की राजधानी क्या है?",
  "answer": "भारत की राजधानी नई दिल्ली है।",
  "sources": [{ "chunk_id": "...", "text": "...", "score": 0.91, "metadata": {} }],
  "language": "hi",
  "grounded": true,
  "confidence": 0.87,
  "status": "SUCCESS",
  "latency": {
    "embedding_ms": 12.3,
    "dense_ms": 4.1,
    "sparse_ms": 1.8,
    "fusion_ms": 0.4,
    "reranking_ms": 8.9,
    "generation_ms": 820.0,
    "grounding_ms": 15.2,
    "total_ms": 862.7
  }
}
```

---

## Chunking Strategies

The pipeline implements three distinct strategies, selectable via `CHUNK_STRATEGY` in `.env` or `--strategy` in the ingestion CLI.

| Strategy | How it works | Best for |
|----------|-------------|---------|
| **`sentence`** | Groups N sentences with overlap into fixed-size windows. Enforces a max character ceiling. | Fast indexing, short passages |
| **`semantic`** | Embeds every sentence, splits where cosine similarity between consecutive sentences drops below a threshold. Produces semantically coherent chunks. | Quality retrieval on dense text |
| **`hierarchical`** | Creates parent chunks (broad context) and smaller child chunks (fine-grained). Child chunks are retrieved; parent text is surfaced for context. | Long passages, complex QA |

---

## Running Evaluations

```bash
# Benchmark retrieval + answer quality against the validation set
python -m evaluation.benchmark \
    --langs hi ta \
    --limit 200 \
    --strategy semantic
```

Outputs `Precision@K`, `Recall@K`, `MRR`, `ROUGE-L`, and `BERTScore` to the console.

---

## Environment Variables Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ELEVENLABS_API_KEY` | — | **Yes** | ElevenLabs Scribe v1 API key |
| `GROQ_API_KEY` | — | **Yes** | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | No | Groq model selection |
| `ENVIRONMENT` | `development` | No | `development` or `production` |
| `QDRANT_URL` | (local disk) | No | Qdrant Cloud endpoint URL |
| `QDRANT_API_KEY` | — | No | Qdrant Cloud API key |
| `CACHE_BACKEND` | `memory` | No | `memory` or `redis` |
| `REDIS_URL` | — | No | Redis connection URL |
| `EMBEDDING_MODE` | `real` | No | `real` (sentence-transformers) or `mock` (testing) |
| `CHUNK_STRATEGY` | `semantic` | No | `sentence`, `semantic`, `hierarchical` |
| `SEMANTIC_THRESHOLD` | `0.65` | No | Cosine sim threshold for semantic chunking |
| `RERANKER_TYPE` | `cosine` | No | `cosine` or `cross_encoder` |
| `INGESTION_LIMIT` | `1000` | No | Docs to ingest per run |
| `ALLOW_SYNTHETIC_FALLBACK` | `false` | No | Use synthetic data if HF fails — **never true for submission** |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**  
Run all commands from the project root (`hhgoa-task2/`), not from inside a subdirectory.

**`RuntimeError: Embedding model initialization failed`**  
Set `EMBEDDING_MODE=mock` in `.env` to bypass model download during testing. For production, ensure `sentence-transformers` is installed and you have internet access on first run.

**Sparse index not found (BM25 returns empty)**  
You must run `python -m ingestion.build_index` before starting the backend.

**ElevenLabs STT returns empty / mock transcription**  
Check that `ELEVENLABS_API_KEY` is correctly set in `.env`. In mock mode (no key), the system uses a predefined test sentence per language.

**Low grounding score / `GROUNDING_VIOLATION` responses**  
The index may be too small. Re-run ingestion with a larger `--limit` or a different `--strategy`. Also check that the query language matches an ingested language.
