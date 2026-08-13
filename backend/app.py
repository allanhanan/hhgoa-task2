import os
import sys

# Inject parent directory into path to support running directly from inside backend/ folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from backend import config
from backend.embeddings.model import EmbeddingModel
from backend.retrieval.dense import QdrantRepository
from backend.retrieval.sparse import BM25SparseRetriever
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.reranker import Reranker
from backend.speech.sarvam import SarvamSTTClient
from backend.generation.llm import LLMGenerator
from backend.cache.cache_adapter import get_cache_adapter
from backend.guardrails.safety import check_query_safety
from backend.guardrails.grounding import GroundingChecker
from backend.observability.metrics import MetricsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG.api")

app = FastAPI(
    title="Multilingual Production RAG Service",
    description="Indic-focused voice/text retrieval and question-answering pipeline with multi-signal guardrails.",
    version="1.0.0"
)

# Enable CORS for frontend interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instances (initialized on startup)
components = {}

class QueryRequest(BaseModel):
    query: str = Field(..., example="भारत की राजधानी क्या है?")
    language: str = Field("hi", example="hi")

@app.on_event("startup")
def startup_event():
    logger.info("Initializing RAG pipeline components...")
    
    # 1. Initialize Embedder
    embedder = EmbeddingModel(
        mode=config.EMBEDDING_MODE,
        model_name=config.EMBEDDING_MODEL_NAME,
        dim=config.EMBEDDING_DIM
    )
    components["embedder"] = embedder
    
    # 2. Initialize Qdrant Persistent Repository
    qdrant_repo = QdrantRepository(
        path=str(config.QDRANT_PATH),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        vector_dim=config.EMBEDDING_DIM
    )
    components["dense_retriever"] = qdrant_repo
    
    # 3. Initialize BM25 Sparse Retriever
    sparse_retriever = BM25SparseRetriever()
    sparse_retriever.load(str(config.BM25_PATH))
    components["sparse_retriever"] = sparse_retriever
    
    # 4. Initialize Reranker
    reranker = Reranker(
        cross_encoder_model_name=config.CROSS_ENCODER_MODEL_NAME if config.RERANKER_TYPE == "cross_encoder" else None,
        embedding_model=embedder
    )
    components["reranker"] = reranker
    
    # 5. Initialize Generator
    components["generator"] = LLMGenerator(api_key=config.GEMINI_API_KEY)
    
    # 6. Initialize Grounding Checker
    components["grounding_checker"] = GroundingChecker(
        embedding_model=embedder,
        relevance_threshold=config.RELEVANCE_THRESHOLD,
        grounding_threshold=config.GROUNDING_THRESHOLD
    )
    
    # 7. Initialize Sarvam STT
    components["stt"] = SarvamSTTClient(api_key=config.SARVAM_API_KEY)
    
    # 8. Initialize Cache
    components["cache"] = get_cache_adapter(
        backend=config.CACHE_BACKEND,
        redis_url=config.REDIS_URL,
        file_path=str(config.CACHE_PATH)
    )
    
    # 9. Initialize Metrics Tracker
    components["metrics"] = MetricsTracker()
    
    logger.info("RAG pipeline initialization complete.")

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "environment": config.ENVIRONMENT,
        "embedding_mode": config.EMBEDDING_MODE,
        "reranker_type": config.RERANKER_TYPE,
        "cache_backend": config.CACHE_BACKEND,
        "timestamp": time.time()
    }

@app.get("/api/v1/metrics")
def get_metrics():
    tracker: MetricsTracker = components.get("metrics")
    if tracker:
        return tracker.get_report()
    return {"error": "Metrics tracker not available."}

@app.post("/api/v1/text/query")
def process_text_query(req: QueryRequest):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    query = req.query.strip()
    language = req.language.strip()
    
    logger.info(f"[{request_id}] Received text query: '{query}' ({language})")
    
    # 1. Safety check (Jailbreak / Prompt Injection check)
    is_safe, safety_status = check_query_safety(query)
    if not is_safe:
        logger.warning(f"[{request_id}] Query blocked by safety guardrail: {safety_status}")
        return {
            "request_id": request_id,
            "query": query,
            "answer": "blocked_by_safety",
            "grounded": False,
            "status": safety_status,
            "latency_ms": 0
        }

    # 2. Check Cache
    cache = components["cache"]
    cache_key = f"rag_cache:{language}:{query}"
    cached_val = cache.get(cache_key)
    if cached_val:
        logger.info(f"[{request_id}] Cache hit! Returning cached response.")
        response = json.loads(cached_val)
        response["request_id"] = request_id
        response["cached"] = True
        return response

    # 3. Execute Pipeline with detailed latency tracking
    metrics: MetricsTracker = components["metrics"]
    latencies = {}
    
    t_start = time.time()

    # Query embedding
    t_emb_0 = time.time()
    embedder = components["embedder"]
    try:
        query_vector = embedder.embed_queries([query])[0]
    except Exception as e:
        logger.error(f"[{request_id}] Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed.")
    latencies["embedding_ms"] = (time.time() - t_emb_0) * 1000

    # Dense Search
    t_dense_0 = time.time()
    dense_ret = components["dense_retriever"]
    dense_res = dense_ret.search(query_vector, limit=config.DENSE_TOP_K, language=language)
    latencies["dense_ms"] = (time.time() - t_dense_0) * 1000

    # Sparse Search (BM25)
    t_sparse_0 = time.time()
    sparse_ret = components["sparse_retriever"]
    sparse_res = sparse_ret.search(query, limit=config.SPARSE_TOP_K, language=language)
    latencies["sparse_ms"] = (time.time() - t_sparse_0) * 1000

    # Hybrid Fusion (RRF)
    t_fusion_0 = time.time()
    fused_res = reciprocal_rank_fusion(dense_res, sparse_res, k=config.RRF_CONSTANT, limit=config.FUSED_TOP_K)
    latencies["fusion_ms"] = (time.time() - t_fusion_0) * 1000

    # Reranking
    t_rerank_0 = time.time()
    reranker = components["reranker"]
    reranked_res = reranker.rerank(query, fused_res, limit=config.FINAL_TOP_K)
    latencies["reranking_ms"] = (time.time() - t_rerank_0) * 1000

    # Relevance guardrail check
    max_relevance = max([c.get("rerank_score", 0.0) for c in reranked_res]) if reranked_res else 0.0
    effective_threshold = 0.0 if config.EMBEDDING_MODE == "mock" else config.RELEVANCE_THRESHOLD
    if max_relevance < effective_threshold:
        logger.info(f"[{request_id}] Low relevance ({max_relevance:.2f}). Refusing query.")

        total_time = (time.time() - t_start) * 1000
        latencies["total_ms"] = total_time
        response = {
            "request_id": request_id,
            "query": query,
            "answer": "NOT_SUPPORTED",
            "sources": [],
            "language": language,
            "grounded": True,
            "confidence": 1.0,
            "status": "OUT_OF_SCOPE",
            "latency": latencies,
            "pipeline_steps": {
                "dense_candidates": len(dense_res),
                "sparse_candidates": len(sparse_res),
                "fused_candidates": len(fused_res),
                "reranked_candidates": 0
            }
        }
        # Log out-of-scope metric
        metrics.log_request(total_time, is_grounded=True, cache_hit=False)
        return response

    # Grounded Generation
    t_gen_0 = time.time()
    generator = components["generator"]
    answer, cited_chunk_ids = generator.generate_answer(query, reranked_res)
    latencies["generation_ms"] = (time.time() - t_gen_0) * 1000

    # Grounding Checker
    t_ground_0 = time.time()
    checker = components["grounding_checker"]
    is_grounded, grounding_score, grounding_details = checker.verify_grounding(
        query=query,
        answer=answer,
        retrieved_chunks=reranked_res,
        llm_judge_fn=generator.verify_grounding_via_llm
    )
    latencies["grounding_ms"] = (time.time() - t_ground_0) * 1000

    # Final latency compute
    total_time = (time.time() - t_start) * 1000
    latencies["total_ms"] = total_time

    # Build response sources with proper chunk metadata mappings
    sources = []
    for c in reranked_res:
        sources.append({
            "chunk_id": c.get("chunk_id", ""),
            "text": c["payload"]["text"],
            "score": c.get("rerank_score", c.get("score", 0.0)),
            "metadata": c["payload"].get("metadata", {})
        })

    response = {
        "request_id": request_id,
        "query": query,
        "answer": answer if is_grounded else "NOT_SUPPORTED",
        "sources": sources,
        "language": language,
        "grounded": is_grounded,
        "confidence": grounding_score,
        "status": "SUCCESS" if is_grounded else "GROUNDING_VIOLATION",
        "latency": latencies,
        "pipeline_steps": {
            "dense_candidates": len(dense_res),
            "sparse_candidates": len(sparse_res),
            "fused_candidates": len(fused_res),
            "reranked_candidates": len(reranked_res)
        },
        "grounding_details": grounding_details
    }

    # Save to Cache
    try:
        cache.set(cache_key, json.dumps(response), ttl=3600)
    except Exception as e:
        logger.error(f"[{request_id}] Cache set failure: {e}")

    # Track metrics
    metrics.log_request(total_time, is_grounded=is_grounded, cache_hit=False)

    return response

@app.post("/api/v1/voice/query")
async def process_voice_query(
    file: UploadFile = File(...),
    language: str = Form("hi")
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{request_id}] Received voice query in '{language}'")
    
    t_start = time.time()
    
    # Read audio bytes
    audio_bytes = await file.read()
    
    # 1. Voice transcription (Sarvam STT)
    t_stt_0 = time.time()
    stt_client = components["stt"]
    transcription = stt_client.transcribe(audio_bytes, language)
    stt_latency = (time.time() - t_stt_0) * 1000
    
    if not transcription:
        raise HTTPException(status_code=400, detail="Voice transcription failed or returned empty text.")

    # 2. Forward transcription to text query pipeline
    # We invoke process_text_query directly to preserve latency metrics and trace query execution flow
    req = QueryRequest(query=transcription, language=language)
    response = process_text_query(req)
    
    # 3. Add voice specific keys to payload response
    response["transcription"] = transcription
    
    # Record STT latency in breakdown
    response["latency"]["stt_ms"] = stt_latency
    response["latency"]["total_ms"] += stt_latency
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
