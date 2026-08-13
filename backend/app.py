import os
import json
import sys

# Inject parent directory into path to support running directly from inside backend/ folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
import uuid
import logging
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import config
from backend.embeddings.model import EmbeddingModel
from backend.retrieval.dense import QdrantRepository
from backend.retrieval.sparse import BM25SparseRetriever
from backend.retrieval.reranker import Reranker
from backend.speech.elevenlabs import ElevenLabsSTTClient, ElevenLabsTTSClient
from backend.generation.providers import GroqLLMProvider, MockLLMProvider
from backend.generation.router import AdaptiveModelRouter
from backend.cache.semantic_cache import SemanticLSHCache
from backend.guardrails.grounding import GroundingChecker
from backend.observability.metrics import MetricsTracker
from backend.harness import PipelineHarness, PipelineResult
from backend.pipeline import AsyncRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG.api")

app = FastAPI(
    title="Multilingual Production RAG Service",
    description="Indic-focused voice/text retrieval and question-answering pipeline with sub-200ms retrieval and real-time audio streaming.",
    version="2.1.0"
)

# Enable CORS for frontend interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Container for Dependency Injection
container = {}

class QueryRequest(BaseModel):
    query: str = Field(..., example="भारत की राजधानी क्या है?")
    language: str = Field("hi", example="hi")
    latency_sensitive: bool = Field(False, example=False)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RAG pipeline with Dependency Injection Container...")

    # 1. Embedder
    embedder = EmbeddingModel(
        mode=config.EMBEDDING_MODE,
        model_name=config.EMBEDDING_MODEL_NAME,
        dim=config.EMBEDDING_DIM
    )
    container["embedder"] = embedder

    # 2. Dense Retriever
    dense_retriever = QdrantRepository(
        path=str(config.QDRANT_PATH),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        vector_dim=config.EMBEDDING_DIM
    )
    container["dense_retriever"] = dense_retriever

    # 3. Sparse Retriever
    sparse_retriever = BM25SparseRetriever()
    sparse_retriever.load(str(config.BM25_PATH))
    container["sparse_retriever"] = sparse_retriever

    # 4. Reranker
    reranker = Reranker(
        cross_encoder_model_name=config.CROSS_ENCODER_MODEL_NAME if config.RERANKER_TYPE == "cross_encoder" else None,
        embedding_model=embedder
    )
    container["reranker"] = reranker

    # 5. LLM Providers & Router (Dependency Injection)
    groq_provider = GroqLLMProvider(api_key=config.GROQ_API_KEY, default_model=config.GROQ_MODEL)
    mock_provider = MockLLMProvider()
    
    providers = {
        "groq": groq_provider,
        "mock": mock_provider
    }
    
    router = AdaptiveModelRouter(
        providers=providers,
        default_provider_name="groq" if config.GROQ_API_KEY else "mock",
        fast_model=os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
        quality_model=os.getenv("GROQ_QUALITY_MODEL", config.GROQ_MODEL)
    )
    container["router"] = router

    # 6. Pipeline Harness
    harness = PipelineHarness(
        router=router,
        embedder=embedder,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        max_retries=2
    )
    container["harness"] = harness

    # 7. Grounding Checker
    grounding_checker = GroundingChecker(
        embedding_model=embedder,
        relevance_threshold=config.RELEVANCE_THRESHOLD,
        grounding_threshold=config.GROUNDING_THRESHOLD
    )
    container["grounding_checker"] = grounding_checker

    # 8. Speech Clients (STT & TTS)
    stt_client = ElevenLabsSTTClient(api_key=config.ELEVENLABS_API_KEY)
    tts_client = ElevenLabsTTSClient(api_key=config.ELEVENLABS_API_KEY)
    container["stt"] = stt_client
    container["tts"] = tts_client

    # 9. Semantic LSH Cache (~2ms match)
    semantic_cache = SemanticLSHCache(file_path=str(config.CACHE_PATH), max_hamming_distance=3)
    container["semantic_cache"] = semantic_cache

    # 10. Metrics Tracker
    metrics_tracker = MetricsTracker()
    container["metrics"] = metrics_tracker

    # 11. Async Orchestrated Pipeline with Audio Streaming
    pipeline = AsyncRAGPipeline(
        harness=harness,
        embedder=embedder,
        reranker=reranker,
        grounding_checker=grounding_checker,
        semantic_cache=semantic_cache,
        metrics_tracker=metrics_tracker,
        stt_client=stt_client,
        tts_client=tts_client
    )
    container["pipeline"] = pipeline

    logger.info("Async RAG Pipeline with Real-time Audio Streaming initialized.")

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": config.ENVIRONMENT,
        "embedding_mode": config.EMBEDDING_MODE,
        "reranker_type": config.RERANKER_TYPE,
        "router_models": {
            "fast": os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
            "quality": config.GROQ_MODEL
        },
        "timestamp": time.time()
    }

@app.get("/api/v1/metrics")
async def get_metrics():
    tracker: MetricsTracker = container.get("metrics")
    if tracker:
        return tracker.get_report()
    return {"error": "Metrics tracker not available."}

@app.post("/api/v1/text/query")
async def process_text_query(req: QueryRequest):
    pipeline: AsyncRAGPipeline = container["pipeline"]
    try:
        result = await pipeline.process_text_query(
            query=req.query, 
            language=req.language, 
            latency_sensitive=req.latency_sensitive
        )
        return result.dict()
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/query")
async def process_voice_query(
    file: UploadFile = File(...),
    language: str = Form("hi")
):
    pipeline: AsyncRAGPipeline = container["pipeline"]
    try:
        audio_bytes = await file.read()
        result = await pipeline.process_voice_query(audio_bytes=audio_bytes, language=language)
        return result.dict()
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/text/stream")
async def stream_text_audio(req: QueryRequest):
    """
    Streams audio/mpeg synthesized spoken answer in real-time from text query.
    Pipes LLM tokens into ElevenLabs TTS Stream.
    """
    pipeline: AsyncRAGPipeline = container["pipeline"]
    return StreamingResponse(
        pipeline.process_text_audio_stream(query=req.query, language=req.language),
        media_type="audio/mpeg"
    )

@app.post("/api/v1/voice/stream")
async def stream_voice_audio(
    file: UploadFile = File(...),
    language: str = Form("hi")
):
    """
    Streams audio/mpeg synthesized spoken answer in real-time from voice input upload.
    Pipeline: Voice STT -> Vector Retrieval -> LLM Stream -> ElevenLabs TTS Stream.
    """
    pipeline: AsyncRAGPipeline = container["pipeline"]
    audio_bytes = await file.read()
    return StreamingResponse(
        pipeline.process_voice_audio_stream(audio_bytes=audio_bytes, language=language),
        media_type="audio/mpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
