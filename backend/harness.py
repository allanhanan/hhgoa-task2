import time
import inspect
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from backend.interfaces import ILLMProvider, IModelRouter, IDenseRetriever, ISparseRetriever, IEmbeddingModel

logger = logging.getLogger("RAG.harness")

# -------------------------------------------------------------------
# Structured Input & Output Schema Definitions
# -------------------------------------------------------------------

class SourceCitation(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DetailedLatency(BaseModel):
    stt_ms: Optional[float] = 0.0
    embedding_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    retrieval_total_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_ms: float = 0.0

class PipelineResult(BaseModel):
    request_id: str
    query: str
    transcription: Optional[str] = None
    answer: str
    sources: List[SourceCitation]
    language: str
    grounded: bool
    confidence: float
    status: str
    cached: bool = False
    latency: DetailedLatency
    pipeline_steps: Dict[str, int]
    grounding_details: Optional[Dict[str, Any]] = None

# -------------------------------------------------------------------
# Circuit Breaker Implementation for Resilience
# -------------------------------------------------------------------

class CircuitBreaker:
    """
    Prevents cascading failures when downstream LLM APIs experience outages or rate limits.
    """
    def __init__(self, failure_threshold: int = 3, recovery_time: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.state = "CLOSED" # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = 0.0

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit Breaker TRIPPED to OPEN state after {self.failure_count} failures!")

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = "HALF-OPEN"
                logger.info("Circuit Breaker transitioned to HALF-OPEN for trial request.")
                return True
            return False
        return True # HALF-OPEN

# -------------------------------------------------------------------
# Orchestration Harness
# -------------------------------------------------------------------

class PipelineHarness:
    """
    Production Orchestration Harness around the RAG models.
    Provides:
    - Structured Pydantic I/O validation
    - Exponential backoff retries on transient errors
    - Deprecation model fallback routing
    - Circuit breaker pattern for API protection
    - Fallback error recovery across retrieval modules
    """
    def __init__(
        self,
        router: IModelRouter,
        embedder: IEmbeddingModel,
        dense_retriever: Any,
        sparse_retriever: Any,
        max_retries: int = 2
    ):
        self.router = router
        self.embedder = embedder
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()

    async def execute_retrieval_with_fallbacks(
        self, 
        query: str, 
        query_vector: Any, 
        language: str,
        strategy: Optional[str] = None,
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
        """
        Executes Dense and Sparse retrieval in parallel with graceful fault tolerance.
        Works seamlessly with both async and sync retriever methods via asyncio.to_thread.
        """
        async def fetch_dense():
            t0 = time.time()
            try:
                if query_vector is None:
                    return [], 0.0
                if inspect.iscoroutinefunction(self.dense_retriever.search):
                    res = await self.dense_retriever.search(query_vector, limit=dense_top_k, language=language, strategy=strategy)
                else:
                    res = await asyncio.to_thread(self.dense_retriever.search, query_vector, limit=dense_top_k, language=language, strategy=strategy)
                return res, (time.time() - t0) * 1000
            except Exception as e:
                logger.error(f"Dense retrieval error ({e}). Using empty fallback.")
                return [], (time.time() - t0) * 1000

        async def fetch_sparse():
            t0 = time.time()
            try:
                if inspect.iscoroutinefunction(self.sparse_retriever.search):
                    res = await self.sparse_retriever.search(query, limit=sparse_top_k, language=language, strategy=strategy)
                else:
                    res = await asyncio.to_thread(self.sparse_retriever.search, query, limit=sparse_top_k, language=language, strategy=strategy)
                return res, (time.time() - t0) * 1000
            except Exception as e:
                logger.error(f"Sparse retrieval error ({e}). Using empty fallback.")
                return [], (time.time() - t0) * 1000

        # Execute both retrieval streams in parallel using asyncio.gather
        (dense_res, t_dense), (sparse_res, t_sparse) = await asyncio.gather(
            fetch_dense(),
            fetch_sparse()
        )

        return dense_res, sparse_res, t_dense, t_sparse

    async def generate_with_harness(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        language: str,
        latency_sensitive: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Harness-controlled LLM generation with circuit breaker, exponential retries,
        and deprecation model fallbacks.
        """
        if not retrieved_chunks:
            return "NOT_SUPPORTED", []

        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker is OPEN. Fast-failing to safe fallback.")
            text = retrieved_chunks[0]["payload"]["text"]
            return f"{text[:150]}... [1]", [retrieved_chunks[0].get("chunk_id", "chunk_0")]

        # Select provider and model from Router
        provider, model_name = self.router.select_model(query, language, latency_sensitive=latency_sensitive)

        # Retry loop with exponential backoff and model fallbacks
        models_to_try = [model_name]
        if hasattr(self.router, "get_fallback_models"):
            models_to_try.extend(self.router.get_fallback_models(model_name))

        last_error = None
        for current_model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info(f"Harness attempting generation with model '{current_model}' (Attempt {attempt+1})...")
                    answer, citations = await provider.generate_answer(
                        query=query, 
                        retrieved_chunks=retrieved_chunks, 
                        model_name=current_model
                    )
                    self.circuit_breaker.record_success()
                    return answer, citations

                except Exception as e:
                    last_error = e
                    logger.warning(f"Generation failed on '{current_model}' attempt {attempt+1}: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.2 * (2 ** attempt))

            logger.warning(f"Exhausted retries for model '{current_model}'. Shifting to deprecation fallback model...")

        self.circuit_breaker.record_failure()
        logger.error(f"All LLM generation retries & fallbacks failed: {last_error}")
        text = retrieved_chunks[0]["payload"]["text"]
        return f"{text[:150]}... [1]", [retrieved_chunks[0].get("chunk_id", "chunk_0")]
