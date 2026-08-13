import time
import json
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator, AsyncIterator

from backend import config
from backend.harness import PipelineHarness, PipelineResult, SourceCitation, DetailedLatency
from backend.cache.semantic_cache import SemanticLSHCache
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.guardrails.safety import check_query_safety
from backend.guardrails.grounding import GroundingChecker
from backend.observability.metrics import MetricsTracker

logger = logging.getLogger("RAG.pipeline")

class AsyncRAGPipeline:
    """
    Async Parallel Pipeline Orchestrator for Sub-200ms Retrieval and Real-time Voice/Audio Streaming.
    Features:
    - Semantic LSH Cache lookup (~2ms)
    - Async Parallel Dense + Sparse Retrieval (via asyncio.gather)
    - Harness-controlled LLM execution with Circuit Breakers & Model Router
    - Direct ElevenLabs TTS Streaming from LLM token generator for low Time-To-First-Audio (TTFA)
    """
    def __init__(
        self,
        harness: PipelineHarness,
        embedder: Any,
        reranker: Any,
        grounding_checker: GroundingChecker,
        semantic_cache: SemanticLSHCache,
        metrics_tracker: MetricsTracker,
        stt_client: Optional[Any] = None,
        tts_client: Optional[Any] = None
    ):
        self.harness = harness
        self.embedder = embedder
        self.reranker = reranker
        self.grounding_checker = grounding_checker
        self.semantic_cache = semantic_cache
        self.metrics_tracker = metrics_tracker
        self.stt_client = stt_client
        self.tts_client = tts_client

    async def process_text_query(self, query: str, language: str = "hi", latency_sensitive: bool = False) -> PipelineResult:
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        query = query.strip()
        language = language.strip()

        t_start = time.time()
        latencies = DetailedLatency()

        # 1. Safety Check
        is_safe, safety_status = check_query_safety(query)
        if not is_safe:
            logger.warning(f"[{request_id}] Query blocked by safety guardrail: {safety_status}")
            latencies.total_ms = (time.time() - t_start) * 1000
            return PipelineResult(
                request_id=request_id,
                query=query,
                answer="blocked_by_safety",
                sources=[],
                language=language,
                grounded=False,
                confidence=0.0,
                status=safety_status,
                cached=False,
                latency=latencies,
                pipeline_steps={"dense_candidates": 0, "sparse_candidates": 0, "fused_candidates": 0, "reranked_candidates": 0}
            )

        # 2. Asynchronous Query Embedding
        t_emb_0 = time.time()
        try:
            query_vector = await asyncio.to_thread(self.embedder.embed_queries, [query])
            query_vec = query_vector[0]
        except Exception as e:
            logger.error(f"[{request_id}] Embedding generation failed: {e}")
            query_vec = None
        latencies.embedding_ms = (time.time() - t_emb_0) * 1000

        # 3. Check Semantic LSH Cache
        if query_vec is not None:
            cached_match = self.semantic_cache.find_semantic_match(query_vec, language=language)
            if cached_match:
                cached_json, sim_score = cached_match
                try:
                    res_dict = json.loads(cached_json)
                    total_t = (time.time() - t_start) * 1000
                    res_dict["request_id"] = request_id
                    res_dict["cached"] = True
                    res_dict["latency"]["total_ms"] = total_t
                    
                    self.metrics_tracker.log_request(total_t, is_grounded=True, cache_hit=True)
                    return PipelineResult(**res_dict)
                except Exception as e:
                    logger.error(f"Failed to parse cached payload: {e}")

        # 4. Async Parallel Dense & Sparse Retrieval
        t_ret_0 = time.time()
        dense_res, sparse_res, t_dense, t_sparse = await self.harness.execute_retrieval_with_fallbacks(
            query=query,
            query_vector=query_vec,
            language=language,
            dense_top_k=config.DENSE_TOP_K,
            sparse_top_k=config.SPARSE_TOP_K
        )
        latencies.dense_ms = t_dense
        latencies.sparse_ms = t_sparse

        # 5. RRF Fusion & Reranking
        t_fusion_0 = time.time()
        fused_res = reciprocal_rank_fusion(dense_res, sparse_res, k=config.RRF_CONSTANT, limit=config.FUSED_TOP_K)
        latencies.fusion_ms = (time.time() - t_fusion_0) * 1000

        t_rerank_0 = time.time()
        reranked_res = await asyncio.to_thread(self.reranker.rerank, query, fused_res, config.FINAL_TOP_K)
        latencies.reranking_ms = (time.time() - t_rerank_0) * 1000

        # Measure Retrieval Sub-200ms Target Latency!
        retrieval_time = (time.time() - t_ret_0) * 1000
        latencies.retrieval_total_ms = retrieval_time
        logger.info(f"[{request_id}] RETRIEVAL STAGE COMPLETED IN {retrieval_time:.2f} ms (Target: < 200 ms)")

        # Relevance Guardrail check
        max_relevance = max([c.get("rerank_score", 0.0) for c in reranked_res]) if reranked_res else 0.0
        effective_threshold = 0.0 if config.EMBEDDING_MODE == "mock" else config.RELEVANCE_THRESHOLD

        if max_relevance < effective_threshold:
            logger.info(f"[{request_id}] Low relevance ({max_relevance:.2f}). Refusing query.")
            total_time = (time.time() - t_start) * 1000
            latencies.total_ms = total_time
            
            result = PipelineResult(
                request_id=request_id,
                query=query,
                answer="NOT_SUPPORTED",
                sources=[],
                language=language,
                grounded=True,
                confidence=1.0,
                status="OUT_OF_SCOPE",
                cached=False,
                latency=latencies,
                pipeline_steps={
                    "dense_candidates": len(dense_res),
                    "sparse_candidates": len(sparse_res),
                    "fused_candidates": len(fused_res),
                    "reranked_candidates": 0
                }
            )
            self.metrics_tracker.log_request(total_time, is_grounded=True, cache_hit=False)
            return result

        # 6. Harness-orchestrated LLM Generation
        t_gen_0 = time.time()
        answer, cited_chunk_ids = await self.harness.generate_with_harness(
            query=query,
            retrieved_chunks=reranked_res,
            language=language,
            latency_sensitive=latency_sensitive
        )
        latencies.generation_ms = (time.time() - t_gen_0) * 1000

        # 7. Grounding Verification
        t_ground_0 = time.time()
        provider, model_name = self.harness.router.select_model(query, language, latency_sensitive=latency_sensitive)
        async def judge_fn(ans, ctx):
            return await provider.verify_grounding(ans, ctx, model_name=model_name)

        is_grounded, grounding_score, grounding_details = await self.grounding_checker.verify_grounding(
            query=query,
            answer=answer,
            retrieved_chunks=reranked_res,
            llm_judge_fn=judge_fn
        )
        latencies.grounding_ms = (time.time() - t_ground_0) * 1000

        # Total Pipeline Latency
        total_time = (time.time() - t_start) * 1000
        latencies.total_ms = total_time

        sources = [
            SourceCitation(
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                text=c["payload"]["text"],
                score=c.get("rerank_score", c.get("score", 0.0)),
                metadata=c["payload"].get("metadata", {})
            )
            for i, c in enumerate(reranked_res)
        ]

        result = PipelineResult(
            request_id=request_id,
            query=query,
            answer=answer if is_grounded else "NOT_SUPPORTED",
            sources=sources,
            language=language,
            grounded=is_grounded,
            confidence=grounding_score,
            status="SUCCESS" if is_grounded else "GROUNDING_VIOLATION",
            cached=False,
            latency=latencies,
            pipeline_steps={
                "dense_candidates": len(dense_res),
                "sparse_candidates": len(sparse_res),
                "fused_candidates": len(fused_res),
                "reranked_candidates": len(reranked_res)
            },
            grounding_details=grounding_details
        )

        if is_grounded and query_vec is not None:
            try:
                res_json = result.json()
                self.semantic_cache.set_semantic(query, query_vec, language, res_json, ttl=3600)
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")

        self.metrics_tracker.log_request(total_time, is_grounded=is_grounded, cache_hit=False)
        return result

    async def process_text_query_sse(
        self,
        query: str,
        language: str = "hi",
        latency_sensitive: bool = False
    ) -> AsyncIterator[str]:
        """
        Yields Server-Sent Events (SSE) for real-time pipeline tracing and token streaming.
        Event format: "data: <json>\n\n"
        Event types:
          - {type: "stage", stage: "embedding", latency_ms: X}
          - {type: "stage", stage: "retrieval", latency_ms: X, ...}
          - {type: "token", text: "..."}
          - {type: "done", latency: {...}, status: "...", grounded: bool, confidence: float, sources: [...]}
          - {type: "error", message: "..."}
        """
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        request_id = f"req_{uuid.uuid4().hex[:8]}"
        query = query.strip()
        language = language.strip()
        t_start = time.time()
        latencies = DetailedLatency()

        # 1. Safety Check
        is_safe, safety_status = check_query_safety(query)
        if not is_safe:
            yield sse({"type": "done", "status": safety_status, "answer": "blocked_by_safety",
                       "grounded": False, "confidence": 0.0, "sources": [],
                       "latency": {"total_ms": (time.time() - t_start) * 1000},
                       "pipeline_steps": {}})
            return

        # 2. Embedding
        t_emb = time.time()
        try:
            query_vector = await asyncio.to_thread(self.embedder.embed_queries, [query])
            query_vec = query_vector[0]
        except Exception as e:
            logger.error(f"[{request_id}] Embedding failed: {e}")
            query_vec = None
        latencies.embedding_ms = (time.time() - t_emb) * 1000
        yield sse({"type": "stage", "stage": "embedding", "latency_ms": round(latencies.embedding_ms, 2)})

        # 3. Semantic Cache Check
        if query_vec is not None:
            cached_match = self.semantic_cache.find_semantic_match(query_vec, language=language)
            if cached_match:
                cached_json, sim_score = cached_match
                try:
                    res_dict = json.loads(cached_json)
                    total_t = (time.time() - t_start) * 1000
                    res_dict["request_id"] = request_id
                    res_dict["cached"] = True
                    res_dict["latency"]["total_ms"] = total_t
                    self.metrics_tracker.log_request(total_t, is_grounded=True, cache_hit=True)
                    yield sse({"type": "cache_hit", "latency_ms": round(total_t, 2)})
                    # Emit the cached answer as token for streaming effect
                    cached_answer = res_dict.get("answer", "")
                    yield sse({"type": "token", "text": cached_answer})
                    yield sse({"type": "done", **res_dict})
                    return
                except Exception as e:
                    logger.error(f"Cache parse error: {e}")

        # 4. Retrieval
        t_ret = time.time()
        dense_res, sparse_res, t_dense, t_sparse = await self.harness.execute_retrieval_with_fallbacks(
            query=query, query_vector=query_vec, language=language,
            dense_top_k=config.DENSE_TOP_K, sparse_top_k=config.SPARSE_TOP_K
        )
        latencies.dense_ms = t_dense
        latencies.sparse_ms = t_sparse
        yield sse({"type": "stage", "stage": "dense_retrieval", "latency_ms": round(t_dense, 2), "candidates": len(dense_res)})
        yield sse({"type": "stage", "stage": "sparse_retrieval", "latency_ms": round(t_sparse, 2), "candidates": len(sparse_res)})

        # 5. Fusion + Reranking
        t_fuse = time.time()
        fused_res = reciprocal_rank_fusion(dense_res, sparse_res, k=config.RRF_CONSTANT, limit=config.FUSED_TOP_K)
        latencies.fusion_ms = (time.time() - t_fuse) * 1000
        yield sse({"type": "stage", "stage": "fusion", "latency_ms": round(latencies.fusion_ms, 2), "candidates": len(fused_res)})

        t_rerank = time.time()
        reranked_res = await asyncio.to_thread(self.reranker.rerank, query, fused_res, config.FINAL_TOP_K)
        latencies.reranking_ms = (time.time() - t_rerank) * 1000
        latencies.retrieval_total_ms = (time.time() - t_ret) * 1000
        yield sse({"type": "stage", "stage": "reranking", "latency_ms": round(latencies.reranking_ms, 2), "candidates": len(reranked_res)})

        # Relevance guardrail
        max_relevance = max([c.get("rerank_score", 0.0) for c in reranked_res]) if reranked_res else 0.0
        effective_threshold = 0.0 if config.EMBEDDING_MODE == "mock" else config.RELEVANCE_THRESHOLD
        if max_relevance < effective_threshold:
            total_time = (time.time() - t_start) * 1000
            latencies.total_ms = total_time
            self.metrics_tracker.log_request(total_time, is_grounded=True, cache_hit=False)
            yield sse({"type": "done", "status": "OUT_OF_SCOPE", "answer": "NOT_SUPPORTED",
                       "grounded": True, "confidence": 1.0, "sources": [],
                       "latency": latencies.__dict__,
                       "pipeline_steps": {"dense_candidates": len(dense_res), "sparse_candidates": len(sparse_res),
                                          "fused_candidates": len(fused_res), "reranked_candidates": 0}})
            return

        # 6. LLM Streaming Generation (token-by-token)
        yield sse({"type": "stage", "stage": "generation_start", "latency_ms": 0})
        t_gen = time.time()
        provider, model_name = self.harness.router.select_model(query, language, latency_sensitive=latency_sensitive)
        full_answer = ""
        try:
            token_stream = provider.generate_answer_stream(query, reranked_res, model_name=model_name)
            async for token in token_stream:
                full_answer += token
                yield sse({"type": "token", "text": token})
        except Exception as e:
            logger.error(f"[{request_id}] Streaming generation error: {e}")
            full_answer = "NOT_SUPPORTED"
        latencies.generation_ms = (time.time() - t_gen) * 1000
        yield sse({"type": "stage", "stage": "generation", "latency_ms": round(latencies.generation_ms, 2)})

        # 7. Grounding
        t_ground = time.time()
        async def judge_fn(ans, ctx):
            return await provider.verify_grounding(ans, ctx, model_name=model_name)
        is_grounded, grounding_score, grounding_details = await self.grounding_checker.verify_grounding(
            query=query, answer=full_answer, retrieved_chunks=reranked_res, llm_judge_fn=judge_fn
        )
        latencies.grounding_ms = (time.time() - t_ground) * 1000
        yield sse({"type": "stage", "stage": "grounding", "latency_ms": round(latencies.grounding_ms, 2),
                   "grounded": is_grounded, "confidence": round(grounding_score, 3)})

        total_time = (time.time() - t_start) * 1000
        latencies.total_ms = total_time

        sources = [
            SourceCitation(
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                text=c["payload"]["text"],
                score=c.get("rerank_score", c.get("score", 0.0)),
                metadata=c["payload"].get("metadata", {})
            ).dict()
            for i, c in enumerate(reranked_res)
        ]

        final_answer = full_answer if is_grounded else "NOT_SUPPORTED"
        self.metrics_tracker.log_request(total_time, is_grounded=is_grounded, cache_hit=False)

        # Cache if grounded
        if is_grounded and query_vec is not None and final_answer != "NOT_SUPPORTED":
            try:
                result = PipelineResult(
                    request_id=request_id, query=query, answer=final_answer, sources=[
                        SourceCitation(**s) for s in sources
                    ], language=language, grounded=is_grounded, confidence=grounding_score,
                    status="SUCCESS", cached=False, latency=latencies,
                    pipeline_steps={"dense_candidates": len(dense_res), "sparse_candidates": len(sparse_res),
                                    "fused_candidates": len(fused_res), "reranked_candidates": len(reranked_res)},
                    grounding_details=grounding_details
                )
                self.semantic_cache.set_semantic(query, query_vec, language, result.json(), ttl=3600)
            except Exception as e:
                logger.error(f"Cache write error: {e}")

        yield sse({
            "type": "done",
            "request_id": request_id,
            "status": "SUCCESS" if is_grounded else "GROUNDING_VIOLATION",
            "answer": final_answer,
            "grounded": is_grounded,
            "confidence": round(grounding_score, 3),
            "sources": sources,
            "latency": latencies.__dict__,
            "pipeline_steps": {
                "dense_candidates": len(dense_res),
                "sparse_candidates": len(sparse_res),
                "fused_candidates": len(fused_res),
                "reranked_candidates": len(reranked_res)
            },
            "grounding_details": grounding_details
        })

    async def process_voice_query(self, audio_bytes: bytes, language: str = "hi") -> PipelineResult:
        t_stt_0 = time.time()
        if not self.stt_client:
            raise RuntimeError("Speech-to-Text client is not initialized.")

        transcription = await self.stt_client.transcribe(audio_bytes, language)
        stt_latency = (time.time() - t_stt_0) * 1000

        if not transcription:
            raise ValueError("Voice transcription returned empty text.")

        result = await self.process_text_query(transcription, language=language)
        result.transcription = transcription
        result.latency.stt_ms = stt_latency
        result.latency.total_ms += stt_latency
        return result

    async def process_text_audio_stream(
        self, 
        query: str, 
        language: str = "hi", 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ) -> AsyncGenerator[bytes, None]:
        """
        Pipes streaming LLM answer tokens into ElevenLabs TTS Stream for real-time audio output.
        Yields audio/mpeg bytes chunks as soon as first phrase is synthesized.
        """
        if not self.tts_client:
            raise RuntimeError("Text-To-Speech client is not initialized.")

        # 1. Embed & Retrieve Context in Sub-200ms
        query_vector = await asyncio.to_thread(self.embedder.embed_queries, [query])
        query_vec = query_vector[0] if query_vector else None

        dense_res, sparse_res, _, _ = await self.harness.execute_retrieval_with_fallbacks(
            query=query, query_vector=query_vec, language=language,
            dense_top_k=config.DENSE_TOP_K, sparse_top_k=config.SPARSE_TOP_K
        )
        fused_res = reciprocal_rank_fusion(dense_res, sparse_res, k=config.RRF_CONSTANT, limit=config.FUSED_TOP_K)
        reranked_res = await asyncio.to_thread(self.reranker.rerank, query, fused_res, config.FINAL_TOP_K)

        if not reranked_res:
            async for chunk in self.tts_client.stream_tts("Sufficient evidence was not found to answer your question.", voice_id=voice_id):
                yield chunk
            return

        # 2. Get Provider & Model from Router
        provider, model_name = self.harness.router.select_model(query, language, latency_sensitive=True)

        # 3. Stream LLM tokens directly into ElevenLabs TTS Audio Stream!
        token_stream = provider.generate_answer_stream(query, reranked_res, model_name=model_name)
        async for audio_chunk in self.tts_client.stream_tts_from_tokens(token_stream, voice_id=voice_id):
            yield audio_chunk

    async def process_voice_audio_stream(
        self, 
        audio_bytes: bytes, 
        language: str = "hi", 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ) -> AsyncGenerator[bytes, None]:
        """
        Full End-to-End Voice Input -> STT -> Vector Retrieval -> LLM Stream -> TTS Audio Stream!
        """
        if not self.stt_client:
            raise RuntimeError("STT client is not initialized.")

        transcription = await self.stt_client.transcribe(audio_bytes, language)
        if not transcription:
            async for chunk in self.tts_client.stream_tts("Sorry, voice transcription failed.", voice_id=voice_id):
                yield chunk
            return

        async for audio_chunk in self.process_text_audio_stream(transcription, language=language, voice_id=voice_id):
            yield audio_chunk
