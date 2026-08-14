import asyncio
import time
import sys
import os

# Inject parent directory into path to support running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import container, startup_event, shutdown_event
from backend.harness import PipelineHarness
from backend import config

async def run_benchmark():
    await startup_event()
    print("Startup complete.")

    try:
        # Check index composition
        dense_retriever = container["dense_retriever"]
        sparse_retriever = container["sparse_retriever"]
        
        q_count = 0
        if dense_retriever.client:
            try:
                col_info = dense_retriever.client.get_collection(dense_retriever.collection_name)
                q_count = col_info.points_count or 0
                print(f"Qdrant vector count: {q_count}")
            except Exception as e:
                print(f"Qdrant error: {e}")
                
        b_count = len(getattr(sparse_retriever, 'chunks', []))
        print(f"BM25 document count: {b_count}")
        
        # We can analyze chunks by language and strategy from BM25
        lang_counts = {}
        strat_counts = {}
        for c in getattr(sparse_retriever, 'chunks', []):
            meta = c.get("metadata", {})
            l = meta.get("language", "hi")
            s = meta.get("strategy", "semantic")
            lang_counts[l] = lang_counts.get(l, 0) + 1
            strat_counts[s] = strat_counts.get(s, 0) + 1
            
        print("BM25 Index Composition:")
        print(" Languages:", lang_counts)
        print(" Strategies:", strat_counts)

        # Profile harness
        harness = container["harness"]
        embedder = container["embedder"]
        reranker = container["reranker"]

        query = "भारत की राजधानी क्या है?"
        lang = "hi"
        
        # 1. Embedding
        t0 = time.time()
        query_vector = embedder.embed_queries([query])[0]
        t1 = time.time()
        print(f"Embedding time: {(t1-t0)*1000:.2f} ms")

        # 2. Retrieval
        dense_res, sparse_res, t_dense, t_sparse = await harness.execute_retrieval_with_fallbacks(
            query=query, 
            query_vector=query_vector, 
            language=lang,
            dense_top_k=20,
            sparse_top_k=20
        )
        print(f"Dense time: {t_dense:.2f} ms | Candidates: {len(dense_res)}")
        print(f"Sparse time: {t_sparse:.2f} ms | Candidates: {len(sparse_res)}")
        
        # 3. Fusion
        from backend.retrieval.hybrid import reciprocal_rank_fusion
        t0 = time.time()
        fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60, limit=15)
        t1 = time.time()
        print(f"Fusion time: {(t1-t0)*1000:.2f} ms | Candidates: {len(fused)}")

        # 4. Reranking
        t0 = time.time()
        reranked = await asyncio.to_thread(reranker.rerank, query, fused, 3)
        t1 = time.time()
        print(f"Reranking time: {(t1-t0)*1000:.2f} ms | Candidates: {len(reranked)}")
        
        if reranked:
            print("\nTop 1 result:")
            print("Score:", reranked[0].get("rerank_score", "N/A"))
            print("Text:", reranked[0]["payload"]["text"][:100] + "...")
            print("Language:", reranked[0]["payload"].get("language"))
            
    finally:
        await shutdown_event()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
