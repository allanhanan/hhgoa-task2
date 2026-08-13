import os
import time
import logging
import numpy as np
from typing import List, Dict, Any

# Ensure parent directory imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.embeddings.model import EmbeddingModel
from backend.retrieval.dense import QdrantRepository
from backend.retrieval.sparse import BM25SparseRetriever
from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.reranker import Reranker
from evaluation.retrieval_metrics import evaluate_retrieval

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("RAG.evaluation")

def run_benchmark(langs: List[str] = ["hi"], limit: int = 50):
    """
    Runs benchmarking over a subset of the dataset.
    Compares: Dense, Sparse, Hybrid, and Reranked.
    Reports: Recall@1, Recall@5, MRR, NDCG@5, and Latency.
    """
    print("=" * 60)
    print("           RAG RETRIEVAL BENCHMARK SUITE             ")
    print("=" * 60)
    
    # 1. Initialize models
    embedder = EmbeddingModel(mode=config.EMBEDDING_MODE, model_name=config.EMBEDDING_MODEL_NAME, dim=config.EMBEDDING_DIM)
    
    qdrant_repo = QdrantRepository(
        path=str(config.QDRANT_PATH),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        vector_dim=config.EMBEDDING_DIM
    )
    
    sparse_retriever = BM25SparseRetriever()
    if not sparse_retriever.load(str(config.BM25_PATH)):
        print("Error: Sparse index not found. Run ingestion/build_index.py first.")
        return

    reranker = Reranker(embedding_model=embedder)

    # 2. Extract test queries from HuggingFace dataset
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets package is not installed.")
        return
        
    test_cases = []
    print(f"Loading evaluation test queries from HF datasets ({langs})...")
    
    for lang in langs:
        try:
            dataset = load_dataset(config.DATASET_NAME, lang, split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue

        for idx, item in enumerate(dataset):
            if len(test_cases) >= limit:
                break
            
            query = item.get("query")
            doc_id = f"msmarco_{lang}_{idx}"
            
            if query:
                test_cases.append({
                    "query": query,
                    "ground_truth_doc_id": doc_id,
                    "language": lang
                })
                
    if not test_cases:
        print("No evaluation test cases found. Aborting.")
        return

    print(f"Loaded {len(test_cases)} evaluation test cases. Commencing retrieval...")
    
    # Metrics aggregators
    strategies = ["dense", "sparse", "hybrid", "reranked"]
    metrics_by_strat = {s: [] for s in strategies}
    latencies_by_strat = {s: [] for s in strategies}
    
    for idx, tc in enumerate(test_cases):
        query = tc["query"]
        gt_doc_id = tc["ground_truth_doc_id"]
        lang = tc["language"]
        
        # --- Dense Search ---
        t0 = time.time()
        q_emb = embedder.embed_queries([query])[0]
        dense_res = qdrant_repo.search(q_emb, limit=config.DENSE_TOP_K, language=lang)
        t_dense = (time.time() - t0) * 1000
        latencies_by_strat["dense"].append(t_dense)
        metrics_by_strat["dense"].append(evaluate_retrieval(dense_res, [gt_doc_id]))
        
        # --- Sparse Search ---
        t0 = time.time()
        sparse_res = sparse_retriever.search(query, limit=config.SPARSE_TOP_K, language=lang)
        t_sparse = (time.time() - t0) * 1000
        latencies_by_strat["sparse"].append(t_sparse)
        metrics_by_strat["sparse"].append(evaluate_retrieval(sparse_res, [gt_doc_id]))
        
        # --- Hybrid (RRF) ---
        t0 = time.time()
        hybrid_res = reciprocal_rank_fusion(dense_res, sparse_res, k=config.RRF_CONSTANT, limit=config.FUSED_TOP_K)
        t_hybrid = (time.time() - t0) * 1000 + (t_dense + t_sparse) # Combine retriever latencies
        latencies_by_strat["hybrid"].append(t_hybrid)
        metrics_by_strat["hybrid"].append(evaluate_retrieval(hybrid_res, [gt_doc_id]))
        
        # --- Reranked ---
        t0 = time.time()
        reranked_res = reranker.rerank(query, hybrid_res, limit=config.FINAL_TOP_K)
        t_rerank = (time.time() - t0) * 1000 + t_hybrid
        latencies_by_strat["reranked"].append(t_rerank)
        metrics_by_strat["reranked"].append(evaluate_retrieval(reranked_res, [gt_doc_id]))

    # 3. Report Results
    print("\n" + "=" * 60)
    print("                 BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<12} | {'Recall@1':<8} | {'Recall@5':<8} | {'MRR':<8} | {'NDCG@5':<8} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
    print("-" * 60)
    
    for s in strategies:
        recall_1 = np.mean([m["recall_1"] for m in metrics_by_strat[s]])
        recall_5 = np.mean([m["recall_5"] for m in metrics_by_strat[s]])
        mrr = np.mean([m["mrr"] for m in metrics_by_strat[s]])
        ndcg_5 = np.mean([m["ndcg_5"] for m in metrics_by_strat[s]])
        
        latencies = latencies_by_strat[s]
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        print(f"{s.capitalize():<12} | {recall_1:<8.3f} | {recall_5:<8.3f} | {mrr:<8.3f} | {ndcg_5:<8.3f} | {p50:<8.1f} | {p95:<8.1f}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate retrieval models.")
    parser.add_argument("--langs", nargs="+", default=["hi"], help="Languages to evaluate.")
    parser.add_argument("--limit", type=int, default=50, help="Number of queries to run evaluation on.")
    
    args = parser.parse_args()
    run_benchmark(langs=args.langs, limit=args.limit)
