import logging
import hashlib
from typing import List, Dict, Any

logger = logging.getLogger("RAG.retrieval.hybrid")
logger.setLevel(logging.INFO)

def get_text_hash(text: str) -> str:
    """Helper to compute stable text hash for dedup."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse search results using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1 / (k + rank)) for each rank list.
    """
    rrf_scores: Dict[str, float] = {}
    candidates: Dict[str, Dict[str, Any]] = {}
    
    # Process dense results
    for rank, item in enumerate(dense_results):
        text = item["payload"]["text"]
        text_hash = get_text_hash(text)
        
        if text_hash not in rrf_scores:
            rrf_scores[text_hash] = 0.0
            candidates[text_hash] = item
            
        # Add reciprocal rank score (rank is 0-indexed, so add 1)
        rrf_scores[text_hash] += 1.0 / (k + (rank + 1))
        
    # Process sparse results
    for rank, item in enumerate(sparse_results):
        text = item["payload"]["text"]
        text_hash = get_text_hash(text)
        
        if text_hash not in rrf_scores:
            rrf_scores[text_hash] = 0.0
            candidates[text_hash] = item
            
        # Add reciprocal rank score (rank is 0-indexed, so add 1)
        rrf_scores[text_hash] += 1.0 / (k + (rank + 1))
        
    # Sort candidates by their RRF score
    sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
    
    fused_results = []
    for h in sorted_hashes[:limit]:
        item = candidates[h]
        # Store metadata about RRF score
        item["rrf_score"] = rrf_scores[h]
        fused_results.append(item)
        
    logger.info(f"Fused {len(dense_results)} dense and {len(sparse_results)} sparse candidates down to {len(fused_results)} fused candidates.")
    return fused_results
