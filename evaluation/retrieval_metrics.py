import numpy as np
from typing import List, Dict, Any, Set

def calculate_recall_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K.
    Recall = (Relevant retrieved items up to rank K) / (Total relevant items)
    """
    if not ground_truth_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_doc_ids[:k])
    relevant_retrieved = retrieved_at_k.intersection(ground_truth_ids)
    return len(relevant_retrieved) / len(ground_truth_ids)

def calculate_mrr(retrieved_doc_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).
    MRR = 1 / rank of the first relevant document.
    """
    if not ground_truth_ids:
        return 0.0
        
    for rank, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in ground_truth_ids:
            return 1.0 / (rank + 1)
            
    return 0.0

def calculate_ndcg_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum(rel_i / log2(i + 1))
    """
    if not ground_truth_ids:
        return 0.0
        
    retrieved_k = retrieved_doc_ids[:k]
    
    # Calculate DCG
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_k):
        if doc_id in ground_truth_ids:
            # Binary relevance (1 if relevant, 0 if not)
            dcg += 1.0 / np.log2(idx + 2) # idx is 0-based, so rank is idx + 1, index in log is rank + 1 = idx + 2
            
    # Calculate Ideal DCG (IDCG)
    idcg = 0.0
    total_relevant = min(len(ground_truth_ids), k)
    for idx in range(total_relevant):
        idcg += 1.0 / np.log2(idx + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def evaluate_retrieval(retrieved: List[Dict[str, Any]], ground_truth_ids: List[str]) -> Dict[str, float]:
    """
    Evaluates retrieval metrics given retrieved items and a ground truth document ID list.
    """
    retrieved_ids = [item["payload"]["document_id"] for item in retrieved]
    gt_set = set(ground_truth_ids)
    
    return {
        "recall_1": calculate_recall_at_k(retrieved_ids, gt_set, 1),
        "recall_5": calculate_recall_at_k(retrieved_ids, gt_set, 5),
        "recall_10": calculate_recall_at_k(retrieved_ids, gt_set, 10),
        "mrr": calculate_mrr(retrieved_ids, gt_set),
        "ndcg_5": calculate_ndcg_at_k(retrieved_ids, gt_set, 5),
        "ndcg_10": calculate_ndcg_at_k(retrieved_ids, gt_set, 10),
    }
