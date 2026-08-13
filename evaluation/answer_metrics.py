import re
import numpy as np
from typing import List, Dict, Any, Optional

def calculate_token_f1(generated: str, reference: str) -> float:
    """
    Computes word-level token alignment F1-score.
    """
    gen_tokens = re.findall(r"\w+", generated.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    
    if not gen_tokens or not ref_tokens:
        return 0.0
        
    gen_set = set(gen_tokens)
    ref_set = set(ref_tokens)
    
    intersection = gen_set.intersection(ref_set)
    if not intersection:
        return 0.0
        
    precision = len(intersection) / len(gen_set)
    recall = len(intersection) / len(ref_set)
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_faithfulness(generated: str, contexts: List[str]) -> float:
    """
    Measures faithfulness (groundedness) of the answer relative to the retrieved contexts.
    Checks what percentage of content terms in the generated answer appear in the context.
    """
    combined_context = " ".join(contexts).lower()
    gen_tokens = re.findall(r"\w+", generated.lower())
    
    # Remove short/stop words for relevance
    content_tokens = [t for t in gen_tokens if len(t) > 2]
    if not content_tokens:
        return 1.0 # Vacuously faithful if empty/short
        
    hits = sum(1 for t in content_tokens if t in combined_context)
    return hits / len(content_tokens)

def evaluate_generation(
    generated: str, 
    reference: str, 
    contexts: List[str], 
    embedding_model = None
) -> Dict[str, float]:
    """
    Returns correctness and faithfulness scores.
    """
    f1 = calculate_token_f1(generated, reference)
    faith = calculate_faithfulness(generated, contexts)
    
    semantic_sim = 0.0
    if embedding_model and generated and reference:
        try:
            embs = embedding_model.embed_documents([generated, reference])
            norm1 = embs[0] / (np.linalg.norm(embs[0]) or 1.0)
            norm2 = embs[1] / (np.linalg.norm(embs[1]) or 1.0)
            semantic_sim = float(np.dot(norm1, norm2))
        except Exception:
            semantic_sim = f1  # Fallback to word F1 if embedding fails

    return {
        "token_f1": f1,
        "faithfulness": faith,
        "semantic_correctness": semantic_sim,
        # Weighted overall answer quality score
        "answer_score": (semantic_sim * 0.7) + (faith * 0.3)
    }
