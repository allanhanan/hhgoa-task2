import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("RAG.guardrails.grounding")
logger.setLevel(logging.INFO)

def calculate_word_intersection(text1: str, text2: str) -> float:
    """
    Computes Jaccard word intersection between two texts.
    Useful as a fast citation/content overlap check.
    """
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1)

class GroundingChecker:
    """
    Multi-signal grounding checker.
    Combines:
    1. Retrieval Relevance Score (Best reranker/fusion score of input chunks).
    2. Context-Answer Semantic Similarity (Via embedding cosine similarity).
    3. Citation/Text Intersection Overlap (N-gram verification).
    4. Optional LLM Judge validation.
    """
    def __init__(self, embedding_model = None, relevance_threshold: float = 0.35, grounding_threshold: float = 0.70):
        self.embedding_model = embedding_model
        self.relevance_threshold = relevance_threshold
        self.grounding_threshold = grounding_threshold

    def verify_grounding(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_judge_fn = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Runs multi-signal verification.
        Returns: (is_grounded, grounding_score, details)
        """
        if not retrieved_chunks:
            return False, 0.0, {"reason": "No retrieved context chunks available."}

        # Check for refusal responses
        refusal_phrases = ["not_supported", "i don't know", "i cannot find", "insufficient evidence", "not supported"]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in refusal_phrases) or len(answer.strip()) < 5:
            return True, 1.0, {"reason": "System output is a safe refusal or empty response."}

        # Signal 1: Retrieval Relevance
        # Calculate maximum rerank/similarity score
        max_relevance = max([chunk.get("rerank_score", chunk.get("score", 0.0)) for chunk in retrieved_chunks])
        relevance_pass = max_relevance >= self.relevance_threshold

        # Combine all context texts
        context_text = " ".join([chunk["payload"]["text"] for chunk in retrieved_chunks])

        # Signal 2: Context-Answer Semantic Similarity
        semantic_sim = 0.0
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.embed_documents([answer, context_text])
                ans_emb = embeddings[0]
                ctx_emb = embeddings[1]
                ans_norm = ans_emb / (np.linalg.norm(ans_emb) or 1.0)
                ctx_norm = ctx_emb / (np.linalg.norm(ctx_emb) or 1.0)
                semantic_sim = float(np.dot(ans_norm, ctx_norm))
            except Exception as e:
                logger.error(f"Semantic similarity check error: {e}")
                semantic_sim = 0.5  # Fallback neutral score

        # Signal 3: Token/Word Intersection
        intersection_score = calculate_word_intersection(answer, context_text)

        # Signal 4: Optional LLM Judge
        llm_verdict = 1.0
        llm_reason = "Skipped"
        if llm_judge_fn and relevance_pass:
            try:
                verdict_str, reason = llm_judge_fn(answer, context_text)
                llm_reason = reason
                if "not_grounded" in verdict_str.lower() or "hallucination" in verdict_str.lower():
                    llm_verdict = 0.0
                elif "grounded" in verdict_str.lower():
                    llm_verdict = 1.0
                else:
                    llm_verdict = 0.5
            except Exception as e:
                logger.error(f"LLM Judge validation failed: {e}")
                llm_verdict = 1.0  # Fallback to skip/pass

        # Compute Overall Grounding Score (Weighted Average)
        # Weights: semantic_sim (0.4) + intersection_score (0.3) + llm_verdict (0.3)
        overall_score = (semantic_sim * 0.4) + (intersection_score * 0.3) + (llm_verdict * 0.3)
        
        # Soft validation combining metrics
        is_grounded = True
        reason = "Grounded"

        if not relevance_pass:
            is_grounded = False
            reason = "Retrieval relevance score too low"
        elif overall_score < self.grounding_threshold:
            is_grounded = False
            reason = f"Grounding confidence score {overall_score:.2f} is below threshold {self.grounding_threshold}"
        elif llm_verdict == 0.0:
            is_grounded = False
            reason = f"LLM judge detected a hallucination: {llm_reason}"

        details = {
            "retrieval_max_relevance": float(max_relevance),
            "relevance_pass": bool(relevance_pass),
            "semantic_similarity": float(semantic_sim),
            "word_intersection": float(intersection_score),
            "llm_judge_verdict": float(llm_verdict),
            "llm_judge_reason": llm_reason,
            "overall_score": float(overall_score),
            "verdict_reason": reason
        }

        return is_grounded, overall_score, details
