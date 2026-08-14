import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("RAG.retrieval.reranker")
logger.setLevel(logging.INFO)

class Reranker:
    """
    Reranks candidate documents.
    Supports two methods:
    1. Cross-Encoder reranking (using transformers CrossEncoder, if installed and loaded).
    2. Embedding Cosine Similarity (using the existing EmbeddingModel, highly efficient and multilingual).
    """
    def __init__(self, cross_encoder_model_name: Optional[str] = None, embedding_model = None):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        self.is_cross_encoder_active = False
        
        if cross_encoder_model_name:
            try:
                logger.info(f"Attempting to load Cross-Encoder: {cross_encoder_model_name}...")
                from sentence_transformers import CrossEncoder
                try:
                    self.cross_encoder = CrossEncoder(cross_encoder_model_name, local_files_only=True)
                except Exception:
                    self.cross_encoder = CrossEncoder(cross_encoder_model_name, local_files_only=False)
                self.is_cross_encoder_active = True
                logger.info("Cross-Encoder loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load Cross-Encoder ({e}). Falling back to embedding-based similarity reranking.")


    def rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int = 3, query_vector: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks against a query.
        """
        if not candidates:
            return []
            
        start_time = time.time()
        reranked = []
        
        if self.is_cross_encoder_active and self.cross_encoder:
            try:
                # Prepare inputs: list of pairs [query, text]
                pairs = [[query, item["payload"]["text"]] for item in candidates]
                scores = self.cross_encoder.predict(pairs)
                
                for item, score in zip(candidates, scores):
                    # Copy item to prevent modifying original cached objects
                    item_copy = item.copy()
                    item_copy["rerank_score"] = float(score)
                    reranked.append(item_copy)
                    
            except Exception as e:
                logger.error(f"Error during Cross-Encoder reranking ({e}). Falling back to embedding similarity.")
                reranked = self._rerank_via_embedding(query, candidates, query_vector)
        else:
            reranked = self._rerank_via_embedding(query, candidates, query_vector)

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Reranked {len(candidates)} candidates down to {min(limit, len(reranked))} in {duration_ms:.2f}ms.")
        return reranked[:limit]

    def _rerank_via_embedding(self, query: str, candidates: List[Dict[str, Any]], query_vector: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Helper method to compute similarity reranking using embedding models.
        """
        np = import_numpy()

        if not self.embedding_model:
            # Absolute fallback if no embedding model is passed: keep original scores
            reranked = []
            for item in candidates:
                item_copy = item.copy()
                item_copy["rerank_score"] = item.get("score", 0.0)
                reranked.append(item_copy)
            return reranked

        # Get query embedding
        if query_vector is not None:
            query_emb = query_vector
        else:
            query_emb = self.embedding_model.embed_queries([query])[0]
        
        query_norm = query_emb / (np.linalg.norm(query_emb) or 1.0)

        # Get chunk embeddings (from cache/Qdrant payloads if saved, or recomputed)
        texts_to_embed = []
        indices_to_embed = []
        doc_embs = [None] * len(candidates)
        
        for i, item in enumerate(candidates):
            if "vector" in item and item["vector"] is not None:
                doc_embs[i] = np.array(item["vector"])
            else:
                texts_to_embed.append(item["payload"]["text"])
                indices_to_embed.append(i)

        if texts_to_embed:
            new_embs = self.embedding_model.embed_documents(texts_to_embed)
            for idx, emb in zip(indices_to_embed, new_embs):
                doc_embs[idx] = emb

        reranked = []
        for item, doc_emb in zip(candidates, doc_embs):
            item_copy = item.copy()
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) or 1.0)
            score = float(np.dot(query_norm, doc_norm))
            item_copy["rerank_score"] = score
            reranked.append(item_copy)

        return reranked

def import_numpy():
    import numpy as np
    return np
