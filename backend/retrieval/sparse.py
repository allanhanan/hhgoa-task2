import os
import pickle
import logging
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("RAG.retrieval.sparse")
logger.setLevel(logging.INFO)

def tokenize_indic_and_english(text: str) -> List[str]:
    """
    Tokenizer supporting English words and Indic word boundaries.
    """
    if not text:
        return []
    # Lowercase English, remove basic punctuation, split by space and punctuation
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s\u0900-\u0D7F]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 0]

class SparseRetriever(ABC):
    """
    Abstract interface for sparse retrieval engines to allow scaling beyond rank_bm25 in production.
    """
    @abstractmethod
    def build_index(self, chunks: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass

    @abstractmethod
    def search(
        self, 
        query: str, 
        limit: int = 20, 
        language: Optional[str] = None, 
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

class BM25SparseRetriever(SparseRetriever):
    """
    BM25 implementation of SparseRetriever using rank_bm25.
    If rank_bm25 is unavailable, uses a lightweight TF-IDF / term-overlap fallback.
    """
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self.is_mock = False
        
        try:
            import rank_bm25
        except ImportError:
            logger.warning("rank_bm25 package not found. Using TF-IDF mock fallback for sparse retriever.")
            self.is_mock = True

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index over a list of chunks.
        """
        if not chunks:
            return
            
        self.chunks = chunks
        
        if self.is_mock:
            logger.info("MOCK Sparse Index built (memory overlap index).")
            return
            
        try:
            from rank_bm25 import BM25Okapi
            corpus_tokens = [tokenize_indic_and_english(chunk.get("text", "")) for chunk in chunks]
            self.bm25 = BM25Okapi(corpus_tokens)
            logger.info(f"Built BM25 index with {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}. Falling back to mock.")
            self.is_mock = True

    def add_chunks_batch(self, batch_chunks: List[Dict[str, Any]]):
        """
        Appends a batch of chunks and updates the BM25 index over accumulated chunks.
        """
        if not batch_chunks:
            return
        self.chunks.extend(batch_chunks)
        self.build_index(self.chunks)


    def save(self, file_path: str):
        """
        Serializes the index and chunks list to a file.
        """
        if not self.chunks:
            logger.warning("Empty sparse index. Save skipped.")
            return
            
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "is_mock": self.is_mock, "bm25": self.bm25}, f)
            logger.info(f"Sparse index saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save sparse index to {file_path}: {e}")

    def load(self, file_path: str):
        """
        Loads the index from a serialized file.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Sparse index file {file_path} not found. Skipping load.")
            return False
            
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data.get("chunks", [])
            self.is_mock = data.get("is_mock", False)
            self.bm25 = data.get("bm25")
            logger.info(f"Loaded sparse index with {len(self.chunks)} chunks from {file_path}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load sparse index from {file_path}: {e}")
            return False

    def search(
        self, 
        query: str, 
        limit: int = 20, 
        language: Optional[str] = None, 
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the BM25 index, applying language and strategy filtering.
        """
        if not self.chunks:
            return []

        # Tokenize query
        query_tokens = tokenize_indic_and_english(query)
        if not query_tokens:
            return []

        if self.is_mock or not self.bm25:
            # Term overlap search fallback
            results = []
            q_set = set(query_tokens)
            for idx, chunk in enumerate(self.chunks):
                # Filter by language
                chunk_lang = chunk.get("metadata", {}).get("language", "hi")
                if language and chunk_lang != language:
                    continue
                # Filter by strategy
                chunk_strat = chunk.get("metadata", {}).get("strategy", "sentence")
                if strategy and chunk_strat != strategy:
                    continue

                doc_tokens = tokenize_indic_and_english(chunk.get("text", ""))
                overlap = len(q_set.intersection(doc_tokens))
                if overlap > 0:
                    score = overlap / (len(q_set) + len(doc_tokens) - overlap) # Jaccard similarity
                    results.append({
                        "id": idx,
                        "score": float(score),
                        "payload": {
                            "text": chunk.get("text", ""),
                            "strategy": chunk_strat,
                            "language": chunk_lang,
                            "document_id": chunk.get("metadata", {}).get("document_id", "doc"),
                            "parent_id": chunk.get("metadata", {}).get("parent_id"),
                            "parent_text": chunk.get("metadata", {}).get("parent_text"),
                            "metadata": chunk.get("metadata", {})
                        }
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

        try:
            # Get raw BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Map scores to chunks and apply filters
            scored_candidates = []
            for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
                if score <= 0.0:
                    continue
                
                # Filter by language
                chunk_lang = chunk.get("metadata", {}).get("language", "hi")
                if language and chunk_lang != language:
                    continue
                # Filter by strategy
                chunk_strat = chunk.get("metadata", {}).get("strategy", "sentence")
                if strategy and chunk_strat != strategy:
                    continue

                scored_candidates.append({
                    "id": idx,
                    "score": float(score),
                    "payload": {
                        "text": chunk.get("text", ""),
                        "strategy": chunk_strat,
                        "language": chunk_lang,
                        "document_id": chunk.get("metadata", {}).get("document_id", "doc"),
                        "parent_id": chunk.get("metadata", {}).get("parent_id"),
                        "parent_text": chunk.get("metadata", {}).get("parent_text"),
                        "metadata": chunk.get("metadata", {})
                    }
                })
            
            # Sort descending
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            return scored_candidates[:limit]
        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []
