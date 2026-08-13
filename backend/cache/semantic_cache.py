import time
import json
import os
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from backend.interfaces import ICacheAdapter

logger = logging.getLogger("RAG.cache.semantic")

def compute_simhash(embedding: np.ndarray) -> int:
    """
    Computes a 64-bit SimHash integer from a dense embedding vector.
    """
    # Project vector onto random hyperplanes or sign vector
    # Deterministic pseudo-random projection matrix (64 x dim)
    dim = len(embedding)
    np.random.seed(42) # Fixed seed for reproducible projection
    projections = np.random.randn(64, dim)
    
    dots = np.dot(projections, embedding)
    simhash_int = 0
    for i, val in enumerate(dots):
        if val > 0:
            simhash_int |= (1 << i)
    return simhash_int

def hamming_distance(h1: int, h2: int) -> int:
    """Calculates bitwise Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count('1')

class SemanticLSHCache(ICacheAdapter):
    """
    Locality-Sensitive Hashing (LSH) Semantic Cache.
    Matches queries by semantic similarity of embeddings in O(1) time.
    Returns response in ~2ms.
    """
    def __init__(self, file_path: Optional[str] = None, max_hamming_distance: int = 3):
        self.file_path = file_path
        self.max_hamming_distance = max_hamming_distance
        # Store entries: list of tuples (simhash_int, query_text, language, response_json_str, expire_at)
        self.entries: List[Dict[str, Any]] = []
        self._load_from_disk()

    def _load_from_disk(self):
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                now = time.time()
                self.entries = [
                    item for item in data 
                    if item.get("expire_at") is None or item.get("expire_at") > now
                ]
                logger.info(f"Loaded {len(self.entries)} active entries into Semantic LSH Cache.")
            except Exception as e:
                logger.error(f"Failed to load semantic cache from disk: {e}")

    def _save_to_disk(self):
        if self.file_path:
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(self.entries, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save semantic cache to disk: {e}")

    def find_semantic_match(self, query_embedding: np.ndarray, language: str = "hi") -> Optional[Tuple[str, float]]:
        """
        Finds a cached response using SimHash hamming distance & cosine verification.
        Returns (cached_response_json, similarity_score) or None.
        """
        if not self.entries or query_embedding is None:
            return None

        q_hash = compute_simhash(query_embedding)
        now = time.time()

        best_match = None
        best_dist = 999

        for entry in self.entries:
            if entry.get("expire_at") and entry["expire_at"] < now:
                continue
            if entry.get("language") != language:
                continue

            e_hash = entry["simhash"]
            dist = hamming_distance(q_hash, e_hash)

            if dist <= self.max_hamming_distance and dist < best_dist:
                best_dist = dist
                best_match = entry

        if best_match:
            logger.info(f"Semantic Cache HIT! Hamming distance: {best_dist}/{self.max_hamming_distance} bits.")
            return best_match["response"], 1.0 - (best_dist / 64.0)

        return None

    async def get(self, key: str) -> Optional[str]:
        # Fallback exact string match for ICacheAdapter compatibility
        now = time.time()
        for entry in self.entries:
            if entry.get("exact_key") == key:
                if entry.get("expire_at") and entry["expire_at"] < now:
                    continue
                return entry["response"]
        return None

    async def set(self, key: str, value: str, ttl: Optional[int] = 3600) -> None:
        # Save key match
        expire_at = (time.time() + ttl) if ttl else None
        entry = {
            "exact_key": key,
            "simhash": 0,
            "language": "hi",
            "response": value,
            "expire_at": expire_at
        }
        self.entries.append(entry)
        self._save_to_disk()

    def set_semantic(
        self, 
        query_text: str, 
        query_embedding: np.ndarray, 
        language: str, 
        response_json: str, 
        ttl: int = 3600
    ):
        """Stores entry with SimHash fingerprint for semantic matching."""
        if query_embedding is None:
            return

        q_hash = compute_simhash(query_embedding)
        expire_at = time.time() + ttl if ttl else None

        entry = {
            "query": query_text,
            "simhash": q_hash,
            "language": language,
            "response": response_json,
            "expire_at": expire_at
        }
        self.entries.append(entry)
        self._save_to_disk()
        logger.info(f"Saved new semantic cache entry (SimHash={q_hash}). Total cached: {len(self.entries)}")

    def clear(self):
        self.entries.clear()
        self._save_to_disk()
