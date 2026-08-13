import time
import logging
from typing import List
import numpy as np

logger = logging.getLogger("RAG.embeddings")
logger.setLevel(logging.INFO)

class EmbeddingModel:
    """
    Multilingual Embedding Model wrapper. Uses sentence-transformers.
    Fails fast with RuntimeError if configured in 'real' mode and model cannot be loaded.
    Mock mode only triggers if EMBEDDING_MODE is explicitly set to 'mock'.
    """
    def __init__(self, mode: str = "real", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dim: int = 384):
        self.mode = mode.lower()
        self.model_name = model_name
        self.dim = dim
        self.model = None
        
        if self.mode == "mock":
            logger.info("Initializing embedding model in explicit MOCK mode.")
            return

        # Real mode loading
        try:
            logger.info(f"Loading embedding model in REAL mode: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load sentence-transformers model: {e}")
            raise RuntimeError(
                f"Embedding model initialization failed: {e}. "
                "Ensure sentence-transformers is installed correctly or set EMBEDDING_MODE=mock for testing."
            )

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
            
        if self.mode == "mock":
            # Generate deterministic mock embeddings based on text characters
            embeddings = []
            for text in texts:
                text_hash = sum(ord(c) for c in text) % 10000
                np.random.seed(text_hash)
                vec = np.random.randn(self.dim)
                vec /= np.linalg.norm(vec) or 1.0
                embeddings.append(vec.tolist())
            return [np.array(e) for e in embeddings]
        
        if not self.model:
            raise RuntimeError("Embedding model is not loaded.")
            
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [np.array(e) for e in embeddings]
        except Exception as e:
            logger.error(f"Error encoding embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
