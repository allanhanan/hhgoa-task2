import time
import logging
import os
import asyncio
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from backend import config
from backend.interfaces import IEmbeddingModel

logger = logging.getLogger("RAG.embeddings")
logger.setLevel(logging.INFO)

class EmbeddingModel(IEmbeddingModel):
    """
    Multilingual Embedding Model wrapper implementing IEmbeddingModel interface.
    Supports ONNX runtime acceleration, sentence-transformers, and mock mode.
    """
    def __init__(self, mode: str = "real", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dim: int = 384):
        self.mode = mode.lower()
        self.model_name = model_name
        self.dim = dim
        self.model = None
        self.onnx_session = None

        if self.mode == "mock":
            logger.info("Initializing embedding model in explicit MOCK mode.")
            return

        try:
            start_time = time.time()
            try:
                # Load directly from local HuggingFace cache (instant startup, zero network requests)
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                # If model is not yet in local cache, download once
                logger.info(f"Model not found in local cache. Downloading {self.model_name}...")
                self.model = SentenceTransformer(self.model_name, local_files_only=False)
            logger.info(f"Loaded embedding model in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load sentence-transformers model: {e}")
            raise RuntimeError(
                f"Embedding model initialization failed: {e}. "
                "Ensure sentence-transformers is installed correctly or set EMBEDDING_MODE=mock for testing."
            )

    async def embed_queries_async(self, texts: List[str]) -> List[np.ndarray]:
        return await asyncio.to_thread(self.embed_queries, texts)

    async def embed_documents_async(self, texts: List[str]) -> List[np.ndarray]:
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []

        if self.mode == "mock":
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
            batch_size = getattr(config, "EMBEDDING_BATCH_SIZE", 32)
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            return [np.array(e) for e in embeddings]
        except Exception as e:
            logger.error(f"Error encoding embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

