import logging
import hashlib
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
    HAS_QDRANT_CLIENT = True
except ImportError:
    HAS_QDRANT_CLIENT = False
    QdrantClient = None
    VectorParams = None
    Distance = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None

logger = logging.getLogger("RAG.retrieval.dense")
logger.setLevel(logging.INFO)

class QdrantRepository:
    """
    Manages Qdrant client connections, collection lifecycle, and vector lookups.
    Abstracts local persistent storage vs. cloud/server storage.
    """
    def __init__(self, path: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None, vector_dim: int = 384, allow_mock: bool = False):
        self.vector_dim = vector_dim
        self.collection_name = "rag_chunks"
        self.client = None
        self.is_mock = False
        self.allow_mock = allow_mock
        
        if not HAS_QDRANT_CLIENT:
            if not allow_mock:
                raise RuntimeError("qdrant-client package is not installed and allow_mock=False.")
            logger.warning("qdrant-client package not installed. Falling back to local in-memory MOCK repository.")
            self.is_mock = True
            self.mock_store = []
            return

        try:
            local_path = path or "qdrant_storage"
            if url and url.strip():
                try:
                    logger.info(f"Connecting to Qdrant Cloud/Server at {url}...")
                    self.client = QdrantClient(url=url, api_key=api_key)
                    if not self.client.collection_exists(self.collection_name):
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                        )
                except Exception as net_err:
                    logger.warning(f"Failed to connect to Qdrant server at {url} ({net_err}). Falling back to local disk Qdrant at '{local_path}'.")
                    self.client = QdrantClient(path=local_path)
            else:
                logger.info(f"Initializing persistent local Qdrant at {local_path}...")
                self.client = QdrantClient(path=local_path)

            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
        except Exception as e:
            if not allow_mock:
                logger.error(f"CRITICAL: Failed to initialize Qdrant repository: {e}")
                raise RuntimeError(f"Qdrant initialization failed: {e}") from e
            logger.warning(f"Failed to initialize Qdrant client ({e}). Falling back to local in-memory MOCK repository.")
            self.is_mock = True
            self.mock_store = []  # List of dicts for local memory fallback

    def close(self):
        """Closes Qdrant client connection and releases disk storage locks."""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception as e:
                logger.debug(f"Error closing Qdrant client: {e}")
            self.client = None


    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """
        Upserts a list of text chunks and their embeddings into the Qdrant database.
        Each chunk is stored with payload containing its text, strategy, language, etc.
        Uses deterministic point IDs to enable idempotent writes.
        """
        if not chunks:
            return
            
        if self.is_mock or not self.client:
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                self.mock_store.append({
                    "id": len(self.mock_store),
                    "vector": emb,
                    "payload": {
                        "text": chunk.get("text", ""),
                        "strategy": chunk.get("metadata", {}).get("strategy", "sentence"),
                        "language": chunk.get("metadata", {}).get("language", "hi"),
                        "document_id": chunk.get("metadata", {}).get("document_id", "doc"),
                        "parent_id": chunk.get("metadata", {}).get("parent_id"),
                        "parent_text": chunk.get("metadata", {}).get("parent_text"),
                        "chunk_id": chunk.get("chunk_id"),
                        "metadata": chunk.get("metadata", {})
                    }
                })
            logger.info(f"Upserted {len(chunks)} chunks into MOCK store. Total: {len(self.mock_store)}")
            return

        try:
            points = []
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    doc_id = chunk.get("metadata", {}).get("document_id", "doc")
                    text_snippet = chunk.get("text", "")[:30]
                    chunk_id = f"{doc_id}_{idx}_{text_snippet}"
                
                # Deterministic 64-bit int point ID from chunk_id
                point_id = int(hashlib.md5(chunk_id.encode("utf-8")).hexdigest()[:16], 16) & 0x7fffffffffffffff
                
                payload = {
                    "text": chunk.get("text", ""),
                    "strategy": chunk.get("metadata", {}).get("strategy", "sentence"),
                    "language": chunk.get("metadata", {}).get("language", "hi"),
                    "document_id": chunk.get("metadata", {}).get("document_id", "doc"),
                    "parent_id": chunk.get("metadata", {}).get("parent_id"),
                    "parent_text": chunk.get("metadata", {}).get("parent_text"),
                    "chunk_id": chunk_id,
                    "metadata": chunk.get("metadata", {})
                }
                points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))
                
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully upserted {len(chunks)} chunks into Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error upserting chunks into Qdrant: {e}")
            if not self.allow_mock:
                raise RuntimeError(f"Failed to upsert chunks into Qdrant: {e}") from e

    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int = 20, 
        language: Optional[str] = None, 
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes an ANN search, optionally filtering by language and/or strategy.
        """
        if self.is_mock or not self.client:
            # Linear scan for mock search
            results = []
            for item in self.mock_store:
                # Filter by language
                if language and item["payload"]["language"] != language:
                    continue
                # Filter by strategy
                if strategy and item["payload"]["strategy"] != strategy:
                    continue
                
                # Compute Cosine similarity
                vec1 = query_vector
                vec2 = item["vector"]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                score = np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                results.append({
                    "id": item["id"],
                    "score": float(score),
                    "payload": item["payload"]
                })
            
            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

        try:
            conditions = []
            if language:
                conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
            if strategy:
                conditions.append(FieldCondition(key="strategy", match=MatchValue(value=strategy)))
                
            query_filter = Filter(must=conditions) if conditions else None
            
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=limit
                )
                search_results = response.points
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=limit
                )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in search_results
            ]
        except Exception as e:
            logger.error(f"Error during Qdrant search: {e}")
            return []


