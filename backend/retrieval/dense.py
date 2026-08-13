import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("RAG.retrieval.dense")
logger.setLevel(logging.INFO)

class QdrantRepository:
    """
    Manages Qdrant client connections, collection lifecycle, and vector lookups.
    Abstracts local persistent storage vs. cloud/server storage.
    """
    def __init__(self, path: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None, vector_dim: int = 384):
        self.vector_dim = vector_dim
        self.collection_name = "rag_chunks"
        self.client = None
        self.is_mock = False
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            if url:
                logger.info(f"Connecting to Qdrant Cloud/Server at {url}...")
                self.client = QdrantClient(url=url, api_key=api_key)
            elif path:
                logger.info(f"Initializing persistent local Qdrant at {path}...")
                self.client = QdrantClient(path=path)
            else:
                logger.info("Initializing in-memory Qdrant client...")
                self.client = QdrantClient(":memory:")
                
            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant client ({e}). Falling back to local in-memory MOCK repository.")
            self.is_mock = True
            self.mock_store = []  # List of dicts for local memory fallback

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """
        Upserts a list of text chunks and their embeddings into the Qdrant database.
        Each chunk is stored with payload containing its text, strategy, language, etc.
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
                        "metadata": chunk.get("metadata", {})
                    }
                })
            logger.info(f"Upserted {len(chunks)} chunks into MOCK store. Total: {len(self.mock_store)}")
            return

        try:
            from qdrant_client.models import PointStruct
            points = []
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id = hash(chunk.get("text", "") + str(idx)) & 0xffffffffffff  # Ensure 64-bit int compatibility
                payload = {
                    "text": chunk.get("text", ""),
                    "strategy": chunk.get("metadata", {}).get("strategy", "sentence"),
                    "language": chunk.get("metadata", {}).get("language", "hi"),
                    "document_id": chunk.get("metadata", {}).get("document_id", "doc"),
                    "parent_id": chunk.get("metadata", {}).get("parent_id"),
                    "parent_text": chunk.get("metadata", {}).get("parent_text"),
                    "metadata": chunk.get("metadata", {})
                }
                points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))
                
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully upserted {len(chunks)} chunks into Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error upserting chunks into Qdrant: {e}")

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
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
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

