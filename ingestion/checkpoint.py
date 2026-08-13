import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from backend import config

logger = logging.getLogger("RAG.ingestion.checkpoint")

class IngestionCheckpointManager:
    """
    Manages persistent ingestion checkpoint state to enable crash recovery and resume capabilities.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        if not checkpoint_path:
            index_dir = os.path.dirname(config.BM25_PATH)
            os.makedirs(index_dir, exist_ok=True)
            checkpoint_path = os.path.join(index_dir, "checkpoint.json")
        self.checkpoint_path = checkpoint_path

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint from {self.checkpoint_path}: processed={data.get('documents_processed', 0)}")
                    return data
            except Exception as e:
                logger.error(f"Failed to read checkpoint file ({e}). Starting fresh.")
        return {
            "dataset": config.DATASET_NAME,
            "revision": "main",
            "split": "validation",
            "last_processed_identifier": "",
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_upserted": 0,
            "status": "not_started",
            "created_at": datetime.now().isoformat()
        }

    def save(self, state: Dict[str, Any]):
        state["last_updated"] = datetime.now().isoformat()
        try:
            temp_path = f"{self.checkpoint_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update(
        self,
        dataset: str,
        split: str,
        last_identifier: str,
        docs_processed: int,
        chunks_created: int,
        vectors_upserted: int,
        status: str = "running",
        limit: int = 0,
        strategy: str = "semantic"
    ):
        checkpoint = self.load()
        checkpoint.update({
            "dataset": dataset,
            "revision": "main",
            "split": split,
            "last_processed_identifier": last_identifier,
            "documents_processed": docs_processed,
            "chunks_created": chunks_created,
            "vectors_upserted": vectors_upserted,
            "status": status,
            "limit": limit,
            "chunk_strategy": strategy
        })
        self.save(checkpoint)
