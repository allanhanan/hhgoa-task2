import os
import json
import time
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger("RAG.cache")
logger.setLevel(logging.INFO)

class CacheAdapter(ABC):
    """
    Abstract interface for cache adapters.
    """
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None):
        pass

    @abstractmethod
    def clear(self):
        pass

class MemoryCacheAdapter(CacheAdapter):
    """
    Memory cache adapter with optional persistent local JSON backup.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.store: Dict[str, Dict[str, Any]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check for expired keys
                now = time.time()
                for k, v in data.items():
                    expire_at = v.get("expire_at")
                    if expire_at is None or expire_at > now:
                        self.store[k] = v
                logger.info(f"Loaded {len(self.store)} active cache entries from disk ({self.file_path}).")
            except Exception as e:
                logger.error(f"Failed to load cache from disk: {e}")

    def _save_to_disk(self):
        if self.file_path:
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(self.store, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to write cache to disk: {e}")

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        if key in self.store:
            entry = self.store[key]
            expire_at = entry.get("expire_at")
            if expire_at is None or expire_at > now:
                return entry.get("value")
            # Expired
            del self.store[key]
            self._save_to_disk()
        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        expire_at = (time.time() + ttl) if ttl else None
        self.store[key] = {
            "value": value,
            "expire_at": expire_at
        }
        self._save_to_disk()

    def clear(self):
        self.store.clear()
        self._save_to_disk()

class RedisCacheAdapter(CacheAdapter):
    """
    Redis cache adapter for production environments.
    """
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None
        self.is_fallback = False
        self.fallback_cache = None
        
        try:
            import redis
            logger.info(f"Connecting to Redis at {self.redis_url}...")
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis ({e}). Falling back to MemoryCacheAdapter.")
            self.is_fallback = True
            self.fallback_cache = MemoryCacheAdapter()

    def get(self, key: str) -> Optional[str]:
        if self.is_fallback or not self.client:
            return self.fallback_cache.get(key)
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get failed: {e}. Falling back to memory.")
            return self.fallback_cache.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        if self.is_fallback or not self.client:
            self.fallback_cache.set(key, value, ttl)
            return
        try:
            self.client.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis set failed: {e}. Falling back to memory.")
            self.fallback_cache.set(key, value, ttl)

    def clear(self):
        if self.is_fallback or not self.client:
            self.fallback_cache.clear()
            return
        try:
            self.client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")

def get_cache_adapter(backend: str = "memory", redis_url: Optional[str] = None, file_path: Optional[str] = None) -> CacheAdapter:
    """
    Factory function returning the appropriate cache adapter.
    """
    if backend == "redis" and redis_url:
        return RedisCacheAdapter(redis_url)
    return MemoryCacheAdapter(file_path)
