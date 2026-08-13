import logging
from typing import Dict, Tuple, Optional, List
from backend.interfaces import ILLMProvider, IModelRouter
from backend import config

logger = logging.getLogger("RAG.generation.router")

# Deprecation fallback maps for popular models
MODEL_DEPRECATION_FALLBACKS: Dict[str, List[str]] = {
    "llama-3.3-70b-versatile": ["llama-3.1-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"],
    "llama-3.1-8b-instant": ["llama3-8b-8192", "gemma2-9b-it"],
    "llama3-70b-8192": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "llama3-8b-8192": ["llama-3.1-8b-instant", "gemma2-9b-it"]
}

class AdaptiveModelRouter(IModelRouter):
    """
    Intelligent Model Router that dynamically routes queries based on:
    1. Latency mode (fast vs balanced vs high-accuracy)
    2. Query length & language complexity
    3. Deprecation fallback chains (gracefully shifts if Groq deprecates a model tag)
    """
    def __init__(
        self, 
        providers: Dict[str, ILLMProvider],
        default_provider_name: str = "groq",
        fast_model: str = "llama-3.1-8b-instant",
        quality_model: str = "llama-3.3-70b-versatile"
    ):
        self.providers = providers
        self.default_provider_name = default_provider_name
        self.fast_model = os_getenv("GROQ_FAST_MODEL", fast_model)
        self.quality_model = os_getenv("GROQ_QUALITY_MODEL", quality_model)

    def select_model(
        self, 
        query: str, 
        language: str = "hi", 
        latency_sensitive: bool = False
    ) -> Tuple[ILLMProvider, str]:
        provider = self.providers.get(self.default_provider_name) or next(iter(self.providers.values()))
        
        # Simple heuristic for routing:
        # Short simple query or latency_sensitive mode -> fast 8B model
        # Longer query or non-English complex query -> 70B quality model
        word_count = len(query.split())
        
        if latency_sensitive or word_count < 6:
            chosen_model = self.fast_model
            logger.info(f"Router selected FAST model: '{chosen_model}' for query (words={word_count})")
        else:
            chosen_model = self.quality_model
            logger.info(f"Router selected QUALITY model: '{chosen_model}' for query (words={word_count})")
            
        return provider, chosen_model

    def get_fallback_models(self, primary_model: str) -> List[str]:
        """Returns ordered fallback models if primary model is deprecated or fails."""
        fallbacks = MODEL_DEPRECATION_FALLBACKS.get(primary_model, ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        return [m for m in fallbacks if m != primary_model]

def os_getenv(key: str, default: str) -> str:
    import os
    return os.getenv(key, default)
