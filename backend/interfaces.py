import abc
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator

class ISTTClient(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, audio_bytes: bytes, language_code: str = "hi") -> str:
        pass

class ITTSClient(abc.ABC):
    @abc.abstractmethod
    async def stream_tts(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> AsyncGenerator[bytes, None]:
        pass

    @abc.abstractmethod
    async def stream_tts_from_tokens(self, token_stream: AsyncGenerator[str, None], voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> AsyncGenerator[bytes, None]:
        pass

class IEmbeddingModel(abc.ABC):
    @abc.abstractmethod
    async def embed_queries(self, texts: List[str]) -> List[Any]:
        pass

    @abc.abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[Any]:
        pass

class IDenseRetriever(abc.ABC):
    @abc.abstractmethod
    def search(self, query_vector: Any, limit: int = 20, language: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

class ISparseRetriever(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, limit: int = 20, language: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

class ILLMProvider(abc.ABC):
    """
    Abstract LLM Provider interface for dependency injection.
    Allows seamlessly swapping Groq, OpenAI, Anthropic, Ollama, etc.
    """
    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        pass

    @abc.abstractmethod
    async def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        model_name: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        pass

    @abc.abstractmethod
    async def generate_answer_stream(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        model_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        pass

    @abc.abstractmethod
    async def verify_grounding(self, answer: str, context_text: str, model_name: Optional[str] = None) -> Tuple[str, str]:
        pass

class IModelRouter(abc.ABC):
    """
    Abstract Model Router to dynamically select the best LLM provider & model
    based on query features, latency budget, or deprecation fallback chains.
    """
    @abc.abstractmethod
    def select_model(self, query: str, language: str, latency_sensitive: bool = False) -> Tuple[ILLMProvider, str]:
        pass

class ICacheAdapter(abc.ABC):
    @abc.abstractmethod
    async def get(self, key: str) -> Optional[str]:
        pass

    @abc.abstractmethod
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        pass
