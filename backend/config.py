import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from root or backend directory
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# Environment setup
ENVIRONMENT = os.getenv("ENVIRONMENT", "development") # "development" or "production"

# Base directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATASET_NAME = "ai4bharat/MSMARCO-XI"
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "en": "English"
}
DEFAULT_LANGUAGE = "hi"

# Ingestion configuration
# Configurable limit of records to pull for stage benchmarking:
# 0 means unlimited / ingest full dataset
INGEST_LIMIT = int(os.getenv("INGEST_LIMIT", os.getenv("INGESTION_LIMIT", "1000")))
INGESTION_LIMIT = INGEST_LIMIT  # Alias for backward compatibility
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Fallback parameter - NEVER set this to True for benchmarks or final submissions
ALLOW_SYNTHETIC_FALLBACK = os.getenv("ALLOW_SYNTHETIC_FALLBACK", "false").lower() == "true"

# Embedding settings
# EMBEDDING_MODE: "real" or "mock". Real uses sentence-transformers; Mock uses deterministic random vectors.
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "real") 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Chunking settings
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic") # "sentence", "semantic", "hierarchical"
CHUNK_SIZE_SENTENCES = 3
CHUNK_OVERLAP_SENTENCES = 1
MAX_CHUNK_CHARACTERS = 600  # Enforce max chunk limit
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))

# Retrieval settings
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FUSED_TOP_K = 15
FINAL_TOP_K = 3

# Reranking settings
# RERANKER_TYPE: "cross_encoder" or "cosine".
RERANKER_TYPE = os.getenv("RERANKER_TYPE", "cosine")
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF Parameter
RRF_CONSTANT = 60

# Cache configurations
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory") # "memory" or "redis"
REDIS_URL = os.getenv("REDIS_URL")
CACHE_PATH = DATA_DIR / "cache.json"

# Qdrant client configurations
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_storage"))

# BM25 Index Path
BM25_PATH = DATA_DIR / "indexes" / "v1" / "bm25_index.pkl"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Guardrail thresholds
RELEVANCE_THRESHOLD = 0.35  # Min score for relevant context
GROUNDING_THRESHOLD = 0.70  # Min grounding confidence score
