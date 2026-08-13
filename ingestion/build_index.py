import os
import json
import time
import logging
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure parent directory imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.embeddings.model import EmbeddingModel
from backend.retrieval.dense import QdrantRepository
from backend.retrieval.sparse import BM25SparseRetriever
from ingestion.preprocess import clean_text, detect_language, Deduplicator
from ingestion.chunking import chunk_by_sentence, chunk_semantically, chunk_hierarchical
from ingestion.checkpoint import IngestionCheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG.ingestion")

ISO_MAP = {
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "gu": "guj",
    "bn": "ben",
    "pa": "pan",
    "or": "ory",
    "as": "asm",
    "en": "eng"
}

FLORES_MAPPING = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "en": "eng_Latn"
}

def get_rss_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import ctypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', ctypes.c_ulong),
                    ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            ctypes.windll.kernel32.K32GetProcessMemoryCounters(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                ctypes.sizeof(counters)
            )
            return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            return 0.0

def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def get_synthetic_data(languages: List[str]) -> List[Dict[str, Any]]:
    """Synthetic dataset fallback for offline testing only."""
    logger.info("Generating synthetic documents for offline testing...")
    synthetic_corpus = [
        {
            "query": "भारत की राजधानी क्या है?",
            "target_lang": "hin_Deva",
            "passages": {
                "Translated_passages": [
                    "भारत की राजधानी नई दिल्ली है। नई दिल्ली भारत सरकार की कार्यकारी, विधायी और न्यायिक शाखाओं का केंद्र है। इसे आधिकारिक तौर पर 1931 में स्थापित किया गया था।",
                    "नई दिल्ली भारत का राजधानी शहर है जो केंद्र शासित प्रदेश दिल्ली का एक हिस्सा है। यह शहर अपने ऐतिहासिक स्मारकों जैसे लाल किला, कुतुब मीनार और इंडिया गेट के लिए जाना जाता है।"
                ]
            }
        },
        {
            "query": "இந்தியாவின் தலைநகரம் எது?",
            "target_lang": "tam_Taml",
            "passages": {
                "Translated_passages": [
                    "இந்தியாவின் தலைநகரம் புது தில்லி ஆகும். புது தில்லி இந்திய அரசின் நிர்வாக, சட்டமன்ற மற்றும் நீதித்துறை கிளைகளின் மையமாக உள்ளது. இது 1931 இல் அதிகாரப்பூர்வமாக நிறுவப்பட்டது.",
                    "புது தில்லி இந்தியாவின் தலைநகரமாக விளங்குகிறது. இது தில்லி யூனியன் பிரதேசத்தின் ஒரு பகுதியாகும். இந்த நகரம் செங்கோட்டை, குதுப் மினார் மற்றும் இந்தியா கேட் போன்ற வரலாற்று நினைவுச் சின்னங்களுக்கு பிரபலமானது."
                ]
            }
        },
        {
            "query": "what is the capital of India?",
            "target_lang": "eng_Latn",
            "passages": {
                "Translated_passages": [
                    "The capital of India is New Delhi. New Delhi is the seat of the executive, legislative, and judicial branches of the Government of India. It was officially inaugurated in 1931.",
                    "New Delhi serves as the capital city of India. It is a territory within Delhi. The city is famous for monuments like Red Fort, Qutub Minar, and India Gate."
                ]
            }
        }
    ]
    flores_targets = [FLORES_MAPPING.get(l) for l in languages if l in FLORES_MAPPING]
    return [row for row in synthetic_corpus if row["target_lang"] in flores_targets]

def process_batch(
    batch_chunks: List[Dict[str, Any]],
    embedder: EmbeddingModel,
    qdrant_repo: QdrantRepository,
    sparse_retriever: BM25SparseRetriever,
    embedding_batch_size: int
) -> int:
    """Embeds and upserts a single batch of chunks directly into Qdrant & BM25."""
    if not batch_chunks:
        return 0

    chunks_to_embed = []
    embed_indices = []

    for i, c in enumerate(batch_chunks):
        if c.get("metadata", {}).get("is_parent", False):
            continue
        chunks_to_embed.append(c["text"])
        embed_indices.append(i)

    vectors_upserted = 0
    if chunks_to_embed:
        embeddings = []
        for b_start in range(0, len(chunks_to_embed), embedding_batch_size):
            sub_texts = chunks_to_embed[b_start : b_start + embedding_batch_size]
            sub_embs = embedder.embed_documents(sub_texts)
            embeddings.extend(sub_embs)

        embedded_chunks = [batch_chunks[i] for i in embed_indices]
        qdrant_repo.upsert_chunks(embedded_chunks, embeddings)
        vectors_upserted = len(embedded_chunks)

    sparse_retriever.add_chunks_batch(batch_chunks)
    sparse_retriever.save(str(config.BM25_PATH))
    return vectors_upserted

def run_ingestion(
    languages: List[str] = ["hi"],
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    strategy: str = "semantic",
    use_validation_split: bool = True
):
    if limit is None:
        limit = config.INGEST_LIMIT
    if batch_size is None:
        batch_size = config.INGEST_BATCH_SIZE
    if embedding_batch_size is None:
        embedding_batch_size = config.EMBEDDING_BATCH_SIZE

    limit_str = "Unlimited" if limit <= 0 else str(limit)
    logger.info(
        f"Starting Streaming Ingestion. Strategy={strategy}, Limit={limit_str}, "
        f"IngestBatchSize={batch_size}, EmbedBatchSize={embedding_batch_size}, Languages={languages}"
    )

    start_time = time.time()
    checkpoint_mgr = IngestionCheckpointManager()
    checkpoint = checkpoint_mgr.load()

    # Determine split & target files
    split_name = "validation" if use_validation_split else "train"
    suffix = "val" if use_validation_split else "train"
    valid_langs = [l for l in languages if l in ISO_MAP and l != "en"]
    if not valid_langs:
        valid_langs = ["hi", "ta"]
    target_files = [f"{split_name}/{ISO_MAP[l]}{suffix}.parquet" for l in valid_langs]

    # Initialize modules (Fail Fast if Qdrant fails)
    logger.info("Initializing Embedding Model...")
    embedder = EmbeddingModel(mode=config.EMBEDDING_MODE, model_name=config.EMBEDDING_MODEL_NAME, dim=config.EMBEDDING_DIM)

    logger.info("Initializing persistent Qdrant Repository (Fail Fast)...")
    qdrant_repo = QdrantRepository(
        path=str(config.QDRANT_PATH),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        vector_dim=config.EMBEDDING_DIM,
        allow_mock=False
    )

    sparse_retriever = BM25SparseRetriever()
    if os.path.exists(config.BM25_PATH):
        sparse_retriever.load(str(config.BM25_PATH))

    deduplicator = Deduplicator()

    # Resume handling
    resume_id = checkpoint.get("last_processed_identifier", "")
    processed_count = checkpoint.get("documents_processed", 0)
    chunk_count = checkpoint.get("chunks_created", 0)
    vectors_upserted_total = checkpoint.get("vectors_upserted", 0)
    skipping = bool(resume_id and checkpoint.get("status") in ["running", "in_progress", "interrupted"])

    if skipping:
        logger.info(f"Resuming ingestion from checkpoint. Last document ID: '{resume_id}' (Processed: {processed_count})")

    checkpoint_mgr.update(
        dataset=config.DATASET_NAME,
        split=split_name,
        last_identifier=resume_id,
        docs_processed=processed_count,
        chunks_created=chunk_count,
        vectors_upserted=vectors_upserted_total,
        status="running",
        limit=limit,
        strategy=strategy
    )

    use_synthetic = False
    batch_chunks: List[Dict[str, Any]] = []
    batch_doc_count = 0
    last_doc_id = resume_id

    try:
        from datasets import load_dataset
        flores_targets = {FLORES_MAPPING[l]: l for l in languages if l in FLORES_MAPPING}

        logger.info(f"Streaming dataset '{config.DATASET_NAME}' ({split_name}). Target files: {target_files}")
        dataset = load_dataset(config.DATASET_NAME, data_files={split_name: target_files}, split=split_name, streaming=True)

        for idx, item in enumerate(dataset):
            if limit > 0 and processed_count >= limit:
                logger.info(f"Reached configured limit of {limit} documents. Stopping stream.")
                break

            target_lang = item.get("target_lang")
            if target_lang not in flores_targets:
                continue

            lang_short = flores_targets[target_lang]
            doc_id = f"msmarco_{lang_short}_{idx}"

            # If resuming, skip items until we reach the last checkpointed document
            if skipping:
                if doc_id == resume_id:
                    skipping = False
                    logger.info(f"Fast-forward completed. Resuming stream right after document '{doc_id}'.")
                continue

            passages = item.get("passages", {})
            translated_passages = passages.get("Translated_passages", [])
            if not translated_passages:
                continue

            doc_chunks_created = 0
            for text in translated_passages:
                cleaned_text = clean_text(text)
                if not cleaned_text or deduplicator.is_duplicate(cleaned_text):
                    continue

                if strategy == "sentence":
                    doc_chunks = chunk_by_sentence(cleaned_text, doc_id, lang_short, config.CHUNK_SIZE_SENTENCES, config.CHUNK_OVERLAP_SENTENCES, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "semantic":
                    doc_chunks = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "hierarchical":
                    doc_chunks = chunk_hierarchical(cleaned_text, doc_id, lang_short)
                elif strategy == "vast":
                    c_sent = chunk_by_sentence(cleaned_text, doc_id, lang_short, config.CHUNK_SIZE_SENTENCES, config.CHUNK_OVERLAP_SENTENCES, config.MAX_CHUNK_CHARACTERS)
                    c_sem = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                    c_hier = chunk_hierarchical(cleaned_text, doc_id, lang_short)
                    doc_chunks = c_sent + c_sem + c_hier

                for c in doc_chunks:
                    c["metadata"]["source_query"] = item.get("query", "")
                    c["metadata"]["split"] = split_name
                    batch_chunks.append(c)
                    doc_chunks_created += 1

            chunk_count += doc_chunks_created
            processed_count += 1
            batch_doc_count += 1
            last_doc_id = doc_id

            # Flush batch to Qdrant & release RAM
            if batch_doc_count >= batch_size:
                upserted = process_batch(batch_chunks, embedder, qdrant_repo, sparse_retriever, embedding_batch_size)
                vectors_upserted_total += upserted

                checkpoint_mgr.update(
                    dataset=config.DATASET_NAME,
                    split=split_name,
                    last_identifier=last_doc_id,
                    docs_processed=processed_count,
                    chunks_created=chunk_count,
                    vectors_upserted=vectors_upserted_total,
                    status="running",
                    limit=limit,
                    strategy=strategy
                )

                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining_sec = ((limit - processed_count) / rate) if limit > 0 and rate > 0 else 0
                rss_mb = get_rss_memory_mb()

                logger.info(
                    f"Progress Report | Docs: {processed_count:,} | Chunks: {chunk_count:,} | "
                    f"Vectors: {vectors_upserted_total:,} | Batch: {batch_doc_count} | "
                    f"Elapsed: {format_time(elapsed)} | Rate: {rate:.1f} docs/sec | "
                    f"ETA: {format_time(remaining_sec) if limit > 0 else 'N/A'} | "
                    f"RSS Memory: {rss_mb / 1024:.2f} GB ({rss_mb:.1f} MB)"
                )

                batch_chunks.clear()
                batch_doc_count = 0
                gc.collect()

        # Flush final remaining batch
        if batch_chunks:
            upserted = process_batch(batch_chunks, embedder, qdrant_repo, sparse_retriever, embedding_batch_size)
            vectors_upserted_total += upserted
            batch_chunks.clear()
            gc.collect()

        if processed_count == 0 and not skipping:
            raise RuntimeError("Hugging Face dataset stream returned 0 records.")

    except Exception as e:
        logger.error(f"Hugging Face streaming ingestion failed: {e}")
        if config.ALLOW_SYNTHETIC_FALLBACK:
            logger.info("ALLOW_SYNTHETIC_FALLBACK is enabled. Seeding synthetic data...")
            use_synthetic = True
        else:
            checkpoint_mgr.update(
                dataset=config.DATASET_NAME,
                split=split_name,
                last_identifier=last_doc_id,
                docs_processed=processed_count,
                chunks_created=chunk_count,
                vectors_upserted=vectors_upserted_total,
                status="failed",
                limit=limit,
                strategy=strategy
            )
            raise RuntimeError(f"Streaming ingestion failed: {e}") from e

    # Synthetic fallback processing if explicitly enabled
    if use_synthetic:
        synthetic_rows = get_synthetic_data(languages)
        for idx, item in enumerate(synthetic_rows):
            target_lang = item["target_lang"]
            lang_short = next(k for k, v in FLORES_MAPPING.items() if v == target_lang)
            doc_id = f"synth_{lang_short}_{idx}"
            for text in item["passages"]["Translated_passages"]:
                cleaned_text = clean_text(text)
                if not cleaned_text or deduplicator.is_duplicate(cleaned_text):
                    continue
                c_list = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                for c in c_list:
                    c["metadata"]["source_query"] = item.get("query", "")
                    batch_chunks.append(c)
                    chunk_count += 1
            processed_count += 1

        if batch_chunks:
            upserted = process_batch(batch_chunks, embedder, qdrant_repo, sparse_retriever, embedding_batch_size)
            vectors_upserted_total += upserted
            batch_chunks.clear()

    total_duration = time.time() - start_time
    checkpoint_mgr.update(
        dataset=config.DATASET_NAME,
        split=split_name,
        last_identifier=last_doc_id,
        docs_processed=processed_count,
        chunks_created=chunk_count,
        vectors_upserted=vectors_upserted_total,
        status="completed",
        limit=limit,
        strategy=strategy
    )

    rss_mb = get_rss_memory_mb()
    logger.info(
        f"=== INGESTION COMPLETE ===\n"
        f"Documents Processed: {processed_count:,}\n"
        f"Chunks Created:     {chunk_count:,}\n"
        f"Vectors Upserted:   {vectors_upserted_total:,}\n"
        f"Total Elapsed Time: {format_time(total_duration)}\n"
        f"Final RSS Memory:   {rss_mb / 1024:.2f} GB ({rss_mb:.1f} MB)\n"
        f"Status:             COMPLETED"
    )
    if hasattr(qdrant_repo, "close"):
        qdrant_repo.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Streaming bounded RAG dataset ingestion CLI.")
    parser.add_argument("--limit", type=int, default=config.INGEST_LIMIT, help="Number of documents to process (0 for unlimited).")
    parser.add_argument("--batch-size", type=int, default=config.INGEST_BATCH_SIZE, help="Ingestion document batch size.")
    parser.add_argument("--embed-batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--strategy", type=str, default=config.CHUNK_STRATEGY, choices=["sentence", "semantic", "hierarchical", "vast"], help="Chunking strategy.")
    parser.add_argument("--langs", nargs="+", default=["hi", "ta"], help="Languages to ingest.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="Dataset split to pull from.")

    args = parser.parse_args()
    use_val = (args.split == "validation")
    run_ingestion(
        languages=args.langs,
        limit=args.limit,
        batch_size=args.batch_size,
        embedding_batch_size=args.embed_batch_size,
        strategy=args.strategy,
        use_validation_split=use_val
    )
