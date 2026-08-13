import os
import json
import time
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Any

# Ensure parent directory imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.embeddings.model import EmbeddingModel
from backend.retrieval.dense import QdrantRepository
from backend.retrieval.sparse import BM25SparseRetriever
from ingestion.preprocess import clean_text, detect_language, Deduplicator
from ingestion.chunking import chunk_by_sentence, chunk_semantically, chunk_hierarchical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG.ingestion")

# ISO 3-letter mapping for dataset files
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

def get_manifest_path() -> str:
    index_dir = os.path.dirname(config.BM25_PATH)
    os.makedirs(index_dir, exist_ok=True)
    return os.path.join(index_dir, "manifest.json")

def load_manifest() -> Dict[str, Any]:
    manifest_path = get_manifest_path()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
    return {}

def save_manifest(manifest: Dict[str, Any]):
    manifest_path = get_manifest_path()
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated ingestion manifest: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

def get_synthetic_data(languages: List[str]) -> List[Dict[str, Any]]:
    """
    Synthetic dataset for offline development/testing only.
    """
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
        },
        {
            "query": "भारत की वित्तीय राजधानी कौन सी है?",
            "target_lang": "hin_Deva",
            "passages": {
                "Translated_passages": [
                    "मुंबई भारत की वित्तीय राजधानी है। यह महाराष्ट्र राज्य की राजधानी है और भारत का सबसे अधिक आबादी वाला शहर है। यहाँ भारतीय रिजर्व बैंक और बॉम्बे स्टॉक एक्सचेंज स्थित हैं।"
                ]
            }
        },
        {
            "query": "இந்தியாவின் நிதித் தலைநகரம் எது?",
            "target_lang": "tam_Taml",
            "passages": {
                "Translated_passages": [
                    "மும்பை இந்தியாவின் நிதித் தலைநகரம் ஆகும். இது மகாராஷ்டிர மாநிலத்தின் தலைநகரமாகவும், இந்தியாவின் அதிக மக்கள் தொகை கொண்ட நகரமாகவும் உள்ளது. இங்கு ரிசர்வ் வங்கி மற்றும் பம்பாய் பங்குச் சந்தை அமைந்துள்ளன."
                ]
            }
        },
        {
            "query": "what is the financial capital of India?",
            "target_lang": "eng_Latn",
            "passages": {
                "Translated_passages": [
                    "Mumbai is the financial capital of India. It is the capital city of Maharashtra state and the most populous city in India. The Reserve Bank of India and Bombay Stock Exchange are located here."
                ]
            }
        }
    ]
    
    flores_targets = [FLORES_MAPPING.get(l) for l in languages if l in FLORES_MAPPING]
    return [row for row in synthetic_corpus if row["target_lang"] in flores_targets]

def run_ingestion(
    languages: List[str] = ["hi"],
    limit: int = 1000,
    strategy: str = "semantic",
    use_validation_split: bool = True
):
    logger.info(f"Starting ingestion process. Strategy={strategy}, Limit={limit}, Languages={languages}, UseValidationSplit={use_validation_split}")
    
    start_time = time.time()
    
    # 1. Check manifest
    manifest = load_manifest()
    if manifest.get("status") == "complete" and manifest.get("chunk_strategy") == strategy and manifest.get("limit") == limit:
        logger.info("Ingestion already completed according to manifest. Skipping.")
        return
        
    manifest.update({
        "dataset_name": config.DATASET_NAME,
        "limit": limit,
        "chunk_strategy": strategy,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": config.EMBEDDING_DIM,
        "languages": languages,
        "status": "in_progress",
        "started_at": datetime.now().isoformat()
    })
    save_manifest(manifest)

    # 2. Initialize modules
    logger.info("Initializing embedding model...")
    embedder = EmbeddingModel(mode=config.EMBEDDING_MODE, model_name=config.EMBEDDING_MODEL_NAME, dim=config.EMBEDDING_DIM)
    
    logger.info("Initializing vector database repository...")
    qdrant_repo = QdrantRepository(
        path=str(config.QDRANT_PATH),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        vector_dim=config.EMBEDDING_DIM
    )
    
    deduplicator = Deduplicator()
    all_chunks = []
    
    # 3. Stream dataset or fallback
    processed_count = 0
    chunk_count = 0
    use_synthetic = False
    
    try:
        from datasets import load_dataset
        
        flores_targets = {FLORES_MAPPING[l]: l for l in languages if l in FLORES_MAPPING}
        
        # Staged logic: Validation files are ~400MB and load instantly; Train files are ~3.7GB.
        # Use validation split for Stage 1 & 2 local testing.
        split_name = "validation" if use_validation_split else "train"
        suffix = "val" if use_validation_split else "train"
        
        target_files = [f"{split_name}/{ISO_MAP[l]}{suffix}.parquet" for l in languages if l in ISO_MAP]
        
        logger.info(f"Streaming from Hugging Face dataset ({split_name}). Target files: {target_files}")
        dataset = load_dataset(config.DATASET_NAME, data_files={split_name: target_files}, split=split_name, streaming=True)
        
        for idx, item in enumerate(dataset):
            if processed_count >= limit:
                break
                
            target_lang = item.get("target_lang")
            if target_lang not in flores_targets:
                continue
                
            lang_short = flores_targets[target_lang]
            doc_id = f"msmarco_{lang_short}_{idx}"
            
            passages = item.get("passages", {})
            translated_passages = passages.get("Translated_passages", [])
            
            if not translated_passages:
                continue

            for p_idx, text in enumerate(translated_passages):
                cleaned_text = clean_text(text)
                if not cleaned_text or deduplicator.is_duplicate(cleaned_text):
                    continue

                doc_chunks = []
                if strategy == "sentence":
                    doc_chunks = chunk_by_sentence(cleaned_text, doc_id, lang_short, config.CHUNK_SIZE_SENTENCES, config.CHUNK_OVERLAP_SENTENCES, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "semantic":
                    doc_chunks = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "hierarchical":
                    doc_chunks = chunk_hierarchical(cleaned_text, doc_id, lang_short)
                elif strategy == "vast":
                    # Vast multi-strategy indexing: combines sentence, semantic, and hierarchical chunks
                    c_sent = chunk_by_sentence(cleaned_text, doc_id, lang_short, config.CHUNK_SIZE_SENTENCES, config.CHUNK_OVERLAP_SENTENCES, config.MAX_CHUNK_CHARACTERS)
                    c_sem = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                    c_hier = chunk_hierarchical(cleaned_text, doc_id, lang_short)
                    doc_chunks = c_sent + c_sem + c_hier
                    
                for c in doc_chunks:
                    c["metadata"]["source_query"] = item.get("query", "")
                    all_chunks.append(c)
                    chunk_count += 1
            
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Ingested {processed_count} documents from HuggingFace. Created {chunk_count} chunks.")
                
        if processed_count == 0:
            raise RuntimeError("Hugging Face stream returned zero matching records.")
            
    except Exception as e:
        logger.error(f"Hugging Face streaming failed: {e}")
        if config.ALLOW_SYNTHETIC_FALLBACK:
            logger.info("ALLOW_SYNTHETIC_FALLBACK is enabled. Activating synthetic data fallback...")
            use_synthetic = True
        else:
            logger.error("ALLOW_SYNTHETIC_FALLBACK is disabled. Ingestion aborted (FAIL FAST).")
            manifest.update({
                "status": "failed",
                "reason": f"Streaming failed: {e}",
                "completed_at": datetime.now().isoformat()
            })
            save_manifest(manifest)
            raise RuntimeError(f"Ingestion failed: {e}")

    # 4. Process synthetic fallback if enabled
    if use_synthetic:
        synthetic_rows = get_synthetic_data(languages)
        for idx, item in enumerate(synthetic_rows):
            target_lang = item["target_lang"]
            lang_short = next(k for k, v in FLORES_MAPPING.items() if v == target_lang)
            doc_id = f"msmarco_{lang_short}_{idx}"
            
            translated_passages = item["passages"]["Translated_passages"]
            for p_idx, text in enumerate(translated_passages):
                cleaned_text = clean_text(text)
                if not cleaned_text or deduplicator.is_duplicate(cleaned_text):
                    continue

                doc_chunks = []
                if strategy == "sentence":
                    doc_chunks = chunk_by_sentence(cleaned_text, doc_id, lang_short, config.CHUNK_SIZE_SENTENCES, config.CHUNK_OVERLAP_SENTENCES, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "semantic":
                    doc_chunks = chunk_semantically(cleaned_text, doc_id, lang_short, embedder.embed_documents, config.SEMANTIC_THRESHOLD, config.MAX_CHUNK_CHARACTERS)
                elif strategy == "hierarchical":
                    doc_chunks = chunk_hierarchical(cleaned_text, doc_id, lang_short)
                    
                for c in doc_chunks:
                    c["metadata"]["source_query"] = item["query"]
                    all_chunks.append(c)
                    chunk_count += 1
            processed_count += 1

    # 5. Embed and save indexes
    if all_chunks:
        logger.info(f"Total chunks created: {len(all_chunks)}. Generating batch embeddings...")
        
        chunks_to_embed = []
        embed_indices = []
        
        for i, c in enumerate(all_chunks):
            if c.get("metadata", {}).get("is_parent", False):
                continue
            chunks_to_embed.append(c["text"])
            embed_indices.append(i)
            
        logger.info(f"Embedding {len(chunks_to_embed)} vector chunks in batches...")
        embeddings = []
        batch_size = 64
        for b_start in range(0, len(chunks_to_embed), batch_size):
            batch_texts = chunks_to_embed[b_start : b_start + batch_size]
            batch_embs = embedder.embed_documents(batch_texts)
            embeddings.extend(batch_embs)
            
        embedded_chunks = [all_chunks[i] for i in embed_indices]
        qdrant_repo.upsert_chunks(embedded_chunks, embeddings)
        
        logger.info("Building BM25 Sparse Index...")
        sparse_retriever = BM25SparseRetriever()
        sparse_retriever.build_index(all_chunks)
        sparse_retriever.save(str(config.BM25_PATH))
        
        duration = time.time() - start_time
        logger.info(f"Ingestion completed in {duration:.2f} seconds.")
        
        manifest.update({
            "status": "complete",
            "processed_count": processed_count,
            "chunk_count": chunk_count,
            "embedded_count": len(embedded_chunks),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": float(duration)
        })
        save_manifest(manifest)
    else:
        logger.warning("No chunks generated. Ingestion aborted.")
        manifest.update({
            "status": "failed",
            "reason": "No chunks generated",
            "completed_at": datetime.now().isoformat()
        })
        save_manifest(manifest)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI dataset and build indices.")
    parser.add_argument("--limit", type=int, default=config.INGESTION_LIMIT, help="Number of documents to process.")
    parser.add_argument("--strategy", type=str, default=config.CHUNK_STRATEGY, choices=["sentence", "semantic", "hierarchical", "vast"], help="Chunking strategy.")
    parser.add_argument("--langs", nargs="+", default=["hi", "ta"], help="Languages to ingest.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"], help="Dataset split to pull from.")
    
    args = parser.parse_args()
    
    use_val = (args.split == "validation")
    run_ingestion(languages=args.langs, limit=args.limit, strategy=args.strategy, use_validation_split=use_val)
