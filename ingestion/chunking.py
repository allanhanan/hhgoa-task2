import re
import hashlib
from typing import List, Dict, Any, Optional, Callable
import numpy as np

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using English sentence boundaries (. ! ?)
    and Indic sentence boundaries (Devanagari/Bengali/Assamese purna viram । or ॥) 
    as well as newlines. Handles spaces after punctuation.
    """
    if not text:
        return []
    
    # Split using a regex that captures common endings or newlines
    sentence_endings = re.compile(r'(?<=[.!?।॥\n])\s*')
    sentences = sentence_endings.split(text)
    
    # Filter empty or whitespace-only matches
    return [s.strip() for s in sentences if s.strip()]

def generate_chunk_id(document_id: str, strategy: str, text: str, idx: int) -> str:
    """
    Generates a globally unique deterministic chunk ID based on doc ID, strategy, index, and text hash.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}_{strategy}_{idx}_{text_hash}"

def chunk_by_sentence(
    text: str, 
    document_id: str,
    language: str,
    chunk_size: int = 3, 
    chunk_overlap: int = 1,
    max_chars: int = 600
) -> List[Dict[str, Any]]:
    """
    Groups sentences into fixed-size chunks with a rolling overlap, 
    respecting a maximum character count.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1

    chunk_idx = 0
    for i in range(0, len(sentences), step):
        group = sentences[i:i + chunk_size]
        if not group:
            continue
        
        # Enforce max character limit by dropping or splitting sentences if they exceed max_chars
        current_sentences = []
        current_len = 0
        
        for s in group:
            if current_len + len(s) + 1 > max_chars and current_sentences:
                # Flush current accumulation if it would exceed limits
                chunk_text = " ".join(current_sentences)
                chunk_id = generate_chunk_id(document_id, "sentence", chunk_text, chunk_idx)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "document_id": document_id,
                        "strategy": "sentence",
                        "language": language,
                        "sentence_start": i,
                        "sentence_end": i + len(current_sentences) - 1,
                        "char_count": len(chunk_text),
                        "is_parent": False
                    }
                })
                chunk_idx += 1
                current_sentences = [s]
                current_len = len(s)
            else:
                current_sentences.append(s)
                current_len += len(s) + 1
        
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = generate_chunk_id(document_id, "sentence", chunk_text, chunk_idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "strategy": "sentence",
                    "language": language,
                    "sentence_start": i,
                    "sentence_end": min(i + chunk_size, len(sentences)) - 1,
                    "char_count": len(chunk_text),
                    "is_parent": False
                }
            })
            chunk_idx += 1
            
    return chunks

def chunk_semantically(
    text: str,
    document_id: str,
    language: str,
    embed_fn: Callable[[List[str]], List[np.ndarray]],
    threshold: float = 0.65,
    max_chars: int = 600
) -> List[Dict[str, Any]]:
    """
    Performs semantic chunking by batch-embedding sentences, calculating
    cosine similarities between consecutive sentences, and splitting
    when similarity drops below a threshold or when character limit is reached.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
        
    if len(sentences) == 1:
        chunk_text = sentences[0]
        chunk_id = generate_chunk_id(document_id, "semantic", chunk_text, 0)
        return [{
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "document_id": document_id,
                "strategy": "semantic",
                "language": language,
                "splits_detected": 0,
                "char_count": len(chunk_text),
                "is_parent": False
            }
        }]

    # 1. Batch embed all sentences in a single model call
    embeddings = embed_fn(sentences)
    
    # 2. Compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i]
        vec2 = embeddings[i + 1]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        sim = float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(sim)

    # 3. Split sentences based on similarity and character limit
    chunks = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])
    splits = 0
    chunk_idx = 0
    
    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        # Check if similarity drops or length exceeds limit
        if sim < threshold or (current_len + len(next_sentence) + 1 > max_chars):
            # Split here
            chunk_text = " ".join(current_sentences)
            chunk_id = generate_chunk_id(document_id, "semantic", chunk_text, chunk_idx)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "strategy": "semantic",
                    "language": language,
                    "splits_detected": splits,
                    "char_count": len(chunk_text),
                    "is_parent": False
                }
            })
            chunk_idx += 1
            current_sentences = [next_sentence]
            current_len = len(next_sentence)
            splits += 1
        else:
            current_sentences.append(next_sentence)
            current_len += len(next_sentence) + 1
            
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_id = generate_chunk_id(document_id, "semantic", chunk_text, chunk_idx)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "document_id": document_id,
                "strategy": "semantic",
                "language": language,
                "splits_detected": splits,
                "char_count": len(chunk_text),
                "is_parent": False
            }
        })
        
    return chunks

def chunk_hierarchical(
    text: str,
    document_id: str,
    language: str,
    parent_size: int = 6,
    parent_overlap: int = 2,
    child_size: int = 2,
    child_overlap: int = 0,
    max_parent_chars: int = 1500,
    max_child_chars: int = 400
) -> List[Dict[str, Any]]:
    """
    Creates hierarchical parent and child chunks.
    Child chunks contain parent_id only to avoid metadata bloat.
    Parent chunks are stored as separate items tagged with is_parent=True.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    
    # Create parent chunks
    parent_step = parent_size - parent_overlap
    if parent_step <= 0:
        parent_step = 1

    parent_idx = 0
    child_idx = 0
    
    for pi in range(0, len(sentences), parent_step):
        parent_group = sentences[pi:pi + parent_size]
        if not parent_group:
            continue
            
        parent_text = " ".join(parent_group)
        if len(parent_text) > max_parent_chars:
            parent_text = parent_text[:max_parent_chars]
            
        parent_id = f"{document_id}_parent_{parent_idx}"
        
        # Save parent chunk
        chunks.append({
            "chunk_id": parent_id,
            "text": parent_text,
            "metadata": {
                "document_id": document_id,
                "strategy": "hierarchical_parent",
                "language": language,
                "is_parent": True,
                "char_count": len(parent_text)
            }
        })
        
        # Create child chunks within this parent window
        child_step = child_size - child_overlap
        if child_step <= 0:
            child_step = 1
            
        child_in_parent_idx = 0
        for ci in range(pi, min(pi + parent_size, len(sentences)), child_step):
            child_group = sentences[ci:ci + child_size]
            if not child_group:
                continue
            child_text = " ".join(child_group)
            if len(child_text) > max_child_chars:
                child_text = child_text[:max_child_chars]
                
            child_id = generate_chunk_id(document_id, "hierarchical_child", child_text, child_idx)
            
            chunks.append({
                "chunk_id": child_id,
                "text": child_text,
                "metadata": {
                    "document_id": document_id,
                    "strategy": "hierarchical_child",
                    "language": language,
                    "parent_id": parent_id,
                    "is_parent": False,
                    "child_index": child_in_parent_idx,
                    "char_count": len(child_text)
                }
            })
            child_idx += 1
            child_in_parent_idx += 1
            
        parent_idx += 1
        
    return chunks
