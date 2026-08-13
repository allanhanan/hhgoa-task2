import re
import hashlib
from typing import Dict, Any

def clean_text(text: str) -> str:
    """
    Cleans text by removing duplicate whitespaces, HTML/markdown boilerplate,
    invalid characters, and normalizing unicode.
    """
    if not text:
        return ""
    
    # 1. Decode bytes if input is bytes
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError:
            text = text.decode("latin-1", errors="ignore")

    # 2. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Standardize whitespace
    text = re.sub(r"\s+", " ", text)
    
    # 4. Remove malformed characters or surrogate pairs
    text = text.encode("utf-8", "ignore").decode("utf-8")
    
    return text.strip()

def detect_language(text: str, declared_lang: str = None) -> Dict[str, str]:
    """
    Detects language script and maps to native name. Falls back to declared_lang if provided.
    Indic scripts have distinct Unicode ranges:
    - Devanagari (Hindi, Marathi): \u0900-\u097F
    - Bengali/Assamese: \u0980-\u09FF
    - Punjabi (Gurmukhi): \u0A00-\u0A7F
    - Gujarati: \u0A80-\u0AFF
    - Odia: \u0B00-\u0B7F
    - Tamil: \u0B80-\u0BFF
    - Telugu: \u0C00-\u0C7F
    - Kannada: \u0C80-\u0CFF
    - Malayalam: \u0D00-\u0D7F
    """
    if not text:
        return {"language": "en", "script": "Latin"}

    # Heuristic based on Unicode ranges
    unicode_ranges = {
        "hi": (0x0900, 0x097F, "Devanagari"),
        "mr": (0x0900, 0x097F, "Devanagari"),
        "bn": (0x0980, 0x09FF, "Bengali"),
        "as": (0x0980, 0x09FF, "Bengali"),
        "pa": (0x0A00, 0x0A7F, "Gurmukhi"),
        "gu": (0x0A80, 0x0AFF, "Gujarati"),
        "or": (0x0B00, 0x0B7F, "Odia"),
        "ta": (0x0B80, 0x0BFF, "Tamil"),
        "te": (0x0C00, 0x0C7F, "Telugu"),
        "kn": (0x0C80, 0x0CFF, "Kannada"),
        "ml": (0x0D00, 0x0D7F, "Malayalam"),
    }

    counts = {lang: 0 for lang in unicode_ranges}
    total_indic = 0

    for char in text:
        val = ord(char)
        for lang, (start, end, script) in unicode_ranges.items():
            if start <= val <= end:
                counts[lang] += 1
                total_indic += 1
                break

    if total_indic > 2:  # Safe threshold for script detection
        detected_lang = max(counts, key=counts.get)
        _, _, script_name = unicode_ranges[detected_lang]
        
        # Refine if user declared another language using the same script
        if declared_lang in ["hi", "mr"] and detected_lang in ["hi", "mr"]:
            return {"language": declared_lang, "script": script_name}
        if declared_lang in ["bn", "as"] and detected_lang in ["bn", "as"]:
            return {"language": declared_lang, "script": script_name}
            
        return {"language": detected_lang, "script": script_name}

    if declared_lang:
        # Check if declared is English
        if declared_lang.startswith("en"):
            return {"language": "en", "script": "Latin"}
        return {"language": declared_lang, "script": "Other"}

    return {"language": "en", "script": "Latin"}

class Deduplicator:
    """
    Deduplicates text passages using a hash set.
    """
    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        cleaned = clean_text(text)
        text_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def reset(self):
        self.seen_hashes.clear()
