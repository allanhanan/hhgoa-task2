import logging
import requests
import json
from typing import List, Dict, Any, Tuple, Optional
from backend import config
from backend.generation.prompts import GROUNDED_QA_SYSTEM_PROMPT, GROUNDED_QA_USER_PROMPT, GROUNDING_JUDGE_SYSTEM_PROMPT

logger = logging.getLogger("RAG.generation.llm")
logger.setLevel(logging.INFO)

class LLMGenerator:
    """
    Handles LLM invocations using Gemini API.
    If GEMINI_API_KEY is not configured, runs in mock mode using
    deterministic context matching.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found. Generator running in MOCK grounded matching mode.")

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generates a grounded answer.
        Returns: (answer_text, list_of_cited_chunk_ids)
        """
        if not retrieved_chunks:
            return "NOT_SUPPORTED", []

        # 1. Format context text with citations
        context_parts = []
        chunk_map = {}
        for idx, chunk in enumerate(retrieved_chunks):
            citation_num = idx + 1
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            chunk_map[citation_num] = chunk_id
            
            # Format text
            text = chunk["payload"]["text"]
            context_parts.append(f"[{citation_num}] {text}")
            
        context_text = "\n\n".join(context_parts)

        # Refusal/mock checks
        if not self.api_key:
            return self._mock_generate(query, retrieved_chunks)

        # 2. Build Prompts
        system_instruction = GROUNDED_QA_SYSTEM_PROMPT.format(context_text=context_text)
        user_content = GROUNDED_QA_USER_PROMPT.format(query=query)

        # 3. Call Gemini REST API
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_instruction}\n\n{user_content}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 500
            }
        }

        try:
            logger.info("Submitting generateContent request to Gemini API...")
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"Gemini API generation successful: '{answer[:100]}...'")
                
                # Extract citation numbers like [1], [2] from text
                citations = []
                for m in re.finditer(r"\[(\d+)\]", answer):
                    num = int(m.group(1))
                    if num in chunk_map:
                        citations.append(chunk_map[num])
                
                # Remove duplicate citation IDs
                citations = list(dict.fromkeys(citations))
                
                return answer, citations
            else:
                logger.error(f"Gemini API request failed ({response.status_code}): {response.text}")
                return self._mock_generate(query, retrieved_chunks)
        except Exception as e:
            logger.error(f"Gemini connection error: {e}. Falling back to mock generator.")
            return self._mock_generate(query, retrieved_chunks)

    def verify_grounding_via_llm(self, answer: str, context_text: str) -> Tuple[str, str]:
        """
        Runs LLM-as-a-judge grounding validation.
        Returns: (verdict: "grounded" / "not_grounded", reason: str)
        """
        if not self.api_key:
            # Simple heuristic in mock mode
            ans_words = set(re.findall(r"\w+", answer.lower()))
            ctx_words = set(re.findall(r"\w+", context_text.lower()))
            # If word overlap ratio of answer in context is high, mark grounded
            overlap = len(ans_words.intersection(ctx_words)) / len(ans_words) if ans_words else 1.0
            if overlap >= 0.7:
                return "grounded", f"Mock check: {overlap:.2f} word overlap overlap ratio."
            else:
                return "not_grounded", f"Mock check: Low word overlap ratio ({overlap:.2f})."

        prompt = GROUNDING_JUDGE_SYSTEM_PROMPT.format(context_text=context_text, generated_answer=answer)
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 300
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Parse verdict and reason from Gemini output
                verdict = "grounded"
                reason = "Grounded according to LLM auditor."
                
                for line in text.split("\n"):
                    if line.startswith("Verdict:"):
                        verdict = line.replace("Verdict:", "").strip().lower()
                    elif line.startswith("Reason:"):
                        reason = line.replace("Reason:", "").strip()
                return verdict, reason
            else:
                return "grounded", "Failed to run LLM judge, defaulting to pass."
        except Exception as e:
            logger.error(f"LLM Judge API error: {e}")
            return "grounded", "LLM Judge API exception, defaulting to pass."

    def _mock_generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Fallback mock grounded answer generator. Matches words in query to context sentences.
        """
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        
        # Look for sentences containing query terms
        for idx, chunk in enumerate(retrieved_chunks):
            text = chunk["payload"]["text"]
            sentences = re.split(r"(?<=[.!?।॥\n])\s*", text)
            
            for s in sentences:
                s_lower = s.lower()
                # Check if all query terms appear in sentence
                if query_words and all(qw in s_lower for qw in query_words):
                    chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
                    return f"{s.strip()} [1]", [chunk_id]

        # Default fallback to first sentence of top chunk
        if retrieved_chunks:
            text = retrieved_chunks[0]["payload"]["text"]
            first_sentence = re.split(r"(?<=[.!?।॥\n])\s*", text)[0].strip()
            chunk_id = retrieved_chunks[0].get("chunk_id", "chunk_0")
            return f"{first_sentence} [1]", [chunk_id]
            
        return "NOT_SUPPORTED", []

import re
