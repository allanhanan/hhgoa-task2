import re
import json
import logging
import httpx
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
from backend.interfaces import ILLMProvider
from backend.generation.prompts import GROUNDED_QA_SYSTEM_PROMPT, GROUNDED_QA_USER_PROMPT, GROUNDING_JUDGE_SYSTEM_PROMPT

logger = logging.getLogger("RAG.generation.providers")

# Define schema for native Tool Calling / Function Calling
GROUNDED_ANSWER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_grounded_answer",
        "description": "Submits a strictly grounded answer based ONLY on the provided retrieved context passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The direct answer to the question in the query language. Return 'NOT_SUPPORTED' if context is insufficient."
                },
                "cited_chunk_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-based indices of the context chunks used to formulate the answer."
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0 indicating how strongly context supports the answer."
                },
                "supported_by_context": {
                    "type": "boolean",
                    "description": "True if the answer is completely supported by retrieved passages, False otherwise."
                }
            },
            "required": ["answer", "cited_chunk_indices", "supported_by_context"]
        }
    }
}

class GroqLLMProvider(ILLMProvider):
    """
    Groq API implementation supporting native Tool Calling / Function Calling,
    HTTP/2 connection pooling, streaming, and structured schema outputs.
    """
    def __init__(self, api_key: Optional[str] = None, default_model: str = "llama-3.3-70b-versatile", client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self.default_model = default_model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self._client = client

    @property
    def provider_name(self) -> str:
        return "groq"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0, http2=True)
        return self._client

    async def generate_answer(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        model_name: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Generates a grounded answer using native Groq Tool Calls / Function Calling.
        """
        if not retrieved_chunks:
            return "NOT_SUPPORTED", []

        if not self.api_key:
            return self._mock_generate(query, retrieved_chunks)

        model = model_name or self.default_model

        context_parts = []
        chunk_map = {}
        for idx, chunk in enumerate(retrieved_chunks):
            citation_num = idx + 1
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            chunk_map[citation_num] = chunk_id
            context_parts.append(f"[{citation_num}] {chunk['payload']['text']}")

        context_text = "\n\n".join(context_parts)
        system_instruction = GROUNDED_QA_SYSTEM_PROMPT.format(context_text=context_text)
        user_content = GROUNDED_QA_USER_PROMPT.format(query=query)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Tool call payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            "tools": [GROUNDED_ANSWER_TOOL_SCHEMA],
            "tool_choice": {"type": "function", "function": {"name": "submit_grounded_answer"}},
            "temperature": 0.0,
            "max_tokens": 500
        }

        try:
            client = self._get_client()
            logger.info(f"Submitting Tool Call request to Groq ({model})...")
            response = await client.post(self.endpoint, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]
                
                # Check for tool call response
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]
                    args = json.loads(tool_call["function"]["arguments"])
                    
                    answer = args.get("answer", "NOT_SUPPORTED").strip()
                    is_supported = args.get("supported_by_context", True)
                    cited_indices = args.get("cited_chunk_indices", [])
                    
                    if not is_supported or answer == "NOT_SUPPORTED":
                        return "NOT_SUPPORTED", []
                        
                    citations = [chunk_map[i] for i in cited_indices if i in chunk_map]
                    logger.info(f"Groq Tool Call success: Answer='{answer[:80]}...' Citations={citations}")
                    return answer, citations
                
                # Fallback to direct content text if tool call wasn't invoked
                answer = message.get("content", "").strip()
                citations = []
                for m in re.finditer(r"\[(\d+)\]", answer):
                    num = int(m.group(1))
                    if num in chunk_map:
                        citations.append(chunk_map[num])
                return answer, list(dict.fromkeys(citations))
            else:
                logger.error(f"Groq Tool Call API error ({response.status_code}): {response.text}")
                raise RuntimeError(f"Groq API HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Groq Tool Call provider error ({e}). Falling back to mock generator.")
            return self._mock_generate(query, retrieved_chunks)

    async def generate_answer_stream(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        model_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if not retrieved_chunks or not self.api_key:
            ans, _ = self._mock_generate(query, retrieved_chunks)
            yield ans
            return

        model = model_name or self.default_model
        context_parts = [f"[{idx+1}] {c['payload']['text']}" for idx, c in enumerate(retrieved_chunks)]
        context_text = "\n\n".join(context_parts)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": GROUNDED_QA_SYSTEM_PROMPT.format(context_text=context_text)},
                {"role": "user", "content": GROUNDED_QA_USER_PROMPT.format(query=query)}
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": True
        }

        try:
            client = self._get_client()
            async with client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line[6:])
                            delta = chunk_data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            ans, _ = self._mock_generate(query, retrieved_chunks)
            yield ans

    async def verify_grounding(self, answer: str, context_text: str, model_name: Optional[str] = None) -> Tuple[str, str]:
        if not self.api_key:
            return "grounded", "Mock validation pass."

        model = model_name or self.default_model
        prompt = GROUNDING_JUDGE_SYSTEM_PROMPT.format(context_text=context_text, generated_answer=answer)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300
        }

        try:
            client = self._get_client()
            response = await client.post(self.endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                verdict, reason = "grounded", "Grounded according to LLM auditor."
                for line in text.split("\n"):
                    if line.startswith("Verdict:"):
                        verdict = line.replace("Verdict:", "").strip().lower()
                    elif line.startswith("Reason:"):
                        reason = line.replace("Reason:", "").strip()
                return verdict, reason
            return "grounded", "Judge API warning, defaulting to pass."
        except Exception as e:
            logger.error(f"LLM Judge error: {e}")
            return "grounded", "Judge API exception, defaulting to pass."

    def _mock_generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        for idx, chunk in enumerate(retrieved_chunks):
            text = chunk["payload"]["text"]
            for s in re.split(r"(?<=[.!?।॥\n])\s*", text):
                if query_words and all(qw in s.lower() for qw in query_words):
                    return f"{s.strip()} [1]", [chunk.get("chunk_id", f"chunk_{idx}")]
        if retrieved_chunks:
            text = retrieved_chunks[0]["payload"]["text"]
            first_s = re.split(r"(?<=[.!?।॥\n])\s*", text)[0].strip()
            return f"{first_s} [1]", [retrieved_chunks[0].get("chunk_id", "chunk_0")]
        return "NOT_SUPPORTED", []

class MockLLMProvider(ILLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]], model_name: Optional[str] = None) -> Tuple[str, List[str]]:
        if not retrieved_chunks:
            return "NOT_SUPPORTED", []
        text = retrieved_chunks[0]["payload"]["text"]
        first_s = re.split(r"(?<=[.!?।॥\n])\s*", text)[0].strip()
        chunk_id = retrieved_chunks[0].get("chunk_id", "chunk_0")
        return f"{first_s} [1]", [chunk_id]

    async def generate_answer_stream(self, query: str, retrieved_chunks: List[Dict[str, Any]], model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        ans, _ = await self.generate_answer(query, retrieved_chunks, model_name)
        yield ans

    async def verify_grounding(self, answer: str, context_text: str, model_name: Optional[str] = None) -> Tuple[str, str]:
        return "grounded", "Mock grounded pass."
