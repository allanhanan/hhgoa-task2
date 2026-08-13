import re
import logging
from typing import Tuple

logger = logging.getLogger("RAG.guardrails.safety")
logger.setLevel(logging.INFO)

# List of common jailbreak/prompt injection indicators
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a\s+dan",
    r"do\s+anything\s+now",
    r"bypass\s+(?:your\s+)?filter",
    r"system\s+prompt",
    r"you\s+must\s+forget",
    r"translate\s+the\s+system\s+instruction",
    r"acting\s+as\s+a\s+developer",
    r"ignore\s+above\s+and\s+do",
]

# Off-topic patterns (e.g., explicit hacking requests, malware creation)
OFF_TOPIC_PATTERNS = [
    r"hack\s+someone",
    r"write\s+(?:a\s+)?malware",
    r"create\s+(?:a\s+)?virus",
    r"bypass\s+security",
    r"steal\s+credentials",
    r"sql\s+injection\s+exploit",
    r"crack\s+password",
]

def check_query_safety(query: str) -> Tuple[bool, str]:
    """
    Evaluates safety of input queries.
    Returns: (is_safe, safety_status)
    """
    if not query:
        return True, "SAFE"

    query_lower = query.lower()

    # 1. Check for prompt injection
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Safety violation: PROMPT_INJECTION matched '{pattern}'")
            return False, "PROMPT_INJECTION_VIOLATION"

    # 2. Check for off-topic / malicious intent
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Safety violation: OFF_TOPIC matched '{pattern}'")
            return False, "OFF_TOPIC_MALICIOUS_VIOLATION"

    return True, "SAFE"
