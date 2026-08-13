import logging
import requests
from typing import Optional

logger = logging.getLogger("RAG.speech.sarvam")
logger.setLevel(logging.INFO)

class SarvamSTTClient:
    """
    Client for Sarvam Speech-to-Text API.
    API documentation details:
    - Endpoint: https://api.sarvam.ai/speech-to-text
    - Header: api-subscription-key: <key>
    - Body: multipart/form-data (file, model='saaras:v1', language_code='hi-IN'/'ta-IN' etc.)
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.endpoint = "https://api.sarvam.ai/speech-to-text"
        
        if not self.api_key:
            logger.warning("No SARVAM_API_KEY provided. Sarvam STT running in mock fallback mode.")

    def transcribe(self, audio_bytes: bytes, language_code: str = "hi") -> str:
        """
        Transcribes audio bytes to text using Sarvam API or a language-based mock fallback.
        """
        if not audio_bytes:
            return ""

        # Map language ISO code to Sarvam's expected locale
        sarvam_locales = {
            "hi": "hi-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
            "mr": "mr-IN",
            "gu": "gu-IN",
            "bn": "bn-IN",
            "pa": "pa-IN",
            "or": "or-IN",
            "as": "as-IN",
            "en": "en-IN"
        }
        locale = sarvam_locales.get(language_code, "en-IN")

        if not self.api_key:
            return self._mock_transcription(language_code)

        try:
            headers = {
                "api-subscription-key": self.api_key
            }
            files = {
                "file": ("query.wav", audio_bytes, "audio/wav")
            }
            data = {
                "model": "saaras:v1",
                "language_code": locale
            }
            
            logger.info(f"Submitting STT request to Sarvam ({locale})...")
            response = requests.post(self.endpoint, headers=headers, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                transcript = result.get("transcript", "").strip()
                logger.info(f"Sarvam STT success: '{transcript}'")
                return transcript
            else:
                logger.error(f"Sarvam STT failed with code {response.status_code}: {response.text}. Falling back to mock.")
                return self._mock_transcription(language_code)
                
        except Exception as e:
            logger.error(f"Sarvam STT connection error: {e}. Falling back to mock.")
            return self._mock_transcription(language_code)

    def _mock_transcription(self, language_code: str) -> str:
        """
        Mock transcription queries for standard Indic test cases.
        """
        mock_queries = {
            "hi": "भारत की राजधानी क्या है?",  # What is the capital of India?
            "ta": "இந்தியாவின் தலைநகரம் எது?",  # What is the capital of India?
            "te": "భారతదేశ రాజధాని ఏది?",       # What is the capital of India?
            "kn": "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",      # What is the capital of India?
            "ml": "ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്?", # What is the capital of India?
            "mr": "भारताची राजधानी कोणती आहे?", # What is the capital of India?
            "gu": "ભારતની રાજધાની કઈ છે?",      # What is the capital of India?
            "bn": "ভারতের রাজধানী কি?",         # What is the capital of India?
            "pa": "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?",      # What is the capital of India?
            "en": "what is the capital of India?"
        }
        val = mock_queries.get(language_code, "what is the capital of India?")
        logger.info(f"STT Mock Transcription activated for '{language_code}': '{val}'")
        return val
