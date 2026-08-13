import re
import logging
import httpx
import asyncio
from typing import Optional, AsyncGenerator
from backend.interfaces import ISTTClient, ITTSClient

logger = logging.getLogger("RAG.speech.elevenlabs")
logger.setLevel(logging.INFO)

class ElevenLabsSTTClient(ISTTClient):
    """
    Client for ElevenLabs Speech-to-Text (Scribe v1) API implementing ISTTClient interface.
    Uses async httpx connection pool for non-blocking STT requests.
    """
    ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

    SUPPORTED_LOCALES = {
        "hi": "hi",
        "ta": "ta",
        "te": "te",
        "kn": "kn",
        "ml": "ml",
        "mr": "mr",
        "gu": "gu",
        "bn": "bn",
        "pa": "pa",
        "or": "or",
        "as": "as",
        "en": "en",
    }

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self._client = client
        if not self.api_key:
            logger.warning(
                "No ELEVENLABS_API_KEY provided. "
                "ElevenLabs STT running in mock fallback mode."
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20.0, http2=True)
        return self._client

    async def transcribe(self, audio_bytes: bytes, language_code: str = "hi") -> str:
        if not audio_bytes:
            return ""

        locale = self.SUPPORTED_LOCALES.get(language_code, "en")

        if not self.api_key:
            return self._mock_transcription(language_code)

        try:
            headers = {"xi-api-key": self.api_key}
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
            data = {
                "model_id": "scribe_v1",
                "language_code": locale,
            }

            logger.info(f"Submitting STT request to ElevenLabs Scribe ({locale}) ...")
            client = self._get_client()
            response = await client.post(
                self.ENDPOINT,
                headers=headers,
                files=files,
                data=data,
            )

            if response.status_code == 200:
                result = response.json()
                transcript = result.get("text", "").strip()
                logger.info(f"ElevenLabs STT success: '{transcript}'")
                return transcript

            logger.error(
                f"ElevenLabs STT failed — HTTP {response.status_code}: "
                f"{response.text}. Falling back to mock."
            )
            return self._mock_transcription(language_code)

        except Exception as exc:
            logger.error(f"ElevenLabs STT connection error: {exc}. Falling back to mock.")
            return self._mock_transcription(language_code)

    def _mock_transcription(self, language_code: str) -> str:
        mock_queries = {
            "hi": "भारत की राजधानी क्या है?",
            "ta": "இந்தியாவின் தலைநகரம் எது?",
            "te": "భారతదేశ రాజధాని ఏది?",
            "kn": "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",
            "ml": "ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്?",
            "mr": "भारताची राजधानी कोणती आहे?",
            "gu": "ભારતની રાજધાની કઈ છે?",
            "bn": "ভারতের রাজধানী কি?",
            "pa": "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ?",
            "en": "what is the capital of India?",
        }
        val = mock_queries.get(language_code, "what is the capital of India?")
        logger.info(f"STT mock activated for '{language_code}': '{val}'")
        return val


class ElevenLabsTTSClient(ITTSClient):
    """
    Client for ElevenLabs Text-to-Speech Streaming API implementing ITTSClient interface.
    Supports streaming audio directly from text or LLM token streams for real-time voice response synthesis.
    """
    BASE_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0, http2=True)
        return self._client

    async def stream_tts(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> AsyncGenerator[bytes, None]:
        """
        Streams audio/mpeg bytes for a complete text string via ElevenLabs TTS stream endpoint.
        """
        if not text:
            return

        if not self.api_key:
            async for chunk in self._mock_audio_stream():
                yield chunk
            return

        url = f"{self.BASE_ENDPOINT}/{voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        try:
            client = self._get_client()
            logger.info(f"Initiating ElevenLabs TTS audio stream (voice={voice_id})...")
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code == 200:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                else:
                    logger.error(f"ElevenLabs TTS HTTP Error ({response.status_code}). Falling back to mock audio.")
                    async for chunk in self._mock_audio_stream():
                        yield chunk
        except Exception as e:
            logger.error(f"ElevenLabs TTS stream exception ({e}). Falling back to mock audio.")
            async for chunk in self._mock_audio_stream():
                yield chunk

    async def stream_tts_from_tokens(
        self, 
        token_stream: AsyncGenerator[str, None], 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ) -> AsyncGenerator[bytes, None]:
        """
        Buffers streaming LLM tokens into sentence phrases and pipes them into ElevenLabs TTS Stream
        for ultra-low Time-To-First-Audio (TTFA).
        """
        buffer = ""
        sentence_delimiters = re.compile(r'(?<=[.!?।॥\n])\s*')

        async for token in token_stream:
            buffer += token
            sentences = sentence_delimiters.split(buffer)
            if len(sentences) > 1:
                # Complete sentence phrase ready for TTS synthesis
                complete_sentence = sentences[0].strip()
                buffer = "".join(sentences[1:])

                if complete_sentence:
                    logger.info(f"Piping sentence phrase to ElevenLabs TTS: '{complete_sentence[:50]}...'")
                    async for audio_chunk in self.stream_tts(complete_sentence, voice_id=voice_id):
                        yield audio_chunk

        # Flush remaining buffer text
        if buffer.strip():
            logger.info(f"Flushing final text buffer to ElevenLabs TTS: '{buffer.strip()[:50]}...'")
            async for audio_chunk in self.stream_tts(buffer.strip(), voice_id=voice_id):
                yield audio_chunk

    async def _mock_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Generates mock synthetic audio bytes (silent/header placeholder) for offline testing."""
        # Yield a tiny dummy MP3 header frame
        dummy_frame = b'\xff\xfb\x90\xc4' + (b'\x00' * 128)
        for _ in range(5):
            await asyncio.sleep(0.05)
            yield dummy_frame
