"""
Ollama Streaming Client
Phase 5.1: Async HTTP client for the local Ollama API

Streams tokens from POST /api/chat with streaming enabled.
Supports vision (image attachments) for Phase 7.
"""

import json
import logging
import os
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger("sylph.llm.ollama")

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"
DEFAULT_MODEL = os.environ.get("MODEL_TAG", "qwen3.5:9b")


class OllamaClient:
    """Async Ollama chat client with streaming support."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def stream_chat(
        self,
        messages: list[dict],
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat tokens from Ollama.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            images: Optional list of base64-encoded images (for vision models)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Yields:
            Individual token strings as they arrive
        """
        # If images are provided, attach them to the last user message
        if images:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["images"] = images
                    break

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        client = await self._get_client()

        try:
            async with client.stream("POST", self.chat_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        if chunk.get("done"):
                            logger.debug(
                                "Stream complete: %s tokens",
                                chunk.get("eval_count", "?"),
                            )
                            return
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama chunk: %s", line[:100])
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", self.base_url)
            yield "[Error: Ollama not running. Start with `ollama serve`]"
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s", e)
            yield f"[Error: Ollama returned {e.response.status_code}]"

    async def stream_chat_raw(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> AsyncGenerator[dict, None]:
        """Stream raw JSON chunks from Ollama, including content and tool calls."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # Qwen3.x reasoning mode adds 2-10s of <think> tokens before the
            # first audible word. For a real-time companion that is pure
            # latency with no quality benefit on chat turns — disable it.
            "think": False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repetition_penalty,
                "num_predict": 320,
            },
        }
        if tools:
            payload["tools"] = tools
        if images:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["images"] = images
                    break

        client = await self._get_client()
        try:
            async with client.stream("POST", self.chat_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse raw Ollama chunk: %s", line[:100])
        except Exception as e:
            logger.error("Ollama stream_chat_raw failed: %s", e)
            raise e

    async def chat(
        self,
        messages: list[dict],
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming chat — returns the full response as a single string."""
        parts = []
        async for token in self.stream_chat(messages, images, temperature):
            parts.append(token)
        return "".join(parts)

    async def chat_raw(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
    ) -> dict:
        """Non-streaming chat that returns the raw Ollama response dict."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if tools:
            payload["tools"] = tools
        if images:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    msg["images"] = images
                    break

        client = await self._get_client()
        try:
            r = await client.post(self.chat_url, json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("Ollama chat_raw failed: %s", e)
            raise e


    async def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/api/tags")
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    async def preload_model(self) -> bool:
        """Force Ollama to load the model into memory/VRAM on startup."""
        payload = {
            "model": self.model,
            "keep_alive": -1  # Keep loaded indefinitely to prevent unloading
        }
        client = await self._get_client()
        try:
            logger.info("Requesting Ollama to preload model %s...", self.model)
            r = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=120.0)
            if r.status_code == 200:
                logger.info("Successfully preloaded LLM model %s into memory", self.model)
                return True
            else:
                logger.warning("Failed to preload LLM model %s: HTTP %d", self.model, r.status_code)
        except Exception as e:
            logger.error("Error preloading LLM model %s: %s", self.model, e)
        return False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
