"""
Ollama Streaming Client — Cloud Edition
Phase 5.1: Async HTTP client for the Ollama chat API
Phase 11:  Migrated from local Ollama to Ollama Cloud (https://ollama.com)

The cloud host serves the same native API surface as a local daemon
(/api/chat, /api/tags), so this client works against either — the base URL
decides. Differences handled here:

- Auth: cloud requires `Authorization: Bearer $OLLAMA_API_KEY`.
- keep_alive / preload are local-daemon concepts — never sent to the cloud.
- Network failures are now expected operating conditions, not exceptional:
  connection attempts retry with backoff INSIDE this client (the planner's
  old retry wrapped lazy generator creation, which can never fail), and
  errors are classified into typed exceptions carrying a speakable
  `user_message` so the avatar can explain what's wrong in character.
- Free-tier quota (429) and auth failures (401/403) are terminal per turn:
  they are never retried, so a capped account doesn't get hammered.

Supports vision (image attachments) for Phase 7.
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger("sylph.llm.ollama")

# Phase A: local-first defaults. The .env still wins (loaded before this
# module), so these only bite when it's absent — point them at the local
# daemon + small multimodal brain rather than the cloud host.
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("MODEL_TAG", "qwen3.5:4b")

# Connection: fail fast. Read: generous — a busy cloud queue can pause the
# stream between chunks. Total turn latency is bounded by the planner's UX,
# not by this timeout.
CLOUD_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)

# Connection-phase retry schedule (seconds between attempts).
RETRY_BACKOFF = [1.5, 4.0]


class OllamaError(Exception):
    """Base class for classified Ollama failures.

    `user_message` is a short, in-character-neutral line the TTS pipeline can
    speak so the user learns what happened without reading logs.
    """

    user_message = "I'm having trouble reaching my language model right now."

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.user_message)
        self.detail = detail


class OllamaAuthError(OllamaError):
    user_message = (
        "My cloud brain rejected the API key. "
        "Check the OLLAMA_API_KEY entry in the environment file."
    )


class OllamaQuotaError(OllamaError):
    user_message = (
        "I've hit the free usage limit on my cloud brain for now. "
        "It resets on its own — give it a little while."
    )


class OllamaUnavailableError(OllamaError):
    user_message = (
        "I can't reach my cloud brain — the network or the service looks down. "
        "I'll be back to normal once the connection recovers."
    )


def _classify_http_error(e: httpx.HTTPStatusError) -> OllamaError:
    code = e.response.status_code
    if code in (401, 403):
        return OllamaAuthError(f"HTTP {code}")
    if code == 429:
        return OllamaQuotaError("HTTP 429 — free-tier usage cap reached")
    return OllamaUnavailableError(f"HTTP {code}")


class OllamaClient:
    """Async Ollama chat client with streaming support (cloud or local)."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Resolve at construction time (after env_config has loaded .env),
        # not at import time.
        self.model = model or os.environ.get("MODEL_TAG", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
        self.chat_url = f"{self.base_url}/api/chat"
        self._client: Optional[httpx.AsyncClient] = None

        if self.is_cloud and not self.api_key:
            logger.warning(
                "Ollama Cloud selected (%s) but OLLAMA_API_KEY is empty — "
                "requests will fail with 401. Create a key at "
                "https://ollama.com/settings/keys and add it to .env",
                self.base_url,
            )
        logger.info(
            "Ollama client: model=%s host=%s (%s)",
            self.model, self.base_url, "cloud" if self.is_cloud else "local",
        )

    @property
    def is_cloud(self) -> bool:
        """True when talking to a remote host rather than a local daemon."""
        return not any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))

    @property
    def is_cloud_model(self) -> bool:
        """
        True when the MODEL runs in the cloud — either because the host is
        remote, or because a local daemon is proxying a `-cloud`/`:cloud`
        stub (signed-in daemon path). Local-daemon concepts like keep_alive
        and VRAM preloading don't apply to these models either way.
        """
        return self.is_cloud or self.model.endswith("-cloud") or self.model.endswith(":cloud")

    def _headers(self) -> dict:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=CLOUD_TIMEOUT, headers=self._headers())
        return self._client

    # ------------------------------------------------------------------
    # Payload assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_images(messages: list[dict], images: Optional[list[str]]) -> None:
        if not images:
            return
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["images"] = images
                break

    def _base_payload(self, messages: list[dict], stream: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            # Qwen3.x reasoning mode adds seconds of <think> tokens before the
            # first audible word — pure latency for a realtime companion, and
            # on the cloud it also burns free-tier GPU-time. Disable it.
            "think": False,
        }
        # keep_alive pins the model in a LOCAL daemon's VRAM. The cloud pool
        # manages its own residency; sending it there is meaningless.
        if not self.is_cloud_model:
            payload["keep_alive"] = -1
        return payload

    # ------------------------------------------------------------------
    # Connection with retry — the single funnel every request goes through
    # ------------------------------------------------------------------

    async def _post_with_retry(self, payload: dict) -> httpx.Response:
        """Non-streaming POST with connection retries. Raises OllamaError."""
        client = await self._get_client()
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate([0.0] + RETRY_BACKOFF):
            if delay:
                await asyncio.sleep(delay)
            try:
                r = await client.post(self.chat_url, json=payload)
                r.raise_for_status()
                return r
            except httpx.HTTPStatusError as e:
                err = _classify_http_error(e)
                if isinstance(err, (OllamaAuthError, OllamaQuotaError)):
                    raise err  # never retry auth/quota
                last_exc = err
                logger.warning("Ollama HTTP %s (attempt %d)", e.response.status_code, attempt + 1)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
                last_exc = e
                logger.warning("Ollama connection failed (attempt %d): %s", attempt + 1, e)
        raise OllamaUnavailableError(str(last_exc))

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    async def stream_chat_raw(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream raw JSON chunks (content and tool calls).

        Connection-phase failures retry with backoff; once the stream is
        open, a mid-stream failure raises immediately (retrying would replay
        the whole response). All failures surface as typed OllamaError.
        """
        payload = self._base_payload(messages, stream=True)
        payload["options"] = {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,
            # Phase A: the 320 cap existed to conserve free-tier cloud GPU-time.
            # Local inference has no quota, so let replies breathe. Still bounded
            # so a runaway generation can't monologue forever in a voice UI.
            "num_predict": 768,
        }
        if tools:
            payload["tools"] = tools
        self._attach_images(messages, images)

        client = await self._get_client()
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate([0.0] + RETRY_BACKOFF):
            if delay:
                await asyncio.sleep(delay)
            stream_opened = False
            try:
                async with client.stream("POST", self.chat_url, json=payload) as response:
                    if response.status_code >= 400:
                        # Read the body so the error is complete, then classify.
                        await response.aread()
                        response.raise_for_status()
                    stream_opened = True
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Unparseable Ollama chunk: %s", line[:100])
                    return
            except httpx.HTTPStatusError as e:
                err = _classify_http_error(e)
                if isinstance(err, (OllamaAuthError, OllamaQuotaError)):
                    logger.error("Ollama request rejected: %s", err.detail)
                    raise err
                last_exc = err
                logger.warning("Ollama HTTP %s (attempt %d)", e.response.status_code, attempt + 1)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
                if stream_opened:
                    # Mid-stream death: tokens were already consumed downstream,
                    # a retry would duplicate speech. Fail loudly instead.
                    logger.error("Ollama stream died mid-response: %s", e)
                    raise OllamaUnavailableError(f"stream interrupted: {e}")
                last_exc = e
                logger.warning("Ollama connect failed (attempt %d): %s", attempt + 1, e)

        logger.error("Ollama unreachable after %d attempts: %s", len(RETRY_BACKOFF) + 1, last_exc)
        raise OllamaUnavailableError(str(last_exc))

    async def stream_chat(
        self,
        messages: list[dict],
        images: Optional[list[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """
        Stream plain content tokens. Soft-fail wrapper around stream_chat_raw:
        on error it yields a bracketed error string instead of raising, which
        callers like the screen-reaction generator treat as degraded output.
        """
        try:
            async for chunk in self.stream_chat_raw(
                messages, images=images, temperature=temperature, top_p=top_p,
            ):
                if chunk.get("done"):
                    logger.debug("Stream complete: %s tokens", chunk.get("eval_count", "?"))
                    return
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except OllamaError as e:
            logger.error("stream_chat failed: %s", e.detail)
            yield f"[Error: {e.detail or e.user_message}]"

    # ------------------------------------------------------------------
    # Non-streaming API
    # ------------------------------------------------------------------

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
        payload = self._base_payload(messages, stream=False)
        payload["options"] = {"temperature": temperature}
        if tools:
            payload["tools"] = tools
        self._attach_images(messages, images)
        r = await self._post_with_retry(payload)
        return r.json()

    # ------------------------------------------------------------------
    # Health / startup
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Check the API host is reachable (and, for cloud, the key works)."""
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
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
        """
        Local daemon: force-load the model into VRAM.
        Cloud: cheap readiness probe instead — validates connectivity, the API
        key, and that MODEL_TAG exists in the live catalog. Zero GPU-time.
        """
        if self.is_cloud_model:
            try:
                models = await self.list_models()
                if not models:
                    logger.warning(
                        "Ollama Cloud readiness probe got no catalog — "
                        "check network and OLLAMA_API_KEY (direct) or that the "
                        "signed-in local daemon is running (proxy)"
                    )
                    return False
                base = self.model.split(":")[0]
                if not any(m == self.model or m.startswith(base + ":") for m in models):
                    logger.warning(
                        "MODEL_TAG '%s' not in the cloud catalog %s — requests may 404",
                        self.model, models,
                    )
                    return False
                logger.info("Ollama Cloud ready: '%s' present in live catalog", self.model)
                return True
            except Exception as e:
                logger.warning("Ollama Cloud readiness probe failed: %s", e)
                return False

        payload = {"model": self.model, "keep_alive": -1}
        client = await self._get_client()
        try:
            logger.info("Requesting local Ollama to preload %s...", self.model)
            r = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=120.0)
            if r.status_code == 200:
                logger.info("Preloaded %s into local VRAM", self.model)
                return True
            logger.warning("Failed to preload %s: HTTP %d", self.model, r.status_code)
        except Exception as e:
            logger.error("Error preloading %s: %s", self.model, e)
        return False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
