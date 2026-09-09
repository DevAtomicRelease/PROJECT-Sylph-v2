"""
Kokoro TTS Synthesizer
Optimized: GPU-first execution + true streaming output

Key changes vs. the batch version:
- Tries CUDA first. Kokoro-82M is ~330MB on GPU — trivially co-resident with
  the 9B LLM on an 8GB RTX 4060 and roughly 10-20x faster than 4-thread CPU.
- `synthesize_stream()` is an async generator that yields each audio segment
  the moment Kokoro produces it, instead of synthesizing the whole clause
  list and returning a batch. Time-to-first-audio drops from
  O(full clause set) to O(first segment).
- CPU fallback uses a real thread budget instead of a hard-coded 4.
- The legacy `synthesize()` batch API is preserved for the /speak test
  endpoint and any other batch callers.
"""

import asyncio
import logging
import os
import queue as thread_queue
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

from .audio_post import apply_emotion_fx

logger = logging.getLogger("sylph.tts.synthesizer")

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_bella"

# Legacy emotion→speed shaping for the Kokoro fallback path. Kokoro cannot
# express emotion natively, so label-driven speed + audio_post pitch tricks
# are the best it can do. (Zonos conditions on emotion directly and never
# uses this.)
EMOTION_SPEED_MAP: dict[str, float] = {
    "happy":     1.08,
    "angry":     1.12,
    "sad":       0.88,
    "relaxed":   0.95,
    "surprised": 1.10,
    "neutral":   1.00,
    "shy":       0.92,
    "bored":     0.90,
}

_SENTINEL = object()

_tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker")


class Synthesizer:
    """
    Kokoro TTS engine. Produces audio + phoneme data per clause.
    Lazy-loads on first synthesis call. GPU-first, CPU fallback.
    """

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self._pipeline = None
        self._loaded = False
        self._device = "cpu"
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the Kokoro pipeline (GPU-first)."""
        with self._lock:
            if self._loaded:
                return
            from kokoro import KPipeline
            import torch

            if torch.cuda.is_available():
                try:
                    logger.info("Loading Kokoro TTS on CUDA (voice='%s')...", self.voice)
                    self._pipeline = KPipeline(lang_code="a", device="cuda")
                    self._device = "cuda"
                    # Warm-up pass so the first real clause doesn't pay CUDA init cost
                    for _ in self._pipeline("Ready.", voice=self.voice):
                        pass
                    self._loaded = True
                    logger.info("Kokoro TTS loaded on CUDA and warmed up")
                    return
                except Exception as e:
                    logger.warning("Kokoro CUDA load failed (%s) — falling back to CPU", e)
                    self._pipeline = None

            # CPU fallback: give PyTorch a real thread budget
            cpu_threads = max(4, (os.cpu_count() or 8) - 2)
            torch.set_num_threads(cpu_threads)
            logger.info(
                "Loading Kokoro TTS on CPU (%d torch threads, voice='%s')...",
                cpu_threads, self.voice,
            )
            self._pipeline = KPipeline(lang_code="a")
            self._device = "cpu"
            self._loaded = True
            logger.info("Kokoro TTS loaded on CPU")

    # ------------------------------------------------------------------
    # Internal: normalize Kokoro output to float32 numpy
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(audio) -> np.ndarray:
        if hasattr(audio, "numpy"):
            audio_np = audio.detach().cpu().numpy().astype(np.float32)
        elif isinstance(audio, np.ndarray):
            audio_np = audio.astype(np.float32)
        else:
            audio_np = np.array(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        return audio_np

    # ------------------------------------------------------------------
    # Streaming API (preferred): yields segments as Kokoro produces them
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding {"graphemes", "phonemes", "audio"} dicts
        as each segment finishes synthesis on the worker thread.

        emotion_label/emotion_intensity match the ZonosSynthesizer signature;
        Kokoro approximates them with its legacy speed map + audio FX.
        """
        if not self._loaded:
            await asyncio.get_running_loop().run_in_executor(_tts_pool, self.load)
        if self._pipeline is None:
            return

        effective_speed = speed * EMOTION_SPEED_MAP.get(emotion_label, 1.0)

        q: thread_queue.Queue = thread_queue.Queue(maxsize=4)

        def _producer():
            try:
                for graphemes, phonemes, audio in self._pipeline(
                    text, voice=self.voice, speed=effective_speed
                ):
                    audio_np = self._to_numpy(audio)
                    if emotion_label != "neutral":
                        audio_np = apply_emotion_fx(audio_np, emotion_label, SAMPLE_RATE)
                    q.put({
                        "graphemes": graphemes,
                        "phonemes": phonemes,
                        "audio": audio_np,
                    })
            except Exception as e:
                logger.error("Kokoro synthesis error: %s", e)
            finally:
                q.put(_SENTINEL)

        loop = asyncio.get_running_loop()
        producer_future = loop.run_in_executor(_tts_pool, _producer)

        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _SENTINEL:
                    break
                yield item
        finally:
            await producer_future

    # ------------------------------------------------------------------
    # Batch API (legacy): kept for /speak endpoint and tests
    # ------------------------------------------------------------------

    def synthesize_sync(self, text: str, speed: float = 1.0) -> list[dict]:
        if not self._loaded:
            self.load()
        if self._pipeline is None:
            return []
        results = []
        for graphemes, phonemes, audio in self._pipeline(
            text, voice=self.voice, speed=speed
        ):
            results.append({
                "graphemes": graphemes,
                "phonemes": phonemes,
                "audio": self._to_numpy(audio),
            })
        return results

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> list[dict]:
        loop = asyncio.get_running_loop()
        effective_speed = speed * EMOTION_SPEED_MAP.get(emotion_label, 1.0)
        return await loop.run_in_executor(
            _tts_pool, self.synthesize_sync, text, effective_speed
        )

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device
