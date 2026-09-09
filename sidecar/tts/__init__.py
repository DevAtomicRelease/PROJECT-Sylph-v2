"""Sylph TTS (Text-to-Speech) Module — Phase 4 / Phase 12 (Zonos)"""

import asyncio
import logging
import os
from typing import AsyncGenerator

from .synthesizer import Synthesizer
from .duration_estimator import estimate_durations, parse_phoneme_string
from .viseme_builder import build_viseme_timeline, timeline_to_compact
from .payload import build_dual_payload

logger = logging.getLogger("sylph.tts")


class ResilientSynthesizer:
    """
    Facade over the primary engine (Zonos, emotional) with automatic
    fallback to Kokoro (flat but battle-tested).

    Fallback triggers:
    - load() failure (missing package, no CUDA, model download failed)
    - a generation failure at runtime (e.g. CUDA OOM mid-session)

    Once fallen back, the session stays on Kokoro — flapping between engines
    mid-conversation would change the voice repeatedly, which is worse than
    losing emotion conditioning.
    """

    def __init__(self):
        self._engine_name = "zonos"
        try:
            from .zonos_synthesizer import ZonosSynthesizer
            self._active = ZonosSynthesizer()
        except Exception as e:
            logger.warning("Zonos unavailable at import (%s) — using Kokoro", e)
            self._active = Synthesizer()
            self._engine_name = "kokoro"

    # -- engine management -------------------------------------------------

    @property
    def engine(self) -> str:
        return self._engine_name

    def _fall_back(self, reason: Exception) -> None:
        if self._engine_name == "kokoro":
            return
        logger.error(
            "Zonos failed (%s) — falling back to Kokoro for this session", reason
        )
        self._active = Synthesizer()
        self._engine_name = "kokoro"

    def load(self) -> None:
        try:
            self._active.load()
        except Exception as e:
            self._fall_back(e)
            self._active.load()

    # -- delegated API -----------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> AsyncGenerator[dict, None]:
        try:
            async for seg in self._active.synthesize_stream(
                text, speed=speed,
                emotion_label=emotion_label, emotion_intensity=emotion_intensity,
            ):
                yield seg
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._fall_back(e)

        # Retry the clause once on the fallback engine so the utterance
        # isn't silently swallowed.
        async for seg in self._active.synthesize_stream(
            text, speed=speed,
            emotion_label=emotion_label, emotion_intensity=emotion_intensity,
        ):
            yield seg

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> list[dict]:
        out = []
        async for seg in self.synthesize_stream(
            text, speed=speed,
            emotion_label=emotion_label, emotion_intensity=emotion_intensity,
        ):
            out.append(seg)
        return out

    # -- attribute passthrough (voice, _loaded, sample_rate, device) -------

    @property
    def sample_rate(self) -> int:
        return self._active.sample_rate

    @property
    def device(self) -> str:
        return self._active.device

    @property
    def _loaded(self) -> bool:
        return getattr(self._active, "_loaded", False)

    @property
    def voice(self) -> str:
        return getattr(self._active, "voice", "af_bella")

    @voice.setter
    def voice(self, value: str) -> None:
        try:
            self._active.voice = value
        except Exception:
            pass


def create_synthesizer():
    """Build the configured TTS engine (TTS_ENGINE=zonos|kokoro, default zonos)."""
    engine = os.environ.get("TTS_ENGINE", "zonos").strip().lower()
    if engine == "kokoro":
        logger.info("TTS engine: Kokoro (explicit via TTS_ENGINE)")
        return Synthesizer()
    return ResilientSynthesizer()


__all__ = [
    "Synthesizer",
    "ResilientSynthesizer",
    "create_synthesizer",
    "estimate_durations",
    "parse_phoneme_string",
    "build_viseme_timeline",
    "timeline_to_compact",
    "build_dual_payload",
]
