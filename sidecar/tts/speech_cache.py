"""
Emotional Speech Cache
Phase 12b: Real Zonos emotion for the fixed personality lines, at zero latency

Benchmarks on the RTX 4060 Laptop put Zonos at RTF ≈ 8 eager (≈ 3 with CUDA
graphs) — too slow to synthesize live replies in realtime. But the autonomous
mumbles ("*sighs*", "So quiet today...") are a FIXED, known set: they can be
rendered once in the background with the full Zonos emotion vector and then
replayed instantly from disk forever after.

Result: live conversation stays realtime on the fallback voice, while the
personality moments — the lines that most need to *feel* something — get the
genuinely emotional voice. First playback of a given line uses the realtime
engine; the Zonos take exists by the next time.

Only active when the live TTS engine is NOT Zonos (otherwise it's redundant).
"""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("sylph.tts.speech_cache")

# Unload the background Zonos model after this long without a render — it
# holds ~3.6 GB of an 8 GB card, and renders are rare once the cache warms.
_UNLOAD_AFTER_SECONDS = 600

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CACHE_DIR = os.path.join(_WORKSPACE_ROOT, "models", "zonos", "mumble_cache")
_MUMBLES_PATH = os.path.join(_WORKSPACE_ROOT, "assets", "config", "mumbles.json")

# One take per (line, emotion); rendered expressive — these ARE the emotional
# moments, a timid take defeats the purpose.
_CACHE_INTENSITY = 0.9


class EmotionalSpeechCache:
    """Lazy background Zonos renderer + disk cache for fixed spoken lines."""

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR):
        self._dir = cache_dir
        self._eligible: set[str] = set()
        self._inflight: set[str] = set()
        self._zonos = None
        self._failed = False
        self._lock = threading.Lock()
        self._last_render = 0.0
        self._reaper_started = False

        try:
            with open(_MUMBLES_PATH, "r", encoding="utf-8") as f:
                self._eligible = {m["text"] for m in json.load(f)}
            logger.info("Speech cache: %d eligible lines", len(self._eligible))
        except Exception as e:
            logger.warning("Speech cache: could not load mumbles (%s) — disabled", e)
            self._failed = True

    # ------------------------------------------------------------------

    def _key(self, text: str, emotion: str) -> str:
        h = hashlib.sha1(f"{text}|{emotion}".encode("utf-8")).hexdigest()[:20]
        return os.path.join(self._dir, f"{h}.npz")

    def get(self, text: str, emotion: str) -> Optional[list[dict]]:
        """Return cached segments for this exact line+emotion, or None."""
        if self._failed or text not in self._eligible:
            return None
        path = self._key(text, emotion)
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=False)
            return [{
                "audio": data["audio"].astype(np.float32),
                "phonemes": str(data["phonemes"]),
                "sample_rate": int(data["sample_rate"]),
            }]
        except Exception as e:
            logger.warning("Speech cache read failed (%s) — dropping entry", e)
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def maybe_enqueue(self, text: str, emotion: str, intensity: float) -> None:
        """If this line is cacheable and missing, render it in the background."""
        if self._failed or text not in self._eligible:
            return
        path = self._key(text, emotion)
        with self._lock:
            if os.path.exists(path) or path in self._inflight:
                return
            self._inflight.add(path)

        # Serialize on the Zonos worker pool — generation is not reentrant.
        from .zonos_synthesizer import _zonos_pool
        _zonos_pool.submit(self._render, text, emotion, path)

    # ------------------------------------------------------------------

    def _start_reaper(self) -> None:
        """Daemon that frees the Zonos VRAM once rendering has gone idle."""
        if self._reaper_started:
            return
        self._reaper_started = True

        def _loop():
            from .zonos_synthesizer import _zonos_pool
            while True:
                time.sleep(60)
                if self._zonos is not None and \
                        time.time() - self._last_render > _UNLOAD_AFTER_SECONDS:
                    _zonos_pool.submit(self._unload)  # serialized with renders

        threading.Thread(target=_loop, name="speech-cache-reaper", daemon=True).start()

    def _unload(self) -> None:
        if self._zonos is None or time.time() - self._last_render <= _UNLOAD_AFTER_SECONDS:
            return
        self._zonos = None
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Speech cache: Zonos unloaded after %ds idle (VRAM freed)",
                    _UNLOAD_AFTER_SECONDS)

    def _render(self, text: str, emotion: str, path: str) -> None:
        try:
            if self._zonos is None:
                from .zonos_synthesizer import ZonosSynthesizer
                z = ZonosSynthesizer()
                z.load()  # raises without CUDA — marks cache failed below
                self._zonos = z
                self._start_reaper()
                logger.info("Speech cache: Zonos loaded for background rendering")
            self._last_render = time.time()

            result = self._zonos._generate_sync(text, emotion, _CACHE_INTENSITY, 1.0)
            if result is None:
                return
            os.makedirs(self._dir, exist_ok=True)
            np.savez(
                path,
                audio=result["audio"],
                phonemes=np.str_(result["phonemes"]),
                sample_rate=np.int64(self._zonos.sample_rate),
            )
            logger.info(
                "Speech cache: rendered '%s' [%s] (%.1fs audio)",
                text[:30], emotion, len(result["audio"]) / self._zonos.sample_rate,
            )
        except Exception as e:
            logger.warning("Speech cache render failed (%s) — cache disabled", e)
            self._failed = True
        finally:
            with self._lock:
                self._inflight.discard(path)
