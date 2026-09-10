"""
faster-whisper Transcriber
Phase 3.3: CPU-based speech-to-text with low-priority threads

Uses faster-whisper (CTranslate2) with distil-large-v3 model for
high-accuracy English transcription on CPU with INT8 quantization.

Key design:
- Runs on a dedicated ThreadPoolExecutor with below-normal priority
- Lazy-loads the model on first transcription request
- Returns plain text transcription from audio numpy arrays
"""

import logging
import os
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger("sylph.stt.transcriber")

SAMPLE_RATE = 16000
DEFAULT_MODEL = "distil-large-v3"
FALLBACK_MODEL = "distil-small.en"

# Anti-hallucination gates. Whisper invents phrases ("Thank you.", "Thanks for
# watching.") from near-silence; the VAD upstream can still pass quiet noise.
# Skip clips below a peak-amplitude floor outright, and drop any transcribed
# segment Whisper itself flags as probably-not-speech.
MIN_PEAK_AMPLITUDE = 0.02
NO_SPEECH_PROB_THRESHOLD = 0.6


def _init_low_priority_worker():
    """No longer downgrades priority to ensure fast response times."""
    pass


# Thread pool for STT
_stt_pool = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="stt-worker",
    initializer=_init_low_priority_worker,
)


class Transcriber:
    """
    faster-whisper transcription engine.
    Lazy-loads the model on first use to avoid blocking startup.
    """

    def __init__(self, model_size: str = DEFAULT_MODEL):
        import threading
        # Phase C: STT device/model are env-configurable. With a local LLM +
        # GPU TTS the 8 GB budget is tight; STT_DEVICE=cpu keeps whisper off the
        # GPU entirely (frees ~1 GB) at the cost of slower transcription.
        #   STT_DEVICE = auto (default: GPU first, CPU fallback) | cuda | cpu
        #   STT_MODEL  = faster-whisper model id (default distil-large-v3)
        self._model_size = os.environ.get("STT_MODEL", model_size)
        self._device_pref = os.environ.get("STT_DEVICE", "auto").strip().lower()
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the Whisper model. Can be called eagerly at startup or lazily."""
        with self._lock:
            if self._loaded:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        """Actual load logic. Caller must hold self._lock."""
        from faster_whisper import WhisperModel

        # Try to load on GPU first (device="cuda") — unless STT_DEVICE=cpu, in
        # which case skip the GPU attempt entirely (no wasted download/verify,
        # no VRAM pressure on the shared 8 GB budget).
        if self._device_pref != "cpu":
            try:
                logger.info("Attempting to load faster-whisper model '%s' on GPU (CUDA, float16)...", self._model_size)
                model = WhisperModel(
                    self._model_size,
                    device="cuda",
                    compute_type="int8_float16",
                    cpu_threads=4,
                )
                # Verify CUDA execution works (CTranslate2 loads cuBLAS/cuDNN lazily on first transcribe)
                dummy_audio = np.zeros(16000, dtype=np.float32)
                list(model.transcribe(dummy_audio, beam_size=1)[0])

                self._model = model
                self._loaded = True
                logger.info("faster-whisper '%s' loaded on GPU successfully and verified", self._model_size)
                return
            except Exception as cuda_err:
                if self._device_pref == "cuda":
                    logger.error("STT_DEVICE=cuda but GPU load failed: %s — falling back to CPU anyway", cuda_err)
                else:
                    logger.warning("Failed to load or verify faster-whisper on GPU: %s. Falling back to CPU...", cuda_err)
        else:
            logger.info("STT_DEVICE=cpu — skipping GPU, loading faster-whisper on CPU")

        # Fallback to CPU. If the requested model is large, switch to a smaller model for CPU to keep latency low.
        cpu_model = self._model_size
        if "large" in cpu_model or "medium" in cpu_model:
            logger.info("Model '%s' is too heavy for CPU. Switching to '%s' for CPU execution.", cpu_model, FALLBACK_MODEL)
            cpu_model = FALLBACK_MODEL

        logger.info("Loading faster-whisper model '%s' on CPU (INT8)...", cpu_model)
        try:
            cpu_threads = max(4, (os.cpu_count() or 8) // 2)
            self._model = WhisperModel(
                cpu_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )
            self._model_size = cpu_model
            self._loaded = True
            logger.info("faster-whisper '%s' loaded on CPU successfully", cpu_model)
        except Exception as cpu_err:
            logger.error("Failed to load '%s' on CPU: %s", cpu_model, cpu_err)
            if cpu_model != "base.en":
                logger.info("Trying absolute fallback 'base.en' on CPU...")
                try:
                    self._model = WhisperModel(
                        "base.en",
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                    )
                    self._model_size = "base.en"
                    self._loaded = True
                    logger.info("faster-whisper 'base.en' loaded on CPU successfully as absolute fallback")
                except Exception as final_err:
                    logger.critical("All whisper loading attempts failed: %s", final_err)

    def transcribe_sync(self, audio: np.ndarray) -> str:
        """
        Transcribe audio synchronously (runs on calling thread).

        Args:
            audio: float32 numpy array, 16kHz mono, values in [-1, 1]

        Returns:
            Transcribed text string
        """
        if not self._loaded:
            self.load()

        if self._model is None:
            return ""

        # faster-whisper expects float32 numpy array
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Cheap pre-gate: reject near-silence before paying for a transcribe.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < MIN_PEAK_AMPLITUDE:
            logger.info("STT: audio too quiet (peak=%.4f < %.2f) — treating as silence", peak, MIN_PEAK_AMPLITUDE)
            return ""

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=1,                       # greedy: ~3-5x faster than beam=5
                language="en",
                vad_filter=False,                  # We already do VAD upstream
                without_timestamps=True,
                condition_on_previous_text=False,  # prevents slow cross-segment attention
            )

            # Collect segment texts, dropping any Whisper itself thinks is silence.
            text_parts = []
            for segment in segments:
                nsp = getattr(segment, "no_speech_prob", 0.0)
                if nsp > NO_SPEECH_PROB_THRESHOLD:
                    logger.info("STT: dropping likely-silence segment (no_speech_prob=%.2f): '%s'",
                                nsp, segment.text.strip()[:40])
                    continue
                text_parts.append(segment.text.strip())

            result = " ".join(text_parts).strip()
            logger.info("Transcription: '%s' (lang=%s, prob=%.2f)",
                         result, info.language, info.language_probability)
            return result
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            # Dynamic fallback to CPU base.en if active GPU model failed during run
            if self._loaded and self._model_size != FALLBACK_MODEL:
                logger.warning("Active model failed during execution. Triggering fallback model '%s' on CPU...", FALLBACK_MODEL)
                self._loaded = False
                self._model = None
                self._model_size = FALLBACK_MODEL
                self.load()
                return self.transcribe_sync(audio)
            else:
                raise

    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio asynchronously on the low-priority thread pool.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            Transcribed text string
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_stt_pool, self.transcribe_sync, audio)
