"""
Zonos TTS Synthesizer
Phase 12: Emotionally expressive local TTS on the freed GPU

Replaces Kokoro as the primary voice. Zonos-v0.1 (Apache 2.0) conditions on a
real 8-dim emotion vector plus independent pitch variation and speaking rate,
so mood reaches the *voice* instead of being faked by resampling tricks.

Interface-compatible with tts.synthesizer.Synthesizer: `synthesize_stream()`
yields {"graphemes", "phonemes", "audio"} dicts, so the downstream viseme
pipeline (duration_estimator → viseme_builder → dual payload) is untouched.
Zonos itself produces no phonemes — misaki (Kokoro's G2P, same IPA alphabet
the ipa_to_viseme.json map expects) provides them, keeping lip-sync exact.

Speaker identity: Zonos without a speaker embedding drifts between clauses.
On first load we synthesize a short reference with Kokoro's af_bella (the
voice users already know), derive a speaker embedding from it, and cache the
embedding at models/zonos/speaker_ref.pt — consistent voice, zero downloads,
no external audio needed.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np

from .emotion_map import mood_to_voice

logger = logging.getLogger("sylph.tts.zonos")

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SPEAKER_CACHE = os.path.join(_WORKSPACE_ROOT, "models", "zonos", "speaker_ref.pt")

DEFAULT_MODEL_ID = os.environ.get("ZONOS_MODEL", "Zyphra/Zonos-v0.1-transformer")

# Zonos emits ~86 codec frames per second of audio; clauses are short, so a
# 20 s ceiling both bounds worst-case latency and stops runaway generations.
_MAX_SECONDS_PER_CLAUSE = 20
_TOKENS_PER_SECOND = 86

# Single worker: Zonos generation is not reentrant, and serializing keeps
# clause order stable (same pattern as the Kokoro pool).
_zonos_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zonos-worker")

_SENTINEL = object()


def _wire_espeak() -> None:
    """Point phonemizer at the pip-bundled espeak-ng (no system install)."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    lib = espeakng_loader.get_library_path()
    data = espeakng_loader.get_data_path()
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(lib))
    os.environ.setdefault("ESPEAK_DATA_PATH", str(data))
    EspeakWrapper.set_library(str(lib))
    try:
        EspeakWrapper.set_data_path(str(data))
    except AttributeError:
        pass  # older phonemizer — env var above covers it
    logger.info("espeak-ng wired from espeakng-loader: %s", lib)


class ZonosSynthesizer:
    """
    Zonos TTS engine with native emotion conditioning.
    Lazy-loads on first call. CUDA required (CPU generation is far below
    realtime); load() raises on failure so the factory can fall back to Kokoro.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.voice = "af_bella"  # compat with /api/settings/voice; identity ref
        self._model = None
        self._g2p = None
        self._speaker = None
        self._loaded = False
        self._device = "cpu"
        self._lock = threading.Lock()
        self._sample_rate = 44100

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Zonos requires CUDA and torch.cuda.is_available() is False")

            _wire_espeak()
            from zonos.model import Zonos

            logger.info("Loading Zonos model '%s' on CUDA...", self.model_id)
            self._model = Zonos.from_pretrained(self.model_id, device="cuda")
            self._model.requires_grad_(False).eval()
            self._device = "cuda"
            self._sample_rate = int(self._model.autoencoder.sampling_rate)

            # misaki G2P — same IPA alphabet as the Kokoro pipeline fed the
            # viseme map, so mouth shapes stay correct.
            from misaki import en
            self._g2p = en.G2P(trf=False, british=False, fallback=None)

            self._speaker = self._load_or_build_speaker_embedding()

            # Warm-up: first generation pays CUDA graph/kernel init.
            self._generate_sync("Ready.", "neutral", 1.0, 1.0)
            self._loaded = True
            logger.info(
                "Zonos loaded on CUDA (sr=%d, speaker=%s)",
                self._sample_rate,
                "ref-embedding" if self._speaker is not None else "unconditioned",
            )

    def _load_or_build_speaker_embedding(self):
        """Load the cached speaker embedding, or derive one from Kokoro."""
        import torch

        if os.path.exists(_SPEAKER_CACHE):
            try:
                emb = torch.load(_SPEAKER_CACHE, map_location="cuda", weights_only=True)
                logger.info("Loaded cached Zonos speaker embedding: %s", _SPEAKER_CACHE)
                return emb
            except Exception as e:
                logger.warning("Failed to load speaker cache (%s) — rebuilding", e)

        try:
            from .synthesizer import Synthesizer as KokoroSynth

            logger.info("Deriving Zonos speaker embedding from Kokoro '%s'...", self.voice)
            ref = KokoroSynth(voice=self.voice)
            chunks = ref.synthesize_sync(
                "Hello there. This is my voice, and it stays consistent "
                "no matter what mood I happen to be in today."
            )
            if not chunks:
                raise RuntimeError("Kokoro produced no reference audio")
            wav = np.concatenate([c["audio"] for c in chunks]).astype(np.float32)
            wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
            emb = self._model.make_speaker_embedding(wav_t, ref.sample_rate)

            os.makedirs(os.path.dirname(_SPEAKER_CACHE), exist_ok=True)
            torch.save(emb, _SPEAKER_CACHE)
            logger.info("Speaker embedding cached at %s", _SPEAKER_CACHE)
            return emb
        except Exception as e:
            logger.warning(
                "Could not derive speaker embedding (%s) — using unconditioned voice", e
            )
            return None

    # ------------------------------------------------------------------
    # Core generation (worker thread)
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        text: str,
        emotion_label: str,
        emotion_intensity: float,
        speed: float,
    ) -> Optional[dict]:
        """Blocking single-clause synthesis. Runs on the zonos worker thread."""
        import torch
        from zonos.conditioning import make_cond_dict

        text = text.strip()
        if not text:
            return None

        voice = mood_to_voice(emotion_label, emotion_intensity)
        # User speed preference multiplies the mood-driven base rate.
        rate = max(8.0, min(24.0, voice.speaking_rate * speed))

        cond = make_cond_dict(
            text=text,
            speaker=self._speaker,
            language="en-us",
            emotion=voice.emotion,
            pitch_std=voice.pitch_std,
            speaking_rate=rate,
        )

        with torch.inference_mode():
            conditioning = self._model.prepare_conditioning(cond)
            codes = self._model.generate(
                conditioning,
                max_new_tokens=_MAX_SECONDS_PER_CLAUSE * _TOKENS_PER_SECOND,
                progress_bar=False,
                # Windows: no MSVC ⇒ torch.compile's inductor backend can't
                # build ("Compiler: cl is not found"), and CUDA graphs are
                # unsupported here — eager decode is the reliable path.
                disable_torch_compile=True,
            )
            wav = self._model.autoencoder.decode(codes).cpu()

        audio = wav.squeeze().numpy().astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Phonemes for the viseme timeline (misaki returns (phoneme_str, tokens))
        try:
            phonemes, _ = self._g2p(text)
        except Exception as e:
            logger.warning("G2P failed for clause (%s) — lip-sync degraded", e)
            phonemes = ""

        return {"graphemes": text, "phonemes": phonemes or "", "audio": audio}

    # ------------------------------------------------------------------
    # Public API (mirrors tts.synthesizer.Synthesizer)
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> AsyncGenerator[dict, None]:
        """Async generator yielding one segment dict for the clause."""
        loop = asyncio.get_running_loop()
        if not self._loaded:
            await loop.run_in_executor(_zonos_pool, self.load)

        result = await loop.run_in_executor(
            _zonos_pool,
            self._generate_sync,
            text, emotion_label, emotion_intensity, speed,
        )
        if result is not None:
            yield result

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        emotion_label: str = "neutral",
        emotion_intensity: float = 1.0,
    ) -> list[dict]:
        """Batch API for the /speak test endpoint."""
        out = []
        async for seg in self.synthesize_stream(
            text, speed=speed,
            emotion_label=emotion_label, emotion_intensity=emotion_intensity,
        ):
            out.append(seg)
        return out

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def device(self) -> str:
        return self._device
