"""
Chatterbox TTS Synthesizer — isolated-subprocess edition
Phase B: realtime emotional voice on the freed GPU.

Chatterbox hard-pins transformers==5.2.0 and numpy<2.0, which would perturb
the main sidecar venv (Zonos/Kokoro/faster-whisper). So Chatterbox lives in a
DEDICATED venv (.venv-chatterbox) and runs as a persistent subprocess worker
(chatterbox_worker.py). This module — imported by the main venv — never imports
`chatterbox` or `torch` at module load; it only talks to the worker:

  control : JSON lines over the worker's stdin/stdout (marker-prefixed so stray
            library prints on stdout can't corrupt the protocol)
  audio   : the worker writes each clip to a temp .wav; this side reads it back
            with soundfile, then deletes it

The class stays interface-compatible with tts.synthesizer.Synthesizer:
`synthesize_stream()` yields {"graphemes", "phonemes", "audio"} dicts, so the
viseme pipeline is untouched. Phonemes come from misaki (main venv), same IPA
alphabet the ipa_to_viseme.json map expects, so lip-sync stays exact. The
speaker reference is rendered once from Kokoro af_bella (main venv) and passed
to the worker as an audio prompt for a consistent voice.

If anything here fails (missing venv, worker won't start, CUDA absent), load()
raises so ResilientSynthesizer falls back to Kokoro.
"""

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np

from .emotion_map import mood_to_chatterbox

logger = logging.getLogger("sylph.tts.chatterbox")

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SPEAKER_CACHE = os.path.join(_WORKSPACE_ROOT, "models", "chatterbox", "speaker_ref.wav")
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatterbox_worker.py")
_TMP_DIR = os.path.join(_WORKSPACE_ROOT, "models", "chatterbox", "tmp")
_WORKER_LOG = os.path.join(_WORKSPACE_ROOT, "models", "chatterbox", "worker.log")

# Protocol markers — the worker prefixes protocol lines so library/stdout noise
# is trivially filtered out.
_READY = "@@READY@@"
_RESP = "@@RESP@@"

# First from_pretrained() may download ~2 GB of weights; be generous.
_STARTUP_TIMEOUT = float(os.environ.get("CHATTERBOX_STARTUP_TIMEOUT", "600"))
_GENERATE_TIMEOUT = float(os.environ.get("CHATTERBOX_GENERATE_TIMEOUT", "60"))

# Single worker thread: one clause at a time keeps the IPC exchange serialized
# and clause order stable (same pattern as the Kokoro / Zonos pools).
_chatterbox_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatterbox-worker")


def _venv_python() -> str:
    """Resolve the dedicated Chatterbox venv interpreter."""
    override = os.environ.get("CHATTERBOX_PYTHON")
    if override:
        return override
    if sys.platform == "win32":
        return os.path.join(_WORKSPACE_ROOT, ".venv-chatterbox", "Scripts", "python.exe")
    return os.path.join(_WORKSPACE_ROOT, ".venv-chatterbox", "bin", "python")


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
        pass


class ChatterboxSynthesizer:
    """Chatterbox engine driven through an isolated subprocess worker."""

    def __init__(self):
        self.voice = "af_bella"  # compat with /api/settings/voice; identity ref
        self._proc: Optional[subprocess.Popen] = None
        self._resp_q: "queue.Queue[str]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._logf = None
        self._g2p = None
        self._ref_wav: Optional[str] = None
        self._loaded = False
        self._device = "cpu"
        self._lock = threading.Lock()
        self._sample_rate = 24000

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            vpy = _venv_python()
            if not os.path.exists(vpy):
                raise RuntimeError(f"Chatterbox venv interpreter not found at {vpy}")
            if not os.path.exists(_WORKER_SCRIPT):
                raise RuntimeError(f"Chatterbox worker script missing at {_WORKER_SCRIPT}")

            os.makedirs(_TMP_DIR, exist_ok=True)

            # Phonemes come from misaki in THIS venv (worker doesn't need it).
            _wire_espeak()
            from misaki import en
            self._g2p = en.G2P(trf=False, british=False, fallback=None)

            # Speaker reference (rendered once from Kokoro, main venv).
            self._ref_wav = self._load_or_build_reference()

            self._spawn_worker(vpy)

            # Warm-up (also proves the round-trip end to end).
            self._generate_sync("Ready.", "neutral", 1.0, 1.0)
            self._loaded = True
            self._device = "cuda"
            logger.info("Chatterbox worker ready (sr=%d, ref=%s)",
                        self._sample_rate, "kokoro-clip" if self._ref_wav else "builtin")

    def _spawn_worker(self, vpy: str) -> None:
        os.makedirs(os.path.dirname(_WORKER_LOG), exist_ok=True)
        self._logf = open(_WORKER_LOG, "a", encoding="utf-8")
        logger.info("Spawning Chatterbox worker: %s %s", vpy, _WORKER_SCRIPT)
        self._proc = subprocess.Popen(
            [vpy, "-u", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._logf,           # library noise → log file, not the protocol
            cwd=_WORKSPACE_ROOT,
            text=True,
            bufsize=1,                    # line-buffered
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

        # Wait for the READY handshake (model may download on first run).
        line = self._await_line(_READY, _STARTUP_TIMEOUT)
        info = json.loads(line[len(_READY):].strip())
        self._sample_rate = int(info.get("sr", 24000))

    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith(_READY) or line.startswith(_RESP):
                self._resp_q.put(line)
            elif line:
                logger.debug("[chatterbox-worker] %s", line[:200])
        self._resp_q.put(_RESP + ' {"ok": false, "error": "worker stdout closed"}')

    def _await_line(self, marker: str, timeout: float) -> str:
        try:
            line = self._resp_q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"Chatterbox worker: no '{marker}' within {timeout:.0f}s")
        if not line.startswith(marker):
            # A RESP arriving where READY was expected (or vice versa) = failure.
            raise RuntimeError(f"Chatterbox worker protocol error: {line[:200]}")
        return line

    def _load_or_build_reference(self) -> Optional[str]:
        """Path to a speaker reference clip, rendered from Kokoro if absent."""
        if os.path.exists(_SPEAKER_CACHE):
            logger.info("Using cached Chatterbox speaker reference: %s", _SPEAKER_CACHE)
            return _SPEAKER_CACHE
        try:
            import soundfile as sf
            from .synthesizer import Synthesizer as KokoroSynth

            logger.info("Rendering Chatterbox speaker reference from Kokoro '%s'...", self.voice)
            ref = KokoroSynth(voice=self.voice)
            chunks = ref.synthesize_sync(
                "Hello there. This is my voice, and it stays consistent "
                "no matter what mood I happen to be in today."
            )
            if not chunks:
                raise RuntimeError("Kokoro produced no reference audio")
            wav = np.concatenate([c["audio"] for c in chunks]).astype(np.float32)
            os.makedirs(os.path.dirname(_SPEAKER_CACHE), exist_ok=True)
            sf.write(_SPEAKER_CACHE, wav, ref.sample_rate)
            logger.info("Speaker reference cached at %s", _SPEAKER_CACHE)
            return _SPEAKER_CACHE
        except Exception as e:
            logger.warning("Could not build speaker reference (%s) — worker uses default voice", e)
            return None

    # ------------------------------------------------------------------
    # Core generation (worker thread) — one IPC round-trip
    # ------------------------------------------------------------------

    def _generate_sync(
        self,
        text: str,
        emotion_label: str,
        emotion_intensity: float,
        speed: float,
    ) -> Optional[dict]:
        text = text.strip()
        if not text:
            return None
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("Chatterbox worker is not running")

        exaggeration, cfg_weight = mood_to_chatterbox(emotion_label, emotion_intensity)

        fd, out_path = tempfile.mkstemp(suffix=".wav", dir=_TMP_DIR)
        os.close(fd)
        req = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "audio_prompt_path": self._ref_wav,
            "out_path": out_path,
        }
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._await_line(_RESP, _GENERATE_TIMEOUT)
            resp = json.loads(line[len(_RESP):].strip())
            if not resp.get("ok"):
                raise RuntimeError(f"worker generate failed: {resp.get('error')}")

            import soundfile as sf
            audio, _sr = sf.read(out_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

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
        loop = asyncio.get_running_loop()
        if not self._loaded:
            await loop.run_in_executor(_chatterbox_pool, self.load)
        result = await loop.run_in_executor(
            _chatterbox_pool,
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
        out = []
        async for seg in self.synthesize_stream(
            text, speed=speed,
            emotion_label=emotion_label, emotion_intensity=emotion_intensity,
        ):
            out.append(seg)
        return out

    def close(self) -> None:
        """Terminate the worker subprocess (best-effort)."""
        try:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                except Exception:
                    pass
                self._proc.terminate()
        except Exception:
            pass
        finally:
            if self._logf:
                try:
                    self._logf.close()
                except Exception:
                    pass

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def device(self) -> str:
        return self._device
