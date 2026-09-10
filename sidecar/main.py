"""
Sylph Brain Sidecar — FastAPI Entry Point
Phase 2 + 3 + 4 + 5 + 6 + 7 + 8: Full pipeline with LLM + Memory + Vision + Personality

Complete pipeline:
  Audio → VAD → faster-whisper → transcript
  → planner (memories + prompt + Ollama streaming + sentence split)
  → Kokoro TTS → viseme timeline → dual payload → WebSocket → frontend

Run: python sidecar/main.py
"""

import env_config  # noqa: F401  — loads repo-root .env BEFORE other imports read os.environ

import asyncio
import time
import json
from datetime import datetime
import logging
import struct
from contextlib import asynccontextmanager
from typing import Set, Optional
from uuid import uuid4

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from stt.vad import VoiceActivityDetector
from stt.transcriber import Transcriber
from tts import create_synthesizer
from tts.duration_estimator import estimate_durations
from tts.viseme_builder import build_viseme_timeline, timeline_to_compact
from tts.payload import build_dual_payload
from llm.ollama_client import OllamaClient
from llm.planner import ConversationPlanner, infer_emotion
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.sync_worker import MemorySyncWorker
from vision.screen_analyzer import ScreenAnalyzer
from personality.mood import MoodStateMachine
from personality.autonomous import AutonomousBehaviour
from personality.interrupt_gate import InterruptGate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import os
from logging.handlers import RotatingFileHandler

log_format = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"
date_format = "%H:%M:%S"

# Determine workspace root (parent of sidecar dir)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file_path = os.path.join(workspace_root, "sidecar.log")

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, date_format))

file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(log_format, date_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger("sylph.sidecar")
logger.info("Logging initialized: Console + file at %s", log_file_path)

# ---------------------------------------------------------------------------
# ML + Service Instances (lazy-loaded)
# ---------------------------------------------------------------------------

vad = VoiceActivityDetector()
transcriber = Transcriber()
synthesizer = create_synthesizer()  # engine per TTS_ENGINE (.env)
ollama = OllamaClient()

# When the live engine is NOT Zonos (RTF on this GPU rules it out for live
# replies), the fixed personality lines still get real Zonos emotion via a
# lazy background-rendered disk cache. No-op when Zonos is the live engine.
speech_cache = None
if getattr(synthesizer, "engine", "kokoro") == "kokoro" and \
        os.environ.get("SPEECH_CACHE", "on").lower() != "off":
    from tts.speech_cache import EmotionalSpeechCache
    speech_cache = EmotionalSpeechCache()
short_term = ShortTermMemory()
long_term = LongTermMemory()
sync_worker: Optional[MemorySyncWorker] = None
planner: Optional[ConversationPlanner] = None
screen_analyzer = ScreenAnalyzer()
mood_machine = MoodStateMachine()
autonomous = AutonomousBehaviour()
interrupt_gate = InterruptGate()

# Current conversation thread
_current_thread_id = str(uuid4())
_global_speed = 1.0

# Phase C: tool interaction timeouts are env-configurable (seconds). On timeout
# a confirmation defaults to DENY (never silently act) and a capture returns an
# empty result. Raise these on slower machines instead of losing the action.
CONFIRM_TIMEOUT = float(os.environ.get("SYLPH_CONFIRM_TIMEOUT", "30"))
CAPTURE_TIMEOUT = float(os.environ.get("SYLPH_CAPTURE_TIMEOUT", "10"))

# Concurrency & Interruption control
active_response_task: Optional[asyncio.Task] = None
_active_task_started_at: float = 0.0  # time.time() when the active task was registered
_last_user_message_time = 0.0
assistant_state = "idle"
partial_transcript = ""

# Turn ID: monotonically increasing. Every TTS payload is stamped with the
# current turn; cancellation bumps it. The frontend drops any audio chunk
# stamped with a stale turn, which eliminates "zombie clauses" — audio that
# was already inside the synthesis executor when task.cancel() landed and
# would otherwise play over the new response (asyncio cancellation cannot
# reach into run_in_executor work that has already started).
_turn_id = 0

def current_turn_id() -> int:
    return _turn_id

def begin_new_turn() -> int:
    global _turn_id
    _turn_id += 1
    return _turn_id

def update_last_user_message_time():
    global _last_user_message_time
    _last_user_message_time = time.time()
    if sync_worker:
        sync_worker.cancel_active_sync()

def is_user_active_recently() -> bool:
    # User is active if there was an interaction in the last 5 minutes (300s)
    return time.time() - _last_user_message_time < 300.0

def cancel_active_response_task():
    global active_response_task
    stale_turn = current_turn_id()
    begin_new_turn()  # anything stamped <= stale_turn is now invalid
    if active_response_task and not active_response_task.done():
        logger.info("Interrupting active response task (turn %d)", stale_turn)
        active_response_task.cancel()
    # Always notify clients: queued audio may exist even after the task ended
    asyncio.create_task(manager.broadcast_json("stop_audio", {"min_turn_id": current_turn_id()}))
    active_response_task = None

def register_active_response_task(task: asyncio.Task):
    global active_response_task, _active_task_started_at
    active_response_task = task
    _active_task_started_at = time.time()


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections from the Tauri frontend."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("WebSocket client connected  (total: %d)", len(self._active))
        # Immediately push current mood so the avatar reflects the right expression from the start
        await self.send_initial_mood(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("WebSocket client disconnected (total: %d)", len(self._active))

    async def broadcast_json(self, msg_type: str, payload: dict) -> None:
        msg = json.dumps({"type": msg_type, "payload": payload})
        for ws in list(self._active):
            try:
                await ws.send_text(msg)
            except Exception:
                self._active.discard(ws)

    async def send_initial_mood(self, ws: WebSocket) -> None:
        """Push the current mood state to a newly connected client immediately."""
        try:
            state = mood_machine.get_state_for_new_connection()
            await ws.send_text(json.dumps({"type": "mood_update", "payload": state}))
            logger.debug("Sent initial mood to new client: %s", state["mood"])
        except Exception as e:
            logger.debug("Failed to send initial mood: %s", e)

    async def broadcast_bytes(self, data: bytes) -> None:
        for ws in list(self._active):
            try:
                await ws.send_bytes(data)
            except Exception:
                self._active.discard(ws)

    async def send_json(self, ws: WebSocket, msg_type: str, payload: dict) -> None:
        await ws.send_text(json.dumps({"type": msg_type, "payload": payload}))

    @property
    def count(self) -> int:
        return len(self._active)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# TTS Helper — sends synthesized speech to all clients
# ---------------------------------------------------------------------------

async def _emit_tts_chunk(audio, phonemes: str, sample_rate: int, turn: int) -> None:
    """Viseme timeline + dual payload + broadcast for one audio segment."""
    durations = estimate_durations(phonemes, len(audio), sample_rate)
    timeline = build_viseme_timeline(durations)
    compact = timeline_to_compact(timeline)
    dual = build_dual_payload(
        audio, compact,
        sample_rate=sample_rate,
        is_final=False,
        turn_id=turn,
    )
    await manager.broadcast_bytes(dual)


async def send_tts_to_clients(text: str, speed: Optional[float] = None) -> None:
    """
    Synthesize text and broadcast dual payloads to all WebSocket clients.

    Emotion resolution (no LLM involved):
    1. Scan THIS clause for emotional keywords — if the sentence being spoken
       carries a tone, the voice carries it too, in the same breath.
    2. Otherwise fall back to the mood machine's current expression, scaled
       by how strongly that mood is held.
    The label + intensity feed Zonos' native emotion vector (or, on the
    Kokoro fallback, its legacy speed/pitch approximation).

    Args:
        text: Text to synthesize.
        speed: Optional user speed preference (multiplies the emotion-driven
               base rate). If None, uses the global settings speed.
    """
    try:
        if speed is None:
            speed = _global_speed

        clause_emotion = infer_emotion(text)
        if clause_emotion != "neutral":
            emotion_label = clause_emotion
            # Text told us the tone outright — speak it with conviction.
            emotion_intensity = 0.85
        else:
            emotion_label = mood_machine.expression_label
            emotion_intensity = mood_machine.expression_intensity
        logger.debug(
            "TTS emotion: '%s' (intensity %.2f, source=%s)",
            emotion_label, emotion_intensity,
            "clause" if clause_emotion != "neutral" else "mood",
        )

        # Capture the turn this utterance belongs to. If a new turn begins
        # (user interrupts) while we are still synthesizing, we stop emitting.
        turn = current_turn_id()

        # Broadcast tts_start BEFORE synthesis so frontend mutes mic immediately
        # and shows the 'speaking' indicator without waiting for audio generation.
        await manager.broadcast_json("tts_start", {
            "text": text[:80],
            "turn_id": turn,
            "emotion": emotion_label,
        })

        # Cached emotional take? (Zonos-voiced mumbles rendered in the
        # background — full emotion vector, zero synthesis latency.)
        cached_segments = speech_cache.get(text, emotion_label) if speech_cache else None
        if cached_segments is not None:
            logger.info("TTS cache hit: '%s' [%s] — playing Zonos take", text[:40], emotion_label)
            for seg in cached_segments:
                if turn != current_turn_id():
                    return
                await _emit_tts_chunk(seg["audio"], seg["phonemes"], seg["sample_rate"], turn)
        else:
            if speech_cache:
                speech_cache.maybe_enqueue(text, emotion_label, emotion_intensity)
            i = 0
            async for chunk in synthesizer.synthesize_stream(
                text, speed=speed,
                emotion_label=emotion_label, emotion_intensity=emotion_intensity,
            ):
                if turn != current_turn_id():
                    logger.info("TTS aborted mid-stream: turn %d superseded by %d", turn, current_turn_id())
                    return
                await _emit_tts_chunk(
                    chunk["audio"], chunk["phonemes"], synthesizer.sample_rate, turn,
                )
                i += 1
                logger.info("TTS chunk %d: '%s' → %d samples (turn %d)", i, chunk["graphemes"][:30], len(chunk["audio"]), turn)

        # Notify the interrupt gate that speech was successfully delivered
        interrupt_gate.on_speech_delivered()
    except Exception as e:
        logger.error("TTS send failed: %s", e)


# ---------------------------------------------------------------------------
# Memory Retriever — for the planner
# ---------------------------------------------------------------------------

async def retrieve_memories(query: str) -> list[str]:
    """Retrieve relevant facts from long-term memory.

    Runs the Chroma query in an executor: embedding is CPU work (and on a
    fresh install Chroma synchronously downloads its ONNX model first) — on
    the event loop it would freeze every WebSocket during that time.
    """
    try:
        loop = asyncio.get_running_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: long_term.retrieve_relevant(query, top_k=6, min_similarity=0.65)
            ),
            timeout=8.0,
        )
        return [r["text"] for r in results]
    except asyncio.TimeoutError:
        logger.warning("Memory retrieval timed out — continuing without memories")
        return []
    except Exception as e:
        logger.warning("Memory retrieval error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Tool User Confirmation Registry (Phase 9)
# ---------------------------------------------------------------------------

pending_confirmations: dict[str, asyncio.Future] = {}
pending_captures: dict[str, asyncio.Future] = {}

async def wait_for_user_confirmation(action_name: str, details: dict) -> bool:
    """Send a confirmation request to the frontend and wait for approval."""
    action_id = str(uuid4())
    logger.info("Registering confirmation %s for tool %s", action_id, action_name)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending_confirmations[action_id] = fut

    # Broadcast event to frontend
    await manager.broadcast_json("confirm_action", {
        "id": action_id,
        "action": action_name,
        "details": details,
    })

    try:
        approved = await asyncio.wait_for(fut, timeout=CONFIRM_TIMEOUT)
        logger.info("Confirmation %s response: %s", action_id, approved)
        return approved
    except asyncio.TimeoutError:
        logger.warning("Confirmation %s timed out", action_id)
        return False
    finally:
        pending_confirmations.pop(action_id, None)


async def capture_screen_on_demand() -> dict:
    """Send a screen capture request to the client and wait for response.
    
    This coroutine is wrapped with asyncio.shield() at call sites that are
    vulnerable to barge-in cancellation (e.g. tool execution inside the
    response task), so that a user speaking mid-capture doesn't discard the
    in-flight capture request/response.
    """
    capture_id = str(uuid4())
    logger.info("Registering on-demand screen capture request %s", capture_id)
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending_captures[capture_id] = fut

    # Broadcast event to frontend
    await manager.broadcast_json("request_screen_capture", {
        "id": capture_id,
    })

    try:
        res = await asyncio.wait_for(fut, timeout=CAPTURE_TIMEOUT)
        image = res.get("image", "")
        window_name = res.get("window_name", "")
        
        if not image:
            return {"ocr_text": "Failed to capture screen (empty image payload)", "window_name": "Unknown"}
            
        # Run OCR
        result = screen_analyzer.process_capture(image, window_name)
        return {
            "ocr_text": result["ocr_text"],
            "window_name": window_name,
        }
    except asyncio.TimeoutError:
        logger.warning("Screen capture request %s timed out", capture_id)
        return {"ocr_text": "Screen capture timed out", "window_name": "Unknown"}
    finally:
        pending_captures.pop(capture_id, None)


import re as _re

# Patterns that indicate the user wants Sylph to look at / read their screen.
# Kept as module-level compiled regexes for speed.
_SCREEN_QUERY_PATTERNS = [
    _re.compile(r"\b(see|read|look\s+at|check|describe|what('s|\s+is)\s+on)\b.{0,15}\b(my\s+)?(screen|monitor|display|browser|window)\b", _re.I),
    _re.compile(r"\b(screen|monitor|display|browser|window)\b.{0,15}\b(see|read|showing|show)\b", _re.I),
    _re.compile(r"\bwhat\s+(do\s+you|can\s+you)\s+see\b", _re.I),
    _re.compile(r"\bcan\s+you\s+see\b.{0,10}\b(screen|this|it)\b", _re.I),
]


def _is_screen_query(text: str) -> bool:
    """Return True if the user's message is asking Sylph about their screen."""
    for pat in _SCREEN_QUERY_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# App Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global planner, sync_worker, _current_thread_id
    logger.info("=== Sylph Brain Sidecar starting ===\n")

    # Load VAD at startup (lightweight)
    try:
        vad.load()
    except Exception as e:
        logger.warning("VAD load deferred: %s", e)

    # Initialize memory systems
    try:
        short_term.initialize()
        long_term.initialize()
        logger.info("Memory systems initialized")
    except Exception as e:
        logger.warning("Memory init deferred: %s", e)

    # Define the mood callback for the planner — triggers emotion impulse + broadcast after each response
    async def _planner_mood_callback(emotion: str) -> None:
        """Called by the planner after detecting the emotional tone of a response."""
        mood_machine.on_speech_emotion(emotion)
        await mood_machine.force_broadcast()
        logger.debug("Planner emotion callback fired: %s", emotion)

    # Intent callback: deterministic avatar/system commands → frontend
    async def _planner_intent_callback(name: str, params: dict) -> None:
        if name in ("quiet_on", "quiet_off"):
            quiet = name == "quiet_on"
            interrupt_gate.set_quiet_mode(quiet)
            autonomous.set_quiet_mode(quiet)
            await manager.broadcast_json("quiet_mode_update", {"quiet": quiet})
        else:
            await manager.broadcast_json("avatar_command", {"command": name, **params})

    # Initialize planner with callbacks
    planner = ConversationPlanner(
        ollama_client=ollama,
        tts_callback=send_tts_to_clients,
        memory_retriever=retrieve_memories,
        confirm_callback=wait_for_user_confirmation,
        capture_callback=capture_screen_on_demand,
        mood_callback=_planner_mood_callback,
        intent_callback=_planner_intent_callback,
    )
    logger.info("Conversation planner initialized")

    # Phase C: resume the most recent conversation thread across restarts.
    # main.py assigns a fresh in-memory thread id each boot; without this the
    # planner would start amnesiac even though short-term memory persisted the
    # messages. Reattach to the newest thread and seed the planner's window.
    try:
        if short_term._initialized:
            recent_threads = short_term.list_threads(limit=1)
            if recent_threads:
                _current_thread_id = recent_threads[0]["thread_id"]
                recent_msgs = short_term.get_recent_messages(_current_thread_id, limit=20)
                planner.load_history(recent_msgs)
                logger.info("Resumed thread %s (%d messages)", _current_thread_id, len(recent_msgs))
    except Exception as e:
        logger.warning("History restore skipped: %s", e)

    # Start memory sync worker
    sync_worker = MemorySyncWorker(
        short_term, long_term, ollama,
        is_user_active_callback=is_user_active_recently
    )
    await sync_worker.start()

    # Preload ML models in the background to warm up memory/GPU without blocking startup
    async def preload_ml_models():
        # --- STT model: load EAGERLY (blocking) ---
        # The STT transcriber MUST be loaded before the server starts processing
        # audio. If we defer it, the first utterance triggers a live HuggingFace
        # download that can stall the entire STT pipeline indefinitely.
        try:
            loop = asyncio.get_running_loop()
            logger.info("Preloading STT model (this may download on first run)...")
            await loop.run_in_executor(None, transcriber.load)
            if transcriber._loaded:
                logger.info("✓ STT model preloaded successfully (model='%s')", transcriber._model_size)
            else:
                logger.error("✗ STT model preload completed but model is NOT loaded — STT will not work")
        except Exception as e:
            logger.error("✗ STT model preload FAILED: %s — voice input will not work until resolved", e)

        # --- LLM model: background (has its own retry) ---
        try:
            await ollama.preload_model()
        except Exception as e:
            logger.warning("Failed to preload Ollama model: %s", e)

        # --- TTS model: background (lazy-loads fine) ---
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, synthesizer.load)
        except Exception as e:
            logger.warning("Failed to preload Synthesizer: %s", e)

        # --- Embedding model: warm it now so the ~80 MB all-MiniLM ONNX
        #     download happens at startup, not on the FIRST memory query mid
        #     conversation (where it blocked retrieval past its 8s timeout and
        #     flooded the console). Downloads once, then cached. ---
        try:
            loop = asyncio.get_running_loop()
            logger.info("Warming long-term embedding model (one-time download on first run)...")
            await loop.run_in_executor(None, lambda: long_term._embed_texts(["warmup"]))
            logger.info("Embedding model ready")
        except Exception as e:
            logger.warning("Embedding warmup failed: %s", e)

    asyncio.create_task(preload_ml_models())

    # Phase 8: Start personality systems
    async def _mood_broadcast(label: str, values: dict) -> None:
        """Broadcast mood changes to all connected frontends."""
        await manager.broadcast_json("mood_update", {
            "mood": label,
            "values": values,
        })
        # Keep planner in sync
        if planner:
            planner.set_mood(label)

    mood_machine.set_callback(_mood_broadcast)
    await mood_machine.start()

    # Wire autonomous behaviour
    autonomous.set_tts_callback(send_tts_to_clients)
    autonomous.set_broadcast_callback(manager.broadcast_json)
    autonomous.set_gate_check(interrupt_gate.can_speak)
    autonomous.set_visual_idle_check(lambda: not (interrupt_gate.is_quiet or interrupt_gate.is_deep_work))
    autonomous.set_mood_getter(lambda: mood_machine.dominant_mood)
    await autonomous.start()
    logger.info("Personality systems initialized (mood + autonomous + gate)")

    # STT/TTS models load lazily on first request
    logger.info("Listening on http://127.0.0.1:8420")

    yield

    # Shutdown
    await autonomous.stop()
    await mood_machine.stop()
    if sync_worker:
        await sync_worker.stop()
    short_term.close()
    long_term.close()
    await ollama.close()
    # Phase B: tear down the Chatterbox subprocess worker if one is running.
    _synth_close = getattr(synthesizer, "close", None)
    if callable(_synth_close):
        try:
            _synth_close()
        except Exception as e:
            logger.debug("Synthesizer close failed: %s", e)
    logger.info("=== Sylph Brain Sidecar shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Sylph Brain Sidecar", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "Sylph Brain Sidecar",
        "version": "0.3.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "websocket": "/ws",
            "speak": "/speak?text=...",
            "ask": "/ask?text=...",
        },
    }


@app.get("/health")
async def health():
    ollama_ok = await ollama.is_available()
    return {
        "status": "ok",
        "service": "sylph-sidecar",
        "version": "0.3.0",
        "connections": manager.count,
        "models": {
            "vad": vad._loaded,
            "stt": transcriber._loaded,
            "tts": synthesizer._loaded,
            "ollama": ollama_ok,
        },
        "memory": {
            "short_term": short_term._initialized,
            "long_term": long_term._initialized,
            "long_term_stats": long_term.get_collection_stats() if long_term._initialized else {},
        },
        "thread_id": _current_thread_id,
        "personality": {
            "mood": mood_machine.to_dict(),
            "quiet_mode": interrupt_gate.is_quiet,
            "deep_work": interrupt_gate.is_deep_work,
        },
    }



@app.get("/api/memories/short-term")
async def get_short_term():
    if not short_term._initialized:
        return []
    return short_term.get_all_messages()

@app.delete("/api/memories/short-term/{msg_id}")
async def delete_short_term(msg_id: int):
    if not short_term._initialized:
        return {"status": "error", "message": "Short-term database not initialized"}
    short_term.delete_message(msg_id)
    return {"status": "ok"}

@app.get("/api/memories/long-term")
async def get_long_term():
    if not long_term._initialized:
        return []
    return long_term.get_all_facts()

@app.delete("/api/memories/long-term/{fact_id}")
async def delete_long_term(fact_id: str, category: str):
    if not long_term._initialized:
        return {"status": "error", "message": "Long-term database not initialized"}
    long_term.delete_fact(fact_id, category)
    return {"status": "ok"}

@app.post("/api/settings/reset")
async def reset_settings():
    if short_term._initialized:
        short_term.clear_all()
    if long_term._initialized:
        long_term.clear_all()
    if planner:
        planner.clear_history()
    return {"status": "ok"}

@app.get("/api/settings/export")
async def export_settings():
    s_mem = short_term.get_all_messages() if short_term._initialized else []
    l_mem = long_term.get_all_facts() if long_term._initialized else []
    return {
        "short_term": s_mem,
        "long_term": l_mem,
        "exported_at": datetime.utcnow().isoformat()
    }

@app.post("/api/settings/quiet")
async def set_quiet_mode(payload: dict):
    quiet = payload.get("quiet", False)
    interrupt_gate.set_quiet_mode(quiet)
    autonomous.set_quiet_mode(quiet)
    await manager.broadcast_json("quiet_mode_update", {"quiet": quiet})
    return {"status": "ok", "quiet": quiet}

@app.post("/api/settings/voice")
async def set_voice_settings(payload: dict):
    global _global_speed
    voice = payload.get("voice", "af_bella")
    speed = payload.get("speed", 1.0)
    synthesizer.voice = voice
    _global_speed = speed
    return {"status": "ok", "voice": voice, "speed": speed}


@app.post("/api/settings/autonomous")
async def set_autonomous_settings(payload: dict):
    enabled = payload.get("enabled", True)
    frequency = payload.get("frequency", 60.0)
    if enabled:
        await autonomous.start()
    else:
        await autonomous.stop()
    autonomous.set_frequency(frequency)
    return {"status": "ok", "enabled": enabled, "frequency": frequency}


@app.get("/speak")
async def speak_endpoint(text: str):
    """Test endpoint: synthesize text and return info (no audio streaming)."""
    try:
        results = await synthesizer.synthesize(text)
        total_samples = sum(len(r["audio"]) for r in results)
        return {
            "status": "ok",
            "text": text,
            "chunks": len(results),
            "total_samples": total_samples,
            "duration_s": round(total_samples / synthesizer.sample_rate, 2),
            "phonemes": [r["phonemes"] for r in results],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/ask")
async def ask_endpoint(text: str):
    """Test endpoint: send text through the planner and return response."""
    if planner is None:
        return {"status": "error", "error": "Planner not initialized"}
    try:
        response = await planner.process_user_message(text)
        return {"status": "ok", "question": text, "response": response}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# STT Pipeline — Phase 3.4
# ---------------------------------------------------------------------------

async def run_stt_planner_pipeline(ws: WebSocket, utterance: np.ndarray) -> None:
    """Run transcription and planner in a background task to keep WebSocket responsive."""
    global assistant_state, partial_transcript

    STT_TIMEOUT_SECONDS = 15  # Max time for a single transcription

    try:
        assistant_state = "transcribing"
        logger.info("STT pipeline: starting transcription (%d samples, %.1fs audio)...",
                     len(utterance), len(utterance) / 16000)

        try:
            text = await asyncio.wait_for(
                transcriber.transcribe(utterance),
                timeout=STT_TIMEOUT_SECONDS,
            )
            logger.info("STT pipeline: transcription complete — text='%s' (%d chars)",
                         text[:80] if text else "(empty)", len(text) if text else 0)
        except asyncio.TimeoutError:
            logger.error(
                "STT pipeline: transcription TIMED OUT after %ds — "
                "model may still be loading or GPU is overloaded. Resetting state.",
                STT_TIMEOUT_SECONDS,
            )
            assistant_state = "idle"
            return

        if text.strip():
            full_text = f"{partial_transcript} {text}".strip()
            
            if vad.is_speaking:
                logger.info("User resumed speaking. Buffering partial transcript: %s", full_text)
                partial_transcript = full_text
                assistant_state = "idle"
                return

            partial_transcript = ""

            # Send transcript to frontend
            await manager.send_json(ws, "transcript", {"text": full_text})

            # Store in short-term memory
            short_term.add_message(_current_thread_id, "user", full_text)

            # Route through the planner (Phase 5.5)
            if planner:
                assistant_state = "thinking"
                await manager.send_json(ws, "thinking", {"text": full_text})

                # Auto-capture: if the user is asking about their screen,
                # grab a fresh screenshot and inject the OCR text into the
                # LLM's system prompt so it has REAL data to answer with
                # instead of hallucinating.
                screen_context = ""
                if _is_screen_query(full_text):
                    logger.info("Auto-capture: user asked about screen — capturing now")
                    try:
                        cap = await asyncio.shield(capture_screen_on_demand())
                        ocr = cap.get("ocr_text", "")
                        win = cap.get("window_name", "")
                        if ocr:
                            screen_context = f"Active Window: {win}\nScreen OCR Text:\n{ocr}"
                            logger.info("Auto-capture: injected %d chars of OCR from '%s'", len(ocr), win[:30])
                        else:
                            logger.warning("Auto-capture: OCR returned empty")
                    except Exception as e:
                        logger.warning("Auto-capture failed: %s", e)

                response = await planner.process_user_message(full_text, screen_context=screen_context)
                assistant_state = "speaking"
                short_term.add_message(_current_thread_id, "assistant", response)
                await manager.send_json(ws, "turn_complete", {
                    "user_text": full_text,
                    "response": response,
                })
        else:
            logger.info("STT pipeline: transcription returned empty text — ignoring utterance")
    except asyncio.CancelledError:
        logger.info("STT pipeline: task was cancelled (barge-in or shutdown)")
        raise
    except Exception as e:
        logger.error("STT pipeline: unexpected error: %s", e, exc_info=True)
    finally:
        if assistant_state in ("thinking", "speaking", "transcribing"):
            assistant_state = "idle"


async def handle_vad_flush(ws: WebSocket) -> None:
    """Force-emit any buffered speech from VAD and process it in a background task."""
    global assistant_state, partial_transcript

    STT_TIMEOUT_SECONDS = 15

    try:
        utterance = vad.flush()
        if utterance is not None:
            assistant_state = "transcribing"
            logger.info("VAD flush: starting transcription (%d samples, %.1fs audio)...",
                         len(utterance), len(utterance) / 16000)

            try:
                text = await asyncio.wait_for(
                    transcriber.transcribe(utterance),
                    timeout=STT_TIMEOUT_SECONDS,
                )
                logger.info("VAD flush: transcription complete — text='%s' (%d chars)",
                             text[:80] if text else "(empty)", len(text) if text else 0)
            except asyncio.TimeoutError:
                logger.error(
                    "VAD flush: transcription TIMED OUT after %ds — resetting state.",
                    STT_TIMEOUT_SECONDS,
                )
                return
            
            full_text = f"{partial_transcript} {text}".strip()
            partial_transcript = ""
            
            if full_text.strip():
                await manager.send_json(ws, "transcript", {"text": full_text})
                # Store in short-term memory
                short_term.add_message(_current_thread_id, "user", full_text)
                if planner:
                    assistant_state = "thinking"
                    await manager.send_json(ws, "thinking", {"text": full_text})

                    # Auto-capture for screen queries (same as run_stt_planner_pipeline)
                    screen_context = ""
                    if _is_screen_query(full_text):
                        logger.info("Auto-capture (flush): user asked about screen — capturing now")
                        try:
                            cap = await asyncio.shield(capture_screen_on_demand())
                            ocr = cap.get("ocr_text", "")
                            win = cap.get("window_name", "")
                            if ocr:
                                screen_context = f"Active Window: {win}\nScreen OCR Text:\n{ocr}"
                        except Exception as e:
                            logger.warning("Auto-capture (flush) failed: %s", e)

                    response = await planner.process_user_message(full_text, screen_context=screen_context)
                    assistant_state = "speaking"
                    short_term.add_message(_current_thread_id, "assistant", response)
                    await manager.send_json(ws, "turn_complete", {
                        "user_text": full_text,
                        "response": response,
                    })
            else:
                logger.info("VAD flush: transcription returned empty text — ignoring")
    except asyncio.CancelledError:
        logger.info("VAD flush: task was cancelled")
        raise
    except Exception as e:
        logger.error("VAD flush: unexpected error: %s", e, exc_info=True)
    finally:
        vad.reset()
        assistant_state = "idle"


async def handle_audio_bytes(ws: WebSocket, audio_bytes: bytes) -> None:
    """Process incoming audio through the VAD and hand off to background pipeline."""
    try:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        frame_size = 512  # 32ms at 16kHz

        # Diagnostics for tracking microphone input
        if not hasattr(handle_audio_bytes, "frame_count"):
            handle_audio_bytes.frame_count = 0
            handle_audio_bytes.max_amplitude = 0.0

        handle_audio_bytes.frame_count += 1
        if len(audio) > 0:
            current_max = float(np.max(np.abs(audio)))
            if current_max > handle_audio_bytes.max_amplitude:
                handle_audio_bytes.max_amplitude = current_max

        if handle_audio_bytes.frame_count % 50 == 0:
            logger.info(
                "[STT Diagnostician] Received 50 binary chunks. Peak amplitude: %.5f",
                handle_audio_bytes.max_amplitude
            )
            handle_audio_bytes.max_amplitude = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i : i + frame_size]
            if len(frame) < frame_size:
                break

            utterance = vad.process_frame(frame)

            # Barge-in: the moment the user starts talking, kill the active
            # response — don't wait the full utterance + STT round trip.
            if vad.consume_speech_onset():
                if active_response_task and not active_response_task.done():
                    global partial_transcript
                    if assistant_state in ("thinking", "speaking"):
                        logger.info("Barge-in: user speech onset — cancelling active response")
                        cancel_active_response_task()
                        partial_transcript = ""
                    elif assistant_state == "transcribing":
                        # Check if the transcription task has been running too long (stalled)
                        task_age = time.time() - _active_task_started_at
                        if task_age > 15.0:
                            logger.warning(
                                "STT task stalled for %.1fs — cancelling and allowing new utterance",
                                task_age,
                            )
                            cancel_active_response_task()
                            partial_transcript = ""
                        else:
                            logger.info(
                                "User resumed speaking during STT (%.1fs old); preserving active STT task.",
                                task_age,
                            )

            if utterance is not None:
                logger.info("VAD: utterance detected, spawning background STT task")
                update_last_user_message_time()
                if assistant_state in ("thinking", "speaking"):
                    cancel_active_response_task()
                
                register_active_response_task(
                    asyncio.create_task(run_stt_planner_pipeline(ws, utterance))
                )
    except Exception as e:
        logger.error("VAD frame processing error: %s", e)


# ---------------------------------------------------------------------------
# TTS Pipeline — Phase 4.7
# ---------------------------------------------------------------------------

async def handle_speak_request(ws: WebSocket, payload: dict) -> None:
    """Synthesize text → audio + viseme → dual payload → WebSocket."""
    global assistant_state
    text = payload.get("text", "")
    speed = payload.get("speed", _global_speed)

    if not text:
        await manager.send_json(ws, "error", {"message": "No text provided"})
        return

    try:
        assistant_state = "speaking"
        await manager.send_json(ws, "tts_start", {"text": text})

        results = await synthesizer.synthesize(
            text, speed=speed,
            emotion_label=mood_machine.expression_label,
            emotion_intensity=mood_machine.expression_intensity,
        )
        for i, chunk in enumerate(results):
            audio = chunk["audio"]
            phonemes = chunk["phonemes"]
            is_final = (i == len(results) - 1)

            durations = estimate_durations(phonemes, len(audio), synthesizer.sample_rate)
            timeline = build_viseme_timeline(durations)
            compact = timeline_to_compact(timeline)
            dual = build_dual_payload(audio, compact, sample_rate=synthesizer.sample_rate, is_final=is_final)
            await ws.send_bytes(dual)

        await manager.send_json(ws, "tts_complete", {"text": text, "chunks": len(results)})
    except asyncio.CancelledError:
        logger.info("Speak request task was cancelled.")
        raise
    except Exception as e:
        logger.error("TTS pipeline error: %s", e)
        try:
            await manager.send_json(ws, "error", {"message": f"TTS error: {e}"})
        except Exception:
            pass
    finally:
        if assistant_state == "speaking":
            assistant_state = "idle"


# ---------------------------------------------------------------------------
# Chat Pipeline — Phase 5.5 (text-based, no audio)
# ---------------------------------------------------------------------------

async def handle_chat_request(ws: WebSocket, payload: dict) -> None:
    """Process a text chat message through the full pipeline."""
    global assistant_state
    text = payload.get("text", "")
    if not text or planner is None:
        return

    try:
        assistant_state = "thinking"
        await manager.send_json(ws, "thinking", {"text": text})

        # Store user message
        short_term.add_message(_current_thread_id, "user", text)

        # Run through planner (streams sentences → TTS automatically via callback)
        response = await planner.process_user_message(text)
        assistant_state = "speaking"

        # Store assistant response
        short_term.add_message(_current_thread_id, "assistant", response)

        await manager.send_json(ws, "turn_complete", {
            "user_text": text,
            "response": response,
        })
    except asyncio.CancelledError:
        logger.info("Chat request task was cancelled.")
        raise
    except Exception as e:
        logger.error("Chat pipeline error: %s", e)
        try:
            await manager.send_json(ws, "error", {"message": f"Chat error: {e}"})
        except Exception:
            pass
    finally:
        if assistant_state in ("thinking", "speaking"):
            assistant_state = "idle"


def _start_new_thread() -> str:
    """Start a new conversation thread."""
    global _current_thread_id
    _current_thread_id = str(uuid4())
    if planner:
        planner.clear_history()
    logger.info("New thread started: %s", _current_thread_id)
    return _current_thread_id


# ---------------------------------------------------------------------------
# Screen Capture Pipeline — Phase 7.3/7.4
# ---------------------------------------------------------------------------

async def handle_screen_capture(ws: WebSocket, payload: dict) -> None:
    """Process a screen capture: OCR → change detection → rule-based triage → LLM reaction (if needed)."""
    base64_jpeg = payload.get("image", "")
    window_name = payload.get("window_name", "")

    if not base64_jpeg:
        await manager.send_json(ws, "error", {"message": "No image provided"})
        return

    result = screen_analyzer.process_capture(base64_jpeg, window_name)

    if not result["changed"]:
        # No change detected — skip triage
        await manager.send_json(ws, "screen_unchanged", {"hash": result["hash"]})
        return

    # Run rule-based triage (no LLM call — pure heuristics)
    triage_result = screen_analyzer.rule_based_triage(
        ocr_text=result["ocr_text"],
        window_name=window_name,
        prev_ocr_text=result.get("prev_ocr_text", ""),
        prev_window_name=result.get("prev_window_name", ""),
    )

    classification = triage_result["classification"]
    reaction = ""

    # Only call the LLM to draft a reaction when the rule gate says to comment
    if classification in ("CONSIDER_COMMENT", "URGENT"):
        reaction = await screen_analyzer.generate_reaction(
            ocr_text=result["ocr_text"],
            window_name=window_name,
            urgency=classification,
            ollama_client=ollama,
        )

    await manager.send_json(ws, "screen_analyzed", {
        "window_name": window_name,
        "ocr_length": len(result["ocr_text"]),
        "classification": classification,
        "reaction": reaction,
    })

    # If CONSIDER_COMMENT or URGENT, speak the reaction
    if classification in ("CONSIDER_COMMENT", "URGENT") and reaction:
        if interrupt_gate.should_allow("screen_triage"):
            logger.info("Sylph reacts: '%s'", reaction[:60])
            await send_tts_to_clients(reaction)
        else:
            logger.info("Screen reaction suppressed by interrupt gate: '%s'", reaction[:60])


async def handle_vision_request(ws: WebSocket, payload: dict) -> None:
    """Phase 7.5: Full vision analysis for explicit user requests."""
    base64_jpeg = payload.get("image", "")
    user_text = payload.get("text", "What am I looking at?")

    if not base64_jpeg:
        await manager.send_json(ws, "error", {"message": "No image for vision"})
        return

    try:
        # One multimodal call with the actual image. (This used to ALSO run
        # the full planner on the same text — two cloud calls per request for
        # one answer. On a GPU-time-capped free tier that's a pure quota leak.)
        vision_response = await ollama.chat(
            messages=[{"role": "user", "content": user_text}],
            images=[base64_jpeg],
        )
        await manager.send_json(ws, "vision_response", {
            "question": user_text,
            "response": vision_response,
        })
        # Also speak it
        await send_tts_to_clients(vision_response)
    except asyncio.CancelledError:
        logger.info("Vision request task was cancelled.")
        raise
    except Exception as e:
        logger.error("Vision request failed: %s", e)
        try:
            await manager.send_json(ws, "error", {"message": f"Vision error: {e}"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Main IPC channel.

    Text frames (JSON):
      ping, speak, chat, vad_flush, mood_request, stt_config

    Binary frames:
      Incoming: raw PCM audio (float32 16kHz mono)
      Outgoing: dual-payload (audio + viseme timeline)
    """
    await manager.connect(ws)
    try:
        while True:
            message = await ws.receive()

            # Handle disconnect frames (Vite HMR reconnection race)
            msg_type_raw = message.get("type")
            if msg_type_raw == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "unknown")
                payload = data.get("payload", {})

                if msg_type == "ping":
                    await manager.send_json(ws, "pong", {
                        "echo": payload,
                        "connections": manager.count,
                    })

                elif msg_type == "recording_start":
                    logger.info("User started recording: interrupting active response task")
                    update_last_user_message_time()
                    global partial_transcript
                    if assistant_state in ("thinking", "speaking"):
                        cancel_active_response_task()
                        partial_transcript = ""

                elif msg_type == "speak":
                    update_last_user_message_time()
                    cancel_active_response_task()
                    register_active_response_task(
                        asyncio.create_task(handle_speak_request(ws, payload))
                    )

                elif msg_type == "chat":
                    update_last_user_message_time()
                    cancel_active_response_task()
                    register_active_response_task(
                        asyncio.create_task(handle_chat_request(ws, payload))
                    )

                elif msg_type == "vad_flush":
                    update_last_user_message_time()
                    cancel_active_response_task()
                    register_active_response_task(
                        asyncio.create_task(handle_vad_flush(ws))
                    )

                elif msg_type == "mood_request":
                    await manager.send_json(ws, "mood_update", {
                        "mood": "neutral",
                        "values": {
                            "playful": 0.3, "focused": 0.5, "bored": 0.1,
                            "curious": 0.4, "annoyed": 0.0, "enthusiastic": 0.3, "tired": 0.1,
                        },
                    })

                elif msg_type == "new_thread":
                    new_id = _start_new_thread()
                    await manager.send_json(ws, "thread_created", {"thread_id": new_id})

                elif msg_type == "sync_memory":
                    if sync_worker:
                        async def run_sync():
                            count = await sync_worker.sync_once()
                            await manager.send_json(ws, "sync_complete", {"facts_stored": count})
                        asyncio.create_task(run_sync())

                elif msg_type == "screen_capture":
                    asyncio.create_task(handle_screen_capture(ws, payload))

                elif msg_type == "vision_request":
                    update_last_user_message_time()
                    cancel_active_response_task()
                    register_active_response_task(
                        asyncio.create_task(handle_vision_request(ws, payload))
                    )

                elif msg_type == "confirm_response":
                    action_id = payload.get("id")
                    approved = payload.get("approved", False)
                    if action_id in pending_confirmations:
                        pending_confirmations[action_id].set_result(approved)

                elif msg_type == "screen_capture_response":
                    capture_id = payload.get("id")
                    image = payload.get("image", "")
                    window_name = payload.get("window_name", "")
                    if capture_id in pending_captures:
                        pending_captures[capture_id].set_result({
                            "image": image,
                            "window_name": window_name
                        })

                elif msg_type == "log":
                    level = payload.get("level", "info")
                    msg_str = payload.get("message", "")
                    if level == "error":
                        logger.error("[Frontend Error] %s", msg_str)
                    elif level == "warn":
                        logger.warning("[Frontend Warning] %s", msg_str)
                    else:
                        logger.info("[Frontend Log] %s", msg_str)

                else:
                    await manager.send_json(ws, "echo", {"original_type": msg_type, "payload": payload})

            elif "bytes" in message:
                await handle_audio_bytes(ws, message["bytes"])

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def kill_port_owner(port: int):
    """Find and terminate any zombie process listening on the specified port to prevent port conflicts."""
    import os
    import signal
    import subprocess
    import sys
    
    my_pid = os.getpid()
    try:
        if sys.platform == "win32":
            output = subprocess.check_output("netstat -ano", shell=True).decode('utf-8', errors='ignore')
            pids = set()
            for line in output.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1].strip()
                        if pid.isdigit():
                            pids.add(int(pid))
            for pid in pids:
                if pid != my_pid:
                    logger.info("Found zombie process %d holding port %d. Terminating...", pid, port)
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        os.system(f"taskkill /F /PID {pid}")
        else:
            try:
                pids_str = subprocess.check_output(["lsof", "-t", f"-i:{port}"]).decode().strip()
                for pid_str in pids_str.split():
                    pid = int(pid_str)
                    if pid != my_pid:
                        logger.info("Found zombie process %d holding port %d. Terminating...", pid, port)
                        os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Failed to check or kill process on port %d: %s", port, e)


if __name__ == "__main__":
    # Ensure port 8420 is free from zombie processes before starting
    kill_port_owner(8420)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8420,
        log_level="info",
        reload=False,
    )
