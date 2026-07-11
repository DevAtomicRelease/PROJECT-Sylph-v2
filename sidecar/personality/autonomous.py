"""
Autonomous Behaviour System
Phase 8.2: Unprompted avatar actions during idle periods

Generates random autonomous events when the user is not actively
talking to Sylph. These make the avatar feel alive:
  - Mumble: Speak a short pre-written line via TTS
  - Glance: Send a look-at target shift to the avatar
  - Posture shift: Trigger a random idle animation variant
  - Sigh / hum: Play a pre-cached audio clip (when available)

Events fire every 30-90 seconds during idle. Suppressed during
deep work mode or quiet mode (via the InterruptGate).
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("sylph.personality.autonomous")

# Default interval range (seconds) between autonomous events
MIN_EVENT_INTERVAL = 30
MAX_EVENT_INTERVAL = 90

# Load mumbles from config
_MUMBLES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "config", "mumbles.json"
)


def _load_mumbles() -> list[dict]:
    """Load mumble lines from config file."""
    try:
        path = os.path.normpath(_MUMBLES_PATH)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d mumble lines", len(data))
        return data
    except FileNotFoundError:
        logger.warning("Mumbles config not found at %s — using defaults", _MUMBLES_PATH)
        return _default_mumbles()
    except Exception as e:
        logger.error("Failed to load mumbles: %s", e)
        return _default_mumbles()


def _default_mumbles() -> list[dict]:
    """Fallback mumble lines if the config file is missing."""
    return [
        {"text": "Hmm...", "moods": ["curious", "focused"]},
        {"text": "Interesting...", "moods": ["curious"]},
        {"text": "*yawns*", "moods": ["tired", "bored"]},
        {"text": "I wonder...", "moods": ["curious", "playful"]},
        {"text": "Huh.", "moods": ["curious", "bored"]},
    ]


class AutonomousBehaviour:
    """
    Scheduler for unprompted avatar events during idle periods.
    """

    def __init__(self) -> None:
        self._mumbles: list[dict] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_event_time: float = time.time()
        self._quiet_mode = False
        self.min_interval = MIN_EVENT_INTERVAL
        self.max_interval = MAX_EVENT_INTERVAL

        # Callbacks
        self._tts_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._broadcast_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
        self._gate_check: Optional[Callable[[], bool]] = None
        self._visual_idle_check: Optional[Callable[[], bool]] = None
        self._mood_getter: Optional[Callable[[], str]] = None

    def set_frequency(self, avg_seconds: float) -> None:
        """Set the autonomous event frequency based on avg_seconds."""
        self.min_interval = max(10.0, avg_seconds * 0.5)
        self.max_interval = avg_seconds * 1.5
        logger.info("Autonomous frequency updated: min=%.1fs, max=%.1fs", self.min_interval, self.max_interval)


    def set_tts_callback(self, cb: Callable[[str], Awaitable[None]]) -> None:
        """Set the TTS synthesis callback for mumble lines."""
        self._tts_callback = cb

    def set_broadcast_callback(self, cb: Callable[[str, dict], Awaitable[None]]) -> None:
        """Set the WebSocket broadcast callback for avatar events (glance, posture)."""
        self._broadcast_callback = cb

    def set_gate_check(self, cb: Callable[[], bool]) -> None:
        """Set the interrupt gate check function (returns True if events are allowed)."""
        self._gate_check = cb

    def set_visual_idle_check(self, cb: Callable[[], bool]) -> None:
        """Set the visual idle check function (returns True if glances/postures are allowed)."""
        self._visual_idle_check = cb

    def set_mood_getter(self, cb: Callable[[], str]) -> None:
        """Set the function that returns the current dominant mood."""
        self._mood_getter = cb

    def set_quiet_mode(self, quiet: bool) -> None:
        self._quiet_mode = quiet
        logger.info("Quiet mode: %s", "ON" if quiet else "OFF")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._mumbles = _load_mumbles()
        self._running = True
        self._task = asyncio.create_task(self._event_loop())
        logger.info("Autonomous behaviour started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomous behaviour stopped")

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    async def _event_loop(self) -> None:
        while self._running:
            try:
                delay = random.uniform(self.min_interval, self.max_interval)
                await asyncio.sleep(delay)

                # Check gate for visual idle (quiet mode or deep work)
                if self._quiet_mode:
                    continue
                if self._visual_idle_check and not self._visual_idle_check():
                    continue

                await self._fire_random_event()
                self._last_event_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Autonomous event error: %s", e)

    async def _fire_random_event(self) -> None:
        """Pick and execute a random autonomous event."""
        # Weight events: glance is most common, mumble less so
        event_type = random.choices(
            ["glance", "posture_shift", "mumble"],
            weights=[0.45, 0.30, 0.25],
            k=1,
        )[0]

        logger.info("Firing autonomous event: %s", event_type)

        if event_type == "glance":
            await self._do_glance()
        elif event_type == "posture_shift":
            await self._do_posture_shift()
        elif event_type == "mumble":
            await self._do_mumble()

    # ------------------------------------------------------------------
    # Individual events
    # ------------------------------------------------------------------

    async def _do_glance(self) -> None:
        """Send a random look-at target shift to the avatar."""
        target_x = random.uniform(-0.3, 0.3)
        target_y = random.uniform(-0.1, 0.2)
        duration = random.uniform(1.5, 3.0)

        if self._broadcast_callback:
            await self._broadcast_callback("avatar_glance", {
                "target_x": round(target_x, 3),
                "target_y": round(target_y, 3),
                "duration": round(duration, 2),
            })
        logger.info("Glance: (%.2f, %.2f) over %.1fs", target_x, target_y, duration)

    async def _do_posture_shift(self) -> None:
        """Send a posture shift command to the avatar."""
        shift_type = random.choice([
            "weight_shift_left",
            "weight_shift_right",
            "subtle_stretch",
            "head_tilt",
            "shoulder_roll",
        ])

        if self._broadcast_callback:
            await self._broadcast_callback("avatar_posture", {
                "shift_type": shift_type,
                "duration": round(random.uniform(1.0, 2.5), 2),
            })
        logger.info("Posture shift: %s", shift_type)

    async def _do_mumble(self) -> None:
        """Speak a mood-compatible mumble line via TTS."""
        if not self._tts_callback or not self._mumbles:
            return

        # Check interrupt gate before speaking proactively
        if self._gate_check and not self._gate_check():
            logger.debug("Mumble blocked by speech gate")
            return

        # Filter by current mood
        current_mood = self._mood_getter() if self._mood_getter else "neutral"
        compatible = [
            m for m in self._mumbles
            if current_mood in m.get("moods", [])
        ]
        if not compatible:
            # Fall back to any mumble
            compatible = self._mumbles

        line = random.choice(compatible)
        text = line["text"]

        logger.info("Mumble [%s]: '%s'", current_mood, text)
        try:
            await self._tts_callback(text)
        except Exception as e:
            logger.error("Mumble TTS failed: %s", e)
