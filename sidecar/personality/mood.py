"""
Mood State Machine
Phase 8.1: Dynamic mood vector that shifts based on interaction,
time-of-day, activity state, and random walk.

The mood vector is 7-dimensional:
  [playful, focused, bored, curious, annoyed, enthusiastic, tired]

The dominant mood (highest value) maps to a VRM expression label
pushed to the frontend via WebSocket for avatar expression blending.

Transitions occur every MOOD_TICK_SECONDS based on:
  - Time since last user interaction
  - Activity type (from screen triggers)
  - Time of day
  - Small random walk (normal distribution, stddev 0.05)
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("sylph.personality.mood")

MOOD_TICK_SECONDS = 60  # 1 minute — fast enough for expressions to feel alive
MOOD_DIMENSIONS = [
    "playful", "focused", "bored", "curious",
    "annoyed", "enthusiastic", "tired",
]

# Maps dominant mood dimension → VRM expression label
MOOD_TO_EXPRESSION = {
    "playful":      "happy",
    "focused":      "neutral",
    "bored":        "relaxed",
    "curious":      "relaxed",
    "annoyed":      "angry",
    "enthusiastic": "happy",
    "tired":        "relaxed",
}

# Maps emotion label → TTS speed multiplier (personality in voice)
MOOD_TO_TTS_SPEED = {
    "happy":     1.08,   # Slightly upbeat
    "angry":     1.12,   # Faster, crisper
    "sad":       0.88,   # Slower, heavier
    "relaxed":   0.95,   # Calm, measured
    "surprised": 1.10,   # Quick burst
    "neutral":   1.00,   # Baseline
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class MoodStateMachine:
    """
    7-dimensional mood vector with time-driven transitions
    and event-driven impulses.
    """

    def __init__(self) -> None:
        # Default mood vector — neutral, slightly curious
        self.values: dict[str, float] = {
            "playful":      0.30,
            "focused":      0.50,
            "bored":        0.10,
            "curious":      0.40,
            "annoyed":      0.00,
            "enthusiastic": 0.30,
            "tired":        0.10,
        }
        self._last_interaction: float = time.time()
        self._last_tick: float = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_mood_change: Optional[Callable[[str, dict], Awaitable[None]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dominant_mood(self) -> str:
        """Return the dimension name with the highest value."""
        return max(self.values, key=self.values.get)  # type: ignore

    @property
    def expression_label(self) -> str:
        """Return the VRM expression name for the current dominant mood."""
        return MOOD_TO_EXPRESSION.get(self.dominant_mood, "neutral")

    @property
    def tts_speed(self) -> float:
        """Return the TTS speed multiplier for the current expression."""
        return MOOD_TO_TTS_SPEED.get(self.expression_label, 1.0)

    @property
    def expression_intensity(self) -> float:
        """
        How strongly the dominant mood is held (0-1). Drives how far the
        Zonos emotion vector departs from neutral: a barely-dominant mood
        colors the voice subtly, a saturated one is unmistakable.
        """
        return _clamp(self.values.get(self.dominant_mood, 0.5))

    def on_user_interaction(self) -> None:
        """Call when the user speaks or interacts."""
        self._last_interaction = time.time()
        # Interaction boosts curiosity and enthusiasm, reduces boredom
        self._impulse(curious=0.08, enthusiastic=0.05, bored=-0.10, tired=-0.03)

    def on_positive_interaction(self) -> None:
        """Call when the user laughs, thanks, or expresses positivity."""
        self._impulse(playful=0.12, enthusiastic=0.10, annoyed=-0.10)

    def on_negative_interaction(self) -> None:
        """Call when the user expresses frustration or negativity."""
        self._impulse(annoyed=0.10, playful=-0.08, enthusiastic=-0.05)

    def on_speech_emotion(self, emotion: str) -> None:
        """
        Call when the LLM generates a response with a detectable emotional tone.
        Applies a targeted impulse to shift the mood vector toward the
        expressed emotion. This ensures the avatar's face reflects
        what Sylph just said, immediately.

        Args:
            emotion: One of 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'neutral'
        """
        impulse_map = {
            "happy":     {"enthusiastic": 0.10, "playful": 0.08, "annoyed": -0.05, "bored": -0.05},
            "angry":     {"annoyed": 0.15, "enthusiastic": -0.05, "playful": -0.08},
            "sad":       {"bored": 0.08, "tired": 0.06, "enthusiastic": -0.08, "playful": -0.06},
            "relaxed":   {"curious": 0.05, "tired": 0.03, "annoyed": -0.05, "bored": -0.03},
            "surprised": {"curious": 0.12, "enthusiastic": 0.08, "bored": -0.08},
            "neutral":   {"focused": 0.05, "annoyed": -0.03, "bored": -0.03},
        }
        kwargs = impulse_map.get(emotion, {})
        if kwargs:
            self._impulse(**kwargs)
            logger.debug("Speech emotion impulse: %s → %s", emotion, kwargs)

    def set_activity_state(self, state: str) -> None:
        """
        Adjust mood based on detected activity state.
        States: 'active', 'passive', 'idle', 'deep_work', 'sleep'
        """
        if state == "deep_work":
            self._impulse(focused=0.15, bored=-0.10, playful=-0.05)
        elif state == "idle":
            self._impulse(bored=0.10, focused=-0.05, curious=0.05)
        elif state == "sleep":
            self._impulse(tired=0.15, bored=0.05, enthusiastic=-0.10)

    def set_callback(self, callback: Callable[[str, dict], Awaitable[None]]) -> None:
        """Set the callback for mood change broadcasts."""
        self._on_mood_change = callback

    async def force_broadcast(self) -> None:
        """
        Immediately broadcast the current mood to all connected clients.
        Call this after applying an emotion impulse so the frontend updates
        without waiting for the next tick.
        """
        if self._on_mood_change:
            label = self.expression_label
            try:
                await self._on_mood_change(label, dict(self.values))
                logger.debug("Forced mood broadcast: %s", label)
            except Exception as e:
                logger.error("Forced mood broadcast error: %s", e)

    def get_state_for_new_connection(self) -> dict:
        """
        Return the current mood payload formatted for immediate broadcast
        to a newly connected WebSocket client.
        """
        return {
            "mood": self.expression_label,
            "values": {k: round(v, 3) for k, v in self.values.items()},
        }

    # ------------------------------------------------------------------
    # Background tick loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic mood tick."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("Mood state machine started (tick: %ds)", MOOD_TICK_SECONDS)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Mood state machine stopped")

    async def _tick_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(MOOD_TICK_SECONDS)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Mood tick error: %s", e)

    async def _tick(self) -> None:
        """One mood transition tick."""
        now = time.time()
        idle_seconds = now - self._last_interaction
        hour = datetime.now().hour

        # 1. Time-of-day bias
        self._apply_time_bias(hour)

        # 2. Idle drift — boredom and tiredness increase with inactivity
        if idle_seconds > 120:
            idle_factor = min(idle_seconds / 600, 1.0) * 0.05
            self._impulse(bored=idle_factor, tired=idle_factor * 0.5,
                          enthusiastic=-idle_factor * 0.3)

        # 3. Random walk — small stochastic changes for organic feel
        for dim in MOOD_DIMENSIONS:
            noise = random.gauss(0, 0.05)
            self.values[dim] = _clamp(self.values[dim] + noise)

        # 4. Normalize — prevent any dimension from dominating too much
        self._soft_normalize()

        self._last_tick = now

        # Broadcast new mood
        label = self.expression_label
        logger.info(
            "Mood tick: dominant=%s (%s), vector=%s",
            self.dominant_mood, label,
            {k: round(v, 2) for k, v in self.values.items()},
        )

        if self._on_mood_change:
            try:
                await self._on_mood_change(label, dict(self.values))
            except Exception as e:
                logger.error("Mood broadcast error: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impulse(self, **kwargs: float) -> None:
        """Apply an impulse to specific dimensions."""
        for dim, delta in kwargs.items():
            if dim in self.values:
                self.values[dim] = _clamp(self.values[dim] + delta)

    def _apply_time_bias(self, hour: int) -> None:
        """Shift mood based on time of day."""
        if 6 <= hour < 10:
            # Morning: energized
            self._impulse(enthusiastic=0.03, tired=-0.03, playful=0.02)
        elif 10 <= hour < 14:
            # Mid-day: focused
            self._impulse(focused=0.03, playful=-0.01)
        elif 14 <= hour < 17:
            # Afternoon: slight boredom, lower energy
            self._impulse(bored=0.02, tired=0.02, focused=-0.01)
        elif 17 <= hour < 21:
            # Evening: relaxed
            self._impulse(playful=0.02, focused=-0.02, curious=0.02)
        elif 21 <= hour or hour < 2:
            # Late night: tired
            self._impulse(tired=0.05, enthusiastic=-0.03, focused=-0.02)
        elif 2 <= hour < 6:
            # Very late / early morning
            self._impulse(tired=0.08, bored=0.03, enthusiastic=-0.05)

    def _soft_normalize(self) -> None:
        """
        Gently pull values toward a balanced range so no single
        dimension permanently dominates. Targets the mean of all values.
        """
        total = sum(self.values.values())
        if total == 0:
            return
        mean = total / len(self.values)
        for dim in MOOD_DIMENSIONS:
            # Pull 10% toward the mean each tick
            self.values[dim] = _clamp(
                self.values[dim] + (mean - self.values[dim]) * 0.10
            )

    def to_dict(self) -> dict:
        """Serialize for WebSocket broadcast."""
        return {
            "dominant": self.dominant_mood,
            "expression": self.expression_label,
            "values": {k: round(v, 3) for k, v in self.values.items()},
        }
