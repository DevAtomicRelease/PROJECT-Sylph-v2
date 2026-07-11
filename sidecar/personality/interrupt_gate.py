"""
Interrupt Gate
Phase 8.3: Decides whether Sylph should speak proactively

Before Sylph speaks unprompted (screen triage CONSIDER_COMMENT,
autonomous comment, or any proactive behaviour), this gate checks:
  1. Has Sylph spoken in the last 60 seconds? → suppress
  2. Is the user in deep work mode? → suppress
  3. Is quiet mode active? → suppress
  4. Is the comment relevant to context? → allow if yes

Only if ALL gates pass does the comment proceed to TTS.
"""

import logging
import time

logger = logging.getLogger("sylph.personality.gate")

# Minimum seconds between proactive speech
MIN_SPEECH_INTERVAL = 60

# Maximum proactive comments per session before throttling
MAX_PROACTIVE_PER_HOUR = 20


class InterruptGate:
    """
    Controls when Sylph is allowed to speak proactively.
    All checks must pass for a proactive comment to proceed.
    """

    def __init__(self) -> None:
        self._last_spoke_time: float = 0
        self._quiet_mode: bool = False
        self._deep_work: bool = False
        self._proactive_count: int = 0
        self._count_reset_time: float = time.time()

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def on_speech_delivered(self) -> None:
        """Called after Sylph finishes speaking (interactive or proactive)."""
        self._last_spoke_time = time.time()

    def set_quiet_mode(self, quiet: bool) -> None:
        self._quiet_mode = quiet
        logger.info("Interrupt gate: quiet mode %s", "ON" if quiet else "OFF")

    def set_deep_work(self, active: bool) -> None:
        self._deep_work = active
        logger.info("Interrupt gate: deep work %s", "ON" if active else "OFF")

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def should_allow(self, reason: str = "") -> bool:
        """
        Check if a proactive comment is allowed right now.

        Args:
            reason: Optional description for logging

        Returns:
            True if the comment should proceed, False if suppressed
        """
        now = time.time()

        # Reset hourly counter
        if now - self._count_reset_time > 3600:
            self._proactive_count = 0
            self._count_reset_time = now

        # Gate 1: Quiet mode
        if self._quiet_mode:
            logger.debug("Gate BLOCKED [quiet mode]: %s", reason)
            return False

        # Gate 2: Deep work mode
        if self._deep_work:
            logger.debug("Gate BLOCKED [deep work]: %s", reason)
            return False

        # Gate 3: Too soon after last speech
        elapsed = now - self._last_spoke_time
        if self._last_spoke_time > 0 and elapsed < MIN_SPEECH_INTERVAL:
            logger.debug(
                "Gate BLOCKED [spoke %.0fs ago < %ds]: %s",
                elapsed, MIN_SPEECH_INTERVAL, reason,
            )
            return False

        # Gate 4: Hourly rate limit
        if self._proactive_count >= MAX_PROACTIVE_PER_HOUR:
            logger.debug("Gate BLOCKED [hourly limit %d]: %s",
                         MAX_PROACTIVE_PER_HOUR, reason)
            return False

        # All gates pass
        self._proactive_count += 1
        logger.debug("Gate ALLOWED (#%d): %s", self._proactive_count, reason)
        return True

    def can_speak(self) -> bool:
        """Simple check for the autonomous behaviour system."""
        return self.should_allow("autonomous_check")

    @property
    def is_quiet(self) -> bool:
        return self._quiet_mode

    @property
    def is_deep_work(self) -> bool:
        return self._deep_work
