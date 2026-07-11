"""
Silero VAD — Voice Activity Detection
Phase 3.2: Gates incoming audio to detect speech segments

Uses Silero VAD v5 to:
1. Accept 16kHz mono PCM audio frames
2. Detect voiced vs unvoiced frames
3. Buffer voiced frames
4. Emit complete utterances on 600ms of silence after speech
"""

import logging
import numpy as np
import torch

logger = logging.getLogger("sylph.stt.vad")

# Silero VAD constants
SAMPLE_RATE = 16000
FRAME_SIZE_MS = 32          # 32ms frames (512 samples at 16kHz)
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_MS / 1000)
SILENCE_THRESHOLD_MS = 500   # End-of-utterance after 500ms silence (was 1500ms — the
                             # single largest fixed latency cost in the whole pipeline)
SILENCE_FRAMES = int(SILENCE_THRESHOLD_MS / FRAME_SIZE_MS)  # ~15 frames
PRE_ROLL_MS = 256            # Keep ~8 frames of audio before speech onset so the
                             # first phoneme isn't clipped by VAD attack time
PRE_ROLL_FRAMES = int(PRE_ROLL_MS / FRAME_SIZE_MS)
MIN_SPEECH_MS = 250         # Minimum speech duration to emit
MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / FRAME_SIZE_MS)
SPEECH_THRESHOLD = 0.5      # VAD probability threshold


class VoiceActivityDetector:
    """
    Silero VAD wrapper that buffers voiced audio and emits
    complete utterances when silence is detected.
    """

    def __init__(self):
        self.model = None
        self._speech_buffer: list[np.ndarray] = []
        self._pre_roll: list[np.ndarray] = []
        self._silence_count = 0
        self._is_speaking = False
        self._speech_just_started = False
        self._loaded = False

    def load(self) -> None:
        """Load Silero VAD model. Call once at startup."""
        if self._loaded:
            return
        logger.info("Loading Silero VAD v5...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info("Silero VAD loaded (CPU, <2MB)")

    def reset(self) -> None:
        """Reset the VAD state for a new utterance."""
        self._speech_buffer.clear()
        self._pre_roll.clear()
        self._silence_count = 0
        self._is_speaking = False
        self._speech_just_started = False
        if self.model is not None:
            self.model.reset_states()

    def consume_speech_onset(self) -> bool:
        """
        Returns True exactly once when speech begins (for barge-in:
        the caller can cancel the active response the moment the user
        starts talking, instead of waiting for the full utterance).
        """
        if self._speech_just_started:
            self._speech_just_started = False
            return True
        return False

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray | None:
        """
        Process a single audio frame (30ms, 480 samples at 16kHz).

        Args:
            audio_frame: float32 numpy array, 16kHz mono, values in [-1, 1]

        Returns:
            Complete utterance as float32 numpy array when end-of-speech is detected,
            or None if still accumulating.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("VAD not loaded. Call load() first.")

        # Ensure correct frame size
        if len(audio_frame) != FRAME_SIZE_SAMPLES:
            # Pad or trim to exact frame size
            if len(audio_frame) < FRAME_SIZE_SAMPLES:
                audio_frame = np.pad(
                    audio_frame, (0, FRAME_SIZE_SAMPLES - len(audio_frame))
                )
            else:
                audio_frame = audio_frame[:FRAME_SIZE_SAMPLES]

        # Run VAD inference
        tensor = torch.from_numpy(audio_frame.copy()).float()
        speech_prob = self.model(tensor, SAMPLE_RATE).item()

        if speech_prob >= SPEECH_THRESHOLD:
            # Speech detected
            if not self._is_speaking:
                # Speech onset: prepend buffered pre-roll so the first
                # phoneme isn't clipped, and raise the barge-in flag.
                self._speech_just_started = True
                self._speech_buffer.extend(self._pre_roll)
                self._pre_roll.clear()
            self._is_speaking = True
            self._silence_count = 0
            self._speech_buffer.append(audio_frame.copy())
        else:
            if not self._is_speaking:
                # Maintain a short pre-roll ring buffer while idle
                self._pre_roll.append(audio_frame.copy())
                if len(self._pre_roll) > PRE_ROLL_FRAMES:
                    self._pre_roll.pop(0)
            if self._is_speaking:
                # Still buffering during brief silence within speech
                self._silence_count += 1
                self._speech_buffer.append(audio_frame.copy())

                # Check if silence threshold reached → end of utterance
                if self._silence_count >= SILENCE_FRAMES:
                    if len(self._speech_buffer) >= MIN_SPEECH_FRAMES:
                        # Emit the complete utterance
                        utterance = np.concatenate(self._speech_buffer)
                        logger.info(
                            "Utterance detected: %.1fs (%d samples)",
                            len(utterance) / SAMPLE_RATE,
                            len(utterance),
                        )
                        self.reset()
                        return utterance
                    else:
                        # Too short — probably noise, discard
                        logger.debug("Discarding short audio segment")
                        self.reset()

        return None

    def flush(self) -> np.ndarray | None:
        """
        Force-emit any buffered speech (e.g., when push-to-talk is released).
        """
        if self._speech_buffer and len(self._speech_buffer) >= MIN_SPEECH_FRAMES:
            utterance = np.concatenate(self._speech_buffer)
            logger.info(
                "Flushed utterance: %.1fs (%d samples)",
                len(utterance) / SAMPLE_RATE,
                len(utterance),
            )
            self.reset()
            return utterance
        self.reset()
        return None

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
