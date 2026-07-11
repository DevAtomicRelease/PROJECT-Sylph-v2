"""
TTS Emotional Audio Post-Processing
Applies lightweight pitch shift and rate adjustment to Kokoro TTS output
based on the current emotional state.

Uses pure numpy linear interpolation for pitch shifting — no new
dependencies, no new models. Adds ~1-2ms per audio chunk at 24kHz.

Usage:
    from tts.audio_post import apply_emotion_fx
    processed = apply_emotion_fx(audio_array, "happy")
"""

import logging
import numpy as np

logger = logging.getLogger("sylph.tts.audio_post")

# Emotion → audio effect parameters
# pitch_shift: >1.0 = higher pitch, <1.0 = lower pitch
# rate: >1.0 = faster, <1.0 = slower (applied via resampling)
EMOTION_AUDIO_FX: dict[str, dict[str, float]] = {
    "happy":     {"pitch_shift": 1.03, "rate": 1.00},   # brighter
    "angry":     {"pitch_shift": 1.01, "rate": 1.00},   # tiny edge
    "sad":       {"pitch_shift": 0.97, "rate": 0.95},   # lower, slower
    "relaxed":   {"pitch_shift": 1.00, "rate": 0.98},   # calm
    "surprised": {"pitch_shift": 1.04, "rate": 1.02},   # quick, higher
    "neutral":   {"pitch_shift": 1.00, "rate": 1.00},   # no change
    "shy":       {"pitch_shift": 0.99, "rate": 0.97},   # soft, slower
    "bored":     {"pitch_shift": 0.98, "rate": 0.96},   # droning
}

_DEFAULT_FX = {"pitch_shift": 1.00, "rate": 1.00}


def _resample_linear(audio: np.ndarray, factor: float) -> np.ndarray:
    """
    Resample audio by a fractional factor using linear interpolation.
    factor > 1.0 → fewer output samples (higher pitch / faster)
    factor < 1.0 → more output samples (lower pitch / slower)
    """
    if abs(factor - 1.0) < 0.001:
        return audio

    n_in = len(audio)
    n_out = int(n_in / factor)
    if n_out < 2:
        return audio

    # Create interpolation indices
    indices = np.linspace(0, n_in - 1, n_out, dtype=np.float64)
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, n_in - 1)
    frac = (indices - idx_floor).astype(np.float32)

    # Linear interpolation
    return audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac


def apply_emotion_fx(
    audio: np.ndarray,
    emotion: str,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Apply emotional audio post-processing to a TTS audio buffer.

    Pitch shifting is done by resampling the audio (changing sample count)
    then trimming/padding to maintain the original playback duration feel.
    Rate adjustment changes the actual duration of the audio.

    Args:
        audio: float32 numpy array (mono, 24kHz)
        emotion: VRM expression label (happy, angry, sad, etc.)
        sample_rate: audio sample rate (for logging only)

    Returns:
        Processed float32 numpy array
    """
    fx = EMOTION_AUDIO_FX.get(emotion, _DEFAULT_FX)
    pitch = fx["pitch_shift"]
    rate = fx["rate"]

    # Skip processing if no effect needed
    if abs(pitch - 1.0) < 0.001 and abs(rate - 1.0) < 0.001:
        return audio

    # Combined factor: pitch shift changes frequency (resample by pitch factor),
    # rate changes duration (resample by 1/rate factor).
    # Net resampling factor = pitch / rate
    # e.g., pitch=1.03, rate=0.95 → factor=1.084 → fewer samples → higher pitch + slower
    #
    # But we want:
    #   - Pitch shift without changing duration → resample then trim/pad
    #   - Rate change that changes duration → resample
    #
    # Simple approach: apply rate first (changes duration), then pitch (preserves new duration)

    result = audio

    # Step 1: Rate adjustment (changes duration)
    if abs(rate - 1.0) >= 0.001:
        result = _resample_linear(result, rate)

    # Step 2: Pitch shift (preserves duration by resampling + truncating)
    if abs(pitch - 1.0) >= 0.001:
        # Resample to shift pitch
        pitched = _resample_linear(result, pitch)
        # Restore original length to preserve duration
        n_orig = len(result)
        if len(pitched) > n_orig:
            result = pitched[:n_orig]
        elif len(pitched) < n_orig:
            result = np.pad(pitched, (0, n_orig - len(pitched)))
        else:
            result = pitched

    return result.astype(np.float32)
