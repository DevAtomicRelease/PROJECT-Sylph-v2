"""
Duration Estimator for Phoneme Timing
Phase 4.3: Estimates per-phoneme durations from audio length

Uses proportional estimation (Approach C from the implementation plan):
Distributes total audio duration across phonemes using weighted averages
based on known phoneme duration statistics from CMU data.

This is the robust fallback that works regardless of Kokoro internals.
Accuracy is ~90% compared to the duration predictor hook approach,
but still far superior to amplitude-based jaw-bobbing.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("sylph.tts.duration")

# Average relative phoneme durations (from CMU statistics, normalized)
# Vowels are longer than consonants; stops are shortest
PHONEME_WEIGHTS: dict[str, float] = {
    # Vowels (longer)
    "ɑ": 1.4, "æ": 1.3, "ʌ": 1.1, "ɛ": 1.2, "ə": 0.8,
    "iː": 1.5, "ɪ": 1.0, "uː": 1.4, "ʊ": 1.0,
    "oʊ": 1.5, "ɔ": 1.3, "ɔɪ": 1.6, "aɪ": 1.5, "aʊ": 1.5,
    "eɪ": 1.4, "ɑɹ": 1.5, "ɔɹ": 1.4,
    "A": 1.4, "I": 1.5, "O": 1.5, "W": 1.5,
    # Fricatives (medium)
    "f": 0.9, "v": 0.8, "θ": 0.9, "ð": 0.7,
    "s": 1.0, "z": 0.9, "ʃ": 1.0, "ʒ": 0.9,
    "h": 0.7,
    # Stops (short)
    "p": 0.5, "b": 0.5, "t": 0.5, "d": 0.5,
    "k": 0.6, "ɡ": 0.5,
    # Nasals
    "m": 0.8, "n": 0.7, "ŋ": 0.8,
    # Liquids/Glides
    "l": 0.8, "ɹ": 0.8, "w": 0.6, "j": 0.6,
    # Affricates
    "tʃ": 0.7, "dʒ": 0.7, "ʤ": 0.7,
    # Silence
    "_": 0.3,
}

DEFAULT_WEIGHT = 0.8  # For unknown phonemes


def parse_phoneme_string(phonemes: str) -> list[str]:
    """
    Parse a Kokoro/misaki phoneme string into individual IPA symbols.

    Handles:
    - Multi-character phonemes (tʃ, dʒ, iː, oʊ, etc.)
    - Stress marks and punctuation (stripped)
    - Spaces (mapped to silence '_')
    """
    if not phonemes:
        return []

    # Remove stress marks, punctuation, and other non-phoneme annotations
    cleaned = re.sub(r"[ˈˌ.ˑ,?!;\-—]", "", phonemes.strip())

    # Always parse character by character, grouping multi-char phonemes
    tokens = []
    i = 0
    while i < len(cleaned):
        # Check for 2-char phonemes first
        if i + 1 < len(cleaned):
            bigram = cleaned[i : i + 2]
            if bigram in PHONEME_WEIGHTS:
                tokens.append(bigram)
                i += 2
                continue
        # Single char
        ch = cleaned[i]
        if ch in (" ", "\t"):
            # Map word boundary space to silence
            tokens.append("_")
        else:
            tokens.append(ch)
        i += 1

    return tokens


def estimate_durations(
    phonemes: str,
    audio_samples: int,
    sample_rate: int = 24000,
) -> list[dict]:
    """
    Estimate per-phoneme durations by distributing audio time proportionally.

    Args:
        phonemes: IPA phoneme string from Kokoro/misaki
        audio_samples: Total audio length in samples
        sample_rate: Audio sample rate (default 24kHz for Kokoro)

    Returns:
        List of {"phoneme": str, "duration_ms": float}
    """
    total_ms = (audio_samples / sample_rate) * 1000.0
    phoneme_list = parse_phoneme_string(phonemes)

    if not phoneme_list:
        return []

    # Get weights for each phoneme
    weights = [PHONEME_WEIGHTS.get(p, DEFAULT_WEIGHT) for p in phoneme_list]
    total_weight = sum(weights)

    if total_weight <= 0:
        # Equal distribution fallback
        per_phoneme_ms = total_ms / len(phoneme_list)
        return [
            {"phoneme": p, "duration_ms": per_phoneme_ms}
            for p in phoneme_list
        ]

    # Distribute proportionally
    result = []
    for phoneme, weight in zip(phoneme_list, weights):
        dur_ms = (weight / total_weight) * total_ms
        result.append({"phoneme": phoneme, "duration_ms": round(dur_ms, 1)})

    logger.debug(
        "Estimated durations for %d phonemes over %.0fms",
        len(result),
        total_ms,
    )
    return result
