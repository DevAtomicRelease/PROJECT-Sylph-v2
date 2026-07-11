"""
Mood-to-Sampling Parameter Mapping
Maps the current mood dimension to LLM sampling parameters.

These are configurable starting values — tune them based on observed
output quality for each mood state.

Usage:
    from llm.sampling_config import get_sampling_params
    params = get_sampling_params("playful")
    # → {"temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.05}
"""

import logging

logger = logging.getLogger("sylph.llm.sampling")

# Mood dimension → sampling parameters
# Higher temperature → more creative, varied output
# Lower temperature → more focused, deterministic output
MOOD_SAMPLING: dict[str, dict[str, float]] = {
    "playful":      {"temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.05},
    "enthusiastic": {"temperature": 0.85, "top_p": 0.92, "repetition_penalty": 1.05},
    "focused":      {"temperature": 0.55, "top_p": 0.80, "repetition_penalty": 1.10},
    "curious":      {"temperature": 0.75, "top_p": 0.90, "repetition_penalty": 1.05},
    "bored":        {"temperature": 0.72, "top_p": 0.85, "repetition_penalty": 1.08},
    "annoyed":      {"temperature": 0.70, "top_p": 0.85, "repetition_penalty": 1.10},
    "tired":        {"temperature": 0.65, "top_p": 0.82, "repetition_penalty": 1.08},
}

# Fallback for unknown mood states
_DEFAULT_PARAMS = {"temperature": 0.70, "top_p": 0.90, "repetition_penalty": 1.05}


def get_sampling_params(mood: str) -> dict[str, float]:
    """
    Look up sampling parameters for a given mood dimension.

    Args:
        mood: One of the 7 mood dimensions (playful, focused, etc.)

    Returns:
        Dict with temperature, top_p, repetition_penalty
    """
    params = MOOD_SAMPLING.get(mood, _DEFAULT_PARAMS)
    logger.debug("Sampling params for mood '%s': %s", mood, params)
    return dict(params)  # return a copy to prevent mutation
"""
Created 2026-06-21 — Sylph personality improvements, Item 4
"""
