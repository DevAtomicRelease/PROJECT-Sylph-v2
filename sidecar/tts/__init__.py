"""Sylph TTS (Text-to-Speech) Module — Phase 4"""

from .synthesizer import Synthesizer
from .duration_estimator import estimate_durations, parse_phoneme_string
from .viseme_builder import build_viseme_timeline, timeline_to_compact
from .payload import build_dual_payload

__all__ = [
    "Synthesizer",
    "estimate_durations",
    "parse_phoneme_string",
    "build_viseme_timeline",
    "timeline_to_compact",
    "build_dual_payload",
]
