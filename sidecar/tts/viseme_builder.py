"""
Viseme Timeline Builder
Phase 4.4: Converts phoneme durations + IPA-to-viseme map into a viseme timeline

Takes the output of duration_estimator and the IPA-to-viseme JSON map to produce
a frame-by-frame timeline of VRM blendshape weights for the frontend to replay.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("sylph.tts.viseme")

# VRM 1.0 standard viseme morph targets
VISEME_CHANNELS = ["aa", "ee", "ih", "oh", "ou"]

# Path to the IPA-to-viseme mapping file
_VISEME_MAP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "config", "ipa_to_viseme.json"
)

_viseme_map: Optional[dict] = None


def _load_viseme_map() -> dict:
    """Load the IPA-to-viseme mapping from JSON (cached)."""
    global _viseme_map
    if _viseme_map is not None:
        return _viseme_map

    path = os.path.normpath(_VISEME_MAP_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            _viseme_map = json.load(f)
        # Remove comment keys
        _viseme_map = {k: v for k, v in _viseme_map.items() if not k.startswith("_")}
        logger.info("Loaded viseme map: %d entries from %s", len(_viseme_map), path)
    except FileNotFoundError:
        logger.warning("Viseme map not found at %s — using empty mapping", path)
        _viseme_map = {}
    return _viseme_map


def build_viseme_timeline(
    phoneme_durations: list[dict],
    transition_ms: float = 30.0,
) -> list[dict]:
    """
    Build a viseme timeline from phoneme durations.

    Args:
        phoneme_durations: List of {"phoneme": str, "duration_ms": float}
                           from duration_estimator.estimate_durations()
        transition_ms: Crossfade time between visemes (ms)

    Returns:
        List of timeline keyframes:
        [
            {
                "time_ms": float,       # Start time in milliseconds
                "duration_ms": float,   # Duration of this keyframe
                "weights": {            # VRM blendshape weights
                    "aa": 0.0-1.0,
                    "ee": 0.0-1.0,
                    "ih": 0.0-1.0,
                    "oh": 0.0-1.0,
                    "ou": 0.0-1.0,
                }
            },
            ...
        ]
    """
    vmap = _load_viseme_map()
    timeline: list[dict] = []
    current_time_ms = 0.0

    for entry in phoneme_durations:
        phoneme = entry["phoneme"]
        duration_ms = entry["duration_ms"]

        # Look up viseme weights for this phoneme
        weights = vmap.get(phoneme, {})

        # Build weight vector (ensure all channels present)
        weight_vec = {}
        for ch in VISEME_CHANNELS:
            weight_vec[ch] = weights.get(ch, 0.0)

        timeline.append({
            "time_ms": round(current_time_ms, 1),
            "duration_ms": round(duration_ms, 1),
            "weights": weight_vec,
        })

        current_time_ms += duration_ms

    # Add a closing keyframe to return mouth to rest
    if timeline:
        timeline.append({
            "time_ms": round(current_time_ms, 1),
            "duration_ms": round(transition_ms, 1),
            "weights": {ch: 0.0 for ch in VISEME_CHANNELS},
        })

    logger.debug(
        "Built viseme timeline: %d keyframes, %.0fms total",
        len(timeline),
        current_time_ms,
    )
    return timeline


def timeline_to_compact(timeline: list[dict]) -> list:
    """
    Convert timeline to a compact array format for WebSocket transmission.

    Output: [[time_ms, duration_ms, aa, ee, ih, oh, ou], ...]

    Each inner array is 7 floats — much smaller than the dict format.
    """
    compact = []
    for kf in timeline:
        w = kf["weights"]
        compact.append([
            kf["time_ms"],
            kf["duration_ms"],
            w.get("aa", 0),
            w.get("ee", 0),
            w.get("ih", 0),
            w.get("oh", 0),
            w.get("ou", 0),
        ])
    return compact
