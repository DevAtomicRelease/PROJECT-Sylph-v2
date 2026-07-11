"""
Dual-Payload Builder
Phase 4.5: Packs audio + viseme timeline into a single binary WebSocket frame

Binary frame layout:
┌─────────────────┬──────────────────┬───────────────────────┬───────────────┐
│ Header (8 bytes)│ Viseme JSON      │ Audio PCM             │               │
│ magic(2)+ver(1) │ (UTF-8, variable)│ (float32, variable)   │               │
│ +viseme_len(4)  │                  │                       │               │
│ +flags(1)       │                  │                       │               │
└─────────────────┴──────────────────┴───────────────────────┴───────────────┘

This packs both the viseme timeline and the audio waveform into one
WebSocket binary message, allowing the frontend to synchronize lip movement
with audio playback from a single atomic delivery.
"""

import json
import struct
import logging
import numpy as np

logger = logging.getLogger("sylph.tts.payload")

# Magic bytes to identify Sylph audio+viseme frames
MAGIC = b"AV"      # 2 bytes
VERSION = 2         # 1 byte — v2 adds a uint32 turn_id after flags
# Flags
FLAG_NONE = 0x00
FLAG_FINAL_CHUNK = 0x01   # Last chunk of a multi-chunk utterance


def build_dual_payload(
    audio: np.ndarray,
    viseme_timeline: list,
    sample_rate: int = 24000,
    is_final: bool = True,
    turn_id: int = 0,
) -> bytes:
    """
    Pack audio + viseme timeline into a single binary payload.

    Args:
        audio: float32 numpy array (mono, any sample rate)
        viseme_timeline: Compact viseme array from timeline_to_compact()
        sample_rate: Audio sample rate
        is_final: Whether this is the last chunk of a multi-part utterance

    Returns:
        Binary payload bytes ready for WebSocket send_bytes()
    """
    # Encode viseme timeline as compact JSON
    viseme_json = json.dumps({
        "sample_rate": sample_rate,
        "timeline": viseme_timeline,
    }).encode("utf-8")

    viseme_len = len(viseme_json)

    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    audio_bytes = audio.tobytes()

    # Header v2: magic(2) + version(1) + viseme_json_length(4) + flags(1) + turn_id(4) = 12 bytes
    # turn_id lets the frontend discard chunks from a cancelled/superseded turn,
    # eliminating zombie audio that survives backend task cancellation.
    flags = FLAG_FINAL_CHUNK if is_final else FLAG_NONE
    header = struct.pack(
        "<2sBIBI",      # little-endian: 2-char, uint8, uint32, uint8, uint32
        MAGIC,
        VERSION,
        viseme_len,
        flags,
        turn_id & 0xFFFFFFFF,
    )

    payload = header + viseme_json + audio_bytes

    logger.debug(
        "Built dual payload: header=%d, viseme=%d, audio=%d bytes (total=%d)",
        len(header),
        viseme_len,
        len(audio_bytes),
        len(payload),
    )
    return payload


def parse_dual_payload(data: bytes) -> dict:
    """
    Parse a dual payload (used for testing / Python-side debugging).

    Returns:
        {
            "version": int,
            "flags": int,
            "sample_rate": int,
            "timeline": list,
            "audio": np.ndarray,
        }
    """
    # Parse header
    magic = data[0:2]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r}, expected {MAGIC!r}")

    version = data[2]
    if version >= 2:
        _, viseme_len, flags, turn_id = struct.unpack_from("<BIBI", data, 2)
        viseme_start = 12
    else:
        _, viseme_len, flags = struct.unpack_from("<BIB", data, 2)
        turn_id = 0
        viseme_start = 8

    # Parse viseme JSON
    viseme_end = viseme_start + viseme_len
    viseme_data = json.loads(data[viseme_start:viseme_end].decode("utf-8"))

    # Parse audio
    audio_bytes = data[viseme_end:]
    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    return {
        "version": version,
        "flags": flags,
        "turn_id": turn_id,
        "sample_rate": viseme_data.get("sample_rate", 24000),
        "timeline": viseme_data.get("timeline", []),
        "audio": audio,
    }
