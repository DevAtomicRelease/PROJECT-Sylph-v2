"""
Mood → Zonos emotion conditioning
Phase 12: Maps Sylph's expression labels (from the mood state machine and the
per-clause keyword inference) onto Zonos' native conditioning space:

  emotion vector: [happiness, sadness, disgust, fear, surprise, anger, other, neutral]
  pitch_std:      pitch variation in Hz-ish units (Zonos: ~20-45 natural,
                  60-150 expressive)
  speaking_rate:  phonemes per second (Zonos default ~15)

The presets are deliberately anchored to the SAME eight labels the rest of
the system already speaks (VRM expressions + audio_post + mood machine), so
one label drives face, body gesture, and now voice — that coherence is what
reads as "the avatar feels something", not any single channel.

`intensity` (0-1, from the dominant mood dimension's strength) interpolates
each preset from the neutral baseline, so a weakly-held mood sounds like a
hint and a strongly-held one is unmistakable.
"""

from dataclasses import dataclass

# Zonos vector index order (per zonos.conditioning docs)
_H, _SAD, _DISGUST, _FEAR, _SURPRISE, _ANGER, _OTHER, _NEUTRAL = range(8)

NEUTRAL_PITCH_STD = 45.0
NEUTRAL_RATE = 14.5


@dataclass(frozen=True)
class VoiceEmotion:
    emotion: list[float]      # 8-dim Zonos vector
    pitch_std: float
    speaking_rate: float


def _vec(**kw: float) -> list[float]:
    v = [0.0] * 8
    named = {
        "happiness": _H, "sadness": _SAD, "disgust": _DISGUST, "fear": _FEAR,
        "surprise": _SURPRISE, "anger": _ANGER, "other": _OTHER, "neutral": _NEUTRAL,
    }
    for name, value in kw.items():
        v[named[name]] = value
    return v


_NEUTRAL_VEC = _vec(neutral=1.0)

# Full-intensity presets per expression label.
_PRESETS: dict[str, VoiceEmotion] = {
    "happy":     VoiceEmotion(_vec(happiness=0.85, neutral=0.15), pitch_std=72.0, speaking_rate=15.5),
    "angry":     VoiceEmotion(_vec(anger=0.70, disgust=0.10, neutral=0.20), pitch_std=80.0, speaking_rate=16.5),
    "sad":       VoiceEmotion(_vec(sadness=0.75, neutral=0.25), pitch_std=28.0, speaking_rate=12.0),
    "relaxed":   VoiceEmotion(_vec(happiness=0.25, neutral=0.75), pitch_std=38.0, speaking_rate=13.5),
    "surprised": VoiceEmotion(_vec(surprise=0.70, happiness=0.15, neutral=0.15), pitch_std=95.0, speaking_rate=16.0),
    "neutral":   VoiceEmotion(list(_NEUTRAL_VEC), pitch_std=NEUTRAL_PITCH_STD, speaking_rate=NEUTRAL_RATE),
    "shy":       VoiceEmotion(_vec(happiness=0.20, fear=0.15, neutral=0.65), pitch_std=36.0, speaking_rate=13.0),
    "bored":     VoiceEmotion(_vec(sadness=0.20, neutral=0.80), pitch_std=24.0, speaking_rate=12.5),
}


def mood_to_voice(label: str, intensity: float = 1.0) -> VoiceEmotion:
    """
    Resolve an expression label + intensity to Zonos conditioning values.

    intensity 0 → neutral baseline; 1 → full preset; in between, linear
    interpolation of the vector, pitch_std, and rate.
    """
    preset = _PRESETS.get(label, _PRESETS["neutral"])
    t = max(0.0, min(1.0, intensity))
    if t >= 0.999:
        return preset

    emotion = [
        n + (p - n) * t for n, p in zip(_NEUTRAL_VEC, preset.emotion)
    ]
    return VoiceEmotion(
        emotion=emotion,
        pitch_std=NEUTRAL_PITCH_STD + (preset.pitch_std - NEUTRAL_PITCH_STD) * t,
        speaking_rate=NEUTRAL_RATE + (preset.speaking_rate - NEUTRAL_RATE) * t,
    )
