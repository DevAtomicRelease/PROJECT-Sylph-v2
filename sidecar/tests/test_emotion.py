"""Emotion inference + mood->voice mappings (no model, no GPU)."""
from llm.planner import infer_emotion
from tts.emotion_map import mood_to_chatterbox, mood_to_voice, NEUTRAL_PITCH_STD


def test_infer_emotion_positive():
    assert infer_emotion("I'm so happy and excited, this is awesome!") == "happy"


def test_infer_emotion_empty_is_neutral():
    assert infer_emotion("") == "neutral"


def test_infer_emotion_plain_is_neutral():
    assert infer_emotion("the quick brown fox jumps") == "neutral"


def test_chatterbox_neutral_baseline():
    exag, cfg = mood_to_chatterbox("neutral", 1.0)
    assert exag == 0.5
    assert 0.2 <= cfg <= 1.0


def test_chatterbox_intensity_scales_and_clamps():
    # intensity 0 collapses to the neutral expressiveness
    exag0, _ = mood_to_chatterbox("happy", 0.0)
    assert abs(exag0 - 0.5) < 1e-9
    # higher intensity => more exaggeration for an aroused label
    exag1, _ = mood_to_chatterbox("surprised", 1.0)
    assert exag1 > 0.5
    # always within Chatterbox's sane range
    for label in ("happy", "angry", "sad", "shy", "bored", "relaxed", "neutral"):
        e, c = mood_to_chatterbox(label, 1.0)
        assert 0.25 <= e <= 1.5
        assert 0.2 <= c <= 1.0


def test_chatterbox_unknown_label_falls_back_to_neutral():
    assert mood_to_chatterbox("nonsense", 1.0) == mood_to_chatterbox("neutral", 1.0)


def test_mood_to_voice_neutral_baseline():
    v = mood_to_voice("neutral", 1.0)
    assert v.pitch_std == NEUTRAL_PITCH_STD
    assert len(v.emotion) == 8


def test_mood_to_voice_intensity_zero_is_neutralish():
    v = mood_to_voice("happy", 0.0)
    # at intensity 0 the vector collapses to the neutral baseline
    assert v.pitch_std == NEUTRAL_PITCH_STD
