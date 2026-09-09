"""Phoneme parsing, duration estimation, and viseme timeline (pure functions)."""
from tts.duration_estimator import parse_phoneme_string, estimate_durations
from tts.viseme_builder import build_viseme_timeline, timeline_to_compact, VISEME_CHANNELS


def test_parse_empty():
    assert parse_phoneme_string("") == []


def test_parse_oov_marker_becomes_schwa():
    # misaki emits ❓ for OOV words; Phase C maps it to a schwa so the mouth
    # still moves instead of going dead.
    tokens = parse_phoneme_string("❓")
    assert tokens == ["ə"]


def test_parse_groups_multichar_phonemes():
    tokens = parse_phoneme_string("tʃ")
    assert "tʃ" in tokens


def test_estimate_durations_sum_matches_audio_length():
    sr = 24000
    samples = sr  # exactly 1 second
    durs = estimate_durations("hɛlO", samples, sr)
    total = sum(d["duration_ms"] for d in durs)
    assert durs, "expected some phonemes"
    assert abs(total - 1000.0) < 5.0  # ~1000 ms, allow rounding


def test_estimate_durations_empty_phonemes():
    assert estimate_durations("", 24000, 24000) == []


def test_viseme_timeline_channels_and_closing_frame():
    durs = estimate_durations("hɛlO", 24000, 24000)
    timeline = build_viseme_timeline(durs)
    assert timeline
    for kf in timeline:
        assert set(kf["weights"].keys()) == set(VISEME_CHANNELS)
    # last frame returns the mouth to rest (all zero)
    assert all(v == 0.0 for v in timeline[-1]["weights"].values())


def test_timeline_compact_row_shape():
    durs = estimate_durations("hɛlO", 24000, 24000)
    compact = timeline_to_compact(build_viseme_timeline(durs))
    assert compact
    for row in compact:
        assert len(row) == 7  # time, duration, aa, ee, ih, oh, ou
