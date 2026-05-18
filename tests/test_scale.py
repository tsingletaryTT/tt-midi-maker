import pytest
from tt_midi_maker.coherence.scale import (
    parse_key, build_scale_set, nearest_scale_pitch, scale_quantize,
)
from tt_midi_maker.models.track import NoteEvent


def note(pitch, channel=1):
    return NoteEvent(pitch=pitch, velocity=80, start_tick=0, duration_ticks=480, channel=channel)


def test_parse_d_minor():
    root, mode = parse_key("D minor")
    assert root == 2 and mode == "minor"


def test_parse_fsharp_dorian():
    root, mode = parse_key("F# dorian")
    assert root == 6 and mode == "dorian"


def test_parse_invalid_root():
    with pytest.raises(ValueError, match="Cannot parse key"):
        parse_key("X major")


def test_parse_invalid_mode():
    with pytest.raises(ValueError, match="Cannot parse key"):
        parse_key("C ragtime")


def test_c_major_scale_set():
    # C major: C D E F G A B  = 0 2 4 5 7 9 11
    assert build_scale_set(0, "major") == frozenset({0, 2, 4, 5, 7, 9, 11})


def test_d_minor_scale_set():
    # D natural minor: D E F G A Bb C = 2 4 5 7 9 10 0
    assert build_scale_set(2, "minor") == frozenset({0, 2, 4, 5, 7, 9, 10})


def test_nearest_in_scale_returns_same():
    scale = build_scale_set(0, "major")   # C major
    assert nearest_scale_pitch(60, scale) == 60  # C4 is in scale


def test_nearest_out_of_scale():
    scale = build_scale_set(0, "major")   # C major
    # F# = 66, nearest scale tones are F=65 and G=67
    result = nearest_scale_pitch(66, scale)
    assert result in (65, 67)


def test_scale_quantize_drums_unchanged():
    n = note(36, channel=10)
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch == 36


def test_scale_quantize_in_scale_unchanged():
    n = note(60, channel=1)          # C4 in C major
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch == 60


def test_scale_quantize_snaps_off_scale_at_strictness_1():
    n = note(66, channel=1)          # F# not in C major
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch % 12 in build_scale_set(0, "major")


def test_scale_quantize_preserves_at_strictness_0(monkeypatch):
    import tt_midi_maker.coherence.scale as s
    monkeypatch.setattr(s.random, "random", lambda: 0.99)   # always > strictness=0
    n = note(66, channel=1)
    result = scale_quantize([n], "C major", strictness=0.0)
    assert result[0].pitch == 66
