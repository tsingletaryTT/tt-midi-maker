import pytest
from tt_midi_maker.coherence.harmony import (
    parse_chord, chord_at_tick, is_strong_beat, chord_aware_filter,
)
from tt_midi_maker.models.track import NoteEvent

TICKS_PER_BEAT = 480
TICKS_PER_BAR = 1920


def note(pitch, tick, channel=1):
    return NoteEvent(pitch=pitch, velocity=80, start_tick=tick,
                     duration_ticks=TICKS_PER_BEAT - 10, channel=channel)


D_MINOR_SCALE = frozenset({0, 2, 4, 5, 7, 9, 10})


def test_parse_dm():
    assert parse_chord("Dm") == frozenset({2, 5, 9})   # D F A


def test_parse_g7():
    # G7: G(7) B(11) D(2) F(5)
    assert parse_chord("G7") == frozenset({7, 11, 2, 5})


def test_parse_cmaj7():
    # Cmaj7: C(0) E(4) G(7) B(11)
    assert parse_chord("Cmaj7") == frozenset({0, 4, 7, 11})


def test_parse_invalid():
    with pytest.raises(ValueError, match="Cannot parse chord"):
        parse_chord("Xyz99")


def test_strong_beat_1():
    assert is_strong_beat(0, TICKS_PER_BEAT) is True         # beat 1


def test_strong_beat_3():
    assert is_strong_beat(2 * TICKS_PER_BEAT, TICKS_PER_BEAT) is True  # beat 3


def test_weak_beat_2():
    assert is_strong_beat(TICKS_PER_BEAT, TICKS_PER_BEAT) is False


def test_chord_at_tick_bar_0():
    tones = chord_at_tick(0, TICKS_PER_BAR, ["Dm", "Gm", "A7", "Dm"])
    assert tones == parse_chord("Dm")


def test_chord_at_tick_bar_1():
    tones = chord_at_tick(TICKS_PER_BAR, TICKS_PER_BAR, ["Dm", "Gm", "A7", "Dm"])
    assert tones == parse_chord("Gm")


def test_chord_at_tick_wraps():
    tones = chord_at_tick(4 * TICKS_PER_BAR, TICKS_PER_BAR, ["Dm", "Gm"])
    assert tones == parse_chord("Dm")


def test_filter_chord_tone_on_beat1_unchanged():
    # D4=62 is in Dm (pitch class 2 = D)
    n = note(62, tick=0)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 62


def test_filter_off_chord_on_beat1_moves():
    # E4=64 is NOT in Dm; on beat 1 it should move to nearest Dm tone
    n = note(64, tick=0)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch % 12 in parse_chord("Dm")


def test_filter_off_chord_on_beat2_unchanged():
    # E4=64 not in Dm, but beat 2 -> leave it alone
    n = note(64, tick=TICKS_PER_BEAT)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 64


def test_filter_drums_always_unchanged():
    n = note(36, tick=0, channel=10)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 36
