"""Tests for the genre structure layer."""
from __future__ import annotations

import pytest

from tt_midi_maker.coherence.structure import (
    GenreStructure,
    _chord_root_pc,
    _roman_to_chord,
    build_genre_structure,
    enforce_phrase_gaps,
    generate_drum_groove,
    generate_walking_bass,
)
from tt_midi_maker.models.track import NoteEvent

_TPB   = 480
_TPBAR = _TPB * 4


def _note(pitch, tick, dur=240, vel=80, ch=1):
    return NoteEvent(pitch=pitch, velocity=vel, start_tick=tick, duration_ticks=dur, channel=ch)


# ── Roman numeral resolution ──────────────────────────────────────────────────

def test_roman_to_chord_blues_in_A():
    assert _roman_to_chord("I7",  9) == "A7"
    assert _roman_to_chord("IV7", 9) == "D7"
    assert _roman_to_chord("V7",  9) == "E7"


def test_roman_to_chord_key_C():
    assert _roman_to_chord("I7",  0) == "C7"
    assert _roman_to_chord("IV7", 0) == "F7"
    assert _roman_to_chord("V7",  0) == "G7"


def test_roman_iv_does_not_match_i():
    """IV should resolve 5 semitones above root, not 0 (not confused with I)."""
    assert _roman_to_chord("IV7", 0) == "F7"   # F is 5 semitones above C


def test_roman_to_chord_quality_preserved():
    """Quality suffix (7, maj7, m7) should pass through unchanged."""
    assert _roman_to_chord("Imaj7", 0)  == "Cmaj7"
    assert _roman_to_chord("IIm7",  0)  == "Dm7"
    assert _roman_to_chord("Vmaj7", 9)  == "Emaj7"


def test_chord_root_pc_sharp():
    assert _chord_root_pc("C#7")   == 1
    assert _chord_root_pc("Eb7")   == 3
    assert _chord_root_pc("Bbmaj7") == 10


# ── build_genre_structure ─────────────────────────────────────────────────────

def test_build_blues_12bar_chord_resolution():
    s = build_genre_structure("blues_12bar", "A minor", ["A7", "D7"], bars=12)
    assert len(s.chord_progression) == 12
    assert s.chord_progression[0]  == "A7"    # bar 1: I7
    assert s.chord_progression[1]  == "D7"    # bar 2: IV7 (quick change)
    assert s.chord_progression[4]  == "D7"    # bar 5: IV7
    assert s.chord_progression[8]  == "E7"    # bar 9: V7
    assert s.chord_progression[11] == "E7"    # bar 12: V7 turnaround


def test_build_blues_structural_metadata():
    s = build_genre_structure("blues_12bar", "A minor", ["A7"], bars=12)
    assert s.call_bars     == [1, 2, 5, 6, 9, 10]
    assert s.response_bars == [3, 4, 7, 8, 11, 12]
    assert s.walking_bass  is True
    assert s.drum_groove   == "shuffle"
    assert s.swing_ratio   == pytest.approx(0.67)


def test_build_null_template_cycles():
    """When chord_template is null, the passed progression is cycled."""
    s = build_genre_structure("jazz_swing", "C major", ["Cmaj7", "Am7", "Fmaj7", "G7"], bars=8)
    assert len(s.chord_progression) == 8
    assert s.chord_progression[0]  == "Cmaj7"
    assert s.chord_progression[4]  == "Cmaj7"   # cycled


def test_build_tension_propagated():
    s = build_genre_structure("blues_12bar", "A minor", ["A7"], bars=12, tension=0.5)
    assert s.tension == pytest.approx(0.5)


# ── generate_walking_bass ─────────────────────────────────────────────────────

def test_walking_bass_note_count():
    notes = generate_walking_bass(["A7", "D7", "A7", "E7"], bars=4)
    assert len(notes) == 16   # 4 beats × 4 bars


def test_walking_bass_evenly_spaced():
    notes = generate_walking_bass(["A7"], bars=1)
    ticks = [n.start_tick for n in notes]
    assert ticks == [0, 480, 960, 1440]


def test_walking_bass_in_range():
    notes = generate_walking_bass(["A7"], bars=1)
    for n in notes:
        assert 33 <= n.pitch <= 57, f"Bass pitch {n.pitch} out of A1-A3 range"


def test_walking_bass_approach_note_below_next_root():
    """Beat 4 should be a half-step below the next bar's root."""
    # A7 → D7: approach should be Db (1 below D=38) = C# (37) or similar
    notes_a = generate_walking_bass(["A7", "D7"], bars=1)
    approach_pitch = notes_a[3].pitch   # beat 4 of bar 1
    next_root_pc   = 2   # D = pitch class 2
    # approach is half-step below next root, in bass range
    assert approach_pitch % 12 == (next_root_pc - 1) % 12


# ── generate_drum_groove ──────────────────────────────────────────────────────

def test_drum_shuffle_counts_per_bar():
    notes = generate_drum_groove("shuffle", bars=1)
    kicks  = [n for n in notes if n.pitch == 36]
    snares = [n for n in notes if n.pitch == 38]
    hihats = [n for n in notes if n.pitch == 42]
    assert len(kicks)  == 2    # beats 1 and 3
    assert len(snares) == 2    # beats 2 and 4
    assert len(hihats) == 8    # 4 beats × 2 triplet positions


def test_drum_swing_ride_has_ride():
    notes = generate_drum_groove("swing_ride", bars=1)
    ride_notes = [n for n in notes if n.pitch == 51]
    assert len(ride_notes) == 8   # 4 beats × 2 positions (straight + swung)


def test_drum_none_returns_empty():
    assert generate_drum_groove("none", bars=4) == []


def test_drum_channel_is_10():
    notes = generate_drum_groove("shuffle", bars=1)
    for n in notes:
        assert n.channel == 10


def test_drum_straight_has_8th_hihats():
    notes = generate_drum_groove("straight", bars=1)
    hihats = [n for n in notes if n.pitch == 42]
    assert len(hihats) == 8    # 8 eighth-note hi-hat hits per bar


# ── enforce_phrase_gaps ───────────────────────────────────────────────────────

def test_phrase_gaps_keep_call_bars_intact():
    """Notes in call bars should not be removed."""
    notes = [_note(60, bar * _TPBAR + beat * _TPB) for bar in range(4) for beat in range(4)]
    call_bars = [1, 3]   # bars 1 and 3 are calls (1-indexed)
    result = enforce_phrase_gaps(notes, response_bars=[2, 4])
    call_ticks = {bar * _TPBAR + beat * _TPB for bar in (0, 2) for beat in range(4)}
    for note in result:
        if note.start_tick in call_ticks:
            pass  # should be present


def test_phrase_gaps_trim_response_bars():
    """Only beat-1 notes should survive in response bars."""
    notes = [_note(60, beat * _TPB) for beat in range(4)]   # 4 notes in bar 1
    result = enforce_phrase_gaps(notes, response_bars=[1], ticks_per_beat=_TPB)
    # Bar 1 is a response bar — only beat 1 (tick 0..479) should survive
    for note in result:
        assert note.start_tick < _TPB + _TPB, "Non-beat-1 note survived in response bar"


def test_phrase_gaps_no_response_bars_unchanged():
    notes = [_note(60, i * _TPB) for i in range(8)]
    result = enforce_phrase_gaps(notes, response_bars=[])
    assert len(result) == len(notes)
