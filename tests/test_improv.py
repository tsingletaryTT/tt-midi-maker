"""Tests for the improv layer."""
from __future__ import annotations

import pytest

from tt_midi_maker.coherence.improv import (
    add_approach_notes,
    build_tension_curve,
    evolve_phrase,
    vary_rhythm,
)
from tt_midi_maker.models.track import NoteEvent

_TPB   = 480
_TPBAR = _TPB * 4


def _note(pitch, tick, dur=240, vel=80, ch=1):
    return NoteEvent(pitch=pitch, velocity=vel, start_tick=tick, duration_ticks=dur, channel=ch)


# ── build_tension_curve ───────────────────────────────────────────────────────

def test_tension_curve_endpoints():
    curve = build_tension_curve(0.0, 1.0, 3)
    assert curve(0) == pytest.approx(0.0)
    assert curve(2) == pytest.approx(1.0)


def test_tension_curve_midpoint():
    curve = build_tension_curve(0.0, 1.0, 3)
    assert curve(1) == pytest.approx(0.5)


def test_tension_curve_single_bar():
    """When bars=1, curve always returns start regardless of index."""
    curve = build_tension_curve(0.5, 0.9, 1)
    assert curve(0) == pytest.approx(0.5)


def test_tension_curve_clamped():
    """Values should not exceed 0.0–1.0."""
    curve = build_tension_curve(0.0, 2.0, 3)
    assert curve(2) == pytest.approx(1.0)   # clamped at 1.0


# ── add_approach_notes ────────────────────────────────────────────────────────

def test_approach_notes_zero_tension_no_insertion():
    notes = [_note(60, _TPBAR)]   # C4 on bar 2 beat 1
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=0.0)
    assert len(result) == len(notes)


def test_approach_notes_inserts_half_step_below():
    """At tension=1.0, every eligible chord-tone downbeat gets a B3 approach."""
    notes = [_note(60, _TPBAR)]   # C4 on bar 2 beat 1; C4 is in Cmaj7
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0, seed=42)
    pitches = sorted(set(n.pitch for n in result))
    assert 59 in pitches   # B3 = half-step below C4


def test_approach_notes_skips_drums():
    """Channel 10 notes (drums) should never get approach notes."""
    notes = [_note(36, _TPBAR, ch=10)]   # kick on bar 2 beat 1
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0)
    assert len(result) == 1


def test_approach_notes_skips_non_downbeats():
    """Notes not on beat 1 of a bar should not get approach notes."""
    notes = [_note(64, _TPBAR + _TPB)]   # E4 on beat 2 of bar 2 (not beat 1)
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0)
    assert len(result) == 1


def test_approach_notes_skips_non_chord_tones():
    """Notes that are not chord tones should not get approach notes."""
    notes = [_note(61, _TPBAR)]   # C#4 on bar 2 beat 1; C# is NOT in Cmaj7
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0)
    assert len(result) == 1


def test_approach_notes_does_not_go_negative():
    """Notes at tick=0 (bar 1 beat 1) cannot have approach notes — no negative ticks."""
    notes = [_note(60, 0)]   # C4 at the very start
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0)
    # approach would require tick=-240 which is invalid — should not be inserted
    assert all(n.start_tick >= 0 for n in result)


def test_approach_note_velocity_lower():
    """Approach notes should have lower velocity than the target."""
    notes = [_note(60, _TPBAR, vel=80)]
    result = add_approach_notes(notes, ["Cmaj7"], _TPBAR, tension=1.0, seed=42)
    approach_notes = [n for n in result if n.pitch == 59]
    if approach_notes:
        assert approach_notes[0].velocity < 80


# ── vary_rhythm ───────────────────────────────────────────────────────────────

def test_vary_rhythm_zero_tension_unchanged():
    notes = [_note(60, i * _TPB) for i in range(4)]
    result = vary_rhythm(notes, tension=0.0)
    assert [n.start_tick for n in result] == [n.start_tick for n in notes]


def test_vary_rhythm_preserves_count():
    notes = [_note(60, i * _TPB) for i in range(8)]
    result = vary_rhythm(notes, tension=1.0, seed=77)
    assert len(result) == len(notes)


def test_vary_rhythm_no_negative_ticks():
    notes = [_note(60, 0)]   # starts at tick 0
    result = vary_rhythm(notes, tension=1.0, seed=77)
    for n in result:
        assert n.start_tick >= 0


def test_vary_rhythm_skips_drums():
    """Drums (channel 10) should never be shifted."""
    notes = [_note(36, 0, ch=10)]
    result = vary_rhythm(notes, tension=1.0, seed=1)
    assert result[0].start_tick == 0   # unchanged


# ── evolve_phrase ─────────────────────────────────────────────────────────────

def test_evolve_phrase_transposes_melodic():
    notes = [_note(60, 0)]
    result = evolve_phrase(notes, semitone_shift=2)
    assert result[0].pitch == 62   # C4 → D4


def test_evolve_phrase_skips_drums():
    notes = [_note(36, 0, ch=10)]
    result = evolve_phrase(notes, semitone_shift=4)
    assert result[0].pitch == 36   # drums not transposed


def test_evolve_phrase_stays_in_midi_range():
    notes = [_note(108, 0)]   # top of MIDI range
    result = evolve_phrase(notes, semitone_shift=5)
    assert result[0].pitch == 108   # clamped — 108+5=113 > 108, reverts to original


def test_evolve_phrase_negative_shift():
    notes = [_note(60, 0)]
    result = evolve_phrase(notes, semitone_shift=-3)
    assert result[0].pitch == 57   # C4 → A3


def test_evolve_phrase_duration_varies_slightly():
    """Duration should change by up to 12% of original."""
    notes = [_note(60, 0, dur=480)]
    result = evolve_phrase(notes, semitone_shift=0, seed=99)
    dur_diff = abs(result[0].duration_ticks - 480)
    assert dur_diff <= 480 * 0.12 + 1   # ±12% tolerance
