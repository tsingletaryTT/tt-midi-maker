"""Tests for humanize.py velocity and timing functions."""
from tt_midi_maker.coherence.humanize import (
    scale_velocity_by_role,
    _ROLE_VELOCITY_RANGES,
    humanize_velocities,
    nudge_timing,
    swing_timing,
)
from tt_midi_maker.models.track import NoteEvent


def make_notes(n=16, vel=80, channel=1):
    return [
        NoteEvent(pitch=60, velocity=vel, start_tick=i * 480,
                  duration_ticks=470, channel=channel)
        for i in range(n)
    ]


def test_velocities_stay_in_valid_range():
    result = humanize_velocities(make_notes(vel=80), variation=20)
    assert all(1 <= n.velocity <= 127 for n in result)


def test_velocities_are_varied():
    result = humanize_velocities(make_notes(n=32, vel=80), variation=10)
    assert len(set(n.velocity for n in result)) > 1


def test_drums_get_smaller_variation():
    drum_notes = make_notes(n=32, vel=80, channel=10)
    result = humanize_velocities(drum_notes, variation=20)
    # Drum variation is capped at +-4
    assert all(76 <= n.velocity <= 84 for n in result)


def test_nudge_timing_non_negative():
    result = nudge_timing(make_notes(), max_ticks=10)
    assert all(n.start_tick >= 0 for n in result)


def test_drums_not_nudged():
    drum_notes = make_notes(channel=10)
    original = [n.start_tick for n in drum_notes]
    result = nudge_timing(drum_notes, max_ticks=10)
    assert [n.start_tick for n in result] == original


def test_nudge_changes_some_ticks():
    notes = make_notes(n=32)
    result = nudge_timing(notes, max_ticks=10)
    assert [n.start_tick for n in result] != [n.start_tick for n in notes]


def test_empty_input():
    assert humanize_velocities([]) == []
    assert nudge_timing([]) == []


# ---------------------------------------------------------------------------
# scale_velocity_by_role() tests
# ---------------------------------------------------------------------------

def _note(vel: int, channel: int = 1) -> NoteEvent:
    return NoteEvent(pitch=60, velocity=vel, start_tick=0, duration_ticks=240, channel=channel)


def test_melody_velocity_range():
    lo, hi = _ROLE_VELOCITY_RANGES["melody"]
    notes = [_note(1), _note(64), _note(127)]
    result = scale_velocity_by_role(notes, "melody")
    for n in result:
        assert lo <= n.velocity <= hi, f"melody note vel {n.velocity} outside [{lo},{hi}]"


def test_harmony_velocity_lower_than_melody():
    melody_lo, melody_hi = _ROLE_VELOCITY_RANGES["melody"]
    harmony_lo, harmony_hi = _ROLE_VELOCITY_RANGES["harmony"]
    assert harmony_hi <= melody_hi, "harmony ceiling should be at or below melody ceiling"
    assert harmony_lo <= melody_lo, "harmony floor should be at or below melody floor"


def test_bass_velocity_range():
    lo, hi = _ROLE_VELOCITY_RANGES["bass"]
    notes = [_note(1), _note(64), _note(127)]
    result = scale_velocity_by_role(notes, "bass")
    for n in result:
        assert lo <= n.velocity <= hi


def test_unknown_role_uses_default_range():
    notes = [_note(1), _note(127)]
    result = scale_velocity_by_role(notes, "unknown_role")
    assert len(result) == 2
    for n in result:
        assert 1 <= n.velocity <= 127


def test_empty_notes_returns_empty():
    assert scale_velocity_by_role([], "melody") == []


def test_velocity_monotone_preserving():
    """Higher input velocity → higher output velocity (relative ordering preserved)."""
    notes = [_note(20), _note(60), _note(100)]
    result = scale_velocity_by_role(notes, "harmony")
    vels = [n.velocity for n in result]
    assert vels == sorted(vels), f"ordering not preserved: {vels}"


def test_custom_ranges_override_defaults():
    custom = {"melody": (90, 90)}  # fixed velocity
    notes = [_note(1), _note(64), _note(127)]
    result = scale_velocity_by_role(notes, "melody", ranges=custom)
    for n in result:
        assert n.velocity == 90


def test_velocity_at_extremes():
    """vel=1 maps to lo, vel=127 maps to hi."""
    lo, hi = 60, 100
    custom = {"test": (lo, hi)}
    note_lo = scale_velocity_by_role([_note(1)], "test", ranges=custom)[0]
    note_hi = scale_velocity_by_role([_note(127)], "test", ranges=custom)[0]
    assert note_lo.velocity == lo, f"vel=1 should map to {lo}, got {note_lo.velocity}"
    assert note_hi.velocity == hi, f"vel=127 should map to {hi}, got {note_hi.velocity}"
