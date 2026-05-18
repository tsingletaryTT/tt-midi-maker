from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing
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
