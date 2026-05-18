from tt_midi_maker.coherence.stitching import stitch_phrases
from tt_midi_maker.models.track import NoteEvent, RoleTrack

TICKS_PER_BAR = 1920


def make_track(role, channel, n=4, start=0):
    notes = [
        NoteEvent(pitch=60, velocity=80,
                  start_tick=start + i * 480, duration_ticks=470, channel=channel)
        for i in range(n)
    ]
    return RoleTrack(role=role, channel=channel, program=0, notes=notes)


def test_stitch_doubles_note_count():
    existing = [make_track("melody", 1, n=4)]
    new      = [make_track("melody", 1, n=4)]
    result = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody = next(t for t in result if t.role == "melody")
    assert len(melody.notes) == 8


def test_new_notes_start_after_existing():
    existing = [make_track("melody", 1, n=4)]   # last note ends around tick 4*480
    new      = [make_track("melody", 1, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody   = next(t for t in result if t.role == "melody")
    existing_max = max(n.start_tick for n in existing[0].notes)
    new_ticks    = [n.start_tick for n in melody.notes[4:]]
    assert all(t > existing_max for t in new_ticks)


def test_missing_role_in_new_preserved():
    existing = [make_track("melody", 1), make_track("bass", 2)]
    new      = [make_track("melody", 1)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    roles    = {t.role for t in result}
    assert "bass" in roles and "melody" in roles


def test_missing_role_in_existing_appended():
    existing = [make_track("melody", 1)]
    new      = [make_track("melody", 1), make_track("bass", 2)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    roles    = {t.role for t in result}
    assert "bass" in roles


def test_crossfade_reduces_last_bar_velocity():
    existing = [make_track("melody", 1, n=8)]   # 8 notes span 2 bars
    new      = [make_track("melody", 1, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody   = next(t for t in result if t.role == "melody")
    last_bar_start = 7 * 480   # note 8 (index 7)
    last_bar_notes = [n for n in melody.notes[:8] if n.start_tick >= last_bar_start]
    assert all(n.velocity <= 80 for n in last_bar_notes)   # faded from 80


def test_drums_not_crossfaded():
    existing = [make_track("drums", 10, n=8)]
    new      = [make_track("drums", 10, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    drums    = next(t for t in result if t.role == "drums")
    # Original drum velocities should be untouched
    assert all(n.velocity == 80 for n in drums.notes[:8])
