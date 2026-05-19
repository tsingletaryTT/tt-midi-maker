"""Smoke test that demo coherence pipelines accept velocity shaping without error."""
from dataclasses import replace
from tt_midi_maker.models.track import NoteEvent, RoleTrack
from tt_midi_maker.coherence.humanize import scale_velocity_by_role, humanize_velocities


def _make_track(role: str, n_notes: int = 8, channel: int = 1) -> RoleTrack:
    notes = [
        NoteEvent(pitch=60 + i, velocity=64, start_tick=i * 480,
                  duration_ticks=240, channel=channel)
        for i in range(n_notes)
    ]
    return RoleTrack(role=role, channel=channel, program=0, notes=notes)


def test_scale_velocity_by_role_in_coherence_pipeline():
    """Velocity shaping followed by humanize stays in bounds for each role."""
    for role in ["melody", "harmony", "bass", "drums"]:
        track = _make_track(role, n_notes=16)
        notes = scale_velocity_by_role(track.notes, track.role)
        notes = humanize_velocities(notes, variation=8)
        for n in notes:
            assert 1 <= n.velocity <= 127, f"role={role}, velocity out of bounds: {n.velocity}"


def test_coherence_pipeline_with_empty_track():
    """Empty notes list passes through without error."""
    track = _make_track("melody", n_notes=0)
    notes = scale_velocity_by_role(track.notes, track.role)
    assert notes == []
