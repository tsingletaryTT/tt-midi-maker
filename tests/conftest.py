import pytest
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack

TICKS_PER_BEAT = 480
TICKS_PER_BAR = 1920  # 4/4 at 480 ticks/beat


@pytest.fixture
def sample_blueprint():
    return MusicalBlueprint(
        key="D minor", bpm=120, time_signature="4/4",
        style="bossa nova", chord_progression=["Dm", "Gm", "A7", "Dm"],
        bars=8, mode="loop",
        roles={
            "drums":   RoleConfig(density=0.7, velocity_range=(60, 90),  pattern_hint="bossa"),
            "bass":    RoleConfig(density=0.8, velocity_range=(70, 100), pattern_hint="walking"),
            "melody":  RoleConfig(density=1.0, velocity_range=(80, 110), pattern_hint="legato"),
            "harmony": RoleConfig(density=0.0, velocity_range=(50, 80),  pattern_hint="default"),
        },
    )


@pytest.fixture
def chromatic_notes_c4():
    """12 notes C4-B4 (60-71), alternating channels 1 and 10."""
    return [
        NoteEvent(pitch=60+i, velocity=80, start_tick=i*TICKS_PER_BEAT,
                  duration_ticks=TICKS_PER_BEAT - 10,
                  channel=10 if i % 4 == 0 else 1)
        for i in range(12)
    ]


def make_role_track(role: str, channel: int, program: int = 0,
                    n_notes: int = 4, start_offset: int = 0) -> RoleTrack:
    notes = [
        NoteEvent(pitch=60, velocity=80,
                  start_tick=start_offset + i * TICKS_PER_BEAT,
                  duration_ticks=TICKS_PER_BEAT - 10,
                  channel=channel)
        for i in range(n_notes)
    ]
    return RoleTrack(role=role, channel=channel, program=program, notes=notes)
