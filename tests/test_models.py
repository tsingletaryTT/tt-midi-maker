import pytest
from pydantic import ValidationError
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_blueprint(**overrides):
    base = dict(
        key="D minor", bpm=120, time_signature="4/4",
        style="bossa nova", chord_progression=["Dm", "Gm", "A7", "Dm"],
        bars=8, mode="loop",
        roles={"drums": RoleConfig(density=0.7, velocity_range=(60, 90), pattern_hint="bossa")},
    )
    base.update(overrides)
    return MusicalBlueprint(**base)


def test_valid_blueprint():
    b = make_blueprint()
    assert b.key == "D minor"
    assert b.bpm == 120
    assert b.mode == "loop"


def test_bpm_too_low():
    with pytest.raises(ValidationError):
        make_blueprint(bpm=10)


def test_bpm_too_high():
    with pytest.raises(ValidationError):
        make_blueprint(bpm=400)


def test_invalid_mode():
    with pytest.raises(ValidationError):
        make_blueprint(mode="jam")


def test_role_density_out_of_range():
    with pytest.raises(ValidationError):
        RoleConfig(density=1.5)


def test_note_event_fields():
    n = NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=480, channel=1)
    assert n.pitch == 60
    assert n.channel == 1


def test_role_track_default_empty_notes():
    t = RoleTrack(role="bass", channel=2, program=32)
    assert t.notes == []
