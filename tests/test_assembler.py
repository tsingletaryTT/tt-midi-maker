import mido
import tempfile
from pathlib import Path
from tt_midi_maker.assembler import build_midi_file, bpm_to_tempo, TICKS_PER_BEAT
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_track(role, channel, program=0, n=4):
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=channel)
        for i in range(n)
    ]
    return RoleTrack(role=role, channel=channel, program=program, notes=notes)


def test_bpm_to_tempo_120():
    assert bpm_to_tempo(120) == 500_000


def test_bpm_to_tempo_60():
    assert bpm_to_tempo(60) == 1_000_000


def test_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        result = build_midi_file([make_track("melody", 1)], bpm=120, output_path=path)
        assert result == path
        assert path.exists()


def test_output_is_type_1():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1), make_track("bass", 2)], 120, path)
        assert mido.MidiFile(str(path)).type == 1


def test_track_count_includes_tempo_track():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1), make_track("bass", 2)], 120, path)
        mid = mido.MidiFile(str(path))
        assert len(mid.tracks) == 3   # tempo + melody + bass


def test_tempo_is_set_correctly():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1)], bpm=90, output_path=path)
        mid = mido.MidiFile(str(path))
        tempos = [m for m in mid.tracks[0] if m.type == "set_tempo"]
        assert tempos[0].tempo == bpm_to_tempo(90)


def test_drums_get_no_program_change():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("drums", 10)], 120, path)
        mid = mido.MidiFile(str(path))
        drum_track = mid.tracks[1]
        assert not any(m.type == "program_change" for m in drum_track)


def test_melody_gets_program_change():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1, program=0)], 120, path)
        mid = mido.MidiFile(str(path))
        melody_track = mid.tracks[1]
        assert any(m.type == "program_change" for m in melody_track)


def test_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nested" / "dir" / "test.mid"
        build_midi_file([make_track("melody", 1)], 120, path)
        assert path.exists()
