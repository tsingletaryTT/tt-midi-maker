"""
Tests for tt_midi_maker.stream_player.

FluidSynth subprocess and ALSA ports are mocked so no audio hardware is needed.
Timing behaviour is tested with a real LoopPlayer against a mock mido port.
"""
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tt_midi_maker.models.track import NoteEvent, RoleTrack
from tt_midi_maker.assembler import build_midi_file
from tt_midi_maker.errors import MidiMakerError


@pytest.fixture
def midi_file(tmp_path):
    notes = [
        NoteEvent(pitch=60, velocity=80, start_tick=0,    duration_ticks=480, channel=1),
        NoteEvent(pitch=62, velocity=75, start_tick=960,  duration_ticks=480, channel=1),
        NoteEvent(pitch=64, velocity=70, start_tick=1920, duration_ticks=480, channel=1),
    ]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], 120, tmp_path / "test_loop.mid")


# ── _events_from_file ─────────────────────────────────────────────────────────

def test_events_from_file_loads_notes(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, programs, loop_ticks, bpm, tpb = _events_from_file(midi_file)
    on_events = [e for e in events if e[1] == "on"]
    assert len(on_events) == 3
    assert bpm == pytest.approx(120.0, rel=0.01)
    assert tpb == 480
    assert loop_ticks > 0


def test_events_loop_ticks_is_bar_aligned(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    _, _, loop_ticks, _, tpb = _events_from_file(midi_file)
    bar_ticks = tpb * 4
    assert loop_ticks % bar_ticks == 0


def test_events_offs_before_ons_at_same_tick(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, _, _, _, _ = _events_from_file(midi_file)
    by_tick: dict[int, list[str]] = {}
    for tick, etype, *_ in events:
        by_tick.setdefault(tick, []).append(etype)
    for tick, types in by_tick.items():
        # at any tick, offs must come before ons
        last_off = max((i for i, t in enumerate(types) if t == "off"), default=-1)
        first_on = min((i for i, t in enumerate(types) if t == "on"), default=999)
        assert last_off < first_on or last_off == -1, f"on before off at tick {tick}"


def test_events_from_missing_file():
    from tt_midi_maker.stream_player import _events_from_file
    with pytest.raises((FileNotFoundError, Exception)):
        _events_from_file(Path("/nonexistent/file.mid"))


# ── FluidSynthServer ──────────────────────────────────────────────────────────

def test_server_start_finds_fluid_port():
    from tt_midi_maker.stream_player import FluidSynthServer, DEFAULT_SOUNDFONT
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("mido.get_output_names", return_value=["FLUID Synth (9999):0"]), \
         patch("mido.set_backend"):
        server = FluidSynthServer()
        port = server.start()
    assert "FLUID" in port
    assert server.is_running


def test_server_start_no_fluidsynth_raises():
    from tt_midi_maker.stream_player import FluidSynthServer
    with patch("shutil.which", return_value=None):
        server = FluidSynthServer()
        with pytest.raises(MidiMakerError) as exc:
            server.start()
    assert exc.value.code == "FLUIDSYNTH_NOT_FOUND"


def test_server_start_missing_soundfont_raises(tmp_path):
    from tt_midi_maker.stream_player import FluidSynthServer
    with patch("shutil.which", return_value="/usr/bin/fluidsynth"):
        server = FluidSynthServer(soundfont=tmp_path / "no.sf2")
        with pytest.raises(MidiMakerError) as exc:
            server.start()
    assert exc.value.code == "SOUNDFONT_NOT_FOUND"


def test_server_stop():
    from tt_midi_maker.stream_player import FluidSynthServer
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("mido.get_output_names", return_value=["FLUID Synth (9999):0"]), \
         patch("mido.set_backend"):
        server = FluidSynthServer()
        server.start()
        server.stop()
    mock_proc.terminate.assert_called_once()
    assert server.port_name is None


# ── LoopPlayer ────────────────────────────────────────────────────────────────

def _make_player(sent_msgs):
    """Create a LoopPlayer with a mock mido output port."""
    from tt_midi_maker.stream_player import LoopPlayer
    mock_port = MagicMock()
    mock_port.send = lambda m: sent_msgs.append(m)
    with patch("mido.open_output", return_value=mock_port), \
         patch("mido.set_backend"):
        return LoopPlayer("FakePort:0")


def test_player_sends_notes_on_play(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, programs, loop_ticks, bpm, tpb = _events_from_file(midi_file)
    sent = []
    player = _make_player(sent)
    player.play(events, programs, loop_ticks, bpm=240.0, tpb=tpb, file_path=str(midi_file))
    time.sleep(0.8)   # 240 BPM: one bar ≈ 1s; wait for at least some notes
    player.stop(immediately=True)
    note_ons = [m for m in sent if hasattr(m, "type") and m.type == "note_on"]
    assert len(note_ons) > 0


def test_player_loops(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, programs, loop_ticks, bpm, tpb = _events_from_file(midi_file)
    sent = []
    player = _make_player(sent)
    player.play(events, programs, loop_ticks, bpm=960.0, tpb=tpb, file_path=str(midi_file))
    time.sleep(1.3)   # 960 BPM: loop ~0.5s; 1.3s comfortably covers 2 complete loops
    player.stop(immediately=True)
    assert player._loop_count >= 2, "expected at least 2 loop iterations"


def test_player_queue_transitions(midi_file, tmp_path):
    from tt_midi_maker.stream_player import _events_from_file
    from tt_midi_maker.models.track import NoteEvent, RoleTrack
    from tt_midi_maker.assembler import build_midi_file

    # Second pattern with different notes
    notes2 = [NoteEvent(pitch=72, velocity=80, start_tick=0, duration_ticks=480, channel=1)]
    track2 = RoleTrack(role="melody", channel=1, program=0, notes=notes2)
    midi2 = build_midi_file([track2], 120, tmp_path / "next.mid")

    events1, prog1, lt1, bpm1, tpb1 = _events_from_file(midi_file)
    events2, prog2, lt2, bpm2, tpb2 = _events_from_file(midi2)

    sent = []
    player = _make_player(sent)
    player.play(events1, prog1, lt1, bpm=480.0, tpb=tpb1, file_path=str(midi_file))
    time.sleep(0.3)
    player.queue_next(events2, prog2, lt2, bpm=480.0, tpb=tpb2, file_path=str(midi2))
    time.sleep(1.5)   # let loop boundary pass + one more loop
    player.stop(immediately=True)

    assert player._current_file == str(midi2), "pattern should have transitioned"


def test_player_stop_graceful(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, programs, loop_ticks, bpm, tpb = _events_from_file(midi_file)
    sent = []
    player = _make_player(sent)
    player.play(events, programs, loop_ticks, bpm=240.0, tpb=tpb, file_path=str(midi_file))
    time.sleep(0.1)
    player.stop(immediately=False)
    time.sleep(2.0)   # wait for loop to finish
    assert player.state == "stopped"


def test_player_all_notes_off_on_stop(midi_file):
    from tt_midi_maker.stream_player import _events_from_file
    events, programs, loop_ticks, bpm, tpb = _events_from_file(midi_file)
    sent = []
    player = _make_player(sent)
    player.play(events, programs, loop_ticks, bpm=240.0, tpb=tpb, file_path=str(midi_file))
    time.sleep(0.1)
    player.stop(immediately=True)
    time.sleep(0.1)
    cc123 = [m for m in sent if hasattr(m, "control") and m.control == 123]
    assert len(cc123) > 0, "all-notes-off CC#123 should be sent on stop"


# ── module-level API ──────────────────────────────────────────────────────────

def test_start_synth_returns_status():
    from tt_midi_maker import stream_player
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_port = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("mido.get_output_names", return_value=["FLUID Synth (1):0"]), \
         patch("mido.set_backend"), \
         patch("mido.open_output", return_value=mock_port):
        result = stream_player.start_synth()
    assert result["status"] == "started"
    assert "FLUID" in result["port"]
    stream_player._server.stop()
    if stream_player._player:
        stream_player._player.close()


def test_loop_play_requires_synth(midi_file):
    from tt_midi_maker import stream_player
    original_server = stream_player._server
    original_player = stream_player._player
    stream_player._server = None
    stream_player._player = None
    try:
        with pytest.raises(MidiMakerError) as exc:
            stream_player.loop_play(str(midi_file))
        assert exc.value.code == "SYNTH_NOT_STARTED"
    finally:
        stream_player._server = original_server
        stream_player._player = original_player


def test_synth_status_not_started():
    from tt_midi_maker import stream_player
    original = stream_player._server
    stream_player._server = None
    try:
        status = stream_player.synth_status()
        assert status["server"]["running"] is False
    finally:
        stream_player._server = original


# ── server tool wrappers ──────────────────────────────────────────────────────

def test_server_tool_synth_start():
    from tt_midi_maker.server import synth_start
    from tt_midi_maker import stream_player
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_port = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("mido.get_output_names", return_value=["FLUID Synth (2):0"]), \
         patch("mido.set_backend"), \
         patch("mido.open_output", return_value=mock_port):
        result = synth_start()
    assert result["status"] == "started"
    stream_player._server.stop()
    if stream_player._player:
        stream_player._player.close()
