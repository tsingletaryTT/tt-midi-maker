"""
Tests for tt_midi_maker.player — playback dispatch and job management.
FluidSynth subprocess and ALSA ports are mocked so no audio hardware is needed.
"""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tt_midi_maker.models.track import NoteEvent, RoleTrack
from tt_midi_maker.assembler import build_midi_file
from tt_midi_maker.errors import MidiMakerError


@pytest.fixture
def midi_file(tmp_path):
    notes = [NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=480, channel=1)]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], 120, tmp_path / "test.mid")


# ── list_output_ports ────────────────────────────────────────────────────────

def test_list_output_ports_returns_list():
    from tt_midi_maker.player import list_output_ports
    result = list_output_ports()
    assert isinstance(result, list)


def test_list_soundfonts_returns_list():
    from tt_midi_maker.player import list_soundfonts
    result = list_soundfonts()
    assert isinstance(result, list)
    # FluidR3_GM.sf2 is installed on this machine
    assert any("FluidR3_GM" in s for s in result)


# ── fluidsynth backend ───────────────────────────────────────────────────────

def test_play_fluidsynth_returns_job_id(midi_file):
    from tt_midi_maker import player
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=mock_proc):
        result = player.play(str(midi_file), backend="fluidsynth")
    assert "job_id" in result
    assert result["backend"] == "fluidsynth"


def test_play_fluidsynth_missing_file():
    from tt_midi_maker import player
    with pytest.raises(MidiMakerError) as exc:
        player.play("/nonexistent/file.mid", backend="fluidsynth")
    assert exc.value.code == "FILE_NOT_FOUND"


def test_play_fluidsynth_missing_binary(midi_file):
    from tt_midi_maker import player
    with patch("shutil.which", return_value=None):
        with pytest.raises(MidiMakerError) as exc:
            player.play(str(midi_file), backend="fluidsynth")
    assert exc.value.code == "FLUIDSYNTH_NOT_FOUND"


def test_play_fluidsynth_missing_soundfont(midi_file):
    from tt_midi_maker import player
    with pytest.raises(MidiMakerError) as exc:
        player.play(str(midi_file), backend="fluidsynth", soundfont="/no/such.sf2")
    assert exc.value.code == "SOUNDFONT_NOT_FOUND"


def test_play_fluidsynth_blocking_waits(midi_file):
    from tt_midi_maker import player
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = 0
    with patch("subprocess.Popen", return_value=mock_proc):
        result = player.play(str(midi_file), backend="fluidsynth", blocking=True)
    assert result["status"] in ("done", "playing", "starting")


# ── alsa backend ─────────────────────────────────────────────────────────────

def test_play_alsa_no_ports_raises(midi_file):
    from tt_midi_maker import player
    with patch.object(player, "list_output_ports", return_value=[]):
        with pytest.raises(MidiMakerError) as exc:
            player.play(str(midi_file), backend="alsa")
    assert exc.value.code == "NO_MIDI_PORTS"


def test_play_alsa_with_explicit_port(midi_file):
    from tt_midi_maker import player
    import mido

    sent = []
    mock_port = MagicMock()
    mock_port.__enter__ = lambda s: s
    mock_port.__exit__ = MagicMock(return_value=False)
    mock_port.send = lambda m: sent.append(m)

    with patch("mido.open_output", return_value=mock_port), \
         patch.object(player, "_RTMIDI_BACKEND", "mido.backends.rtmidi"), \
         patch("mido.set_backend"):
        result = player.play(str(midi_file), backend="alsa", port="FakePort:0", blocking=True)
    assert "job_id" in result


def test_play_alsa_unknown_backend_raises(midi_file):
    from tt_midi_maker import player
    with pytest.raises(MidiMakerError) as exc:
        player.play(str(midi_file), backend="osc")  # type: ignore
    assert exc.value.code == "UNKNOWN_BACKEND"


# ── stop ─────────────────────────────────────────────────────────────────────

def test_stop_nonexistent_job_raises():
    from tt_midi_maker import player
    with pytest.raises(MidiMakerError) as exc:
        player.stop("nonexistent-job")
    assert exc.value.code == "JOB_NOT_FOUND"


def test_stop_already_done_job_returns_status(midi_file):
    from tt_midi_maker import player
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = 0
    with patch("subprocess.Popen", return_value=mock_proc):
        result = player.play(str(midi_file), backend="fluidsynth", blocking=True)

    job_id = result["job_id"]
    # wait briefly for thread to mark done
    time.sleep(0.1)
    stop_result = player.stop(job_id)
    assert stop_result["job_id"] == job_id


# ── job_status ───────────────────────────────────────────────────────────────

def test_job_status_returns_fields(midi_file):
    from tt_midi_maker import player
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=mock_proc):
        result = player.play(str(midi_file), backend="fluidsynth")
    status = player.job_status(result["job_id"])
    for field in ("job_id", "file", "backend", "status"):
        assert field in status


# ── server tools ─────────────────────────────────────────────────────────────

def test_server_list_midi_devices():
    from tt_midi_maker.server import list_midi_devices
    result = list_midi_devices()
    assert "alsa_ports" in result
    assert "soundfonts" in result
    assert "fluidsynth_available" in result
    assert "active_jobs" in result
    assert result["fluidsynth_available"] is True  # we just installed it


def test_server_play_midi_returns_job(midi_file):
    from tt_midi_maker import server
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = None
    with patch("subprocess.Popen", return_value=mock_proc):
        result = server.play_midi(str(midi_file), backend="fluidsynth")
    assert "job_id" in result


def test_server_stop_playback(midi_file):
    from tt_midi_maker import server
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.poll.return_value = None
    mock_proc.terminate = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc):
        play_result = server.play_midi(str(midi_file), backend="fluidsynth")
    stop_result = server.stop_playback(play_result["job_id"])
    assert stop_result["job_id"] == play_result["job_id"]
