"""
Tests call tool handler functions directly (bypassing MCP protocol).
All external I/O (LLM, hardware, generation) is mocked.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack


STUB_BLUEPRINT = MusicalBlueprint(
    key="C major", bpm=120, time_signature="4/4", style="test",
    chord_progression=["C", "F", "G", "C"], bars=4, mode="loop",
    roles={"melody": RoleConfig(density=1.0), "drums": RoleConfig(density=0.7)},
)

STUB_TRACKS = [
    RoleTrack(role="melody", channel=1, program=0, notes=[
        NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=470, channel=1),
    ]),
    RoleTrack(role="drums", channel=10, program=0, notes=[
        NoteEvent(pitch=36, velocity=80, start_tick=0, duration_ticks=100, channel=10),
    ]),
]


def test_set_musical_context_returns_fields_set():
    from tt_midi_maker.server import _set_musical_context
    result = _set_musical_context(session_id="test1", key="D minor", bpm=90)
    assert result["key"] == "D minor"
    assert result["bpm"] == 90
    assert "key" in result["fields_set"]


def test_set_musical_context_null_clears_field():
    from tt_midi_maker.server import _set_musical_context
    _set_musical_context(session_id="test2", key="C major", bpm=120)
    result = _set_musical_context(session_id="test2", key=None)
    assert result["key"] is None
    assert result["bpm"] == 120


def test_generate_midi_returns_file_path(tmp_path):
    from tt_midi_maker import server
    with patch.object(server, "OUTPUT_DIR", tmp_path), \
         patch("tt_midi_maker.server.build_blueprint", return_value=STUB_BLUEPRINT), \
         patch("tt_midi_maker.server._run_generation", return_value=STUB_TRACKS):
        result = server._generate_midi(
            prompt="test prompt", mode="loop", session_id="test3"
        )
    assert "file_path" in result
    assert result["file_path"].endswith(".mid")
    assert Path(result["file_path"]).exists()


def test_generate_midi_output_has_correct_roles(tmp_path):
    from tt_midi_maker import server
    with patch.object(server, "OUTPUT_DIR", tmp_path), \
         patch("tt_midi_maker.server.build_blueprint", return_value=STUB_BLUEPRINT), \
         patch("tt_midi_maker.server._run_generation", return_value=STUB_TRACKS):
        result = server._generate_midi(prompt="test", mode="loop", session_id="t4")
    assert "melody" in result["roles_generated"]
    assert "drums" in result["roles_generated"]


def test_describe_midi_missing_file():
    from tt_midi_maker.server import _describe_midi
    from tt_midi_maker.errors import MidiMakerError
    with pytest.raises(MidiMakerError) as exc:
        _describe_midi("/nonexistent/path.mid")
    assert exc.value.code == "FILE_NOT_FOUND"


def test_chat_with_midi_returns_answer(tmp_path):
    from tt_midi_maker import server
    from tt_midi_maker.assembler import build_midi_file
    from tt_midi_maker.models.track import NoteEvent, RoleTrack
    notes = [NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=470, channel=1)]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    midi_path = build_midi_file([track], 120, tmp_path / "chat_test.mid")

    with patch("tt_midi_maker.analyzer.call_llm", return_value="It is in C major."):
        result = server._chat_with_midi(str(midi_path), "What key?")
    assert "C major" in result["answer"]
