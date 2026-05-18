import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from tt_midi_maker.analyzer import extract_midi_facts, describe_midi, chat_about_midi
from tt_midi_maker.assembler import build_midi_file, TICKS_PER_BEAT
from tt_midi_maker.errors import MidiMakerError
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_test_midi(tmp_path: Path, bpm: int = 120) -> Path:
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=1)
        for i in range(16)
    ]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], bpm=bpm, output_path=tmp_path / "test.mid")


def test_extract_facts_bpm(tmp_path):
    midi_path = make_test_midi(tmp_path, bpm=90)
    facts = extract_midi_facts(midi_path)
    assert facts["bpm"] == 90


def test_extract_facts_note_count(tmp_path):
    midi_path = make_test_midi(tmp_path)
    facts = extract_midi_facts(midi_path)
    assert facts["note_count"] == 16


def test_extract_facts_channels_used(tmp_path):
    midi_path = make_test_midi(tmp_path)
    facts = extract_midi_facts(midi_path)
    assert 1 in facts["channels_used"]


def test_describe_midi_file_not_found():
    with pytest.raises(MidiMakerError) as exc:
        describe_midi(Path("/nonexistent/file.mid"))
    assert exc.value.code == "FILE_NOT_FOUND"


def test_describe_midi_returns_description(tmp_path):
    midi_path = make_test_midi(tmp_path)
    with patch("tt_midi_maker.analyzer.call_llm", return_value="A simple melody."):
        result = describe_midi(midi_path)
    assert result["description"] == "A simple melody."
    assert result["tempo_bpm"] == 120


def test_chat_about_midi_routes_question(tmp_path):
    midi_path = make_test_midi(tmp_path)
    with patch("tt_midi_maker.analyzer.call_llm", return_value="It is in C major."):
        result = chat_about_midi(midi_path, "What key is this?")
    assert "C major" in result["answer"]
    assert "note_count" in result["analysis_context"]
