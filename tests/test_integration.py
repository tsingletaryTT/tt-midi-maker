"""
End-to-end smoke test: runs the full pipeline with mocked LLM and generation.
Verifies that generate_midi produces a readable, non-empty MIDI file.
"""
from pathlib import Path
from unittest.mock import patch
import mido
from tt_midi_maker.server import _generate_midi
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

BLUEPRINT = MusicalBlueprint(
    key="C major", bpm=120, time_signature="4/4", style="ambient",
    chord_progression=["C", "Am", "F", "G"], bars=4, mode="loop",
    roles={
        "melody": RoleConfig(density=1.0, velocity_range=(70, 100), pattern_hint="legato"),
        "drums":  RoleConfig(density=0.7, velocity_range=(60, 90),  pattern_hint="default"),
    },
)


def test_full_pipeline_produces_valid_midi(tmp_path):
    from tt_midi_maker import server
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch.object(server, "OUTPUT_DIR", tmp_path):
        result = _generate_midi(
            prompt="calm ambient with piano and soft drums",
            mode="loop",
        )

    assert "file_path" in result
    midi_path = Path(result["file_path"])
    assert midi_path.exists(), f"MIDI file not written: {midi_path}"

    mid = mido.MidiFile(str(midi_path))
    assert mid.type == 1, "Expected Type-1 multi-track MIDI"
    assert len(mid.tracks) >= 2, "Expected at least tempo track + one instrument track"

    note_ons = sum(
        1 for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )
    assert note_ons > 0, "MIDI file contains no notes"


def test_full_pipeline_respects_role_filter(tmp_path):
    from tt_midi_maker import server
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch.object(server, "OUTPUT_DIR", tmp_path):
        result = _generate_midi(
            prompt="just melody please",
            mode="loop",
            roles=["melody"],
        )
    assert "drums" not in result["roles_generated"]
    assert "melody" in result["roles_generated"]


def test_generate_returns_metadata_fields(tmp_path):
    from tt_midi_maker import server
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch.object(server, "OUTPUT_DIR", tmp_path):
        result = _generate_midi(prompt="test", mode="loop")
    for field in ("file_path", "bars_generated", "bpm", "key",
                  "roles_generated", "hardware_used"):
        assert field in result, f"Missing field in result: {field}"
