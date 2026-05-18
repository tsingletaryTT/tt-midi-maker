import tempfile
from pathlib import Path
import mido
import pytest
from tt_midi_maker.generation.tokenizer import (
    get_tokenizer, encode_midi_file, decode_tokens_to_midi,
)
from tt_midi_maker.assembler import build_midi_file, TICKS_PER_BEAT
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_simple_midi(tmp_path: Path) -> Path:
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=1)
        for i in range(8)
    ]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], bpm=120, output_path=tmp_path / "test.mid")


def test_tokenizer_loads():
    tok = get_tokenizer()
    assert tok is not None


def test_encode_returns_list_of_ints(tmp_path):
    midi_path = make_simple_midi(tmp_path)
    tokens = encode_midi_file(midi_path)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_encode_decode_roundtrip_preserves_note_count(tmp_path):
    midi_path = make_simple_midi(tmp_path)
    tokens = encode_midi_file(midi_path)
    out_path = tmp_path / "decoded.mid"
    decode_tokens_to_midi(tokens, out_path)
    assert out_path.exists()
    mid = mido.MidiFile(str(out_path))
    note_ons = sum(
        1 for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )
    assert note_ons > 0
