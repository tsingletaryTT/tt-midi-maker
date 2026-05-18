"""
Tests for source MIDI context injection.

Covers _midi_file_to_score, _midi_file_to_prompt_rows, _build_prompt with source
rows, and generate_from_blueprint with source_midi — all on CPU, no download.
"""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ── MIDI file helpers ─────────────────────────────────────────────────────────

def _write_minimal_midi(path: Path, notes: list[tuple], bpm: int = 120, tpb: int = 480):
    """Write a minimal type-0 MIDI file with the given notes.

    notes: list of (start_tick, dur_ticks, channel, pitch, velocity)
    """
    import mido
    mid = mido.MidiFile(type=0, ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("program_change", channel=1, program=32, time=0))

    # Build note_on/note_off messages sorted by tick
    msgs = []
    for start, dur, ch, pitch, vel in notes:
        msgs.append((start, "note_on",  ch, pitch, vel))
        msgs.append((start + dur, "note_off", ch, pitch, 0))
    msgs.sort(key=lambda m: m[0])

    prev_tick = 0
    for tick, mtype, ch, pitch, vel in msgs:
        delta = tick - prev_tick
        prev_tick = tick
        if mtype == "note_on":
            track.append(mido.Message("note_on",  channel=ch, note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", channel=ch, note=pitch, velocity=0,  time=delta))

    mid.save(str(path))
    return path


# ── _midi_file_to_score ───────────────────────────────────────────────────────

def test_midi_file_to_score_basic():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_score
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    _write_minimal_midi(p, [(0, 240, 1, 60, 80), (480, 240, 1, 62, 75)])

    score = _midi_file_to_score(p)
    assert score[0] == 480   # ticks_per_beat
    events = score[1]
    notes = [e for e in events if e[0] == "note"]
    assert len(notes) == 2
    assert notes[0][1] == 0    # start tick
    assert notes[1][1] == 480  # start tick


def test_midi_file_to_score_duration():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_score
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    _write_minimal_midi(p, [(0, 240, 1, 60, 80)])

    score = _midi_file_to_score(p)
    note = [e for e in score[1] if e[0] == "note"][0]
    assert note[2] == 240    # duration


def test_midi_file_to_score_sorted_by_tick():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_score
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    _write_minimal_midi(p, [(960, 240, 1, 64, 80), (0, 240, 1, 60, 80)])

    score = _midi_file_to_score(p)
    notes = [e for e in score[1] if e[0] == "note"]
    ticks = [n[1] for n in notes]
    assert ticks == sorted(ticks)


def test_midi_file_to_score_set_tempo_present():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_score
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    _write_minimal_midi(p, [(0, 240, 1, 60, 80)], bpm=90)

    score = _midi_file_to_score(p)
    tempos = [e for e in score[1] if e[0] == "set_tempo"]
    assert len(tempos) >= 1
    # tempo value should correspond to ~90 BPM (666666 µs/beat)
    assert tempos[0][2] == pytest.approx(666667, abs=1000)


# ── _midi_file_to_prompt_rows ─────────────────────────────────────────────────

def _make_tokenizer():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    return MIDITokenizerV1()


def test_prompt_rows_nonempty():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_prompt_rows
    tok = _make_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    _write_minimal_midi(p, [(0, 240, 1, 60, 80), (480, 240, 1, 62, 75)])

    rows = _midi_file_to_prompt_rows(p, tok)
    assert len(rows) > 0
    # Each row should have tok.max_token_seq elements
    for row in rows:
        assert len(row) == tok.max_token_seq


def test_prompt_rows_last_n_bars_filters():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_prompt_rows
    tok = _make_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    # 8 bars of notes: one note per bar at 480 tpb
    tpb = 480
    notes = [(bar * tpb * 4, tpb, 1, 60 + bar, 80) for bar in range(8)]
    _write_minimal_midi(p, notes, tpb=tpb)

    rows_all  = _midi_file_to_prompt_rows(p, tok, last_n_bars=None)
    rows_last4 = _midi_file_to_prompt_rows(p, tok, last_n_bars=4)

    # Last-4-bars should have fewer token rows than all 8 bars
    assert 0 < len(rows_last4) < len(rows_all)


def test_prompt_rows_rebased_to_zero():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_score, _midi_file_to_prompt_rows
    tok = _make_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    # Two notes far into the future
    _write_minimal_midi(p, [(9600, 240, 1, 60, 80), (10080, 240, 1, 62, 75)])

    # After rebasing, prompt_rows should still be non-empty and valid
    rows = _midi_file_to_prompt_rows(p, tok, last_n_bars=None)
    assert len(rows) > 0


def test_prompt_rows_empty_file():
    from tt_midi_maker.generation.midi_backend import _midi_file_to_prompt_rows
    import mido
    tok = _make_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        p = Path(f.name)
    # Empty MIDI file (no notes)
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    mid.tracks.append(mido.MidiTrack())
    mid.save(str(p))

    rows = _midi_file_to_prompt_rows(p, tok)
    assert rows == []


# ── _build_prompt with source_rows ────────────────────────────────────────────

def _make_blueprint(bars=4):
    from tt_midi_maker.models.blueprint import RoleConfig, MusicalBlueprint
    return MusicalBlueprint(
        key="E minor", bpm=92, bars=bars,
        style="post-rock", mode="loop",
        chord_progression=["Em", "D", "C", "D"],
        roles={
            "bass":   RoleConfig(density=0.8),
            "drums":  RoleConfig(density=0.7),
            "melody": RoleConfig(density=0.0),
        },
    )

ROLES_CONFIG = {
    "bass":   {"channel": 2,  "program": 32, "note_range": [28, 52], "density_default": 0.8},
    "drums":  {"channel": 10, "program": 0,  "note_range": [35, 81], "density_default": 0.7},
    "melody": {"channel": 1,  "program": 0,  "note_range": [60, 96], "density_default": 1.0},
}


def test_build_prompt_without_source():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    tok = _make_tokenizer()
    prompt = _build_prompt(_make_blueprint(), ROLES_CONFIG, tok)
    assert prompt.ndim == 2
    assert prompt.shape[1] == tok.max_token_seq
    assert prompt[0, 0] == tok.bos_id


def test_build_prompt_with_source_rows_appended():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    tok = _make_tokenizer()
    # Make 3 fake source rows (just pad rows)
    fake_rows = [[tok.pad_id] * tok.max_token_seq for _ in range(3)]
    prompt_no_src = _build_prompt(_make_blueprint(), ROLES_CONFIG, tok, source_rows=None)
    prompt_with_src = _build_prompt(_make_blueprint(), ROLES_CONFIG, tok, source_rows=fake_rows)

    # With source rows the prompt should be longer
    assert prompt_with_src.shape[0] == prompt_no_src.shape[0] + 3


def test_build_prompt_source_rows_after_setup():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    tok = _make_tokenizer()
    marker_row = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    prompt = _build_prompt(_make_blueprint(), ROLES_CONFIG, tok, source_rows=[marker_row])

    # The marker row (EOS token) should be the LAST row
    assert prompt[-1, 0] == tok.eos_id
    # BOS should still be first
    assert prompt[0, 0] == tok.bos_id


# ── generate_from_blueprint with source_midi ──────────────────────────────────

def test_generate_with_source_midi(monkeypatch, tmp_path):
    """generate_from_blueprint passes source_midi context into the prompt."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1

    tok = MIDITokenizerV1()

    # Write a simple source MIDI to temp file
    src_path = tmp_path / "source.mid"
    _write_minimal_midi(src_path, [(0, 240, 1, 60, 80), (480, 240, 1, 62, 75)])

    prompt_lengths = []

    class FakeModel:
        tokenizer = tok
        def generate(self, prompt, batch_size=1, max_len=16, **kw):
            prompt_lengths.append(prompt.shape[0])   # shape: (n_events, max_token_seq) — record n_events
            bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            return np.array([[bos, eos]], dtype=np.int64)
        def parameters(self):
            return iter([torch.zeros(1)])

    monkeypatch.setattr(midi_backend, "_model_cache", (FakeModel(), tok))
    monkeypatch.setattr(midi_backend, "_hw_model_cache", None)

    from tt_midi_maker.models.blueprint import RoleConfig, MusicalBlueprint
    bp = MusicalBlueprint(
        key="E minor", bpm=92, bars=4, style="post-rock", mode="loop",
        chord_progression=["Em", "D"],
        roles={"bass": RoleConfig(density=0.8), "drums": RoleConfig(density=0.7)},
    )

    # Generate without source
    midi_backend.generate_from_blueprint(bp, ROLES_CONFIG)
    len_no_src = prompt_lengths[-1]

    # Generate with source MIDI — prompt should be longer (source rows appended)
    midi_backend.generate_from_blueprint(bp, ROLES_CONFIG, source_midi=str(src_path))
    len_with_src = prompt_lengths[-1]

    assert len_with_src > len_no_src, "source_midi should make the prompt longer"


def test_generate_with_source_midi_bad_path(monkeypatch):
    """Bad source_midi path is silently ignored — generation still succeeds."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1

    tok = MIDITokenizerV1()

    class FakeModel:
        tokenizer = tok
        def generate(self, prompt, batch_size=1, max_len=16, **kw):
            bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            return np.array([[bos, eos]], dtype=np.int64)
        def parameters(self):
            return iter([torch.zeros(1)])

    monkeypatch.setattr(midi_backend, "_model_cache", (FakeModel(), tok))
    monkeypatch.setattr(midi_backend, "_hw_model_cache", None)

    from tt_midi_maker.models.blueprint import RoleConfig, MusicalBlueprint
    bp = MusicalBlueprint(
        key="E minor", bpm=92, bars=4, style="post-rock", mode="loop",
        chord_progression=["Em", "D"],
        roles={"bass": RoleConfig(density=0.8)},
    )

    # Should not raise even though source file doesn't exist
    result = midi_backend.generate_from_blueprint(
        bp, ROLES_CONFIG, source_midi="/nonexistent/file.mid"
    )
    assert isinstance(result, list)
