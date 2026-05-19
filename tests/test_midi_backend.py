"""
Tests for tt_midi_maker.generation.midi_backend and supporting modules.

All tests run on CPU without downloading any model weights.
The model is replaced with a tiny randomly-initialised replica.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

ROLES_CONFIG = {
    "melody":  {"channel": 1,  "program": 0,  "note_range": [60, 96], "density_default": 1.0},
    "bass":    {"channel": 2,  "program": 32, "note_range": [28, 52], "density_default": 0.8},
    "harmony": {"channel": 3,  "program": 48, "note_range": [48, 72], "density_default": 0.6},
    "drums":   {"channel": 10, "program": 0,  "note_range": [35, 81], "density_default": 0.7},
}


def _make_blueprint(bpm=120, bars=4, key="C major", active_roles=None):
    """Build a minimal MusicalBlueprint-like namespace for tests."""
    from tt_midi_maker.models.blueprint import RoleConfig, MusicalBlueprint
    active = active_roles or ["melody", "bass", "drums"]
    roles = {}
    for role, cfg in ROLES_CONFIG.items():
        density = cfg["density_default"] if role in active else 0.0
        roles[role] = RoleConfig(density=density)
    return MusicalBlueprint(
        key=key, bpm=bpm, bars=bars,
        style="lo-fi hip hop",
        mode="loop",
        chord_progression=["I", "IV", "V", "I"],
        roles=roles,
    )


def _make_tiny_model():
    """Return a (model, tokenizer) pair with random weights (V1 tokenizer)."""
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    from tt_midi_maker.generation.skytnt_model import MIDIModel, MIDIModelConfig
    from transformers import LlamaConfig

    tokenizer = MIDITokenizerV1()
    net_cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        num_hidden_layers=1, intermediate_size=128,
        pad_token_id=tokenizer.pad_id, max_position_embeddings=512,
        use_cache=False,
    )
    tok_cfg = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        num_hidden_layers=1, intermediate_size=128,
        pad_token_id=tokenizer.pad_id, max_position_embeddings=512,
        use_cache=False,
    )
    config = MIDIModelConfig(tokenizer, net_cfg, tok_cfg)
    model = MIDIModel(config)
    model.eval()
    return model, tokenizer


# ── MIDITokenizerV1 ───────────────────────────────────────────────────────────

def test_v1_tokenizer_vocab_size():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    assert tok.vocab_size > 0
    # pad/bos/eos are the first three IDs
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


def test_v1_event2tokens_roundtrip():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    # encode a set_tempo event (time1=0, time2=0, track=0, bpm=120)
    tokens = tok.event2tokens(["set_tempo", 0, 0, 0, 120])
    assert len(tokens) == tok.max_token_seq
    event = tok.tokens2event(tokens)
    assert event[0] == "set_tempo"
    assert event[4] == 120


def test_v1_event2tokens_note_roundtrip():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    # note: time1=0, time2=0, track=0, duration=16, channel=0, pitch=60, velocity=64
    tokens = tok.event2tokens(["note", 0, 0, 0, 16, 0, 60, 64])
    assert len(tokens) == tok.max_token_seq
    event = tok.tokens2event(tokens)
    assert event[0] == "note"
    assert event[5] == 0    # channel
    assert event[6] == 60   # pitch
    assert event[7] == 64   # velocity


def test_v1_detokenize_returns_midi_score():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    note_tokens = tok.event2tokens(["note", 0, 0, 0, 16, 0, 60, 80])
    eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    midi_seq = [bos, note_tokens, eos]
    score = tok.detokenize(midi_seq)
    # score[0] = ticks_per_beat, score[1:] = tracks
    assert score[0] == 480
    all_notes = [e for track in score[1:] for e in track if e[0] == "note"]
    assert len(all_notes) == 1
    assert all_notes[0][3] == 0   # channel 0
    assert all_notes[0][4] == 60  # pitch


def test_v2_tokenizer_vocab_size():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV2
    tok = MIDITokenizerV2()
    assert tok.vocab_size > 0
    assert tok.pad_id == 0


def test_tokenizer_factory():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizer, MIDITokenizerV1, MIDITokenizerV2
    assert isinstance(MIDITokenizer("v1"), MIDITokenizerV1)
    assert isinstance(MIDITokenizer("v2"), MIDITokenizerV2)
    with pytest.raises(ValueError):
        MIDITokenizer("v3")


# ── _build_prompt ─────────────────────────────────────────────────────────────

def test_build_prompt_starts_with_bos():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    bp = _make_blueprint()
    prompt = _build_prompt(bp, ROLES_CONFIG, tok)
    assert prompt.shape[1] == tok.max_token_seq
    assert prompt[0, 0] == tok.bos_id


def test_build_prompt_includes_set_tempo():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    bp = _make_blueprint(bpm=90)
    prompt = _build_prompt(bp, ROLES_CONFIG, tok)
    # Find a set_tempo event_id token
    tempo_id = tok.event_ids["set_tempo"]
    found = any(row[0] == tempo_id for row in prompt.tolist())
    assert found, "set_tempo event missing from prompt"


def test_build_prompt_no_drums_patch_change():
    """Drums channel should never get a patch_change; all other configured channels always do.

    After the all-channel conditioning change, _build_prompt emits patch_change for every
    non-drum role in roles_config regardless of density.  This keeps the model aware of the
    full instrument palette even when some roles have density=0.  The drums invariant
    (channel 10 / ch9) still holds.
    """
    from tt_midi_maker.generation.midi_backend import _build_prompt
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    bp = _make_blueprint(active_roles=["drums"])  # only drums active; others density=0
    prompt = _build_prompt(bp, ROLES_CONFIG, tok)
    patch_id = tok.event_ids["patch_change"]

    # Drums (ch9, 0-indexed) must never appear in a patch_change row
    drum_ch0 = ROLES_CONFIG["drums"]["channel"] - 1  # 9
    for row in prompt.tolist():
        if row[0] == patch_id:
            # channel token is at index 4 in patch_change rows
            ch_token = row[4]
            ch_ids = tok.parameter_ids["channel"]
            for c, cid in enumerate(ch_ids):
                if cid == ch_token:
                    assert c != drum_ch0, f"drums (ch{drum_ch0}) must not appear in patch_change"
                    break

    # All non-drum roles in ROLES_CONFIG should have a patch_change (even density=0)
    non_drum_roles = {n: c for n, c in ROLES_CONFIG.items() if c["channel"] != 10}
    for role_name, cfg in non_drum_roles.items():
        expected_ch0 = cfg["channel"] - 1
        found_ch = False
        for row in prompt.tolist():
            if row[0] == patch_id:
                ch_token = row[4]
                ch_ids = tok.parameter_ids["channel"]
                for c, cid in enumerate(ch_ids):
                    if cid == ch_token and c == expected_ch0:
                        found_ch = True
                        break
        assert found_ch, f"{role_name} (ch{expected_ch0}) missing patch_change in prompt"


def test_build_prompt_includes_patch_for_melody():
    from tt_midi_maker.generation.midi_backend import _build_prompt
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    tok = MIDITokenizerV1()
    bp = _make_blueprint(active_roles=["melody"])
    prompt = _build_prompt(bp, ROLES_CONFIG, tok)
    patch_id = tok.event_ids["patch_change"]
    found = any(row[0] == patch_id for row in prompt.tolist())
    assert found, "melody should get a patch_change"


# ── _score_to_roletracks ──────────────────────────────────────────────────────

def _fake_score(notes_by_ch0: dict) -> list:
    """Build a minimal midi_score from {ch0: [(t, dur, pitch, vel), ...]}."""
    tracks = []
    for ch0, note_list in notes_by_ch0.items():
        track = [["note", t, dur, ch0, pitch, vel] for t, dur, pitch, vel in note_list]
        tracks.append(track)
    return [480] + tracks


def test_score_to_roletracks_basic():
    from tt_midi_maker.generation.midi_backend import _score_to_roletracks
    # ch0=0 → melody (ch1=1)
    score = _fake_score({0: [(0, 480, 60, 80), (960, 480, 62, 75)]})
    tracks = _score_to_roletracks(score, ROLES_CONFIG)
    assert len(tracks) == 1
    assert tracks[0].role == "melody"
    assert tracks[0].channel == 1
    assert len(tracks[0].notes) == 2


def test_score_to_roletracks_drums_channel():
    from tt_midi_maker.generation.midi_backend import _score_to_roletracks
    # ch0=9 → drums (ch1=10)
    score = _fake_score({9: [(0, 240, 36, 100)]})
    tracks = _score_to_roletracks(score, ROLES_CONFIG)
    assert tracks[0].role == "drums"
    assert tracks[0].channel == 10


def test_score_to_roletracks_max_tick_filter():
    from tt_midi_maker.generation.midi_backend import _score_to_roletracks
    # 4 bars at 480 tpb = 7680 ticks max
    max_tick = 4 * 4 * 480
    score = _fake_score({0: [
        (0, 480, 60, 80),         # inside
        (max_tick - 1, 480, 62, 75),  # start inside, duration clipped
        (max_tick, 480, 64, 70),  # exactly at boundary: excluded
        (max_tick + 960, 480, 65, 65),  # outside
    ]})
    tracks = _score_to_roletracks(score, ROLES_CONFIG, max_tick=max_tick)
    notes = tracks[0].notes
    assert all(n.start_tick < max_tick for n in notes)
    # third note starts at max_tick → excluded
    assert len(notes) == 2


def test_score_to_roletracks_empty_notes():
    from tt_midi_maker.generation.midi_backend import _score_to_roletracks
    score = _fake_score({})  # no notes at all
    tracks = _score_to_roletracks(score, ROLES_CONFIG)
    assert tracks == []


def test_score_to_roletracks_unknown_channel():
    from tt_midi_maker.generation.midi_backend import _score_to_roletracks
    # channel not in ROLES_CONFIG gets role "unknown"
    score = _fake_score({14: [(0, 480, 60, 80)]})
    tracks = _score_to_roletracks(score, ROLES_CONFIG)
    assert tracks[0].role == "unknown"
    assert tracks[0].channel == 15


# ── generate_from_blueprint (mocked model) ────────────────────────────────────

def _build_minimal_generated(tok, ch0=0, n_notes=4):
    """Construct a fake generated numpy array (1, seq, max_token_seq) with n notes."""
    bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    rows = [bos]
    for i in range(n_notes):
        t = i * 2
        tokens = tok.event2tokens(["note", t, 0, 0, 8, ch0, 60 + i, 80])
        if tokens:
            rows.append(tokens)
    eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    rows.append(eos)
    arr = np.array(rows, dtype=np.int64)
    return arr[None, :, :]  # (1, seq_len, max_token_seq)


def test_generate_from_blueprint_uses_midi_backend(monkeypatch):
    """generate_from_blueprint should call _get_model and return RoleTracks."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1

    tok = MIDITokenizerV1()
    fake_generated = _build_minimal_generated(tok, ch0=0, n_notes=4)

    # Patch model to avoid any download
    class FakeModel:
        tokenizer = tok
        def generate(self, prompt, batch_size, max_len, **kw):
            return fake_generated
        def parameters(self):
            return iter([torch.zeros(1)])

    monkeypatch.setattr(midi_backend, "_model_cache", (FakeModel(), tok))

    bp = _make_blueprint(bars=4, active_roles=["melody"])
    tracks = midi_backend.generate_from_blueprint(bp, ROLES_CONFIG)

    assert isinstance(tracks, list)
    # All returned RoleTracks should be within max_tick
    max_tick = 4 * 4 * 480
    for track in tracks:
        for note in track.notes:
            assert note.start_tick < max_tick


def test_generate_from_blueprint_fallback_on_empty(monkeypatch):
    """When model produces no notes, generate_from_blueprint returns empty list
    and server.py falls back to stub — test that _score_to_roletracks handles this."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1

    tok = MIDITokenizerV1()
    # Only BOS + EOS, no notes
    bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    arr = np.array([[bos, eos]], dtype=np.int64)

    class FakeModel:
        tokenizer = tok
        def generate(self, prompt, **kw):
            return arr
        def parameters(self):
            return iter([torch.zeros(1)])

    monkeypatch.setattr(midi_backend, "_model_cache", (FakeModel(), tok))

    bp = _make_blueprint(bars=4, active_roles=["melody"])
    tracks = midi_backend.generate_from_blueprint(bp, ROLES_CONFIG)
    assert tracks == []


# ── hardware.py ───────────────────────────────────────────────────────────────

def test_detect_tt_devices_no_smi(monkeypatch):
    """Returns empty list when tt-smi is not found."""
    import subprocess
    from tt_midi_maker.generation.hardware import detect_tt_devices
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no tt-smi")),
    )
    assert detect_tt_devices() == []


def test_detect_tt_devices_parses_indices():
    """Returns list indices of available devices, not a dict field."""
    import subprocess
    import json as _json
    from unittest.mock import MagicMock
    from tt_midi_maker.generation.hardware import detect_tt_devices

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _json.dumps({
        "device_info": [
            {"bus_id": "0000:01:00.0", "status": "available"},
            {"bus_id": "0000:02:00.0", "status": "unavailable"},
            {"bus_id": "0000:03:00.0", "status": "available"},
        ]
    })

    orig_run = subprocess.run

    def fake_run(cmd, **kw):
        if "tt-smi" in cmd:
            return mock_result
        return orig_run(cmd, **kw)

    import subprocess as sp
    sp.run = fake_run
    try:
        result = detect_tt_devices()
    finally:
        sp.run = orig_run

    assert result == [0, 2]  # list indices of "available" devices
