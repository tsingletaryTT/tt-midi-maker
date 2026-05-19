"""Tests for _build_prompt all-channel conditioning and _compute_active_channels."""
import numpy as np
import pytest


def _make_tokenizer():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    return MIDITokenizerV1()


def _make_blueprint(active_roles: dict):
    """Build a minimal MusicalBlueprint with given role densities."""
    from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
    roles = {}
    for name, density in active_roles.items():
        roles[name] = RoleConfig(density=density)
    return MusicalBlueprint(
        key="C major",
        bpm=120,
        bars=4,
        chord_progression=["C", "Am", "F", "G"],
        style="test",
        mode="loop",
        roles=roles,
    )


_ROLES_CFG = {
    "melody":  {"channel": 1,  "program": 40, "note_range": ["C4", "C7"]},
    "harmony": {"channel": 2,  "program": 0,  "note_range": ["C3", "C5"]},
    "bass":    {"channel": 3,  "program": 32, "note_range": ["C2", "C4"]},
    "drums":   {"channel": 10, "program": 0,  "note_range": ["C2", "C7"]},
}


def test_build_prompt_includes_patch_change_for_all_non_drum_roles():
    """All non-drum channels in roles_config appear as patch_change rows even if density=0."""
    from tt_midi_maker.generation.midi_backend import _build_prompt
    tok = _make_tokenizer()
    # melody active, harmony+bass inactive (density=0), drums active
    bp = _make_blueprint({"melody": 1.0, "harmony": 0.0, "bass": 0.0, "drums": 1.0})

    rows = _build_prompt(bp, _ROLES_CFG, tok)

    # Decode rows to find patch_change events
    patch_channels = set()
    for row in rows:
        eid = row[0]
        if eid == tok.event_ids.get("patch_change"):
            # patch_change params: time1, time2, track, channel, patch
            # channel token is at position 4 (index 4 in the row after event id)
            ch_token = row[4]
            ch_ids = tok.parameter_ids["channel"]
            for c, cid in enumerate(ch_ids):
                if cid == ch_token:
                    patch_channels.add(c)
                    break

    # All three melodic channels (0-indexed: 0=melody ch1, 1=harmony ch2, 2=bass ch3)
    assert 0 in patch_channels, f"melody (ch0) missing from patch_changes: {patch_channels}"
    assert 1 in patch_channels, f"harmony (ch1) missing from patch_changes: {patch_channels}"
    assert 2 in patch_channels, f"bass (ch2) missing from patch_changes: {patch_channels}"
    # Drums (channel 10 = ch9 in 0-indexed) should NOT have patch_change
    assert 9 not in patch_channels, f"drums got unexpected patch_change"


def test_compute_active_channels_returns_only_density_positive_roles():
    """_compute_active_channels only includes channels for roles with density > 0."""
    from tt_midi_maker.generation.midi_backend import _compute_active_channels
    bp = _make_blueprint({"melody": 1.0, "harmony": 0.5, "bass": 0.0, "drums": 1.0})
    active = _compute_active_channels(bp, _ROLES_CFG)
    assert 0 in active, "melody (ch0) should be active"
    assert 1 in active, "harmony (ch1) should be active"
    assert 2 not in active, "bass (density=0) should NOT be active"
    assert 9 in active, "drums (ch9) should be active"


def test_compute_active_channels_empty_blueprint():
    """All density=0 → empty active channels set."""
    from tt_midi_maker.generation.midi_backend import _compute_active_channels
    bp = _make_blueprint({"melody": 0.0, "bass": 0.0})
    active = _compute_active_channels(bp, _ROLES_CFG)
    assert active == set()
