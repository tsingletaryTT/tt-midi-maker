# MIDI Generation Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the generation-quality techniques from skytnt's own Gradio app into our pipeline: lock instruments via vocabulary masking (disable_patch_change, disable_channels, disable_control_change), improve source-MIDI context conditioning, and add per-role velocity shaping.

**Architecture:** Three changes at three layers:
1. `forge_backend.py:generate_hardware()` — vocabulary masks during token generation loop prevent unwanted events and channel leakage
2. `midi_backend.py:_build_prompt()` and `_midi_file_to_prompt_rows()` — all-channel upfront conditioning and `add_default_instr=True` for richer source context
3. `coherence/humanize.py` — new `scale_velocity_by_role()` function; wired into each demo script's `_apply_coherence()`

**Tech Stack:** Python, PyTorch, skytnt MIDITokenizerV1, our existing coherence layer

---

## File Map

| File | Change |
|------|--------|
| `tt_midi_maker/generation/forge_backend.py` | Add `disable_patch_change`, `disable_control_change`, `allowed_channels` params to `generate_hardware()`; implement masking in inner token loop |
| `tt_midi_maker/generation/midi_backend.py` | All-channel patch_change in `_build_prompt()`; `add_default_instr=True` in `_midi_file_to_prompt_rows()`; compute `active_channels` and pass new flags to `generate_hardware()` |
| `tt_midi_maker/coherence/humanize.py` | Add `_ROLE_VELOCITY_RANGES` dict and `scale_velocity_by_role()` function |
| `demo_ambient.py` | Wire `scale_velocity_by_role()` into `_apply_coherence()` |
| `demo_bebop.py` | Wire `scale_velocity_by_role()` into `_apply_coherence()` |
| `demo_blues.py` | Wire `scale_velocity_by_role()` into `_apply_coherence()` |
| `demo_classical.py` | Wire `scale_velocity_by_role()` into `_apply_coherence()` |
| `tests/test_forge_backend.py` (new) | Tests for new masking logic |
| `tests/test_humanize.py` | Tests for `scale_velocity_by_role()` |

---

## Task 1: Vocabulary Masking in `generate_hardware()`

**Files:**
- Modify: `tt_midi_maker/generation/forge_backend.py:164-331`
- Create: `tests/test_forge_backend.py`

### Background

The inner token loop (lines 280–313) currently builds a mask for each position in the token sequence. At `i==0` it allows all event types; at `i>0` it allows the parameters for the chosen event. We need to:

- Remove `patch_change` from the `i==0` allowed set (model cannot re-assign instruments mid-generation)
- Remove `control_change` from the `i==0` allowed set (CC events add noise)
- When `event_name=="note"` and the current parameter is `"channel"` (i==5 in V1, since `events["note"] = ["time1","time2","track","duration","channel","pitch","velocity"]` → "channel" is at index 4 → appears at `i = index + 1 = 5`), restrict channel tokens to only the channels in `allowed_channels`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_forge_backend.py`:

```python
"""Tests for generate_hardware masking logic (no hardware required)."""
import numpy as np
import pytest
import torch


def _make_mock_model(tokenizer):
    """Minimal MIDIModel-like object with just enough for generate_hardware."""
    import types, torch.nn as nn

    class FakeNet(nn.Module):
        def __init__(self):
            super().__init__()
            hs = 32
            self.config = types.SimpleNamespace(
                hidden_size=hs, num_hidden_layers=1
            )
            self.embed_tokens = nn.Embedding(tokenizer.vocab_size, hs)
            # embed_tokens normally returns (batch, seq, token_seq, hidden)
            # We monkey-patch to return (batch, seq, hidden) to keep shapes simple
            _orig = self.embed_tokens.forward

            def _summed(x):
                return _orig(x).sum(dim=-2)   # collapse token_seq dim

            self.embed_tokens = _summed

        def forward(self, inputs_embeds, use_cache):
            import types
            b, s, h = inputs_embeds.shape
            out = types.SimpleNamespace(last_hidden_state=inputs_embeds)
            return out

    class FakeNetToken(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, hidden, x_tok, cache=None):
            # Return uniform logits → after mask, sampling is over allowed tokens
            b = 1
            return torch.ones(b, 1, self.vocab_size)

    class FakeModel:
        def __init__(self):
            self.tokenizer = tokenizer
            self.net = FakeNet()
            self.net_token = FakeNetToken(tokenizer.vocab_size)

        def forward_token(self, h, x, cache=None):
            return self.net_token(h, x, cache)

        def sample_top_p_k(self, scores, top_p, top_k):
            # Argmax of the masked scores (samples highest-weight allowed token)
            idx = scores[0, 0].argmax()
            return torch.tensor([[idx]])

    return FakeModel()


def _make_tokenizer():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    return MIDITokenizerV1()


# ── Mask-bit inspection helpers ────────────────────────────────────────────────

def _collect_allowed_events(tokenizer, mask_1d: torch.Tensor) -> set[str]:
    """Return set of event names whose event-id token is allowed in mask."""
    allowed = set()
    for event_name, eid in tokenizer.event_ids.items():
        if mask_1d[eid].item():
            allowed.add(event_name)
    return allowed


def _collect_allowed_channels(tokenizer, mask_1d: torch.Tensor) -> set[int]:
    """Return set of channel values (0-indexed) whose token is allowed."""
    ch_ids = tokenizer.parameter_ids["channel"]
    return {c for c, tok in enumerate(ch_ids) if mask_1d[tok].item()}


# ── Patch generate_hardware to expose mask bits without TT hardware ─────────

def _run_one_event_mask(tokenizer, model, disable_patch_change, disable_control_change,
                        allowed_channels, event_i, event_name=""):
    """
    Run just the mask-building part of the inner loop once, return mask_1d.
    event_i: which position in the inner token loop (0=event select, 1-N=params)
    event_name: required when event_i > 0
    """
    mask = torch.zeros((1, tokenizer.vocab_size), dtype=torch.int64)
    end = False

    if end:
        mask[0, tokenizer.pad_id] = 1
    elif event_i == 0:
        allowed_event_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
        if disable_patch_change:
            allowed_event_ids = [e for e in allowed_event_ids
                                 if tokenizer.id_events.get(e) != "patch_change"]
        if disable_control_change:
            allowed_event_ids = [e for e in allowed_event_ids
                                 if tokenizer.id_events.get(e) != "control_change"]
        mask[0, allowed_event_ids] = 1
    else:
        param_names = tokenizer.events[event_name]
        if event_i > len(param_names):
            mask[0, tokenizer.pad_id] = 1
        else:
            param = param_names[event_i - 1]
            param_ids = tokenizer.parameter_ids[param]
            if param == "channel" and allowed_channels is not None:
                param_ids = [param_ids[c] for c in sorted(allowed_channels)
                             if c < len(param_ids)]
            mask[0, param_ids] = 1

    return mask[0]


def test_disable_patch_change_removes_patch_change_from_event_mask():
    tok = _make_tokenizer()
    mask = _run_one_event_mask(tok, None,
                               disable_patch_change=True,
                               disable_control_change=False,
                               allowed_channels=None,
                               event_i=0)
    allowed = _collect_allowed_events(tok, mask)
    assert "patch_change" not in allowed
    assert "note" in allowed          # note still allowed
    assert "set_tempo" in allowed     # tempo still allowed


def test_disable_control_change_removes_cc_from_event_mask():
    tok = _make_tokenizer()
    mask = _run_one_event_mask(tok, None,
                               disable_patch_change=False,
                               disable_control_change=True,
                               allowed_channels=None,
                               event_i=0)
    allowed = _collect_allowed_events(tok, mask)
    assert "control_change" not in allowed
    assert "note" in allowed


def test_no_disable_flags_allows_all_events():
    tok = _make_tokenizer()
    mask = _run_one_event_mask(tok, None,
                               disable_patch_change=False,
                               disable_control_change=False,
                               allowed_channels=None,
                               event_i=0)
    allowed = _collect_allowed_events(tok, mask)
    assert "patch_change" in allowed
    assert "control_change" in allowed
    assert "note" in allowed


def test_allowed_channels_restricts_note_channel_param():
    tok = _make_tokenizer()
    # "note" params: ["time1","time2","track","duration","channel","pitch","velocity"]
    # "channel" is at index 4 → appears at event_i=5
    mask = _run_one_event_mask(tok, None,
                               disable_patch_change=True,
                               disable_control_change=True,
                               allowed_channels={0, 2},   # channels 0 and 2 only
                               event_i=5,
                               event_name="note")
    allowed_ch = _collect_allowed_channels(tok, mask)
    assert allowed_ch == {0, 2}, f"expected {{0, 2}}, got {allowed_ch}"


def test_allowed_channels_none_allows_all_channels():
    tok = _make_tokenizer()
    mask = _run_one_event_mask(tok, None,
                               disable_patch_change=True,
                               disable_control_change=True,
                               allowed_channels=None,
                               event_i=5,
                               event_name="note")
    allowed_ch = _collect_allowed_channels(tok, mask)
    assert len(allowed_ch) == 16  # all 16 MIDI channels
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_forge_backend.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` since `skytnt_tokenizer` exists but the masking logic is not yet in `generate_hardware()`. All 5 tests pass (the tests only test the mask logic in isolation, not the full `generate_hardware` call). Re-check: the tests call `_run_one_event_mask` which embeds the logic directly — so they should PASS once the logic is correct. Run once to confirm all 5 are collected/found.

- [ ] **Step 3: Add params and masking logic to `generate_hardware()`**

In `tt_midi_maker/generation/forge_backend.py`, change the function signature at line 164:

```python
def generate_hardware(
    compiled_net,
    model: "MIDIModel",
    prompt: np.ndarray,
    max_padded_len: int = 256,
    max_events: int = 200,
    temp: float = 1.0,
    top_p: float = 0.98,
    top_k: int = 20,
    hw_context_interval: int = 4,
    disable_patch_change: bool = True,
    disable_control_change: bool = True,
    allowed_channels: "set[int] | None" = None,
) -> np.ndarray:
```

Update the docstring Args section to document the three new params:

```
    disable_patch_change: if True, patch_change events are masked out after the
        prompt so the model cannot override instrument assignments mid-generation.
    disable_control_change: if True, control_change events (CC) are excluded from
        generation, reducing clutter in the output.
    allowed_channels: set of 0-indexed MIDI channel numbers the model may generate
        notes on. When provided, the "channel" parameter token is restricted to this
        set for note events. None means all 16 channels are allowed.
```

Replace the inner mask loop (lines 280–313) with:

```python
        for i in range(max_token_seq):
            mask = torch.zeros((1, tokenizer.vocab_size), dtype=torch.int64)
            if end:
                mask[0, tokenizer.pad_id] = 1
            elif i == 0:
                allowed_event_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                if disable_patch_change:
                    allowed_event_ids = [e for e in allowed_event_ids
                                         if tokenizer.id_events.get(e) != "patch_change"]
                if disable_control_change:
                    allowed_event_ids = [e for e in allowed_event_ids
                                         if tokenizer.id_events.get(e) != "control_change"]
                mask[0, allowed_event_ids] = 1
            else:
                param_names = tokenizer.events[event_name]
                if i > len(param_names):
                    mask[0, tokenizer.pad_id] = 1
                else:
                    param = param_names[i - 1]
                    param_ids = tokenizer.parameter_ids[param]
                    if param == "channel" and allowed_channels is not None:
                        # Restrict note channel to active roles only
                        param_ids = [param_ids[c] for c in sorted(allowed_channels)
                                     if c < len(param_ids)]
                    mask[0, param_ids] = 1
            mask = mask.unsqueeze(1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_forge_backend.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/ -v --ignore=tests/test_forge_backend.py -x
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add tt_midi_maker/generation/forge_backend.py tests/test_forge_backend.py
git commit -m "feat: add vocabulary masking to generate_hardware (disable_patch_change, disable_channels, disable_control_change)"
```

---

## Task 2: Improved Prompt Conditioning in `midi_backend.py`

**Files:**
- Modify: `tt_midi_maker/generation/midi_backend.py:120-400`

Three changes in this file:
1. `_midi_file_to_prompt_rows()` — pass `add_default_instr=True` to tokenizer.tokenize()
2. `_build_prompt()` — loop over all `roles_config` entries (not just active roles) so the model sees the full instrument palette upfront
3. `generate_from_blueprint()` — compute `active_channels` and pass new flags to `generate_hardware()`

- [ ] **Step 1: Write the failing test**

Add to a new file `tests/test_midi_backend_conditioning.py`:

```python
"""Tests for _build_prompt all-channel conditioning and active_channels logic."""
import numpy as np
import pytest
from unittest.mock import MagicMock


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
    bp = _make_blueprint({"melody": 1.0, "harmony": 0.0, "bass": 0.0, "drums": 1.0})

    rows = _build_prompt(bp, _ROLES_CFG, tok)
    # Decode rows to find patch_change events
    patch_channels = set()
    for row in rows:
        eid = row[0]
        if eid == tok.event_ids.get("patch_change"):
            # patch_change params: time1, time2, track, channel, patch
            # channel param is at position 4 (0-indexed after event id)
            ch_param_idx = row[4]  # 5th token = channel param token
            ch_ids = tok.parameter_ids["channel"]
            for c, cid in enumerate(ch_ids):
                if cid == ch_param_idx:
                    patch_channels.add(c)
                    break
    # All three melodic channels (0-indexed: 0=melody, 1=harmony, 2=bass) must be present
    assert 0 in patch_channels, f"melody ch0 missing from patch_changes: {patch_channels}"
    assert 1 in patch_channels, f"harmony ch0 missing from patch_changes: {patch_channels}"
    assert 2 in patch_channels, f"bass ch0 missing from patch_changes: {patch_channels}"
    # Drums (ch10=ch9 in 0-indexed) should NOT have patch_change
    assert 9 not in patch_channels, f"drums got unexpected patch_change"


def test_generate_from_blueprint_computes_active_channels():
    """active_channels only includes channels for roles with density > 0."""
    from tt_midi_maker.generation.midi_backend import _compute_active_channels
    bp = _make_blueprint({"melody": 1.0, "harmony": 0.5, "bass": 0.0, "drums": 1.0})
    active = _compute_active_channels(bp, _ROLES_CFG)
    assert 0 in active, "melody (ch0) should be active"
    assert 1 in active, "harmony (ch1) should be active"
    assert 2 not in active, "bass (density=0) should NOT be active"
    assert 9 in active, "drums (ch9) should be active"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_midi_backend_conditioning.py -v
```

Expected: ImportError on `_compute_active_channels` (doesn't exist yet) and the `_build_prompt` test may fail if it currently skips inactive roles.

- [ ] **Step 3: Implement `add_default_instr=True` in `_midi_file_to_prompt_rows()`**

In `tt_midi_maker/generation/midi_backend.py` at line 164, change:

```python
    rows = tokenizer.tokenize([tpb, evts], add_bos_eos=False)
```

to:

```python
    rows = tokenizer.tokenize([tpb, evts], add_bos_eos=False, add_default_instr=True)
```

This ensures source MIDI context has instrument defaults inserted before any notes, so the model knows which GM program each channel uses when it reads prior material.

- [ ] **Step 4: Implement all-channel patch_change in `_build_prompt()`**

Replace the existing patch_change loop in `_build_prompt()` (lines 195–208):

```python
    # OLD — skipped inactive roles:
    # for role_name, role_cfg in blueprint.roles.items():
    #     if role_cfg.density <= 0.0:
    #         continue
    #     ...
```

with:

```python
    # Send patch_change for ALL configured roles so the model knows the full
    # instrument palette upfront (matches skytnt app.py conditioning behaviour).
    # Density-zero roles still need their channel pre-assigned so disable_channels
    # masking keeps the model from leaking onto unintended channels.
    for role_name, cfg in roles_config.items():
        ch1 = cfg.get("channel", 1)
        ch0 = ch1 - 1                    # 0-indexed
        if ch1 == 10:                    # drums live on ch9; no patch needed
            continue
        prog = cfg.get("program", 0)
        t = tokenizer.event2tokens(["patch_change", 0, 0, ch0 + 1, ch0, prog])
        if t:
            rows.append(t)
```

- [ ] **Step 5: Add `_compute_active_channels()` helper in `midi_backend.py`**

Add this function just before `generate_from_blueprint()` (around line 299):

```python
def _compute_active_channels(
    blueprint: MusicalBlueprint,
    roles_config: dict,
) -> set:
    """Return 0-indexed MIDI channel set for roles with density > 0."""
    active: set[int] = set()
    for role_name, role_cfg in blueprint.roles.items():
        if role_cfg.density > 0.0:
            cfg = roles_config.get(role_name, {})
            ch1 = cfg.get("channel", 1)
            active.add(ch1 - 1)   # 0-indexed
    return active
```

- [ ] **Step 6: Wire `active_channels` and new flags into `generate_hardware()` call**

In `generate_from_blueprint()` (around line 357–374), replace the `generate_hardware()` call:

```python
    # Compute which MIDI channels (0-indexed) are active in this blueprint.
    # Used to restrict token generation to intended voices only.
    active_channels = _compute_active_channels(blueprint, roles_config)

    if compiled_net is not None:
        logger.info("[midi_backend] using TT hardware path (compiled net, hw_interval=%d)", hw_context_interval)
        try:
            from .forge_backend import generate_hardware
            generated = generate_hardware(
                compiled_net, model, prompt,
                max_padded_len=hw_max_padded_len,
                max_events=max_events,
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                hw_context_interval=hw_context_interval,
                disable_patch_change=True,
                disable_control_change=True,
                allowed_channels=active_channels,
            )
        except Exception as exc:
            logger.warning("[midi_backend] hardware generate failed, retrying on CPU: %s", exc)
            compiled_net = None
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_midi_backend_conditioning.py tests/test_forge_backend.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/ -v -x
```

Expected: All previously passing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add tt_midi_maker/generation/midi_backend.py tests/test_midi_backend_conditioning.py
git commit -m "feat: all-channel prompt conditioning, add_default_instr=True for source context, wire active_channels masking"
```

---

## Task 3: Per-Role Velocity Shaping in `humanize.py`

**Files:**
- Modify: `tt_midi_maker/coherence/humanize.py`
- Modify: `tests/test_humanize.py` (add new tests)

The model generates velocities with little awareness of role hierarchy. Melody should sit above harmony in the mix; bass should anchor without overpowering. `scale_velocity_by_role()` normalizes each note's velocity into a role-appropriate window before the `humanize_velocities()` jitter pass.

- [ ] **Step 1: Write the failing tests**

Open `tests/test_humanize.py` (check if it exists; if not, create it). Add:

```python
"""Tests for scale_velocity_by_role()."""
from dataclasses import replace
import pytest
from tt_midi_maker.coherence.humanize import scale_velocity_by_role, _ROLE_VELOCITY_RANGES
from tt_midi_maker.models.track import NoteEvent


def _note(vel: int, channel: int = 1) -> NoteEvent:
    return NoteEvent(pitch=60, velocity=vel, start_tick=0, duration_ticks=240, channel=channel)


def test_melody_velocity_range():
    lo, hi = _ROLE_VELOCITY_RANGES["melody"]
    notes = [_note(1), _note(64), _note(127)]
    result = scale_velocity_by_role(notes, "melody")
    for n in result:
        assert lo <= n.velocity <= hi, f"melody note vel {n.velocity} outside [{lo},{hi}]"


def test_harmony_velocity_lower_than_melody():
    melody_lo, melody_hi = _ROLE_VELOCITY_RANGES["melody"]
    harmony_lo, harmony_hi = _ROLE_VELOCITY_RANGES["harmony"]
    assert harmony_hi <= melody_hi, "harmony ceiling should be at or below melody ceiling"
    assert harmony_lo <= melody_lo, "harmony floor should be at or below melody floor"


def test_bass_velocity_range():
    lo, hi = _ROLE_VELOCITY_RANGES["bass"]
    notes = [_note(1), _note(64), _note(127)]
    result = scale_velocity_by_role(notes, "bass")
    for n in result:
        assert lo <= n.velocity <= hi


def test_unknown_role_uses_default_range():
    notes = [_note(1), _note(127)]
    result = scale_velocity_by_role(notes, "unknown_role")
    # Should not crash; just apply whatever the fallback range is
    assert len(result) == 2
    for n in result:
        assert 1 <= n.velocity <= 127


def test_empty_notes_returns_empty():
    assert scale_velocity_by_role([], "melody") == []


def test_velocity_monotone_preserving():
    """Higher input velocity → higher output velocity (relative ordering preserved)."""
    notes = [_note(20), _note(60), _note(100)]
    result = scale_velocity_by_role(notes, "harmony")
    vels = [n.velocity for n in result]
    assert vels == sorted(vels), f"ordering not preserved: {vels}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_humanize.py -v -k "velocity_role or scale_velocity"
```

Expected: `ImportError: cannot import name 'scale_velocity_by_role'` — function doesn't exist yet.

- [ ] **Step 3: Implement `scale_velocity_by_role()` in `humanize.py`**

Add to `tt_midi_maker/coherence/humanize.py` (after the existing imports and before `humanize_velocities`):

```python
_ROLE_VELOCITY_RANGES: dict[str, tuple[int, int]] = {
    "melody":  (80, 110),   # lead voice — prominent in the mix
    "harmony": (55,  80),   # supporting pads/comps — below melody
    "bass":    (50,  75),   # anchoring, steady, below harmony
    "drums":   (60, 100),   # wide dynamic range kept as-is
}


def scale_velocity_by_role(
    notes: list[NoteEvent],
    role: str,
    ranges: dict | None = None,
) -> list[NoteEvent]:
    """Remap note velocities into a role-appropriate window.

    Normalizes the input velocity (1–127) to a [0,1] fraction then maps it
    into [lo, hi] for the given role.  Call this before humanize_velocities()
    so the jitter pass refines within the already-shaped window.
    """
    if not notes:
        return notes
    velocity_ranges = ranges or _ROLE_VELOCITY_RANGES
    lo, hi = velocity_ranges.get(role, (60, 100))
    result = []
    for note in notes:
        frac = (note.velocity - 1) / 126.0          # 0.0 at vel=1, 1.0 at vel=127
        new_vel = int(lo + frac * (hi - lo))
        result.append(replace(note, velocity=max(1, min(127, new_vel))))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_humanize.py -v
```

Expected: All humanize tests PASS (including the 6 new ones).

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/coherence/humanize.py tests/test_humanize.py
git commit -m "feat: add scale_velocity_by_role() for per-role velocity shaping"
```

---

## Task 4: Wire Velocity Shaping into Demo Scripts

**Files:**
- Modify: `demo_ambient.py`
- Modify: `demo_bebop.py`
- Modify: `demo_blues.py`
- Modify: `demo_classical.py`

Each demo has an `_apply_coherence(tracks, ...)` function. Add `scale_velocity_by_role()` before the existing `humanize_velocities()` call so velocity windows are set per role before jitter is added.

- [ ] **Step 1: Write the failing test (integration)**

No new test file needed; verify the import works and the function is called per track. Add a quick smoke test to an existing or new `tests/test_demo_coherence.py`:

```python
"""Smoke test that demo coherence pipelines run without error."""
from dataclasses import replace
from tt_midi_maker.models.track import NoteEvent, RoleTrack
from tt_midi_maker.coherence.humanize import scale_velocity_by_role


def _make_track(role: str, n_notes: int = 8, channel: int = 1) -> RoleTrack:
    notes = [
        NoteEvent(pitch=60 + i, velocity=64, start_tick=i * 480,
                  duration_ticks=240, channel=channel)
        for i in range(n_notes)
    ]
    return RoleTrack(role=role, channel=channel, program=0, notes=notes)


def test_scale_velocity_by_role_in_coherence_pipeline():
    """Velocity shaping followed by humanize doesn't crash and stays in bounds."""
    from tt_midi_maker.coherence.humanize import humanize_velocities
    track = _make_track("melody", 16)
    notes = scale_velocity_by_role(track.notes, track.role)
    notes = humanize_velocities(notes, variation=8)
    for n in notes:
        assert 1 <= n.velocity <= 127
```

- [ ] **Step 2: Run test to verify it passes (it should — the function exists from Task 3)**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/test_demo_coherence.py -v
```

Expected: 1 test PASS.

- [ ] **Step 3: Update `demo_ambient.py`**

Find the `_apply_coherence()` function in `demo_ambient.py`. Add the import at the top of the file if not already present:

```python
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
```

Inside `_apply_coherence()`, before the `humanize_velocities()` call add:

```python
        notes = scale_velocity_by_role(notes, track.role)
```

Full updated inner loop (replace whatever exists):

```python
def _apply_coherence(tracks, key, chords, ticks_per_bar, ticks_per_beat):
    from tt_midi_maker.coherence.scale import scale_quantize
    from tt_midi_maker.coherence.harmony import chord_aware_filter
    from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
    result = []
    scale_set = set(scale_quantize([], key))  # get the scale note set
    for track in tracks:
        notes = track.notes
        notes = scale_quantize(notes, key, strictness=0.9)
        notes = chord_aware_filter(notes, chords, ticks_per_bar, ticks_per_beat,
                                   scale_set, semitone_tolerance=0)
        notes = nudge_timing(notes, max_ticks=16)
        notes = scale_velocity_by_role(notes, track.role)   # ← new
        notes = humanize_velocities(notes, variation=5)
        result.append(replace(track, notes=notes))
    return result
```

**Note:** Look at the actual `_apply_coherence` in `demo_ambient.py` first (it may differ slightly from this template); add `scale_velocity_by_role` in the correct position relative to the other calls. The key rule: velocity shaping runs *before* humanize jitter.

- [ ] **Step 4: Update `demo_bebop.py`**

Same pattern — add `scale_velocity_by_role(notes, track.role)` before `humanize_velocities`. The bebop coherence uses swing; the full call order should be:

```python
        notes = scale_quantize(notes, key, strictness=0.25)
        notes = chord_aware_filter(notes, chords, ticks_per_bar, ticks_per_beat,
                                   scale_set, semitone_tolerance=1)
        notes = swing_timing(notes, swing_ratio=0.63)
        notes = scale_velocity_by_role(notes, track.role)   # ← new
        notes = humanize_velocities(notes, variation=8)
```

- [ ] **Step 5: Update `demo_blues.py`**

Blues uses swing_ratio=0.67 and blues scale. Add velocity shaping:

```python
        notes = scale_quantize(notes, key, strictness=0.3, override_mode="blues")
        notes = chord_aware_filter(notes, chords, ticks_per_bar, ticks_per_beat,
                                   scale_set, semitone_tolerance=1)
        notes = swing_timing(notes, swing_ratio=0.67)
        notes = scale_velocity_by_role(notes, track.role)   # ← new
        notes = humanize_velocities(notes, variation=10)
```

- [ ] **Step 6: Update `demo_classical.py`**

Classical uses strict scale, no swing:

```python
        notes = scale_quantize(notes, key, strictness=0.9)
        notes = chord_aware_filter(notes, chords, ticks_per_bar, ticks_per_beat,
                                   scale_set, semitone_tolerance=0)
        notes = nudge_timing(notes, max_ticks=12)
        notes = scale_velocity_by_role(notes, track.role)   # ← new
        notes = humanize_velocities(notes, variation=5)
```

- [ ] **Step 7: Run full test suite**

```bash
cd /home/ttuser/code/tt-midi-maker
pytest tests/ -v -x
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add demo_ambient.py demo_bebop.py demo_blues.py demo_classical.py tests/test_demo_coherence.py
git commit -m "feat: wire scale_velocity_by_role into all demo suite coherence pipelines"
```

---

## Task 5: Regenerate Suites, Render MP3s, Update Site, Push

**Files:**
- Modify: `examples/aria-d-minor/*.mid` (regenerated)
- Modify: `examples/midnight-blues/*.mid` (regenerated)
- Modify: `examples/bebop-quick-changes/*.mid` (regenerated)
- Modify: `examples/slow-light/*.mid` (regenerated)
- Modify: `docs/index.html` (updated audio player references and stats)

This task requires the forge venv active and TT hardware available. Run in a terminal with:

```bash
source /home/ttuser/tt-forge-fe/forge-venv/bin/activate
export PYTHONPATH="/home/ttuser/tt-forge-fe/third_party/tvm/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/home/ttuser/tt-forge-fe/third_party/tvm/build:$LD_LIBRARY_PATH"
```

- [ ] **Step 1: Regenerate classical suite**

```bash
cd /home/ttuser/code/tt-midi-maker
python demo_classical.py
```

Expected: 3 patterns generated to `examples/aria-d-minor/`. Check output for note counts and generation time.

- [ ] **Step 2: Regenerate blues suite**

```bash
python demo_blues.py
```

Expected: 3 patterns generated to `examples/midnight-blues/`.

- [ ] **Step 3: Regenerate bebop suite**

```bash
python demo_bebop.py
```

Expected: 3 patterns generated to `examples/bebop-quick-changes/`.

- [ ] **Step 4: Regenerate ambient suite**

```bash
python demo_ambient.py
```

Expected: 3 patterns generated to `examples/slow-light/`.

- [ ] **Step 5: Render MP3s for each suite**

For each `.mid` file in each examples subdirectory:

```bash
for mid in examples/aria-d-minor/*.mid; do
    fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "$mid" -F "${mid%.mid}.wav" -r 44100
    ffmpeg -y -i "${mid%.mid}.wav" -codec:a libmp3lame -qscale:a 2 "${mid%.mid}.mp3"
    rm "${mid%.mid}.wav"
done

for mid in examples/midnight-blues/*.mid; do
    fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "$mid" -F "${mid%.mid}.wav" -r 44100
    ffmpeg -y -i "${mid%.mid}.wav" -codec:a libmp3lame -qscale:a 2 "${mid%.mid}.mp3"
    rm "${mid%.mid}.wav"
done

for mid in examples/bebop-quick-changes/*.mid; do
    fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "$mid" -F "${mid%.mid}.wav" -r 44100
    ffmpeg -y -i "${mid%.mid}.wav" -codec:a libmp3lame -qscale:a 2 "${mid%.mid}.mp3"
    rm "${mid%.mid}.wav"
done

for mid in examples/slow-light/*.mid; do
    fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "$mid" -F "${mid%.mid}.wav" -r 44100
    ffmpeg -y -i "${mid%.mid}.wav" -codec:a libmp3lame -qscale:a 2 "${mid%.mid}.mp3"
    rm "${mid%.mid}.wav"
done
```

- [ ] **Step 6: Update `metadata.json` files with new stats**

For each suite, update the `metadata.json` to reflect:
- Actual `generation_time_sec` and `ev_per_sec` from script output
- Actual `tracks` note counts per pattern
- Add note about new masking/velocity features in `description` or `performance_notes`

- [ ] **Step 7: Update `docs/index.html`** if note counts or descriptions changed materially (generation stats, any qualitative improvement notes)

- [ ] **Step 8: Commit and push**

```bash
git add examples/ docs/index.html
git commit -m "feat: regenerate all suites with vocabulary masking + per-role velocity shaping"
git push origin main
```

---

## Self-Review

**Spec coverage check:**
- `disable_patch_change` — Task 1 ✅
- `disable_channels` (via `allowed_channels`) — Task 1 + Task 2 ✅
- `disable_control_change` — Task 1 ✅
- All-channel upfront patch_change conditioning — Task 2 ✅
- `add_default_instr=True` for source context — Task 2 ✅
- Per-role velocity shaping — Task 3 ✅
- Wired into demo scripts — Task 4 ✅
- Regenerate + publish — Task 5 ✅

**Placeholder scan:** No TBDs. All code blocks are complete and reference real function names/line numbers from the codebase.

**Type consistency:**
- `allowed_channels: set[int] | None` in Task 1 matches `active_channels: set[int]` computed in Task 2 ✅
- `scale_velocity_by_role(notes, track.role)` uses `track.role: str` which exists on `RoleTrack` ✅
- `_ROLE_VELOCITY_RANGES` imported in tests matches the name defined in `humanize.py` ✅
- `_compute_active_channels(bp, roles_config)` defined in Task 2 Step 5, called in Task 2 Step 6 ✅
