"""
Tests for tt_midi_maker.generation.forge_backend.

All tests run on CPU without forge or TT hardware — forge.compile is monkey-
patched so CI passes everywhere.  Structural correctness (shapes, event types,
EOS handling) is verified independently of actual hardware compilation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_model():
    """Return a tiny (1-layer) MIDIModel with random weights for fast tests."""
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
    return model


def _fake_compiled_net(model, max_padded_len):
    """Return a callable that mimics a forge CompiledModel using the real net."""
    class _FakeCompiled:
        def __init__(self, net, length):
            self._net = net
            self._len = length

        def __call__(self, x_emb):
            # x_emb: (1, max_padded_len, hidden_size)
            with torch.no_grad():
                out = self._net.forward(inputs_embeds=x_emb, use_cache=False)
            return [out.last_hidden_state]  # forge returns a list

    return _FakeCompiled(model.net, max_padded_len)


# ── _NetWrapper ───────────────────────────────────────────────────────────────

def test_net_wrapper_output_shape():
    from tt_midi_maker.generation.forge_backend import _NetWrapper
    model = _make_tiny_model()
    wrapper = _NetWrapper(model.net)
    hidden_size = model.net.config.hidden_size
    x = torch.zeros(1, 16, hidden_size)
    with torch.no_grad():
        out = wrapper(x)
    assert out.shape == (1, 16, hidden_size)


def test_net_wrapper_is_scriptable():
    """_NetWrapper must be a plain nn.Module (no forward-only quirks)."""
    from tt_midi_maker.generation.forge_backend import _NetWrapper
    model = _make_tiny_model()
    wrapper = _NetWrapper(model.net)
    assert isinstance(wrapper, torch.nn.Module)


# ── compile_for_hardware (mocked) ─────────────────────────────────────────────

def test_compile_for_hardware_caches(monkeypatch):
    """Second call with same max_padded_len returns cached result."""
    import tt_midi_maker.generation.forge_backend as fb

    call_count = [0]

    def fake_compile(module, inputs, module_name=""):
        call_count[0] += 1
        return object()

    def fake_import():
        class _M:
            compile = staticmethod(fake_compile)
        return _M()

    monkeypatch.setattr(fb, "_import_forge", fake_import)
    monkeypatch.setattr(fb, "_compiled_cache", None)

    model = _make_tiny_model()
    fb.compile_for_hardware(model, max_padded_len=32)
    fb.compile_for_hardware(model, max_padded_len=32)   # should hit cache

    assert call_count[0] == 1   # forge.compile called only once


def test_compile_for_hardware_recompiles_on_different_len(monkeypatch):
    """Different max_padded_len triggers a new compile."""
    import tt_midi_maker.generation.forge_backend as fb

    call_count = [0]

    def fake_compile(module, inputs, module_name=""):
        call_count[0] += 1
        return object()

    def fake_import():
        class _M:
            compile = staticmethod(fake_compile)
        return _M()

    monkeypatch.setattr(fb, "_import_forge", fake_import)
    monkeypatch.setattr(fb, "_compiled_cache", None)

    model = _make_tiny_model()
    fb.compile_for_hardware(model, max_padded_len=32)
    fb.compile_for_hardware(model, max_padded_len=64)   # different len → recompile

    assert call_count[0] == 2


def test_compile_for_hardware_raises_on_missing_forge(monkeypatch):
    """RuntimeError raised when forge cannot be imported."""
    import tt_midi_maker.generation.forge_backend as fb

    def bad_import():
        raise RuntimeError("forge not available")

    monkeypatch.setattr(fb, "_import_forge", bad_import)
    monkeypatch.setattr(fb, "_compiled_cache", None)

    model = _make_tiny_model()
    with pytest.raises(RuntimeError):
        fb.compile_for_hardware(model, max_padded_len=16)


# ── generate_hardware ─────────────────────────────────────────────────────────

def _build_prompt_array(tok, n_events=2):
    """Build a minimal prompt: BOS + n note events."""
    bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
    rows = [bos]
    for i in range(n_events):
        rows.append(tok.event2tokens(["note", i, 0, 0, 8, 0, 60 + i, 80]))
    return np.array(rows, dtype=np.int64)


def test_generate_hardware_output_shape():
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    prompt = _build_prompt_array(tok, n_events=2)
    compiled = _fake_compiled_net(model, max_padded_len=32)

    out = generate_hardware(compiled, model, prompt, max_padded_len=32, max_events=4)

    assert out.ndim == 3
    assert out.shape[0] == 1                    # batch=1
    assert out.shape[2] == tok.max_token_seq    # token seq dim
    assert out.shape[1] >= prompt.shape[0]      # grew past prompt


def test_generate_hardware_prompt_2d_accepted():
    """2-D prompt (no batch dim) is accepted and produces valid output."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    prompt2d = _build_prompt_array(tok, n_events=2)   # (seq, token_seq)
    compiled = _fake_compiled_net(model, max_padded_len=32)

    out = generate_hardware(compiled, model, prompt2d, max_padded_len=32, max_events=4)

    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[2] == tok.max_token_seq
    assert out.shape[1] >= prompt2d.shape[0]   # at least as long as prompt


def test_generate_hardware_sliding_window():
    """Sliding window: generation does not crash when prompt already exceeds max_padded_len."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    # Prompt larger than the compiled window forces sliding-window logic from step 0.
    prompt = _build_prompt_array(tok, n_events=34)   # 35 rows > max_padded_len=32
    compiled = _fake_compiled_net(model, max_padded_len=32)

    # Must not raise; output must be at least as long as the prompt.
    out = generate_hardware(compiled, model, prompt, max_padded_len=32, max_events=4)
    assert out.shape[1] >= prompt.shape[0]
    assert out.shape[2] == tok.max_token_seq


def test_generate_hardware_forge_list_output():
    """generate_hardware handles forge returning a list of tensors (not a raw tensor)."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    prompt = _build_prompt_array(tok, n_events=2)
    compiled = _fake_compiled_net(model, max_padded_len=32)
    # _fake_compiled_net already returns [tensor] — this just confirms shape
    out = generate_hardware(compiled, model, prompt, max_padded_len=32, max_events=2)
    assert out.shape[2] == tok.max_token_seq


def test_generate_hardware_context_interval_reduces_hw_calls():
    """hw_context_interval=N bounds hardware calls to at most max_events//N + 1."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    prompt = _build_prompt_array(tok, n_events=2)
    inner = _fake_compiled_net(model, max_padded_len=32)

    hw_context_interval = 4
    max_events = 8

    call_count = [0]

    class _CountingNet:
        def __call__(self, x_emb):
            call_count[0] += 1
            return inner(x_emb)

    generate_hardware(
        _CountingNet(), model, prompt,
        max_padded_len=32, max_events=max_events, hw_context_interval=hw_context_interval,
    )

    # With interval=4 and max_events=8 the hardware fires at event_count 0 and 4
    # (and at most once for an EOS attempt).  ceil((max_events+1)/interval) = 3.
    upper = max_events // hw_context_interval + 1
    assert 1 <= call_count[0] <= upper, (
        f"interval={hw_context_interval}: {call_count[0]} hw calls, expected 1–{upper}"
    )
    # Sanity: must be strictly fewer than max_events (proves we're not calling hw every step)
    assert call_count[0] < max_events


def test_generate_hardware_context_interval_1_calls_hw_every_step():
    """hw_context_interval=1 calls hardware once per generation attempt (every step)."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    prompt = _build_prompt_array(tok, n_events=2)
    inner = _fake_compiled_net(model, max_padded_len=32)

    call_count = [0]

    class _CountingNet:
        def __call__(self, x_emb):
            call_count[0] += 1
            return inner(x_emb)

    max_events = 4
    out = generate_hardware(
        _CountingNet(), model, prompt,
        max_padded_len=32, max_events=max_events, hw_context_interval=1,
    )
    events_generated = out.shape[1] - prompt.shape[0]
    # hw fires once per outer-loop iteration: once per generated event, plus
    # one more if EOS terminates the loop (the EOS attempt also calls hw).
    assert events_generated <= call_count[0] <= events_generated + 1, (
        f"expected {events_generated}–{events_generated + 1} hw calls, got {call_count[0]}"
    )


def test_generate_hardware_fresh_buffer_fixes_stride_mismatch():
    """Sliding window delivers a contiguous buffer (new_zeros+copy_, not a slice view)."""
    from tt_midi_maker.generation.forge_backend import generate_hardware
    model = _make_tiny_model()
    tok = model.tokenizer
    hidden_size = model.net.config.hidden_size
    max_padded_len = 32
    # 29-token prompt → sliding window kicks in once cur_len exceeds 32
    prompt = _build_prompt_array(tok, n_events=28)   # 29 rows incl BOS
    inner = _fake_compiled_net(model, max_padded_len=max_padded_len)

    strides_seen = []

    class _StrideCheckNet:
        def __call__(self, x_emb):
            strides_seen.append(x_emb.stride())
            return inner(x_emb)

    generate_hardware(
        _StrideCheckNet(), model, prompt,
        max_padded_len=max_padded_len, max_events=8, hw_context_interval=1,
    )

    # Every buffer passed to hardware must have strides of a contiguous
    # (1, max_padded_len, hidden_size) tensor: (max_padded_len*hidden, hidden, 1)
    expected_outer = max_padded_len * hidden_size
    for i, s in enumerate(strides_seen):
        assert s[0] == expected_outer, (
            f"call {i}: outer stride {s[0]} != {expected_outer} (non-contiguous view)"
        )
        assert s[1] == hidden_size, f"call {i}: seq stride {s[1]} != {hidden_size}"
        assert s[2] == 1, f"call {i}: elem stride {s[2]} != 1"


# ── midi_backend integration (no hardware required) ───────────────────────────

def test_get_compiled_net_no_devices(monkeypatch):
    """_get_compiled_net returns None when no TT devices found."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation import hardware

    monkeypatch.setattr(hardware, "detect_tt_devices", lambda: [])
    monkeypatch.setattr(midi_backend, "_hw_model_cache", None)

    model = _make_tiny_model()
    result = midi_backend._get_compiled_net(model, max_padded_len=32)
    assert result is None


def test_get_compiled_net_forge_failure(monkeypatch):
    """_get_compiled_net returns None and logs warning when forge fails."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation import hardware
    import tt_midi_maker.generation.forge_backend as fb

    monkeypatch.setattr(hardware, "detect_tt_devices", lambda: [0])
    monkeypatch.setattr(midi_backend, "_hw_model_cache", None)
    monkeypatch.setattr(fb, "compile_for_hardware",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no forge")))

    model = _make_tiny_model()
    result = midi_backend._get_compiled_net(model, max_padded_len=32)
    assert result is None


def test_generate_from_blueprint_uses_hardware_when_available(monkeypatch):
    """generate_from_blueprint routes through hardware path when devices present."""
    from tt_midi_maker.generation import midi_backend
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    from tt_midi_maker.models.blueprint import RoleConfig, MusicalBlueprint

    tok = MIDITokenizerV1()

    # Fake model with minimal generate() (CPU)
    class FakeModel:
        tokenizer = tok

        def generate(self, prompt, batch_size=1, max_len=16, **kw):
            bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
            arr = np.array([[bos, eos]], dtype=np.int64)
            return arr

        def parameters(self):
            return iter([torch.zeros(1)])

    hw_called = [False]

    def fake_generate_hardware(compiled_net, model, prompt, **kw):
        hw_called[0] = True
        bos = [tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
        eos = [tok.eos_id] + [tok.pad_id] * (tok.max_token_seq - 1)
        return np.array([[bos, eos]], dtype=np.int64)

    import tt_midi_maker.generation.forge_backend as fb
    monkeypatch.setattr(midi_backend, "_model_cache", (FakeModel(), tok))
    monkeypatch.setattr(midi_backend, "_hw_model_cache", (object(), 256))
    monkeypatch.setattr(fb, "generate_hardware", fake_generate_hardware)

    bp = MusicalBlueprint(
        key="C major", bpm=120, bars=4,
        style="test", mode="loop",
        chord_progression=["I"],
        roles={"melody": RoleConfig(density=1.0)},
    )
    midi_backend.generate_from_blueprint(bp, {"melody": {"channel": 1, "program": 0,
                                                          "note_range": [60, 96],
                                                          "density_default": 1.0}})
    assert hw_called[0], "hardware generate_hardware was not called"


# ── Vocabulary masking tests ──────────────────────────────────────────────────

def _make_tokenizer():
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    return MIDITokenizerV1()


def _collect_allowed_events(tokenizer, mask_1d: torch.Tensor) -> set:
    """Return set of event names whose event-id token is allowed in mask."""
    allowed = set()
    for event_name, eid in tokenizer.event_ids.items():
        if mask_1d[eid].item():
            allowed.add(event_name)
    return allowed


def _collect_allowed_channels(tokenizer, mask_1d: torch.Tensor) -> set:
    """Return set of channel values (0-indexed) whose token is allowed."""
    ch_ids = tokenizer.parameter_ids["channel"]
    return {c for c, tok in enumerate(ch_ids) if mask_1d[tok].item()}


def _build_mask(tokenizer, end, i, event_name, disable_patch_change,
                disable_control_change, allowed_channels):
    """Reproduce the mask-building logic from the generate_hardware inner loop."""
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
                param_ids = [param_ids[c] for c in sorted(allowed_channels)
                             if c < len(param_ids)]
            mask[0, param_ids] = 1
    return mask[0]


def test_disable_patch_change_removes_patch_change_from_event_mask():
    tok = _make_tokenizer()
    mask = _build_mask(tok, end=False, i=0, event_name="",
                       disable_patch_change=True, disable_control_change=False,
                       allowed_channels=None)
    allowed = _collect_allowed_events(tok, mask)
    assert "patch_change" not in allowed
    assert "note" in allowed
    assert "set_tempo" in allowed


def test_disable_control_change_removes_cc_from_event_mask():
    tok = _make_tokenizer()
    mask = _build_mask(tok, end=False, i=0, event_name="",
                       disable_patch_change=False, disable_control_change=True,
                       allowed_channels=None)
    allowed = _collect_allowed_events(tok, mask)
    assert "control_change" not in allowed
    assert "note" in allowed


def test_no_disable_flags_allows_all_events():
    tok = _make_tokenizer()
    mask = _build_mask(tok, end=False, i=0, event_name="",
                       disable_patch_change=False, disable_control_change=False,
                       allowed_channels=None)
    allowed = _collect_allowed_events(tok, mask)
    assert "patch_change" in allowed
    assert "control_change" in allowed
    assert "note" in allowed


def test_allowed_channels_restricts_note_channel_param():
    tok = _make_tokenizer()
    # "note" params: ["time1","time2","track","duration","channel","pitch","velocity"]
    # "channel" is at index 4 → appears at i=5
    mask = _build_mask(tok, end=False, i=5, event_name="note",
                       disable_patch_change=True, disable_control_change=True,
                       allowed_channels={0, 2})
    allowed_ch = _collect_allowed_channels(tok, mask)
    assert allowed_ch == {0, 2}, f"expected {{0, 2}}, got {allowed_ch}"


def test_allowed_channels_none_allows_all_channels():
    tok = _make_tokenizer()
    mask = _build_mask(tok, end=False, i=5, event_name="note",
                       disable_patch_change=True, disable_control_change=True,
                       allowed_channels=None)
    allowed_ch = _collect_allowed_channels(tok, mask)
    assert len(allowed_ch) == 16


def test_generate_hardware_respects_disable_patch_change(monkeypatch):
    """Verify generate_hardware() itself masks patch_change when flag is True."""
    from tt_midi_maker.generation.skytnt_tokenizer import MIDITokenizerV1
    from tt_midi_maker.generation import forge_backend

    tok = MIDITokenizerV1()
    patch_change_id = tok.event_ids["patch_change"]
    sampled_event_masks = []

    # Minimal fake model
    import types, torch.nn as nn

    class FakeNetModule(nn.Module):
        def __init__(self):
            super().__init__()
            # generate_hardware calls model.net.embed_tokens(input_tensor).sum(dim=-2)
            # input_tensor shape: (1, seq, max_token_seq) — embed_tokens must map
            # token IDs to floats.  We use hidden_size=32 to keep the test light.
            self.config = types.SimpleNamespace(hidden_size=32, num_hidden_layers=1)
            self.embed_tokens = nn.Embedding(tok.vocab_size, 32)

        def forward(self, inputs_embeds=None, use_cache=False):
            b, s, h = inputs_embeds.shape
            out = types.SimpleNamespace(last_hidden_state=inputs_embeds)
            return out

    class FakeModel:
        def __init__(self):
            self.tokenizer = tok
            self._net_module = FakeNetModule()

        @property
        def net(self):
            return self._net_module

        @net.setter
        def net(self, v):
            self._net_module = v

        def forward_token(self, h_in, x_tok, cache=None):
            # Return uniform logits — sampling will pick argmax of the masked scores
            return torch.ones(1, 1, tok.vocab_size)

        def sample_top_p_k(self, scores, top_p, top_k):
            # Record the mask that was applied (scores = softmax * mask, so nonzero = allowed)
            event_i_mask = (scores[0, 0] > 0)
            sampled_event_masks.append(event_i_mask.clone())
            # Return the EOS token to end generation quickly
            return torch.tensor([[tok.eos_id]])

    # Fake compiled_net: returns zeros of the right shape
    class FakeCompiledNet:
        def __call__(self, x_emb):
            return x_emb  # pass-through; shape matches hidden expectations

    model = FakeModel()
    compiled_net = FakeCompiledNet()

    # Build a minimal prompt: just BOS
    prompt = np.array([[tok.bos_id] + [tok.pad_id] * (tok.max_token_seq - 1)], dtype=np.int64)

    # Run generate_hardware with disable_patch_change=True.
    # sample_top_p_k returns EOS at i==0, recording one mask then stopping.
    from tt_midi_maker.generation.forge_backend import generate_hardware
    generate_hardware(
        compiled_net, model, prompt,
        max_padded_len=16,
        max_events=1,
        disable_patch_change=True,
        disable_control_change=False,
        allowed_channels=None,
    )

    assert len(sampled_event_masks) >= 1, "sample_top_p_k was never called"
    event_mask = sampled_event_masks[0]
    assert not event_mask[patch_change_id].item(), (
        f"patch_change token {patch_change_id} was allowed in the mask "
        f"even though disable_patch_change=True"
    )
    # Verify note events ARE still allowed
    note_id = tok.event_ids["note"]
    assert event_mask[note_id].item(), "note token should still be allowed"
