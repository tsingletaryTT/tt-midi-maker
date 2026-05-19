"""
Hardware-accelerated MIDI generation using tt-forge on Tenstorrent P300C chips.

The 12-layer LlamaModel (model.net) is compiled with forge.compile and runs on
TT hardware.  The 3-layer token-prediction head (model.net_token) stays on CPU
because it is small (~50 M params) and uses DynamicCache which changes shape
per token — not directly compilable.

Each generation step produces one MIDI event token-sequence.  The hardware
context vector (hidden state at the last real position) is refreshed every
hw_context_interval events to amortise the ~500 ms per-call PCIe + dispatch
overhead across multiple CPU net_token steps.  hw_context_interval=1 gives the
original behaviour (hardware on every step); hw_context_interval=4 (the
default) reduces hardware calls by 4× while preserving musical coherence.

Architecture split:
  - compiled_net  (hardware, called every hw_context_interval events)
      input:  (1, max_padded_len, hidden_size) float32
      output: (1, max_padded_len, hidden_size) float32
  - model.net_token  (CPU, called every step for token prediction)
      input:  hidden (1, hidden_size) + optional prior tokens with DynamicCache
      output: logits (1, 1, vocab_size)

Sliding-window: when sequence length > max_padded_len the last max_padded_len
embeddings are used and the position index points at the final slot.  The model
loses very distant context but the compiled shape stays fixed.  A fresh tensor
(new_zeros + copy_) is used instead of a slice view so that forge receives a
buffer whose strides exactly match max_padded_len × hidden_size.  Slice views
carry the parent tensor's outer stride, which triggers forge's stride-mismatch
check when parent_rows != max_padded_len.

Activation requirements (environment):
    source /home/ttuser/tt-forge-fe/forge-venv/bin/activate
    export PYTHONPATH="/home/ttuser/tt-forge-fe/third_party/tvm/python:$PYTHONPATH"
    export LD_LIBRARY_PATH="/home/ttuser/tt-forge-fe/third_party/tvm/build:$LD_LIBRARY_PATH"

If the forge import fails (e.g. running outside the venv) the functions here
raise RuntimeError, and midi_backend.py falls back to the CPU path.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

if TYPE_CHECKING:
    from .skytnt_model import MIDIModel

logger = logging.getLogger(__name__)

_FORGE_VENV    = "/home/ttuser/tt-forge-fe/forge-venv/bin/activate"
_TVM_PYTHON    = "/home/ttuser/tt-forge-fe/third_party/tvm/python"
_TVM_BUILD_LIB = "/home/ttuser/tt-forge-fe/third_party/tvm/build"

# Lazily compiled singleton: (compiled_net, max_padded_len)
_compiled_cache: tuple | None = None


# ── Torch wrapper ─────────────────────────────────────────────────────────────

class _NetWrapper(torch.nn.Module):
    """Thin wrapper around MIDIModel.net for forge.compile.

    Accepts pre-embedded+summed float input so that the token embedding
    lookup (which references CPU tensors) stays on CPU.  Only the heavy
    12-layer LlamaModel forward pass is compiled for TT hardware.

    Input:  x_emb  (1, seq_len, hidden_size) float32
    Output: last_hidden_state  (1, seq_len, hidden_size) float32
    """

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        return self.net.forward(inputs_embeds=x_emb, use_cache=False).last_hidden_state


# ── Environment helpers ───────────────────────────────────────────────────────

def _ensure_forge_paths() -> None:
    """Inject TVM source-build paths into sys.path / LD_LIBRARY_PATH."""
    import sys
    if _TVM_PYTHON not in sys.path:
        sys.path.insert(0, _TVM_PYTHON)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _TVM_BUILD_LIB not in ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_TVM_BUILD_LIB}:{ld}"


def _import_forge():
    """Import forge, injecting paths first.  Raises RuntimeError on failure."""
    _ensure_forge_paths()
    try:
        import forge as _forge
        return _forge
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import forge. Activate the forge venv and set PYTHONPATH:\n"
            f"  source {_FORGE_VENV}\n"
            f"  export PYTHONPATH={_TVM_PYTHON}:$PYTHONPATH\n"
            f"  export LD_LIBRARY_PATH={_TVM_BUILD_LIB}:$LD_LIBRARY_PATH"
        ) from exc


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_for_hardware(model: "MIDIModel", max_padded_len: int = 256) -> object:
    """Compile model.net with forge for TT hardware.

    The compiled model accepts input of shape (1, max_padded_len, hidden_size).
    Caches the result globally; subsequent calls with the same max_padded_len
    return the cached compiled model instantly.

    Args:
        model: loaded MIDIModel (weights already on CPU).
        max_padded_len: fixed sequence length to compile for.  Each generation
            step pads the embedded sequence to this length.

    Returns:
        forge CompiledModel object (callable as compiled_net(x_emb)).

    Raises:
        RuntimeError: if forge is not importable.
    """
    global _compiled_cache
    if _compiled_cache is not None:
        compiled, cached_len = _compiled_cache
        if cached_len == max_padded_len:
            logger.debug("[forge_backend] reusing cached compiled net (len=%d)", max_padded_len)
            return compiled

    forge = _import_forge()

    hidden_size = model.net.config.hidden_size
    wrapper = _NetWrapper(model.net).eval()
    sample = torch.zeros(1, max_padded_len, hidden_size, dtype=torch.float32)

    logger.info(
        "[forge_backend] compiling net — shape=(1, %d, %d), layers=%d",
        max_padded_len, hidden_size, model.net.config.num_hidden_layers,
    )
    compiled = forge.compile(wrapper, [sample], module_name="midi_net")
    logger.info("[forge_backend] compile complete: %s", type(compiled).__name__)

    _compiled_cache = (compiled, max_padded_len)
    return compiled


def reset_compiled() -> None:
    """Clear the compiled model cache (useful for testing / re-compilation)."""
    global _compiled_cache
    _compiled_cache = None


# ── Generation ────────────────────────────────────────────────────────────────

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
    """Generate MIDI events using TT-compiled net + CPU net_token.

    The 12-layer net runs on TT hardware every hw_context_interval events to
    refresh the context vector (hidden state at the last real token position).
    The 3-layer net_token token-prediction loop runs on CPU at every step,
    conditioned on the most recently computed hardware context vector.

    Reusing the context vector across hw_context_interval steps amortises the
    ~500 ms per-call P300C overhead, yielding proportionally more events per
    second.  hw_context_interval=4 gives ~5 ev/s on P300C vs ~2 ev/s with
    interval=1.  Musical coherence is preserved because the context refreshes
    frequently enough to track the evolving sequence.

    hw_context_interval=1 reproduces original behaviour (hardware every step).

    When the sequence exceeds max_padded_len a trailing window of max_padded_len
    tokens is used.  .clone() ensures forge receives a fresh contiguous buffer
    (avoids the stride-mismatch error that occurs with .contiguous() on sliced
    views when the parent tensor size does not match the slice stride).

    Args:
        compiled_net: output of compile_for_hardware().
        model: MIDIModel instance (net_token used for CPU token prediction).
        prompt: (seq_len, max_token_seq) or (1, seq_len, max_token_seq) int64.
        max_padded_len: must match what was used in compile_for_hardware().
        max_events: maximum new MIDI events to generate beyond the prompt.
        temp / top_p / top_k: sampling hyperparameters.
        hw_context_interval: hardware is called every this many generated events.
            1 = every step (original); 4 = every 4th step (default, 4× speedup).
        disable_patch_change: if True, patch_change events are masked out after the
            prompt so the model cannot override instrument assignments mid-generation.
        disable_control_change: if True, control_change events (CC) are excluded from
            generation, reducing clutter in the output.
        allowed_channels: set of 0-indexed MIDI channel numbers the model may generate
            notes on. When provided, the "channel" parameter token is restricted to
            this set for note events. None means all 16 channels are allowed.

    Returns:
        (1, final_seq_len, max_token_seq) int64 numpy array.
    """
    tokenizer = model.tokenizer
    max_token_seq = tokenizer.max_token_seq

    # Normalise prompt → (1, prompt_len, max_token_seq)
    if prompt.ndim == 2:
        prompt = prompt[None, :]
    prompt = prompt[..., :max_token_seq]
    if prompt.shape[-1] < max_token_seq:
        prompt = np.pad(
            prompt,
            ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
            mode="constant", constant_values=tokenizer.pad_id,
        )

    input_tensor = torch.from_numpy(prompt).long()   # (1, prompt_len, max_token_seq)
    cur_len = input_tensor.shape[1]
    max_len = cur_len + max_events

    logger.info(
        "[forge_backend] generate: prompt_len=%d, max_len=%d, max_padded_len=%d, hw_interval=%d",
        cur_len, max_len, max_padded_len, hw_context_interval,
    )

    hidden: torch.Tensor | None = None
    event_count = 0   # events generated so far; gates hardware-call schedule

    while cur_len < max_len:
        # ── Hardware context refresh ──────────────────────────────────────────
        # Always runs at event_count=0 (first event) and every hw_context_interval
        # events thereafter (0, interval, 2*interval, …).
        if hidden is None or event_count % hw_context_interval == 0:
            with torch.no_grad():
                # embed_tokens: (1, seq, max_token_seq) → (1, seq, max_token_seq, hidden)
                # sum over max_token_seq → (1, seq, hidden)
                x_emb = model.net.embed_tokens(input_tensor).sum(dim=-2)

            if cur_len > max_padded_len:
                # Sliding window: last max_padded_len tokens.
                # Allocate a fresh contiguous buffer and copy the slice into it.
                # Slices carry the parent tensor's outer stride (parent_rows *
                # hidden_size), so neither .contiguous() nor .clone() reliably
                # produces a tensor whose strides match the slice's own shape —
                # forge's stride-mismatch check compares stride[0] against
                # max_padded_len * hidden_size and rejects parent-size strides.
                hidden_sz = x_emb.shape[-1]
                x_emb_padded = x_emb.new_zeros(1, max_padded_len, hidden_sz)
                x_emb_padded[0].copy_(x_emb[0, -max_padded_len:])
                pos = max_padded_len - 1
            else:
                pad_len = max_padded_len - cur_len
                x_emb_padded = F.pad(x_emb, (0, 0, 0, pad_len))
                pos = cur_len - 1   # last real token position

            # Forward through hardware-compiled 12-layer net
            hw_out = compiled_net(x_emb_padded)
            if isinstance(hw_out, (list, tuple)):
                hw_out = hw_out[0]
            if not isinstance(hw_out, torch.Tensor):
                hw_out = torch.as_tensor(hw_out)

            hidden = hw_out[:, pos, :]    # (1, hidden_size) — context vector
            logger.debug(
                "[forge_backend] hw context updated at event %d / step %d",
                event_count, cur_len,
            )

        # ── Token-level prediction on CPU (3-layer net_token) ─────────────────
        next_token_seq = None
        event_name = ""
        end = False
        cache2 = DynamicCache()

        for i in range(max_token_seq):
            mask = torch.zeros((1, tokenizer.vocab_size), dtype=torch.int64)
            if end:
                mask[0, tokenizer.pad_id] = 1
            elif i == 0:
                # Build the allowed event-id list, optionally excluding
                # patch_change and control_change to keep the model on task.
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
                        # Restrict note channel to active roles only; prevents
                        # the model from bleeding onto unassigned MIDI channels.
                        param_ids = [param_ids[c] for c in sorted(allowed_channels)
                                     if c < len(param_ids)]
                    mask[0, param_ids] = 1
            mask = mask.unsqueeze(1)

            x_tok = None if i == 0 else next_token_seq[:, -1:]
            h_in  = hidden if i == 0 else None

            with torch.no_grad():
                logits = model.forward_token(h_in, x_tok, cache=cache2)[:, -1:]

            scores = torch.softmax(logits / temp, dim=-1) * mask
            sample = model.sample_top_p_k(scores, top_p, top_k)

            if i == 0:
                next_token_seq = sample
                eid = sample[0].item()
                if eid == tokenizer.eos_id:
                    end = True
                    break   # stop inner token loop; outer loop will break on end
                event_name = tokenizer.id_events[eid]
            else:
                next_token_seq = torch.cat([next_token_seq, sample], dim=1)
                if len(tokenizer.events[event_name]) == i:
                    break

        if end:
            logger.debug("[forge_backend] EOS at cur_len=%d", cur_len)
            break

        # Pad to max_token_seq and append to sequence
        if next_token_seq.shape[1] < max_token_seq:
            next_token_seq = F.pad(
                next_token_seq,
                (0, max_token_seq - next_token_seq.shape[1]),
                "constant", value=tokenizer.pad_id,
            )
        input_tensor = torch.cat([input_tensor, next_token_seq.unsqueeze(1)], dim=1)
        cur_len += 1
        event_count += 1

    logger.info("[forge_backend] done: generated %d events", cur_len - prompt.shape[1])
    return input_tensor.cpu().numpy()
