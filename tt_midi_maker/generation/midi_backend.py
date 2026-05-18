"""
Orchestration layer: blueprint → skytnt/midi-model → RoleTracks.

CPU path: loads skytnt/midi-model weights from HuggingFace Hub.
TT hardware path (Phase 2): forge-compiled decode steps on P300C cards.

Model weights are cached after first load. To force a reload, call reset_model().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..models.track import NoteEvent, RoleTrack

if TYPE_CHECKING:
    from ..models.blueprint import MusicalBlueprint

logger = logging.getLogger(__name__)

_MODEL_ID = "skytnt/midi-model"
_TICKS_PER_BEAT = 480

# Lazy-loaded singleton: (MIDIModel, tokenizer)
_model_cache: tuple | None = None


def reset_model() -> None:
    """Clear the cached model (useful for testing)."""
    global _model_cache
    _model_cache = None


def _get_model():
    """Load model on first call and cache it.  Always runs on CPU; hardware
    compilation is wired in a separate step (requires user confirmation)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    from .skytnt_model import MIDIModel, MIDIModelConfig
    from transformers import AutoConfig

    logger.info("[midi_backend] loading %s …", _MODEL_ID)
    # Register custom config so from_pretrained resolves the type
    try:
        AutoConfig.register("midi_model", MIDIModelConfig)
    except Exception:
        pass  # already registered

    model = MIDIModel.from_pretrained(_MODEL_ID)
    model.eval()
    tokenizer = model.tokenizer
    logger.info("[midi_backend] model loaded (%s, vocab=%d)", tokenizer.version, tokenizer.vocab_size)

    _model_cache = (model, tokenizer)
    return _model_cache


def _build_prompt(blueprint: MusicalBlueprint, roles_config: dict, tokenizer) -> np.ndarray:
    """Return a 2-D numpy prompt array (n_events, max_token_seq) for the model.

    Conditioning sequence:
      BOS → set_tempo(bpm) → patch_change per active non-drum role
    """
    max_bpm = tokenizer.event_parameters.get("bpm", 256) - 1
    bpm_val = min(int(blueprint.bpm), max_bpm)

    # BOS
    rows = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]

    # set_tempo: time1=0, time2=0, track=0, bpm
    t = tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm_val])
    if t:
        rows.append(t)

    # patch_change for each active melodic/harmonic role
    for role_name, role_cfg in blueprint.roles.items():
        if role_cfg.density <= 0.0:
            continue
        cfg = roles_config.get(role_name, {})
        ch1 = cfg.get("channel", 1)      # 1-indexed
        ch0 = ch1 - 1                    # 0-indexed (what the model uses)
        if ch1 == 10:                    # drums live on ch9 (0-idx); no patch needed
            continue
        prog = cfg.get("program", 0)
        # patch_change: time1, time2, track, channel, patch
        t = tokenizer.event2tokens(["patch_change", 0, 0, ch0 + 1, ch0, prog])
        if t:
            rows.append(t)

    return np.array(rows, dtype=np.int64)


def _score_to_roletracks(
    midi_score: list,
    roles_config: dict,
    max_tick: int | None = None,
) -> list[RoleTrack]:
    """Convert a detokenized midi_score to a list of RoleTracks.

    midi_score format (both V1 and V2 detokenize output):
      [ticks_per_beat, track0_events, track1_events, …]
    Note event format: ["note", t_ticks, dur_ticks, ch0, pitch, velocity]
      where ch0 is 0-indexed.
    """
    # Build reverse map: 0-indexed channel → (role_name, program)
    ch0_to_role: dict[int, tuple[str, int]] = {}
    for role_name, cfg in roles_config.items():
        ch1 = cfg.get("channel", 1)
        ch0_to_role[ch1 - 1] = (role_name, cfg.get("program", 0))

    notes_by_ch0: dict[int, list[NoteEvent]] = {}

    for track in midi_score[1:]:
        for event in track:
            if event[0] != "note":
                continue
            # ["note", t_ticks, dur_ticks, ch0, pitch, velocity]
            _, t, dur, ch0, pitch, vel = event[:6]

            if max_tick is not None and t >= max_tick:
                continue
            if max_tick is not None:
                dur = min(dur, max_tick - t)
            if dur <= 0:
                continue

            notes_by_ch0.setdefault(ch0, []).append(NoteEvent(
                pitch=int(pitch),
                velocity=max(1, min(127, int(vel))),
                start_tick=int(t),
                duration_ticks=int(dur),
                channel=ch0 + 1,          # 1-indexed for our system
            ))

    tracks: list[RoleTrack] = []
    for ch0, notes in sorted(notes_by_ch0.items()):
        role_name, program = ch0_to_role.get(ch0, ("unknown", 0))
        tracks.append(RoleTrack(
            role=role_name,
            channel=ch0 + 1,
            program=program,
            notes=sorted(notes, key=lambda n: n.start_tick),
        ))
    return tracks


def generate_from_blueprint(
    blueprint: MusicalBlueprint,
    roles_config: dict,
    max_events: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.98,
    top_k: int = 20,
) -> list[RoleTrack]:
    """Generate MIDI RoleTracks from a MusicalBlueprint.

    On the first call this downloads ~400 MB of model weights from HuggingFace
    and caches them in ~/.cache/huggingface/.  Subsequent calls reuse the cache.

    max_events controls the generation budget (excluding the prompt).  A longer
    budget produces more notes but takes longer on CPU.  The output is always
    trimmed to blueprint.bars regardless of how many events are generated.
    """
    model, tokenizer = _get_model()

    prompt = _build_prompt(blueprint, roles_config, tokenizer)
    max_tick = blueprint.bars * 4 * _TICKS_PER_BEAT

    logger.info(
        "[midi_backend] generating %d bars, max_events=%d, bpm=%s",
        blueprint.bars, max_events, blueprint.bpm,
    )

    with torch.inference_mode():
        generated = model.generate(
            prompt=prompt,
            batch_size=1,
            max_len=len(prompt) + max_events,
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # generated: (1, seq_len, max_token_seq) as numpy
    midi_seq = generated[0].tolist()
    midi_score = tokenizer.detokenize(midi_seq)

    tracks = _score_to_roletracks(midi_score, roles_config, max_tick=max_tick)
    logger.info(
        "[midi_backend] done: %d tracks, %d total notes",
        len(tracks), sum(len(t.notes) for t in tracks),
    )
    return tracks
