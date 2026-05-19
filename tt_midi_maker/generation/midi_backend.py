"""
Orchestration layer: blueprint → skytnt/midi-model → RoleTracks.

CPU path: loads skytnt/midi-model weights from HuggingFace Hub.
TT hardware path: forge-compiled net on P300C chips when available.
  - model.net (12-layer, ~350 M params) compiled with forge.compile
  - model.net_token (3-layer, ~50 M params) stays on CPU

Model weights are cached after first load. To force a reload, call reset_model().
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..models.track import NoteEvent, RoleTrack

if TYPE_CHECKING:
    from ..models.blueprint import MusicalBlueprint

logger = logging.getLogger(__name__)

_MODEL_ID = "skytnt/midi-model"
_TICKS_PER_BEAT = 480

# Lazy-loaded singletons
_model_cache: tuple | None = None         # (MIDIModel, tokenizer)
_hw_model_cache: tuple | None = None      # (compiled_net, max_padded_len)


def reset_model() -> None:
    """Clear the cached models (useful for testing)."""
    global _model_cache, _hw_model_cache
    _model_cache = None
    _hw_model_cache = None


def _get_model():
    """Load model on first call and cache it (CPU)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    from .skytnt_model import MIDIModel, MIDIModelConfig
    from transformers import AutoConfig

    logger.info("[midi_backend] loading %s …", _MODEL_ID)
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


def _get_compiled_net(model, max_padded_len: int = 256):
    """Compile model.net for TT hardware, or return None if unavailable.

    Checks for available TT devices first.  On failure logs a warning and
    returns None so the caller can fall back to the CPU path.
    """
    global _hw_model_cache
    if _hw_model_cache is not None:
        compiled, cached_len = _hw_model_cache
        if cached_len == max_padded_len:
            return compiled

    from .hardware import detect_tt_devices
    devices = detect_tt_devices()
    if not devices:
        logger.debug("[midi_backend] no TT devices found, using CPU path")
        return None

    logger.info("[midi_backend] TT devices found: %s — compiling net for hardware", devices)
    try:
        from .forge_backend import compile_for_hardware
        compiled = compile_for_hardware(model, max_padded_len)
        _hw_model_cache = (compiled, max_padded_len)
        return compiled
    except Exception as exc:
        logger.warning("[midi_backend] hardware compile failed, using CPU: %s", exc)
        return None


def _midi_file_to_score(path) -> list:
    """Parse a MIDI file into midi_score format for tokenizer.tokenize().

    Returns [ticks_per_beat, track_events] where track_events is a flat list
    of events in the format tokenizer.tokenize() expects:
      note:         ["note",         tick, dur,     ch0, pitch, vel]
      patch_change: ["patch_change", tick, ch0,     program]
      set_tempo:    ["set_tempo",    tick, tempo_us]
    """
    import mido
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    events: list = []

    for track in mid.tracks:
        tick = 0
        pending: dict = {}           # (ch, pitch) -> (start_tick, velocity)
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append(["set_tempo", tick, msg.tempo])
            elif msg.type == "program_change":
                events.append(["patch_change", tick, msg.channel, msg.program])
            elif msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in pending:
                    start, vel = pending.pop(key)
                    dur = max(1, tick - start)
                    events.append(["note", start, dur, msg.channel, msg.note, vel])

    events.sort(key=lambda e: e[1])
    return [tpb, events]


def _midi_file_to_prompt_rows(path, tokenizer, last_n_bars: int | None = None) -> list:
    """Tokenize a MIDI file into token rows suitable for use as a prompt prefix.

    last_n_bars: if set, only the final N bars of the file are included.  This
    limits context length while providing the most musically relevant material.
    Events are re-timed to start at tick 0 after slicing.

    Returns a list of token rows (no BOS/EOS).  Empty list if the file has no
    parseable notes.
    """
    midi_score = _midi_file_to_score(path)
    tpb   = midi_score[0]
    evts  = midi_score[1]

    if not evts:
        return []

    if last_n_bars is not None:
        last_tick   = max(e[1] for e in evts)
        cutoff_tick = max(0, last_tick - last_n_bars * tpb * 4)
        evts = [e for e in evts if e[1] >= cutoff_tick]
        if evts:
            # Re-base times so the context window starts at tick 0
            origin = evts[0][1]
            rebased = []
            for e in evts:
                e2 = list(e)
                e2[1] -= origin
                rebased.append(e2)
            evts = rebased

    if not evts:
        return []

    rows = tokenizer.tokenize([tpb, evts], add_bos_eos=False, add_default_instr=True)
    return rows


def _build_prompt(
    blueprint: MusicalBlueprint,
    roles_config: dict,
    tokenizer,
    source_rows: list | None = None,
) -> np.ndarray:
    """Return a 2-D numpy prompt array (n_events, max_token_seq) for the model.

    Conditioning sequence:
      BOS → set_tempo(bpm) → patch_change per configured non-drum role
      [ → source_rows  (previous MIDI context, if provided) ]

    source_rows: token rows from _midi_file_to_prompt_rows().  Appended after
    the setup tokens so the model sees the previous musical material before
    generating new content.
    """
    max_bpm = tokenizer.event_parameters.get("bpm", 256) - 1
    bpm_val = min(int(blueprint.bpm), max_bpm)

    # BOS
    rows = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]

    # set_tempo: time1=0, time2=0, track=0, bpm
    t = tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm_val])
    if t:
        rows.append(t)

    # Send patch_change for ALL configured roles so the model knows the full
    # instrument palette upfront. Density-zero roles still need their channel
    # pre-assigned so disable_channels masking keeps the model from leaking
    # onto unintended channels.
    for role_name, cfg in roles_config.items():
        ch1 = cfg.get("channel", 1)
        ch0 = ch1 - 1                    # 0-indexed
        if ch1 == 10:                    # drums live on ch9; no patch needed
            continue
        prog = cfg.get("program", 0)
        t = tokenizer.event2tokens(["patch_change", 0, 0, ch0 + 1, ch0, prog])
        if t:
            rows.append(t)

    # Append previous-MIDI context rows after setup tokens
    if source_rows:
        rows.extend(source_rows)
        logger.debug("[midi_backend] prompt includes %d source context rows", len(source_rows))

    return np.array(rows, dtype=np.int64)


def _score_to_roletracks(
    midi_score: list,
    roles_config: dict,
    max_tick: int | None = None,
    active_roles: set[str] | None = None,
) -> list[RoleTrack]:
    """Convert a detokenized midi_score to a list of RoleTracks.

    midi_score format (both V1 and V2 detokenize output):
      [ticks_per_beat, track0_events, track1_events, …]
    Note event format: ["note", t_ticks, dur_ticks, ch0, pitch, velocity]
      where ch0 is 0-indexed.

    active_roles: if provided, tracks on channels not in this set are dropped.
    Each role's note_range from roles_config is enforced — out-of-range notes
    are dropped so the model cannot leak outside the intended register.
    """
    # Build reverse map: 0-indexed channel → (role_name, program, note_range)
    ch0_to_role: dict[int, tuple[str, int, list[int]]] = {}
    for role_name, cfg in roles_config.items():
        ch1 = cfg.get("channel", 1)
        ch0_to_role[ch1 - 1] = (
            role_name,
            cfg.get("program", 0),
            cfg.get("note_range", [0, 127]),
        )

    # Restrict to requested channels only (drop uninvited roles)
    allowed_ch0: set[int] | None = None
    if active_roles is not None:
        allowed_ch0 = {
            ch1 - 1
            for role_name, cfg in roles_config.items()
            if role_name in active_roles
            for ch1 in [cfg.get("channel", 1)]
        }

    notes_by_ch0: dict[int, list[NoteEvent]] = {}

    for track in midi_score[1:]:
        for event in track:
            if event[0] != "note":
                continue
            # ["note", t_ticks, dur_ticks, ch0, pitch, velocity]
            _, t, dur, ch0, pitch, vel = event[:6]

            if allowed_ch0 is not None and ch0 not in allowed_ch0:
                continue

            if max_tick is not None and t >= max_tick:
                continue
            if max_tick is not None:
                dur = min(dur, max_tick - t)
            if dur <= 0:
                continue

            # Enforce note_range for this role
            _, _, note_range = ch0_to_role.get(ch0, ("unknown", 0, [0, 127]))
            if not (note_range[0] <= int(pitch) <= note_range[1]):
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
        role_name, program, _ = ch0_to_role.get(ch0, ("unknown", 0, [0, 127]))
        tracks.append(RoleTrack(
            role=role_name,
            channel=ch0 + 1,
            program=program,
            notes=sorted(notes, key=lambda n: n.start_tick),
        ))
    return tracks


def _compute_active_channels(
    blueprint: "MusicalBlueprint",
    roles_config: dict,
) -> "set[int]":
    """Return 0-indexed MIDI channel set for roles with density > 0."""
    active: set[int] = set()
    for role_name, role_cfg in blueprint.roles.items():
        if role_cfg.density > 0.0:
            cfg = roles_config.get(role_name, {})
            ch1 = cfg.get("channel", 1)
            active.add(ch1 - 1)   # 0-indexed
    return active


def generate_from_blueprint(
    blueprint: MusicalBlueprint,
    roles_config: dict,
    max_events: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.98,
    top_k: int = 20,
    hw_max_padded_len: int = 256,
    hw_context_interval: int = 4,
    source_midi: str | None = None,
    source_context_bars: int | None = 8,
    max_attempts: int = 1,
    judge_threshold: float = 0.55,
) -> list[RoleTrack]:
    """Generate MIDI RoleTracks from a MusicalBlueprint.

    On the first call this downloads ~400 MB of model weights from HuggingFace
    and caches them in ~/.cache/huggingface/.  Subsequent calls reuse the cache.

    When TT hardware is available, model.net is compiled with forge and the
    12-layer forward pass runs on the P300C chips; model.net_token stays on CPU.
    Falls back to pure CPU silently if hardware is unavailable or compile fails.

    max_events: generation budget (notes) beyond the prompt. Trimmed to blueprint.bars.
    hw_max_padded_len: fixed sequence length compiled for hardware.
    hw_context_interval: hardware forward pass is called every this many events.
        1 = every step (accurate but slow, ~2 ev/s on P300C).
        4 = every 4th step (default, ~5 ev/s — enough for 64 events per 13.9s loop).
        8 = every 8th step (~9 ev/s, higher quality gap between hardware refreshes).
    source_midi: path to a MIDI file to use as musical context for generation.
        The file's events are tokenized and prepended to the prompt so the model
        can continue from or respond to the existing material.
    source_context_bars: how many bars from the end of source_midi to include.
        None = use the whole file.  Default 8 (one full loop).
    max_attempts: if > 1, apply rule-based judge after generation and re-roll up to
        this many total tries when the pattern scores below judge_threshold.
        Default 1 = no re-rolling (original behaviour).
    judge_threshold: rule_score floor for acceptance when max_attempts > 1.
        Range 0.0–1.0; 0.55 accepts patterns with up to ~4 rule violations.
    """
    model, tokenizer = _get_model()

    # Rule-based judge used for re-rolling when max_attempts > 1
    _judge = None
    if max_attempts > 1:
        from ..coherence.judge import judge_tracks as _judge_tracks
        _judge = _judge_tracks

    source_rows: list | None = None
    if source_midi is not None:
        try:
            source_rows = _midi_file_to_prompt_rows(
                source_midi, tokenizer, last_n_bars=source_context_bars
            )
            logger.info(
                "[midi_backend] source context: %d rows from %s (last %s bars)",
                len(source_rows), source_midi,
                source_context_bars if source_context_bars else "all",
            )
        except Exception as exc:
            logger.warning("[midi_backend] failed to load source MIDI, ignoring: %s", exc)

    prompt = _build_prompt(blueprint, roles_config, tokenizer, source_rows=source_rows)
    max_tick = blueprint.bars * 4 * _TICKS_PER_BEAT

    logger.info(
        "[midi_backend] generating %d bars, max_events=%d, bpm=%s",
        blueprint.bars, max_events, blueprint.bpm,
    )

    # Try TT hardware path; fall back to CPU on any failure
    compiled_net = _get_compiled_net(model, hw_max_padded_len)

    # Compute which MIDI channels (0-indexed) are active in this blueprint.
    active_channels = _compute_active_channels(blueprint, roles_config)
    active_roles = {r for r, cfg in blueprint.roles.items() if cfg.density > 0.0}

    best_tracks: list[RoleTrack] = []
    best_score: float = -1.0

    for attempt in range(max_attempts):
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

        if compiled_net is None:
            logger.info("[midi_backend] using CPU path")
            # model.generate() has no equivalent masking (disable_patch_change/allowed_channels);
            # channel bleed is caught downstream by _score_to_roletracks active_roles filter.
            with torch.inference_mode():
                generated = model.generate(
                    prompt=prompt,
                    batch_size=1,
                    max_len=len(prompt) + max_events,
                    temp=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

        # Decode and convert to RoleTracks
        midi_seq = generated[0].tolist()
        midi_score = tokenizer.detokenize(midi_seq)
        tracks = _score_to_roletracks(midi_score, roles_config,
                                       max_tick=max_tick, active_roles=active_roles)

        # Quality gate: judge and optionally re-roll
        if _judge is not None:
            report = _judge(tracks, bars=blueprint.bars, bpm=blueprint.bpm,
                            pass_threshold=judge_threshold)
            if report.rule_score > best_score:
                best_score = report.rule_score
                best_tracks = tracks

            if report.passed:
                logger.info(
                    "[midi_backend] judge: attempt %d/%d PASS (score=%.2f)",
                    attempt + 1, max_attempts, report.rule_score,
                )
                break
            else:
                issues_short = " | ".join(report.issues[:3])
                logger.info(
                    "[midi_backend] judge: attempt %d/%d score=%.2f — re-rolling: %s",
                    attempt + 1, max_attempts, report.rule_score, issues_short,
                )
        else:
            best_tracks = tracks
            break

    tracks = best_tracks or tracks  # fallback: keep last attempt if judge never ran

    logger.info(
        "[midi_backend] done: %d tracks, %d total notes",
        len(tracks), sum(len(t.notes) for t in tracks),
    )
    return tracks
