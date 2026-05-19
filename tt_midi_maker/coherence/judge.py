"""
Musical quality judge for generated patterns.

Two-layer analysis:
  1. Rule-based: pitch span, interval variance, note density, silence ratio,
     register overlap, rhythmic clustering, rhythmic diversity.
  2. Perplexity: event-type NLL under the skytnt model, computed on CPU
     with a single batched forward pass (no generation loop).

Usage in generation scripts::

    from tt_midi_maker.coherence.judge import judge_tracks, PatternReport

    for attempt in range(max_attempts):
        tracks = generate_from_blueprint(...)
        report = judge_tracks(tracks, bars=8, bpm=84)
        if report.passed:
            break
        logger.info("  [judge] %s — re-rolling", " | ".join(report.issues))

Usage for post-hoc MIDI file analysis::

    report = judge_midi_file(Path("examples/silicon-road/p2_drift.mid"),
                             bars=8, bpm=84)
    print(report.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mido

from ..models.track import NoteEvent, RoleTrack

_TICKS_PER_BEAT = 480

# ─── thresholds ───────────────────────────────────────────────────────────────

# Notes per bar (per active role, not counting silent density-0 tracks)
DENSITY_MIN = 0.5    # below → sparse/silent
DENSITY_MAX = 30.0   # above → machine-gun

# Melody-specific pitch metrics
PITCH_SPAN_MIN    = 4    # semitones — narrower is monotonous
PITCH_SPAN_MAX    = 36   # semitones — wider is random-feeling
UNIQUE_PITCH_MIN  = 3    # distinct pitches — fewer is a loop of 2-3 notes
MEAN_INTERVAL_MAX = 8    # semitones avg — above this sounds like random leaps
MAX_INTERVAL_MAX  = 24   # semitones single step — 2 octaves in one step is jarring

# Silence ratio: fraction of loop ticks with no notes on the track
SILENCE_MIN = 0.08   # below → no breathing room
SILENCE_MAX = 0.96   # above → barely any notes

# Bass/melody register: how much (semitones) does bass pitch range invade melody's?
REGISTER_OVERLAP_MAX = 6   # semitones

# Rhythmic clustering: fraction of notes landing within 24 ticks (half-beat) of another
CLUSTER_RATIO_MAX = 0.65

# Direction reversals: fraction of intervals that reverse direction (zigzag)
DIRECTION_REVERSAL_MAX = 0.75   # above → random zigzag


# ─── data classes ─────────────────────────────────────────────────────────────

@dataclass
class PatternReport:
    path: str
    bars: int
    bpm: float

    # Rule-based
    rule_score: float           # 0.0–1.0 (higher = fewer issues)
    passed: bool                # True if score ≥ 0.55 (configurable)
    issues: list[str]           # human-readable problem descriptions
    metrics: dict               # raw computed values per role

    # Perplexity (optional — None if skipped)
    event_nll: Optional[float] = None   # mean negative log-prob per event type token

    def summary(self) -> str:
        nll_str = f"  event NLL: {self.event_nll:.3f}" if self.event_nll is not None else ""
        issue_str = "\n    ✗ " + "\n    ✗ ".join(self.issues) if self.issues else "    ✓ no issues"
        return (
            f"{self.path}\n"
            f"  rule score: {self.rule_score:.2f}  {'PASS' if self.passed else 'FAIL'}{nll_str}\n"
            f"{issue_str}"
        )


# ─── rule-based analysis ──────────────────────────────────────────────────────

def _analyze_track(
    track: RoleTrack,
    bars: int,
    bpm: float,
    loop_ticks: int,
) -> tuple[list[str], dict]:
    """
    Return (issues, metrics) for a single RoleTrack.
    issues: list of human-readable strings
    metrics: dict of computed numeric values
    """
    notes = track.notes
    issues: list[str] = []
    m: dict = {}

    if not notes:
        return issues, m

    role = track.role
    is_melody = role in ("melody",)
    is_bass   = role in ("bass",)

    pitches = [n.pitch for n in notes]
    durs    = [n.duration_ticks for n in notes]
    starts  = sorted(n.start_tick for n in notes)

    # ── density ──────────────────────────────────────────────────────────────
    npb = len(notes) / bars
    m["notes_per_bar"] = round(npb, 2)
    if npb < DENSITY_MIN:
        issues.append(f"{role}: too sparse ({npb:.1f} notes/bar)")
    elif npb > DENSITY_MAX:
        issues.append(f"{role}: too dense ({npb:.1f} notes/bar — machine-gun)")

    # ── pitch span and variety ────────────────────────────────────────────────
    if len(pitches) >= 2:
        span = max(pitches) - min(pitches)
        unique = len(set(pitches))
        m["pitch_span"]     = span
        m["unique_pitches"] = unique
        m["pitch_min"]      = min(pitches)
        m["pitch_max"]      = max(pitches)

        if is_melody or is_bass:
            if span < PITCH_SPAN_MIN:
                issues.append(f"{role}: pitch span only {span} semitones (monotonous)")
            elif span > PITCH_SPAN_MAX:
                issues.append(f"{role}: pitch span {span} semitones (too wide — scattered)")

        if is_melody and unique < UNIQUE_PITCH_MIN:
            issues.append(f"{role}: only {unique} distinct pitches (repetitive loop)")

    # ── interval analysis (melody) ────────────────────────────────────────────
    if is_melody and len(pitches) >= 3:
        sorted_notes = sorted(notes, key=lambda n: n.start_tick)
        intervals = [abs(sorted_notes[i+1].pitch - sorted_notes[i].pitch)
                     for i in range(len(sorted_notes)-1)]

        mean_iv = sum(intervals) / len(intervals)
        max_iv  = max(intervals)
        m["mean_interval"] = round(mean_iv, 2)
        m["max_interval"]  = max_iv

        if mean_iv > MEAN_INTERVAL_MAX:
            issues.append(
                f"{role}: mean interval {mean_iv:.1f} semitones (average > P5 — sounds like random leaps)"
            )
        if max_iv > MAX_INTERVAL_MAX:
            issues.append(f"{role}: max interval {max_iv} semitones (>{MAX_INTERVAL_MAX} = jarring jump)")

        # direction reversals (zigzag detector)
        raw_intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                         for i in range(len(sorted_notes)-1)]
        raw_nonzero = [x for x in raw_intervals if x != 0]
        if len(raw_nonzero) >= 3:
            reversals = sum(
                1 for i in range(len(raw_nonzero)-1)
                if raw_nonzero[i] * raw_nonzero[i+1] < 0
            )
            rev_ratio = reversals / (len(raw_nonzero) - 1)
            m["direction_reversal_ratio"] = round(rev_ratio, 3)
            if rev_ratio > DIRECTION_REVERSAL_MAX:
                issues.append(
                    f"{role}: {rev_ratio:.0%} direction reversals (melodic zigzag/jumble)"
                )

    # ── silence ratio ─────────────────────────────────────────────────────────
    if loop_ticks > 0:
        active_ticks = sum(min(n.duration_ticks, loop_ticks) for n in notes)
        silence_ratio = max(0.0, 1.0 - active_ticks / loop_ticks)
        m["silence_ratio"] = round(silence_ratio, 3)
        if silence_ratio < SILENCE_MIN:
            issues.append(
                f"{role}: silence ratio {silence_ratio:.0%} (no breathing room)"
            )
        elif silence_ratio > SILENCE_MAX and not is_bass:
            issues.append(
                f"{role}: silence ratio {silence_ratio:.0%} (barely any notes)"
            )

    # ── rhythmic clustering ───────────────────────────────────────────────────
    if len(starts) >= 3:
        cluster_count = 0
        for i, t in enumerate(starts):
            neighbours = sum(1 for s in starts if 0 < abs(s - t) <= 24)
            if neighbours >= 1:
                cluster_count += 1
        cluster_ratio = cluster_count / len(starts)
        m["cluster_ratio"] = round(cluster_ratio, 3)
        if cluster_ratio > CLUSTER_RATIO_MAX:
            issues.append(
                f"{role}: {cluster_ratio:.0%} notes clustered within half-beat (stacked pile-up)"
            )

    return issues, m


def judge_tracks(
    tracks: list[RoleTrack],
    bars: int,
    bpm: float,
    *,
    pass_threshold: float = 0.55,
    source: str = "<in-memory>",
) -> PatternReport:
    """Judge a list of RoleTrack objects (from generate_from_blueprint)."""
    loop_ticks = int(bars * 4 * _TICKS_PER_BEAT)
    all_issues: list[str] = []
    all_metrics: dict = {}

    melody_range: tuple[int, int] | None = None
    bass_range:   tuple[int, int] | None = None

    for track in tracks:
        if not track.notes:
            continue
        iss, m = _analyze_track(track, bars, bpm, loop_ticks)
        all_issues.extend(iss)
        all_metrics[track.role] = m

        pitches = [n.pitch for n in track.notes]
        if track.role == "melody":
            melody_range = (min(pitches), max(pitches))
        elif track.role == "bass":
            bass_range = (min(pitches), max(pitches))

    # Register overlap check (bass invades melody's register)
    if melody_range and bass_range:
        overlap = max(0, bass_range[1] - melody_range[0])
        all_metrics["register_overlap_semitones"] = overlap
        if overlap > REGISTER_OVERLAP_MAX:
            all_issues.append(
                f"register overlap: bass top={bass_range[1]} invades melody bottom={melody_range[0]} "
                f"by {overlap} semitones"
            )

    score = max(0.0, 1.0 - 0.12 * len(all_issues))
    return PatternReport(
        path=source,
        bars=bars,
        bpm=bpm,
        rule_score=round(score, 3),
        passed=score >= pass_threshold,
        issues=all_issues,
        metrics=all_metrics,
    )


def judge_midi_file(
    path: Path | str,
    bars: int,
    bpm: float,
    *,
    pass_threshold: float = 0.55,
) -> PatternReport:
    """Read a MIDI file and judge it. Converts mido events → RoleTracks."""
    path = Path(path)
    mid  = mido.MidiFile(str(path))
    tpb  = mid.ticks_per_beat

    # Collect notes by channel
    channel_notes: dict[int, list[NoteEvent]] = {}
    for track in mid.tracks:
        tick = 0
        pending: dict[tuple[int,int], tuple[int,int]] = {}
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in pending:
                    start, vel = pending.pop(key)
                    dur = max(1, tick - start)
                    # Normalise tick resolution to 480 tpb
                    norm_start = int(start * _TICKS_PER_BEAT / tpb)
                    norm_dur   = int(dur   * _TICKS_PER_BEAT / tpb)
                    ch = msg.channel
                    channel_notes.setdefault(ch, []).append(
                        NoteEvent(pitch=msg.note, velocity=vel,
                                  start_tick=norm_start, duration_ticks=norm_dur, channel=ch+1)
                    )

    # Assign channel → role heuristically
    # channel 0 → melody, channel 1 → bass, channel 2 → harmony, channel 9 → drums
    CHANNEL_ROLE = {0: "melody", 1: "bass", 2: "harmony", 9: "drums"}
    tracks: list[RoleTrack] = []
    for ch, notes in channel_notes.items():
        role = CHANNEL_ROLE.get(ch, f"ch{ch}")
        tracks.append(RoleTrack(role=role, channel=ch+1, program=0, notes=notes))

    report = judge_tracks(tracks, bars, bpm, pass_threshold=pass_threshold,
                          source=str(path))
    return report


# ─── perplexity scoring ───────────────────────────────────────────────────────

def score_perplexity(path: Path | str, model=None, tokenizer=None) -> float | None:
    """
    Compute mean negative log-probability of event types in the MIDI file.

    Uses the skytnt model in CPU eval mode. A lower score means the model
    "expected" the event sequence more — more coherent relative to training data.

    If model/tokenizer are None, they are loaded on first call (cached singleton).
    Returns None if the MIDI has <3 parseable events.

    Complexity: ONE batched forward pass through model.net (12 layers) +
    ONE batched forward pass through model.net_token (3 layers).
    ~10-30s on CPU for a typical MIDI (50-200 events).
    """
    import sys
    from pathlib import Path as _Path
    import numpy as np
    import torch
    import torch.nn.functional as F

    path = _Path(path)

    # Load model if not provided
    if model is None or tokenizer is None:
        # Reuse the backend's singleton cache
        root = _Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tt_midi_maker.generation.midi_backend import _get_model
        model, tokenizer = _get_model()

    # Read and tokenize MIDI
    import mido as _mido
    mid = _mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    events: list = []
    for track in mid.tracks:
        tick = 0
        pending: dict = {}
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

    if not events:
        return None

    midi_score = [tpb, events]
    rows = tokenizer.tokenize(midi_score, add_bos_eos=True, add_default_instr=True)
    if len(rows) < 3:
        return None

    token_array = np.array(rows, dtype=np.int64)  # (n, max_token_seq)
    n = token_array.shape[0]

    # Forward pass through model.net (12-layer) on full sequence except last event
    # This gives context vectors h[i] = "understanding of events 0..i"
    with torch.no_grad():
        tokens_tensor = torch.from_numpy(token_array[:-1]).unsqueeze(0)  # (1, n-1, max_token_seq)
        hidden = model.forward(tokens_tensor)   # (1, n-1, n_embd)

        # Score each event at position 1..n-1 using the context from 0..i-1
        # forward_token(hidden_state=(batch, n_embd), x=None) → (batch, 1, vocab_size)
        # We batch all n-1 context vectors at once.
        h_batch = hidden[0]   # (n-1, n_embd)
        logits  = model.forward_token(h_batch, x=None)   # (n-1, 1, vocab_size)
        logits  = logits[:, 0, :]                         # (n-1, vocab_size)

        # Actual event-type tokens for positions 1..n-1
        target_event_ids = torch.from_numpy(token_array[1:, 0])  # (n-1,)

        # Filter: only score events that are real events (not pad/bos/eos)
        valid_event_ids = set(tokenizer.event_ids.values())
        mask = torch.tensor(
            [t.item() in valid_event_ids for t in target_event_ids], dtype=torch.bool
        )
        if mask.sum() == 0:
            return None

        log_probs = F.log_softmax(logits[mask], dim=-1)  # (n_valid, vocab_size)
        actual_ids = target_event_ids[mask]              # (n_valid,)
        token_log_probs = log_probs[range(len(actual_ids)), actual_ids]
        mean_nll = -token_log_probs.mean().item()

    return round(mean_nll, 4)
