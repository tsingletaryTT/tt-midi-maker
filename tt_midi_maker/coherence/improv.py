"""Stochastic improv layer — approach notes, rhythmic variation, tension curves, phrase evolution.

All functions use deterministic seeds so output is reproducible given the same
tension value and seed, but different seeds produce musically varied results.
Drums (channel 10) are never modified by any improv function.
"""
from __future__ import annotations

import random
from typing import Callable

from ..models.track import NoteEvent
from .harmony import chord_at_tick


# ── Tension curve ─────────────────────────────────────────────────────────────

def build_tension_curve(start: float, end: float, bars: int) -> Callable[[int], float]:
    """Return fn(bar_index) -> tension (0.0–1.0), linearly ramping start→end.

    bar_index is 0-based.  curve(0)=start, curve(bars-1)=end.
    When bars=1, always returns start regardless of bar_index.
    """
    def tension_at(bar_idx: int) -> float:
        if bars <= 1:
            return start
        t = bar_idx / (bars - 1)
        return max(0.0, min(1.0, start + (end - start) * t))
    return tension_at


# ── Approach notes ────────────────────────────────────────────────────────────

def add_approach_notes(
    notes: list[NoteEvent],
    chord_progression: list[str],
    ticks_per_bar: int,
    ticks_per_beat: int = 480,
    tension: float = 0.0,
    seed: int = 42,
) -> list[NoteEvent]:
    """Insert chromatic approach notes before chord-tone bar downbeats.

    For each melody note on beat 1 of a bar that is a chord tone, insert a
    half-step-below approach note starting 1/2-beat earlier with lower velocity.
    Insertion probability = tension (0.0 = none, 1.0 = every eligible note).

    Drums and non-downbeat notes are never modified.
    """
    if tension <= 0.0:
        return notes

    rng = random.Random(seed)
    inserts: list[NoteEvent] = []
    half_beat = ticks_per_beat // 2

    for note in notes:
        if note.channel == 10:          # drums: skip
            continue
        # Only target notes on beat 1 of a bar (within first beat)
        if note.start_tick % ticks_per_bar >= ticks_per_beat:
            continue
        # Only approach chord tones (approaching non-chord-tones is harsh)
        chord_pcs = chord_at_tick(note.start_tick, ticks_per_bar, chord_progression)
        if note.pitch % 12 not in chord_pcs:
            continue
        if rng.random() > tension:
            continue
        approach_start = note.start_tick - half_beat
        if approach_start < 0:
            continue
        inserts.append(NoteEvent(
            pitch=note.pitch - 1,
            velocity=max(30, note.velocity - 25),
            start_tick=approach_start,
            duration_ticks=half_beat - 15,
            channel=note.channel,
        ))

    result = notes + inserts
    result.sort(key=lambda n: n.start_tick)
    return result


# ── Rhythmic variation ────────────────────────────────────────────────────────

def vary_rhythm(
    notes: list[NoteEvent],
    ticks_per_beat: int = 480,
    tension: float = 0.0,
    seed: int = 77,
) -> list[NoteEvent]:
    """Shift note starts by ±half-beat proportional to tension.

    For each non-drum note, with probability tension×0.45, moves the start
    tick forward or backward by half a beat.  Higher tension = looser feel.
    """
    if tension <= 0.0:
        return notes

    rng = random.Random(seed)
    half_beat = ticks_per_beat // 2
    result = []
    for note in notes:
        if note.channel == 10 or rng.random() > tension * 0.45:
            result.append(note)
            continue
        nudge = half_beat if rng.random() > 0.5 else -half_beat
        result.append(NoteEvent(
            pitch=note.pitch,
            velocity=note.velocity,
            start_tick=max(0, note.start_tick + nudge),
            duration_ticks=note.duration_ticks,
            channel=note.channel,
        ))
    result.sort(key=lambda n: n.start_tick)
    return result


# ── Cross-pattern phrase evolution ────────────────────────────────────────────

def evolve_phrase(
    notes: list[NoteEvent],
    semitone_shift: int = 2,
    tension: float = 0.3,
    seed: int = 99,
) -> list[NoteEvent]:
    """Develop a phrase from a previous pattern for cross-pattern motivic growth.

    Transposes melodic notes up by `semitone_shift` and adds slight duration
    variation to create a "composed variation" feel from P1 to P2 to P3.
    Drums (channel 10) are never transposed.
    Notes that would fall outside MIDI range 21–108 are left at original pitch.
    """
    rng = random.Random(seed)
    result = []
    for note in notes:
        if note.channel == 10:
            result.append(note)
            continue
        new_pitch = note.pitch + semitone_shift
        if not (21 <= new_pitch <= 108):
            new_pitch = note.pitch
        dur_var = int(note.duration_ticks * 0.12 * (rng.random() - 0.5))
        result.append(NoteEvent(
            pitch=new_pitch,
            velocity=note.velocity,
            start_tick=note.start_tick,
            duration_ticks=max(15, note.duration_ticks + dur_var),
            channel=note.channel,
        ))
    return result
