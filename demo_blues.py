#!/usr/bin/env python3
"""
"Midnight in East Texas" — 1930s Delta blues.

Musical design
--------------
Key:    A minor  (blues scale: A C D Eb E G — pentatonic + blue note)
BPM:    80       (slow Delta shuffle)
Bars:   12       (full 12-bar blues form: I-IV-I-I-IV-IV-I-I-V-IV-I-V)
Chords: Full 12-bar timeline resolved from Roman numeral template in key A

Structure (deterministic)
---------
  Walking bass : 4 quarter notes/bar (root → 3rd → 5th → chromatic approach)
  Drum groove  : shuffle (kick 1+3, snare 2+4, triplet hi-hat)
  Call bars    : 1, 2, 5, 6, 9, 10 — full melodic phrase
  Response bars: 3, 4, 7, 8, 11, 12 — brief answer note + silence

Improv (stochastic, deterministic seeds)
------
  Approach notes : chromatic half-step before chord-tone downbeats (tension arc)
  Tension arc    : P1=0.0 (clean statement), P2=0.3 (developing), P3=0.6 (peak)
  Source chaining: P1→P2→P3 so each pattern responds to the last

Instruments
-----------
  melody   — Acoustic Steel Guitar (program 25)  A2–E5  (1930s acoustic lead)
  harmony  — Acoustic Grand Piano (program 0)     A2–A5  (barrelhouse comp)
  bass     — Acoustic Bass (program 32)            A1–A3  (walking bass, deterministic)
  drums    — Standard kit (channel 10)             (shuffle groove, deterministic)
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("blues")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "midnight-blues"
OUTPUT_DIR  = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role, swing_timing
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.coherence.structure import (
    GenreStructure, build_genre_structure,
    generate_drum_groove, generate_walking_bass, enforce_phrase_gaps,
)
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import RoleTrack

KEY    = "A minor"
BPM    = 80
BARS   = 12    # full 12-bar blues form

# Seed chords passed to build_genre_structure; overridden by blues_12bar template
SEED_CHORDS = ["A7", "D7", "A7", "E7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

GUITAR_PROGRAM = 25    # Acoustic Steel Guitar — 1930s Delta blues
GUITAR_RANGE   = [45, 76]  # A2 – E5
PIANO_PROGRAM  = 0     # Acoustic Grand Piano — barrelhouse comp
BASS_PROGRAM   = 32    # Acoustic Bass
BASS_RANGE     = [33, 57]  # A1 – A3

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": GUITAR_PROGRAM, "note_range": GUITAR_RANGE}
    elif name == "harmony":
        overrides = {"program": PIANO_PROGRAM}
    elif name == "bass":
        overrides = {"program": BASS_PROGRAM, "note_range": BASS_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


def _blueprint(active_roles: list[str], structure: GenreStructure) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="blues", mode="loop",
        chord_progression=structure.chord_progression,   # full 12-chord timeline
        roles={
            name: RoleConfig(density=cfg["density_default"] if name in active_roles else 0.0)
            for name, cfg in ROLES_CONFIG.items()
        },
    )


def _apply_coherence(tracks, bp: MusicalBlueprint, structure: GenreStructure) -> list:
    root, mode = parse_key(bp.key)
    scale_set  = build_scale_set(root, mode)
    tpb        = TICKS_PER_BEAT
    tpbar      = tpb * 4
    out        = []
    for track in tracks:
        if track.role in ("bass", "drums"):
            # Deterministic tracks injected directly — skip model-processing passes
            out.append(track)
            continue
        notes = scale_quantize(track.notes, bp.key, strictness=0.65, override_mode="blues")
        notes = chord_aware_filter(notes, bp.chord_progression, tpbar, tpb,
                                   scale_set, semitone_tolerance=1)
        notes = swing_timing(notes, swing_ratio=0.67)
        if track.role == "melody":
            notes = enforce_phrase_gaps(notes, structure.response_bars, ticks_per_beat=tpb)
            notes = add_approach_notes(notes, bp.chord_progression, tpbar,
                                       tension=structure.tension, seed=42)
        notes = scale_velocity_by_role(notes, track.role)
        notes = humanize_velocities(notes, variation=8)
        notes = nudge_timing(notes, max_ticks=18)
        out.append(replace(track, notes=notes))
    return out


def generate(
    filename: str,
    source_midi: str | None = None,
    label: str = "",
    tension: float = 0.0,
    max_events: int = 96,
    hw_context_interval: int = 4,
) -> str | None:
    structure = build_genre_structure("blues_12bar", KEY, SEED_CHORDS, bars=BARS, tension=tension)
    bp        = _blueprint(active_roles=["melody", "harmony"], structure=structure)
    src_label = f"← {Path(source_midi).name}" if source_midi else "cold start"
    bar("─")
    print(f"  {label}  (tension={tension:.1f})")
    print(f"  chords: {' '.join(structure.chord_progression)}")
    print(f"  source: {src_label}")
    t0 = time.time()

    # Model generates melody + harmony only; bass and drums are deterministic
    tracks = generate_from_blueprint(
        bp, SUITE_ROLES,
        max_events=max_events,
        hw_context_interval=hw_context_interval,
        max_attempts=3,
        judge_threshold=0.55,
        source_midi=source_midi,
        source_context_bars=8,
    )
    dt = time.time() - t0

    if not tracks:
        print(f"  [!] model returned empty — skipping {filename}")
        return None

    # Stamp correct programs for model-generated tracks
    stamped = []
    for t in tracks:
        if t.role == "melody":
            stamped.append(replace(t, program=GUITAR_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=PIANO_PROGRAM))
        else:
            stamped.append(t)

    # Inject deterministic walking bass and shuffle drum groove
    bass_notes = generate_walking_bass(
        structure.chord_progression, BARS,
        ticks_per_beat=TICKS_PER_BEAT, velocity=78, channel=2,
    )
    drum_notes = generate_drum_groove("shuffle", BARS, ticks_per_beat=TICKS_PER_BEAT)
    bass_track = RoleTrack(role="bass",  channel=2,  program=BASS_PROGRAM, notes=bass_notes)
    drum_track = RoleTrack(role="drums", channel=10, program=0,             notes=drum_notes)
    all_tracks = stamped + [bass_track, drum_track]

    polished   = _apply_coherence(all_tracks, bp, structure)
    out        = OUTPUT_DIR / filename
    build_midi_file(polished, bp.bpm, out)

    summary    = "  ".join(f"{t.role}:{len(t.notes)}n" for t in polished)
    loop_ratio = dt / LOOP_SECS
    print(f"  {dt:.1f}s  ({max_events/dt:.1f} ev/s)  = {loop_ratio:.2f}× loop")
    print(f"  tracks: {summary}")
    print(f"  saved:  {out}")
    return str(out)


def bar(char="═", width=60):
    print(char * width)


def main():
    bar("═")
    print("  tt-midi-maker  ▸  Midnight in East Texas  (A minor — 12-bar Delta blues)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}   Bars: {BARS} (12-bar form)")
    print(f"  Loop    : {LOOP_SECS:.1f}s per 12-bar phrase")
    print(f"  Roles   : guitar (model) + piano (model) + walking-bass + shuffle-drums")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    # Tension arc: clean statement → developing → peak
    f1 = generate("p1_intro.mid",
                  label="Pattern 1 — intro statement",
                  tension=0.0, max_events=128, hw_context_interval=2)
    f2 = generate("p2_groove.mid",
                  source_midi=f1,
                  label="Pattern 2 — groove development",
                  tension=0.3, max_events=128, hw_context_interval=2)
    f3 = generate("p3_resolution.mid",
                  source_midi=f2,
                  label="Pattern 3 — resolution peak",
                  tension=0.6, max_events=160, hw_context_interval=2)

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
