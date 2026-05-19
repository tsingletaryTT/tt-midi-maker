#!/usr/bin/env python3
"""
"Silicon Road" — Berlin School / Tangerine Dream style, after Hyperborea (1983).

Musical design
--------------
Key:    E minor  (dark, modal, cosmic)
BPM:    84       (slow hypnotic drift — Berlin School tempo)
Bars:   8        (~22.9s loop)
Chords: Em7 → Cmaj7 → Am7 → Bm7  (i7 – VII – vi – vii)

Hyperborea context: Tangerine Dream's 1983 album sits at the intersection of
sequencer-driven Berlin School and early cinematic electronic. Cinnamon Road
(the source inspiration) has floating sawtooth leads over sustained warm pads,
an ostinato synth bass, and no percussion — all texture and drift.

Instruments
-----------
  melody   — Lead 2 Sawtooth  (GM 81)    E4–E7   (floating, gliding lines)
  harmony  — Pad 2 Warm       (GM 89)    B2–B5   (lush sustained pad chords)
  bass     — Synth Bass 1     (GM 38)    E1–E4   (steady ostinato sequences)
  drums    — silent (density 0)

Scale coherence
---------------
E natural minor (Aeolian): E F# G A B C D
Strictness 0.75 — allows passing tones for that organic Berlin School feel.
Chord filter semitone_tolerance=1 — chord tones favoured on strong beats,
chromatic approach notes allowed.
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
logger = logging.getLogger("silicon_road")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME = "silicon-road"
OUTPUT_DIR = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

KEY    = "E minor"
BPM    = 84
BARS   = 8
CHORDS = ["Em7", "Cmaj7", "Am7", "Bm7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)  # ~22.9s

LEAD_PROGRAM    = 81   # Lead 2 Sawtooth — classic Berlin School oscillator timbre
LEAD_RANGE      = [52, 88]   # E3–E6  (model's comfort zone for melodic lines)
PAD_PROGRAM     = 89   # Pad 2 Warm — Jupiter-8-ish sustain chords
PAD_RANGE       = [47, 71]   # B2–B5  (mid-register pads)
BASS_PROGRAM    = 38   # Synth Bass 1 — Moog-style ostinato bass
BASS_RANGE      = [28, 57]   # E1–A3

VELOCITY_RANGES = {
    "melody":  (65, 100),   # lead — expressive, not too loud (floats above pads)
    "harmony": (50,  80),   # pads — sustained, slightly recessed
    "bass":    (70,  90),   # bass — consistent ostinato presence
    "drums":   (60, 100),   # unused
}

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    if name == "melody":
        # Use program 0 (piano) so the model generates notes — stamp to sawtooth post-gen.
        # The model is trained on piano-heavy MIDI and rarely generates GM 81 (sawtooth).
        SUITE_ROLES[name] = {**cfg, "program": 0, "note_range": LEAD_RANGE}
    elif name == "harmony":
        SUITE_ROLES[name] = {**cfg, "program": PAD_PROGRAM, "note_range": PAD_RANGE}
    elif name == "bass":
        SUITE_ROLES[name] = {**cfg, "program": BASS_PROGRAM, "note_range": BASS_RANGE}
    else:
        SUITE_ROLES[name] = {**cfg}


def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="ambient", mode="loop",
        chord_progression=CHORDS,
        roles={
            name: RoleConfig(density=cfg["density_default"] if name in active_roles else 0.0)
            for name, cfg in ROLES_CONFIG.items()
        },
    )


def _apply_coherence(tracks, bp: MusicalBlueprint) -> list:
    root, mode = parse_key(bp.key)
    scale_set  = build_scale_set(root, mode)
    out = []
    for track in tracks:
        # Berlin School: expressive enough for passing tones (0.75 strictness),
        # chord tones favoured on strong beats but chromatic approaches allowed (tolerance=1)
        notes = scale_quantize(track.notes, bp.key, strictness=0.75)
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set,
                                   semitone_tolerance=1)
        notes = scale_velocity_by_role(notes, track.role, ranges=VELOCITY_RANGES)
        notes = humanize_velocities(notes, variation=12, phrase_contour=True)
        notes = nudge_timing(notes, max_ticks=15)  # expressive — Berlin School is loose
        out.append(replace(track, notes=notes))
    return out


def generate(
    active_roles: list[str],
    filename: str,
    source_midi: str | None = None,
    max_events: int = 128,
    hw_context_interval: int = 2,
    label: str = "",
) -> str | None:
    bp        = _blueprint(active_roles)
    src_label = f"← {Path(source_midi).name}" if source_midi else "cold start"
    bar("─")
    print(f"  {label or ', '.join(active_roles)}")
    print(f"  source: {src_label}")
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, SUITE_ROLES,
        max_events=max_events,
        hw_context_interval=hw_context_interval,
        source_midi=source_midi,
        source_context_bars=8,
    )
    dt = time.time() - t0

    if not tracks:
        print(f"  [!] model returned empty — skipping {filename}")
        return None

    stamped = []
    for t in tracks:
        if t.role == "melody":
            stamped.append(replace(t, program=LEAD_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=PAD_PROGRAM))
        elif t.role == "bass":
            stamped.append(replace(t, program=BASS_PROGRAM))
        else:
            stamped.append(t)

    polished   = _apply_coherence(stamped, bp)
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
    print("  tt-midi-maker  ▸  Silicon Road  (Berlin School / Tangerine Dream)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {LOOP_SECS:.1f}s per {BARS}-bar phrase")
    print(f"  Roles   : sawtooth lead + warm pads + synth bass (no drums)")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    ROLES = ["bass", "harmony", "melody"]

    f1 = generate(ROLES, "p1_void.mid",
                  max_events=128, hw_context_interval=2,
                  label="Pattern 1 — void (cold start, sparse emergence)")
    f2 = generate(ROLES, "p2_drift.mid",
                  source_midi=f1,
                  max_events=160, hw_context_interval=2,
                  label="Pattern 2 — drift (seeded from P1, texture develops)")
    f3 = generate(ROLES, "p3_shore.mid",
                  source_midi=f2,
                  max_events=160, hw_context_interval=2,
                  label="Pattern 3 — shore (seeded from P2, arrival)")

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
