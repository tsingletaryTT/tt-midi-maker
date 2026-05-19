#!/usr/bin/env python3
"""
"Aria in D Minor" — three-pattern Baroque/Classical piece.

Musical design
--------------
Key:    D minor  (D E F G A Bb C — natural minor, darkly expressive)
BPM:    76       (Andante — unhurried, with weight)
Bars:   8        (~25.3 s loop)
Chords: Dm → F → Gm → A7   (i – III – iv – V7)
        The A7 creates the sharpened leading tone (C#→D) typical of
        harmonic minor — the chord filter restores it on strong beats.

Instruments
-----------
  melody   — Acoustic Grand Piano (program 0)    D4–D7  (singing treble line)
  harmony  — String Ensemble 1   (program 48)    A2–A5  (lush string chords)
  bass     — Contrabass          (program 43)    D1–A3  (bowed bass continuo)
  drums    — Silent (density 0)  — no percussion in this piece

The model favours generating piano notes — string chord notes are fewer but boosted
to velocity 90-115 so each one is clearly audible in the mix (MuseScore sf3 helps too).
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
logger = logging.getLogger("classical")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "aria-d-minor"
OUTPUT_DIR  = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

KEY    = "D minor"
BPM    = 76
BARS   = 8
CHORDS = ["Dm", "F", "Gm", "A7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

PIANO_PROGRAM     = 0    # Acoustic Grand Piano — singing treble line
PIANO_RANGE       = [62, 98]  # D4 – D7  (high register; model prefers generating here)
STRING_PROGRAM    = 48   # String Ensemble 1 — chord support
STRING_RANGE      = [45, 69]  # A2 – A5  (orchestral string range)
CONTRABASS_PROGRAM = 43  # Contrabass (bowed)
CONTRABASS_RANGE  = [26, 57]  # D1 – A3  (full bass register)

# Strings (harmony) are boosted to be clearly audible despite fewer notes.
# The model prefers generating piano (program 0) on the melody channel —
# that's accepted, with string chords prominent when they appear.
VELOCITY_RANGES = {
    "melody":  (80, 108),  # piano melody — bright, singing
    "harmony": (90, 115),  # strings — boosted so each note is clearly present
    "bass":    (50,  75),  # contrabass — supportive foundation
    "drums":   (60, 100),  # unused (no drums in this piece)
}

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": PIANO_PROGRAM, "note_range": PIANO_RANGE}
    elif name == "harmony":
        overrides = {"program": STRING_PROGRAM, "note_range": STRING_RANGE}
    elif name == "bass":
        overrides = {"program": CONTRABASS_PROGRAM, "note_range": CONTRABASS_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="classical", mode="loop",
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
        # Classical: strict scale adherence (0.9). The A7 chord's C# (leading tone)
        # is restored by chord_aware_filter on strong beats — authentic harmonic minor.
        notes = scale_quantize(track.notes, bp.key, strictness=0.9)
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set,
                                   semitone_tolerance=0)
        notes = scale_velocity_by_role(notes, track.role, ranges=VELOCITY_RANGES)
        notes = humanize_velocities(notes, variation=10, phrase_contour=True)
        notes = nudge_timing(notes, max_ticks=12)   # subtle expressive timing
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
        max_attempts=3, judge_threshold=0.55,
    )
    dt = time.time() - t0

    if not tracks:
        print(f"  [!] model returned empty — skipping {filename}")
        return None

    stamped = []
    for t in tracks:
        if t.role == "melody":
            stamped.append(replace(t, program=PIANO_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=STRING_PROGRAM))
        elif t.role == "bass":
            stamped.append(replace(t, program=CONTRABASS_PROGRAM))
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
    print("  tt-midi-maker  ▸  Aria in D Minor  (Baroque / Classical)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {LOOP_SECS:.1f}s per {BARS}-bar phrase")
    print(f"  Roles   : piano + strings (boosted velocity) + contrabass (no drums)")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    # No drums in a Baroque/Classical continuo ensemble
    ROLES = ["bass", "harmony", "melody"]

    f1 = generate(ROLES, "p1_exposition.mid",
                  max_events=128, hw_context_interval=2,
                  label="Pattern 1 — exposition (cold start)")
    f2 = generate(ROLES, "p2_development.mid",
                  source_midi=f1,
                  max_events=160, hw_context_interval=2,
                  label="Pattern 2 — development (seeded from P1)")
    f3 = generate(ROLES, "p3_recapitulation.mid",
                  source_midi=f2,
                  max_events=160, hw_context_interval=2,
                  label="Pattern 3 — recapitulation (seeded from P2)")

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
