#!/usr/bin/env python3
"""
"Midnight in East Texas" — a three-pattern E minor blues.

Musical design
--------------
Key:    E minor (blues pentatonic: E G A Bb B D)
BPM:    88  (medium shuffle — room to breathe)
Bars:   8   (~10.9 s loop at 88 BPM)
Chords: E7 → A7 → E7 → B7   (8-bar blues cycle)
        Dominant 7ths throughout — that suspended, unresolved tension
        that defines the blues.

Instruments
-----------
  melody   — Electric guitar (jazz, program 26)  E2–G5
  harmony  — Hammond organ (drawbar, program 16)  C3–C5
  bass     — Electric bass finger (program 33)    E1–E3
  drums    — Standard kit (channel 10)
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
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "E minor"
BPM    = 88
BARS   = 8
CHORDS = ["E7", "A7", "E7", "B7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

# ── GM program overrides ───────────────────────────────────────────────────────

GUITAR_PROGRAM = 26   # Electric Guitar (jazz)
GUITAR_RANGE   = [40, 79]  # E2 – G5
HAMMOND_PROGRAM = 16  # Drawbar Organ
BASS_PROGRAM    = 33  # Electric Bass (finger)
BASS_RANGE      = [28, 52]  # E1 – E3

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": GUITAR_PROGRAM, "note_range": GUITAR_RANGE}
    elif name == "harmony":
        overrides = {"program": HAMMOND_PROGRAM}
    elif name == "bass":
        overrides = {"program": BASS_PROGRAM, "note_range": BASS_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="blues", mode="loop",
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
        notes = scale_quantize(track.notes, bp.key)
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set)
        notes = humanize_velocities(notes)
        notes = nudge_timing(notes)
        out.append(replace(track, notes=notes))
    return out


def generate(
    active_roles: list[str],
    filename: str,
    source_midi: str | None = None,
    max_events: int = 96,
    hw_context_interval: int = 4,
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
            stamped.append(replace(t, program=GUITAR_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=HAMMOND_PROGRAM))
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
    print("  tt-midi-maker  ▸  Midnight in East Texas  (E minor blues)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {LOOP_SECS:.1f}s per {BARS}-bar phrase")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    f1 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p1_intro.mid",
        label="Pattern 1 — full band, cold start",
    )
    f2 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p2_groove.mid",
        source_midi=f1,
        label="Pattern 2 — groove (seeded from P1)",
    )
    f3 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p3_resolution.mid",
        source_midi=f2,
        label="Pattern 3 — resolution (seeded from P2)",
        max_events=112,
    )

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
