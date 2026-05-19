#!/usr/bin/env python3
"""
"Slow Light" — a three-pattern spacey ambient piece.

Musical design
--------------
Key:    Eb major (lush, slightly warm — Eb-F-G-Ab-Bb-C-D)
BPM:    62  (very slow, atmospheric)
Bars:   8   (~15.5 s loop at 62 BPM — long, floating phrases)
Chords: Ebmaj7 → Cm7 → Abmaj7 → Bb  (I – vi – IV – V, very settled)
        No dominant tension — these chords resolve into each other
        without urgency. Time expands.

Instruments
-----------
  melody   — String Ensemble 1 (program 48)     Eb3–Eb6  (high, airy)
  harmony  — Pad 2 warm (program 89)            Bb2–Bb4  (wide, sustaining)
  bass     — Synth Bass 1 (program 38)          Eb1–Eb3  (deep, slow)
  drums    — Kit (ch10) — at 62 BPM the model tends to generate
             sparse, brushed-style patterns naturally
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
logger = logging.getLogger("ambient")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "slow-light"
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

KEY    = "Eb major"
BPM    = 62
BARS   = 8
CHORDS = ["Ebmaj7", "Cm7", "Abmaj7", "Bb"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

# ── GM program overrides ───────────────────────────────────────────────────────

STRING_PROGRAM = 48   # String Ensemble 1
STRING_RANGE   = [51, 87]  # Eb3 – Eb6  (high, airy register)
PAD_PROGRAM    = 89   # Pad 2 (warm)
PAD_RANGE      = [34, 70]  # Bb1 – Bb4  (warm mid-low range)
SYNTH_BASS_PROGRAM = 38   # Synth Bass 1
SYNTH_BASS_RANGE   = [27, 55]  # Eb1 – G3

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": STRING_PROGRAM, "note_range": STRING_RANGE}
    elif name == "harmony":
        overrides = {"program": PAD_PROGRAM, "note_range": PAD_RANGE}
    elif name == "bass":
        overrides = {"program": SYNTH_BASS_PROGRAM, "note_range": SYNTH_BASS_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


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
        # Ambient: strict scale (0.9) for clean harmonic consonance, no swing
        notes = scale_quantize(track.notes, bp.key, strictness=0.9)
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set)
        notes = humanize_velocities(notes, variation=5)  # subtle velocity variation
        notes = nudge_timing(notes, max_ticks=16)         # wider micro-timing for float
        out.append(replace(track, notes=notes))
    return out


def generate(
    active_roles: list[str],
    filename: str,
    source_midi: str | None = None,
    max_events: int = 80,
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
            stamped.append(replace(t, program=STRING_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=PAD_PROGRAM))
        elif t.role == "bass":
            stamped.append(replace(t, program=SYNTH_BASS_PROGRAM))
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
    print("  tt-midi-maker  ▸  Slow Light  (Eb major spacey ambient)")
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
        "p1_opening.mid",
        label="Pattern 1 — opening (cold start)",
        max_events=96,
    )
    f2 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p2_drift.mid",
        source_midi=f1,
        label="Pattern 2 — drift (seeded from P1)",
        max_events=96,
    )
    f3 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p3_dissolution.mid",
        source_midi=f2,
        label="Pattern 3 — dissolution (seeded from P2)",
        max_events=80,
    )

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
