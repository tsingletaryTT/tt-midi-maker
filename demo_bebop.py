#!/usr/bin/env python3
"""
"Quick Changes" — a three-pattern bebop jazz suite in Bb major.

Musical design
--------------
Key:    Bb major (standard bebop key — horn-friendly fingerings)
BPM:    200  (true bebop tempo — this is fast)
Bars:   8    (~4.8 s loop at 200 BPM — tight, electric)
Chords: Bbmaj7 → G7 → Cm7 → F7   (I – VI7 – ii7 – V7)
        The classic bebop turnaround. The G7 pulls the ear down toward
        Cm, the F7 pushes it back up to Bb. Harmonic gravity in motion.

Instruments
-----------
  melody   — Alto Saxophone (program 65)   Db3–Ab5  (Bird range)
  harmony  — Acoustic Grand Piano (prog 0)  Bb2–Bb5  (wide comp range)
  bass     — Acoustic Bass (program 32)    Bb1–F3   (walking range)
  drums    — Standard kit (channel 10)     — at 200 BPM, ride-cymbal swing

Note on tempo: 200 BPM means LOOP_SECS ≈ 4.8s. With max_events=96 and
7.8 ev/s generation throughput, this pattern finishes in ~12s — 2.5 loops.
The hardware still stays ahead of real time on a per-pattern basis.
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
logger = logging.getLogger("bebop")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "bebop-quick-changes"
OUTPUT_DIR  = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role, swing_timing
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "Bb major"
BPM    = 200
BARS   = 8
CHORDS = ["Bbmaj7", "G7", "Cm7", "F7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

# ── GM program overrides ───────────────────────────────────────────────────────

ALTO_SAX_PROGRAM = 65   # Alto Sax
ALTO_SAX_RANGE   = [49, 81]  # Db3 – Ab5 (classic alto sax range)
PIANO_PROGRAM    = 0    # Acoustic Grand Piano
PIANO_RANGE      = [34, 82]  # Bb1 – Bb5 (wide comp range)
BASS_PROGRAM     = 32   # Acoustic Bass
BASS_RANGE       = [34, 53]  # Bb1 – F3  (walking bass range)

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": ALTO_SAX_PROGRAM, "note_range": ALTO_SAX_RANGE}
    elif name == "harmony":
        overrides = {"program": PIANO_PROGRAM, "note_range": PIANO_RANGE}
    elif name == "bass":
        overrides = {"program": BASS_PROGRAM, "note_range": BASS_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="jazz", mode="loop",
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
        # Bebop uses heavy chromaticism — very low strictness to preserve approach notes
        notes = scale_quantize(track.notes, bp.key, strictness=0.25)
        # Chromatic approach notes are within 1 semitone of chord tones — leave them
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set,
                                   semitone_tolerance=1)
        notes = scale_velocity_by_role(notes, track.role)
        notes = humanize_velocities(notes)
        notes = swing_timing(notes, swing_ratio=0.63)   # medium bebop swing
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
            stamped.append(replace(t, program=ALTO_SAX_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=PIANO_PROGRAM))
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
    print("  tt-midi-maker  ▸  Quick Changes  (Bb major bebop)")
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
        "p1_head.mid",
        max_events=128, hw_context_interval=2,
        label="Pattern 1 — head (cold start)",
    )
    f2 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p2_solo1.mid",
        source_midi=f1,
        max_events=128, hw_context_interval=2,
        label="Pattern 2 — first chorus (seeded from P1)",
    )
    f3 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p3_solo2.mid",
        source_midi=f2,
        max_events=160, hw_context_interval=2,
        label="Pattern 3 — second chorus (seeded from P2)",
    )

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
