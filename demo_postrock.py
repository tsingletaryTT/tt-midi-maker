#!/usr/bin/env python3
"""
"Smoke & Mirrors" — post-rock bass build.

Musical design
--------------
Key:    E minor
BPM:    92  (mid-tempo post-rock — Mogwai / GY!BE territory)
Bars:   8
Chords: Em → D → C → D  (i → bVII → bVI → bVII)

Structure (3-pattern arc)
-------------------------
P1: Bass foundation + drums — tension=0.0  (clean, grounded)
P2: Add high-register Electric Bass solo  — tension=0.45  (melodic climb)
P3: Full build: bass + solo + strings pad — tension=0.75  (post-rock climax)

Instruments
-----------
  bass     — Acoustic Bass (prog 32)   low foundation
  melody   — Electric Bass Finger (prog 33)  upper register solo (P2, P3)
  strings  — String Ensemble 1 (prog 48)     pad swell (P3 only)
  drums    — Standard kit (ch 10)
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
logger = logging.getLogger("postrock")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "smoke-and-mirrors"
OUTPUT_DIR  = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "E minor"
BPM    = 92
BARS   = 8
CHORDS = ["Em", "D", "C", "D"]   # classic post-rock: i → bVII → bVI → bVII
STYLE  = "post-rock"

# The melody voice (ch1) is repurposed as a high-register bass solo.
# GM program 33 = Electric Bass (Finger) — same family as the low bass (32),
# but the melody note_range [60–96] puts it an octave+ above the bass.
SOLO_PROGRAM = 33


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    roles = {}
    for name, cfg in ROLES_CONFIG.items():
        if name in active_roles:
            roles[name] = RoleConfig(density=cfg["density_default"])
        else:
            roles[name] = RoleConfig(density=0.0)
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style=STYLE, mode="loop",
        chord_progression=CHORDS,
        roles=roles,
    )


def _apply_coherence(tracks, bp: MusicalBlueprint, tension: float = 0.0) -> list:
    root, mode = parse_key(bp.key)
    scale_set  = build_scale_set(root, mode)
    out = []
    for track in tracks:
        notes = scale_quantize(track.notes, bp.key)
        notes = chord_aware_filter(notes, bp.chord_progression,
                                   4 * TICKS_PER_BEAT, TICKS_PER_BEAT, scale_set)
        notes = scale_velocity_by_role(notes, track.role)
        # Tension arc: post-rock builds dramatically — high tension values create
        # dense chromatic approach figures before downbeats (Mogwai/GY!BE climax feel)
        if track.role == "melody" and tension > 0.0:
            notes = add_approach_notes(notes, bp.chord_progression,
                                       4 * TICKS_PER_BEAT, TICKS_PER_BEAT,
                                       tension=tension, seed=42)
        notes = humanize_velocities(notes)
        notes = nudge_timing(notes)
        out.append(replace(track, notes=notes))
    return out


def bar(char="─", width=66):
    print(char * width)


def generate(
    active_roles: list[str],
    filename: str,
    label: str = "",
    override_melody_program: bool = False,
    source_midi: str | None = None,
    tension: float = 0.0,
    max_events: int = 256,
    hw_context_interval: int = 4,
) -> str | None:
    """Generate pattern, apply coherence, save. Returns path or None."""
    bp = _blueprint(active_roles)
    src_label = f"← {Path(source_midi).name}" if source_midi else "cold start"
    print("─" * 66)
    print(f"  {label}  (tension={tension:.2f})")
    print(f"  roles: {active_roles}  source: {src_label}")
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, ROLES_CONFIG, max_events=max_events,
        hw_context_interval=hw_context_interval,
        source_midi=source_midi, source_context_bars=8,
        max_attempts=3, judge_threshold=0.55,
    )
    if not tracks:
        print(f"  [!] model returned empty — skipping {filename}")
        return None

    if override_melody_program:
        tracks = [
            replace(t, program=SOLO_PROGRAM) if t.role == "melody" else t
            for t in tracks
        ]

    polished = _apply_coherence(tracks, bp, tension=tension)
    out = OUTPUT_DIR / filename
    build_midi_file(polished, bp.bpm, out)

    dt = time.time() - t0
    summary = [(t.role, f"{len(t.notes)}n") for t in polished]
    loop_secs = BARS * (60.0 / BPM) * 4
    print(f"  {dt:.1f}s ({dt / loop_secs:.2f}× loop)  {summary}")
    print(f"  saved: {out}")
    return str(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bar("═")
    print("  tt-midi-maker  ▸  Smoke & Mirrors  (post-rock, E minor, 92 BPM)")
    bar("═")

    devices  = detect_tt_devices()
    backend  = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    loop_secs = BARS * (60.0 / BPM) * 4
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}   Bars: {BARS}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {loop_secs:.1f}s")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    # P1: grounded foundation
    f1 = generate(
        ["bass", "drums"],
        "p1_foundation.mid",
        label="P1 — bass + drums  (grounded foundation)",
        tension=0.0,
    )
    if f1 is None:
        print("  ERROR: generation failed"); sys.exit(1)

    # P2: melodic climb — high-register electric bass solo joins
    f2 = generate(
        ["bass", "melody", "drums"],
        "p2_climb.mid",
        label="P2 — bass + solo  (melodic climb)",
        override_melody_program=True,
        source_midi=f1,
        tension=0.45,
    )

    # P3: full build — strings swell at peak tension
    f3 = generate(
        ["bass", "melody", "strings", "drums"],
        "p3_climax.mid",
        label="P3 — full build  (post-rock climax)",
        override_melody_program=True,
        source_midi=f2 or f1,
        tension=0.75,
    )

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
