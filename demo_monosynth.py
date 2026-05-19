#!/usr/bin/env python3
"""
Monosynth / chiptune demo — single-voice generation, TT hardware accelerated.

Musical design
--------------
Key:    C major (pentatonic: C D E G A)
BPM:    120     (uptempo chip feel)
Bars:   4       (tight loop — clean melodic statement)
Chords: Cmaj7 → Am7 → Fmaj7 → G7  (I–vi–IV–V)

Instruments
-----------
  melody   — Lead 1 Square (GM program 80)  — classic chiptune square wave
  all others — density 0.0 (silent)

Why single-instrument focus produces cleaner output
---------------------------------------------------
1. Channel clarity: all other roles at density=0.0 — the model sees no patch_change
   prompts for other channels, so it concentrates generation on ch0/melody.
2. Unambiguous coherence: scale_quantize and chord_aware_filter on one voice have
   no cross-channel pitch conflicts or rhythm sync issues to resolve.
3. Chiptune programs: GM 80 (Square) appears in the model's training data and
   produces coherent melodic lines — authentic chip-register timbre.

Fallback: if no TT hardware is detected, model.generate() runs on CPU.
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
logger = logging.getLogger("monosynth")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME = "cpu-monosynth"
OUTPUT_DIR = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.scale import scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

KEY    = "C major"
BPM    = 120
BARS   = 4
CHORDS = ["Cmaj7", "Am7", "Fmaj7", "G7"]

LEAD_PROGRAM = 80        # Lead 1 (Square) — classic chiptune square wave
LEAD_RANGE   = [60, 96]  # C4–C7  (upper register, bright chip feel)

LOOP_SECS = BARS * 4 * (60.0 / BPM)  # 8.0s

VELOCITY_RANGES = {
    "melody":  (85, 110),
    "harmony": (55,  80),
    "bass":    (50,  75),
    "drums":   (60, 100),
}

# Build suite roles with lead program on melody
SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    if name == "melody":
        SUITE_ROLES[name] = {**cfg, "program": LEAD_PROGRAM, "note_range": LEAD_RANGE}
    else:
        SUITE_ROLES[name] = {**cfg}


def _blueprint() -> MusicalBlueprint:
    """Single active role: melody only. All others at density 0 (silent)."""
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="chiptune", mode="loop",
        chord_progression=CHORDS,
        roles={
            "melody": RoleConfig(density=1.0),
            **{name: RoleConfig(density=0.0) for name in ROLES_CONFIG if name != "melody"},
        },
    )


def _apply_coherence(tracks) -> list:
    out = []
    for track in tracks:
        # Strict C major pentatonic: no chromatic passing tones — clean chip feel
        notes = scale_quantize(track.notes, KEY, strictness=0.9,
                               override_mode="pentatonic_major")
        notes = scale_velocity_by_role(notes, track.role, ranges=VELOCITY_RANGES)
        notes = humanize_velocities(notes, variation=6)
        notes = nudge_timing(notes, max_ticks=6)  # tight rhythmic lock — chip feel
        out.append(replace(track, notes=notes))
    return out


def generate(
    filename: str,
    source_midi: str | None = None,
    label: str = "",
) -> str | None:
    bp        = _blueprint()
    src_label = f"← {Path(source_midi).name}" if source_midi else "cold start"
    print("─" * 60)
    print(f"  {label}")
    print(f"  source: {src_label}")
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, SUITE_ROLES,
        max_events=96,           # hardware path: enough for a full 4-bar melodic phrase
        hw_context_interval=2,   # refresh hardware context every 2 steps for quality
        source_midi=source_midi,
        source_context_bars=4,
    )
    dt = time.time() - t0

    if not tracks:
        print(f"  [!] model returned empty — skipping {filename}")
        return None

    stamped  = [replace(t, program=LEAD_PROGRAM) if t.role == "melody" else t for t in tracks]
    polished = _apply_coherence(stamped)
    out      = OUTPUT_DIR / filename
    build_midi_file(polished, bp.bpm, out)

    melody_tracks = [t for t in polished if t.role == "melody"]
    total_notes   = sum(len(t.notes) for t in melody_tracks)
    loop_ratio    = dt / LOOP_SECS
    print(f"  {dt:.1f}s  ({96 / max(dt, 0.001):.1f} ev/s)  = {loop_ratio:.2f}× loop")
    print(f"  melody notes: {total_notes}")
    print(f"  saved: {out}")
    return str(out)


def bar(char="═", width=60):
    print(char * width)


def main():
    bar("═")
    print("  tt-midi-maker  ▸  CPU Monosynth  (C major chiptune)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU (no hardware)"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}")
    print(f"  Bars    : {BARS}  ({LOOP_SECS:.1f}s loop)")
    print(f"  Program : {LEAD_PROGRAM} (Lead 1 Square — chiptune square wave)")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  max_events: 96  /  hw_context_interval: 2")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    f1 = generate("p1_theme.mid",
                  label="Pattern 1 — theme (cold start)")
    f2 = generate("p2_variation.mid",
                  source_midi=f1,
                  label="Pattern 2 — variation (seeded from P1)")
    f3 = generate("p3_development.mid",
                  source_midi=f2,
                  label="Pattern 3 — development (seeded from P2)")

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
