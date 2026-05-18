#!/usr/bin/env python3
"""
"Smoke & Mirrors" — a four-pattern modal jazz suite in D minor.

Each pattern is generated on TT hardware and seeded from the previous one,
so the musical ideas carry across the whole piece.

Pattern 1  bass + drums            establish the groove
Pattern 2  + harmony (Hammond)     comping enters over the groove
Pattern 3  + melody (tenor sax)    sax solo seeded from P2
Pattern 4  full band variation     continues where P3 left off

Musical design
--------------
Key:    D minor (Aeolian)
BPM:    118  (medium-slow, deliberate)
Bars:   8    (13.1 s loop at 118 BPM)
Chords: Dm7  Am7  Bbmaj7  A7      (i7 – v7 – bVImaj7 – V7)
        Dark, circular, slightly unresolved — classic modal jazz gravity
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
logger = logging.getLogger("perf")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

OUTPUT_DIR = Path.home() / "Music" / "tt-midi-maker" / "smoke-and-mirrors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.stream_player import (
    loop_play, loop_queue, loop_stop, start_synth, synth_status,
)

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "D minor"
BPM    = 118
BARS   = 8
CHORDS = ["Dm7", "Am7", "Bbmaj7", "A7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

# ── Role overrides (all built on top of roles.yaml defaults) ───────────────────

# Hammond organ comping: program 16 = Hammond Organ, wide range for jazz voicings
HAMMOND_PROGRAM = 16
# Tenor sax melody: program 66, range Bb2–F5 (the real instrument's sweet zone)
TENOR_SAX_PROGRAM = 66
TENOR_SAX_RANGE   = [46, 77]   # Bb2 – F5

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "harmony":
        overrides = {"program": HAMMOND_PROGRAM}
    elif name == "melody":
        overrides = {"program": TENOR_SAX_PROGRAM, "note_range": TENOR_SAX_RANGE}
    elif name == "bass":
        # Widen from [28,52] — model rarely goes below E2; accept up to C4 so
        # walking lines in the low-mid register aren't silently dropped.
        overrides = {"note_range": [28, 60]}
    SUITE_ROLES[name] = {**cfg, **overrides}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Generate one pattern, apply coherence passes, save MIDI, return path."""
    bp = _blueprint(active_roles)
    src_label = f"← {Path(source_midi).name}" if source_midi else "cold start"
    title     = label or ", ".join(active_roles)

    bar("─")
    print(f"  {title}")
    print(f"  roles: {', '.join(active_roles)}    source: {src_label}")
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

    # Re-stamp custom programs onto roles that have overrides
    stamped = []
    for t in tracks:
        if t.role == "melody":
            stamped.append(replace(t, program=TENOR_SAX_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=HAMMOND_PROGRAM))
        else:
            stamped.append(t)

    polished = _apply_coherence(stamped, bp)
    out      = OUTPUT_DIR / filename
    build_midi_file(polished, bp.bpm, out)

    # Print per-track note counts
    role_summary = "  ".join(
        f"{t.role}:{len(t.notes)}n" for t in polished
    )
    loop_ratio = dt / LOOP_SECS
    hw_calls   = max(1, max_events // hw_context_interval)
    print(f"  generated in {dt:.1f}s  ({max_events/dt:.1f} ev/s)  "
          f"= {loop_ratio:.2f}× loop   ~{hw_calls} hw calls")
    print(f"  tracks: {role_summary}")
    print(f"  saved:  {out.name}")
    return str(out)


def bar(char="═", width=62):
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bar("═")
    print("  tt-midi-maker  ▸  Smoke & Mirrors  (D minor modal jazz)")
    bar("═")

    devices  = detect_tt_devices()
    backend  = f"tt-forge  ({len(devices)} P300C chip(s))" if devices else "CPU"
    print(f"  Backend  : {backend}")
    print(f"  Key      : {KEY}   BPM: {BPM}")
    print(f"  Chords   : {' → '.join(CHORDS)}")
    print(f"  Loop     : {LOOP_SECS:.1f}s per 8-bar phrase")
    print(f"  Output   : {OUTPUT_DIR}")
    bar()

    files = []

    # All patterns use all 4 roles: the model biases strongly toward generating
    # on the first patch-change channel (melody ch0=0), so omitting melody from
    # P1 collapses everything to one channel and cascades into all later patterns.
    # The musical "build" comes from seeding each pattern from the previous one.

    # ── Pattern 1: cold-start ensemble ────────────────────────────────────────
    f1 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p1_intro.mid",
        label="Pattern 1 — full ensemble, cold start",
    )
    files.append(f1)

    # ── Pattern 2: seeded continuation ────────────────────────────────────────
    f2 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p2_variation.mid",
        source_midi=f1,
        label="Pattern 2 — variation (seeded from P1)",
    )
    files.append(f2)

    # ── Pattern 3: development ────────────────────────────────────────────────
    f3 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p3_development.mid",
        source_midi=f2,
        label="Pattern 3 — development (seeded from P2)",
    )
    files.append(f3)

    # ── Pattern 4: climax — more events, denser texture ───────────────────────
    f4 = generate(
        ["bass", "drums", "harmony", "melody"],
        "p4_climax.mid",
        source_midi=f3,
        label="Pattern 4 — climax variation (seeded from P3)",
        max_events=128,
        hw_context_interval=4,
    )
    files.append(f4)

    bar("═")
    print("  All patterns generated.")
    bar()

    valid = [f for f in files if f]
    if not valid:
        print("  Nothing to play — exiting.")
        return

    # ── Playback ──────────────────────────────────────────────────────────────
    print()
    print("  Starting FluidSynth …")
    result = start_synth(gain=2.5, driver="pulseaudio")
    print(f"  ALSA port: {result['port']}")
    print()

    bar()
    print(f"  ▶  LOOPING  →  {Path(valid[0]).name}")
    loop_play(valid[0])

    for f in valid[1:]:
        loop_queue(f)
        print(f"  ⟳  QUEUED  →  {Path(f).name}")

    bar("═")
    print("  Playing 'Smoke & Mirrors'.  Ctrl+C to stop.\n")
    bar("═")
    print()

    try:
        last = None
        while True:
            st  = synth_status()["player"]
            cur = Path(st["current_file"]).name if st.get("current_file") else "—"
            nxt = Path(st["queued_file"]).name  if st.get("queued_file")  else "—"
            if cur != last:
                print(f"\n  ▶  {cur}")
                last = cur
            sys.stdout.write(
                f"\r  loop #{st['loops_played']:3d}  "
                f"playing: {cur:<30s}  next: {nxt:<30s}"
            )
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        print()

    loop_stop(immediately=True)
    bar("═")
    print("  Stopped.  Files are in:", OUTPUT_DIR)
    bar("═")


if __name__ == "__main__":
    main()
