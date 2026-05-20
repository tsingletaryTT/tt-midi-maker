#!/usr/bin/env python3
"""
tt-midi-maker demo: post-rock bass with gradual high-register solo build.

Pattern 1 (plays immediately):
  Low bass + drums — E minor, 92 BPM, 8 bars

Pattern 2 (queues while P1 loops):
  Low bass + Electric Bass solo in upper register + drums

Pattern 3 (queues while P2 loops):
  Full build — bass + solo + strings pad + drums
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
logger = logging.getLogger("demo")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

OUTPUT_DIR = Path.home() / "Music" / "tt-midi-maker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.stream_player import (
    loop_play, loop_queue, loop_stop, start_synth, synth_status,
)

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


def generate_pattern(
    active_roles: list[str],
    filename: str,
    max_events: int = 256,
    override_melody_program: bool = False,
    source_midi: str | None = None,
    source_context_bars: int | None = 8,
    tension: float = 0.0,
) -> str | None:
    """Generate, apply coherence, save, return path. None on empty output.

    source_midi: path to a previous generation to use as musical context.
    The model sees the last source_context_bars of that file before generating,
    so each pattern builds naturally on the previous one.
    tension: approach-note probability (0.0=none, 0.75=dramatic climax).
    """
    bp = _blueprint(active_roles)
    if source_midi:
        logger.info("generating [%s] max_events=%d (continuing from %s) …",
                    ", ".join(active_roles), max_events, Path(source_midi).name)
    else:
        logger.info("generating [%s] max_events=%d …", ", ".join(active_roles), max_events)
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, ROLES_CONFIG, max_events=max_events,
        source_midi=source_midi, source_context_bars=source_context_bars,
        max_attempts=3, judge_threshold=0.55,
    )
    if not tracks:
        logger.warning("model returned empty — skipping %s", filename)
        return None

    if override_melody_program:
        # Recast the melody voice as a high-register Electric Bass (Finger)
        tracks = [
            replace(t, program=SOLO_PROGRAM) if t.role == "melody" else t
            for t in tracks
        ]

    tracks = _apply_coherence(tracks, bp, tension=tension)
    out = OUTPUT_DIR / filename
    build_midi_file(tracks, bp.bpm, out)

    dt = time.time() - t0
    summary = [(t.role, f"ch{t.channel}", f"prog{t.program}",
                f"{len(t.notes)}n") for t in tracks]
    logger.info("done %.1fs → %s  %s", dt, filename, summary)
    return str(out)


def bar(char="─", width=66):
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bar("═")
    print("  tt-midi-maker  ▸  post-rock bass build demo")
    bar("═")

    devices = detect_tt_devices()
    hw_label = f"tt-forge ({len(devices)} device(s))" if devices else "CPU (no TT hardware)"
    print(f"  Backend  : {hw_label}")
    print(f"  Key      : {KEY}   BPM: {BPM}")
    print(f"  Chords   : {' → '.join(CHORDS)}")
    print(f"  Output   : {OUTPUT_DIR}")
    bar()

    # ── Pattern 1: bass + drums ───────────────────────────────────────────────
    print()
    print("  [1/3] Pattern 1 — post-rock bass + drums")
    f1 = generate_pattern(
        ["bass", "drums"],
        "demo_1_bass_drums.mid",
        max_events=256,
        tension=0.0,
    )
    if f1 is None:
        print("  ERROR: generation failed"); sys.exit(1)

    # Start synth and begin looping immediately
    bar()
    print("  Starting FluidSynth …")
    result = start_synth(gain=2.5, driver="pulseaudio")
    print(f"  ALSA port: {result['port']}")

    bar()
    print(f"  ▶  LOOPING NOW  →  {Path(f1).name}")
    print("     Low bass + drums. E minor. Post-rock groove.")
    loop_play(f1)
    print()

    loop_dur = BARS * (60.0 / BPM) * 4   # seconds per loop

    # ── Pattern 2: add high-register bass solo ────────────────────────────────
    print("  [2/3] Pattern 2 — add high-register Electric Bass solo …")
    print(f"        (loop is {loop_dur:.1f}s; generating while it plays)")
    f2 = generate_pattern(
        ["bass", "melody", "drums"],
        "demo_2_bass_solo.mid",
        max_events=256,
        override_melody_program=True,   # melody voice → prog 33 = Electric Bass Finger
        source_midi=f1,                 # continue from pattern 1
        tension=0.45,
    )
    if f2:
        loop_queue(f2)
        bar()
        print(f"  ⟳  QUEUED  →  {Path(f2).name}")
        print("     High-register Electric Bass solo joins at next loop boundary.")
        print("     (same fingered-bass family as the low line, one octave up)")
        print()

    # ── Pattern 3: full build — add strings swell ─────────────────────────────
    print("  [3/3] Pattern 3 — full build: bass + solo + strings pad …")
    f3 = generate_pattern(
        ["bass", "melody", "strings", "drums"],
        "demo_3_full_build.mid",
        max_events=256,
        override_melody_program=True,
        source_midi=f2 or f1,           # continue from pattern 2 (or 1 if 2 failed)
        tension=0.75,
    )
    if f3:
        loop_queue(f3)
        bar()
        print(f"  ⟳  QUEUED  →  {Path(f3).name}")
        print("     Strings pad layer joins at next boundary for the full build.")
        print()

    bar("═")
    print("  All patterns queued. Transitioning seamlessly at loop boundaries.")
    print("  Press Ctrl+C to stop.\n")
    bar("═")
    print()

    # Live status ticker
    try:
        last_file = None
        while True:
            st = synth_status()["player"]
            cur = Path(st["current_file"]).name if st.get("current_file") else "—"
            nxt = Path(st["queued_file"]).name  if st.get("queued_file")  else "—"
            if cur != last_file:
                print(f"  ▶  {cur}")
                last_file = cur
            # Show loop count update every 2s quietly
            sys.stdout.write(
                f"\r  loop #{st['loops_played']:3d}  "
                f"playing: {cur:<30s}  "
                f"next: {nxt:<30s}  "
            )
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        print()

    loop_stop(immediately=True)
    bar("═")
    print("  Stopped.")
    bar("═")


if __name__ == "__main__":
    main()
