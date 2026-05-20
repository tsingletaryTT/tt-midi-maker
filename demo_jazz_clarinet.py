#!/usr/bin/env python3
"""
Jazz demo: walking bass + piano comping + bass clarinet solo.

Pattern 1: bass + drums + piano comping (establish the groove)
Pattern 2: add bass clarinet solo, seeded from pattern 1 for continuity
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
    format="%(asctime)s  %(name)-16s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("jazz")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

OUTPUT_DIR = Path.home() / "Music" / "tt-midi-maker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role, swing_timing
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.coherence.structure import (
    build_genre_structure, generate_drum_groove, generate_walking_bass,
)
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import RoleTrack
from tt_midi_maker.stream_player import (
    loop_play, loop_queue, loop_stop, start_synth, synth_status,
)

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "C major"
BPM    = 138          # medium-up swing
BARS   = 8
CHORDS = ["Dm7", "G7", "Cmaj7", "Am7"]   # ii-V-I-vi: the beating heart of jazz

# GM program 71 (0-indexed) = Clarinet — closest to bass clarinet in GM
# Note range lowered to C3–C5 for that dark, woody bass clarinet register
BASS_CLARINET_PROGRAM = 71
BASS_CLARINET_RANGE   = [48, 72]   # C3–C5

# Piano comping: program 0 = Acoustic Grand Piano, harmony channel (ch3)
PIANO_PROGRAM = 0

# Jazz-tuned roles: melody → bass clarinet range, harmony → piano
JAZZ_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides = {}
    if name == "melody":
        overrides = {"program": BASS_CLARINET_PROGRAM, "note_range": BASS_CLARINET_RANGE}
    elif name == "harmony":
        overrides = {"program": PIANO_PROGRAM}
    JAZZ_ROLES[name] = {**cfg, **overrides}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blueprint(active_roles: list[str]) -> MusicalBlueprint:
    roles = {
        name: RoleConfig(density=cfg["density_default"] if name in active_roles else 0.0)
        for name, cfg in ROLES_CONFIG.items()
    }
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="jazz", mode="loop",
        chord_progression=CHORDS,
        roles=roles,
    )


def _apply_coherence(tracks, bp: MusicalBlueprint, tension: float = 0.0) -> list:
    root, mode = parse_key(bp.key)
    scale_set  = build_scale_set(root, mode)
    tpb        = TICKS_PER_BEAT
    tpbar      = tpb * 4
    out        = []
    for track in tracks:
        if track.role in ("bass", "drums"):
            out.append(track)
            continue
        notes = scale_quantize(track.notes, bp.key)
        notes = chord_aware_filter(notes, bp.chord_progression, tpbar, tpb, scale_set)
        if track.role == "melody" and tension > 0.0:
            notes = add_approach_notes(notes, bp.chord_progression, tpbar,
                                       tension=tension, seed=42)
        notes = scale_velocity_by_role(notes, track.role)
        notes = humanize_velocities(notes)
        notes = swing_timing(notes, swing_ratio=0.60)   # jazz swing
        notes = nudge_timing(notes)
        out.append(replace(track, notes=notes))
    return out


def generate_pattern(
    active_roles: list[str],
    filename: str,
    source_midi: str | None = None,
    max_events: int = 96,
    hw_context_interval: int = 4,
    tension: float = 0.0,
) -> str | None:
    """Generate melody+harmony, inject deterministic walking bass and swing_ride drums.

    max_events=96 with hw_context_interval=4 targets ≈12s on P300C hardware
    which fits within one 8-bar loop at 138 BPM (13.9s).
    """
    structure = build_genre_structure("jazz_swing", KEY, CHORDS, bars=BARS, tension=tension)
    bp        = _blueprint(["melody", "harmony"])   # model generates melody+harmony only
    label     = f"continuing from {Path(source_midi).name}" if source_midi else "cold start"
    logger.info("generating [melody+harmony]  %s …", label)
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, JAZZ_ROLES, max_events=max_events,
        hw_context_interval=hw_context_interval,
        source_midi=source_midi, source_context_bars=8,
        max_attempts=3, judge_threshold=0.55,
    )
    if not tracks:
        logger.warning("model returned empty — skipping %s", filename)
        return None

    # Stamp programs onto model-generated tracks
    stamped = [
        replace(t, program=BASS_CLARINET_PROGRAM) if t.role == "melody" else t
        for t in tracks
    ]

    # Inject deterministic walking bass and swing_ride drums
    bass_notes = generate_walking_bass(CHORDS, BARS, ticks_per_beat=TICKS_PER_BEAT,
                                        velocity=72, channel=2)
    drum_notes = generate_drum_groove("swing_ride", BARS, ticks_per_beat=TICKS_PER_BEAT)
    bass_track = RoleTrack(role="bass",  channel=2,  program=32, notes=bass_notes)
    drum_track = RoleTrack(role="drums", channel=10, program=0,  notes=drum_notes)
    all_tracks = stamped + [bass_track, drum_track]

    polished = _apply_coherence(all_tracks, bp, tension=tension)
    out      = OUTPUT_DIR / filename
    build_midi_file(polished, bp.bpm, out)

    dt      = time.time() - t0
    summary = [(t.role, f"{len(t.notes)}n") for t in polished]
    logger.info("done %.1fs → %s  %s", dt, filename, summary)
    return str(out)


def bar(char="─", width=60):
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bar("═")
    print("  tt-midi-maker  ▸  jazz bass clarinet demo")
    bar("═")

    devices   = detect_tt_devices()
    hw_label  = f"tt-forge ({len(devices)} device(s))" if devices else "CPU"
    loop_secs = BARS * (60.0 / BPM) * 4

    print(f"  Backend  : {hw_label}")
    print(f"  Key      : {KEY}   BPM: {BPM}")
    print(f"  Chords   : {' → '.join(CHORDS)}")
    print(f"  Loop     : {loop_secs:.1f}s")
    print(f"  Output   : {OUTPUT_DIR}")
    bar()

    # ── Pattern 1: groove foundation ─────────────────────────────────────────
    print()
    print("  [1/2] Groove — walking bass + piano comping + swing_ride drums")
    f1 = generate_pattern(
        ["melody", "harmony"],   # model generates these; bass+drums are deterministic
        "jazz_1_groove.mid",
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
    print("     Walking bass, piano comping, swing_ride drums. C major, 138 BPM.")
    loop_play(f1)
    print()

    # ── Pattern 2: bass clarinet opens up ────────────────────────────────────
    print("  [2/2] Bass clarinet solo — approach notes open up …")
    print(f"        (generating while groove plays — {loop_secs:.0f}s per loop)")
    f2 = generate_pattern(
        ["melody", "harmony"],
        "jazz_2_clarinet_solo.mid",
        source_midi=f1,
        tension=0.35,           # approach notes add jazz chromatic color
    )
    if f2:
        loop_queue(f2)
        bar()
        print(f"  ⟳  QUEUED  →  {Path(f2).name}")
        print("     Bass clarinet solo over the ii-V-I-vi. Dark, woody register.")
        print()

    bar("═")
    print("  Both patterns queued. Ctrl+C to stop.\n")
    bar("═")
    print()

    try:
        last_file = None
        while True:
            st  = synth_status()["player"]
            cur = Path(st["current_file"]).name if st.get("current_file") else "—"
            nxt = Path(st["queued_file"]).name  if st.get("queued_file")  else "—"
            if cur != last_file:
                print(f"  ▶  {cur}")
                last_file = cur
            sys.stdout.write(
                f"\r  loop #{st['loops_played']:3d}  "
                f"playing: {cur:<35s}  "
                f"next: {nxt:<35s}  "
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
