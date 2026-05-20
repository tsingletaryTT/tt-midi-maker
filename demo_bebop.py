#!/usr/bin/env python3
"""
"Quick Changes" — a three-pattern bebop jazz suite in Bb major.

Musical design
--------------
Key:    Bb major (standard bebop key — horn-friendly fingerings)
BPM:    200  (true bebop tempo — this is fast)
Bars:   8    (~4.8 s loop at 200 BPM — tight, electric)
Chords: Bbmaj7 → G7 → Cm7 → F7   (I – VI7 – ii7 – V7)
        The classic bebop turnaround.

Structure (deterministic)
---------
  Walking bass  : 4 quarter notes/bar — locked bebop walk
  Drum groove   : swing_ride (jazz ride pattern, snare 2+4, pedal hi-hat)
  No phrase gaps: bebop is continuous melodic flow, no call-response

Improv (stochastic, deterministic seeds)
------
  Approach notes : chromatic half-step before chord-tone downbeats
  Tension arc    : P1=0.0 (head), P2=0.4 (first chorus), P3=0.7 (peak improv)
  Source chaining: P1→P2→P3 for motivic development

Instruments
-----------
  melody   — Alto Saxophone (program 65)   Db3–Ab5  (Bird range)
  harmony  — Acoustic Grand Piano (prog 0)  Bb2–Bb5  (wide comp range)
  bass     — Acoustic Bass (program 32)    Bb1–F3   (walking range, deterministic)
  drums    — Standard kit (channel 10)     swing_ride groove (deterministic)
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
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.coherence.structure import (
    build_genre_structure, generate_drum_groove, generate_walking_bass,
)
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import RoleTrack

# ── Musical parameters ─────────────────────────────────────────────────────────

KEY    = "Bb major"
BPM    = 200
BARS   = 8
CHORDS = ["Bbmaj7", "G7", "Cm7", "F7"]

LOOP_SECS = BARS * 4 * (60.0 / BPM)

ALTO_SAX_PROGRAM = 65   # Alto Sax
ALTO_SAX_RANGE   = [49, 81]  # Db3 – Ab5 (classic alto sax range)
PIANO_PROGRAM    = 0    # Acoustic Grand Piano
PIANO_RANGE      = [34, 82]  # Bb1 – Bb5
BASS_PROGRAM     = 32   # Acoustic Bass
BASS_RANGE       = (34, 53)  # Bb1 – F3 (walking bass range for bebop)

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": ALTO_SAX_PROGRAM, "note_range": ALTO_SAX_RANGE}
    elif name == "harmony":
        overrides = {"program": PIANO_PROGRAM, "note_range": PIANO_RANGE}
    elif name == "bass":
        overrides = {"program": BASS_PROGRAM, "note_range": list(BASS_RANGE)}
    SUITE_ROLES[name] = {**cfg, **overrides}


def _blueprint(active_roles: list[str], chord_progression: list[str]) -> MusicalBlueprint:
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="jazz", mode="loop",
        chord_progression=chord_progression,
        roles={
            name: RoleConfig(density=cfg["density_default"] if name in active_roles else 0.0)
            for name, cfg in ROLES_CONFIG.items()
        },
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
        # Bebop uses heavy chromaticism — very low strictness to preserve approach notes
        notes = scale_quantize(track.notes, bp.key, strictness=0.25)
        notes = chord_aware_filter(notes, bp.chord_progression, tpbar, tpb,
                                   scale_set, semitone_tolerance=1)
        if track.role == "melody" and tension > 0.0:
            notes = add_approach_notes(notes, bp.chord_progression, tpbar,
                                       tension=tension, seed=55)
        notes = scale_velocity_by_role(notes, track.role)
        notes = humanize_velocities(notes)
        notes = swing_timing(notes, swing_ratio=0.58)   # bebop swing ratio
        out.append(replace(track, notes=notes))
    return out


def generate(
    filename: str,
    source_midi: str | None = None,
    max_events: int = 96,
    hw_context_interval: int = 4,
    label: str = "",
    tension: float = 0.0,
) -> str | None:
    structure  = build_genre_structure("bebop", KEY, CHORDS, bars=BARS, tension=tension)
    bp         = _blueprint(active_roles=["melody", "harmony"], chord_progression=CHORDS)
    src_label  = f"← {Path(source_midi).name}" if source_midi else "cold start"
    bar("─")
    print(f"  {label}  (tension={tension:.1f})")
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
            stamped.append(replace(t, program=ALTO_SAX_PROGRAM))
        elif t.role == "harmony":
            stamped.append(replace(t, program=PIANO_PROGRAM))
        else:
            stamped.append(t)

    # Inject deterministic walking bass and swing_ride drums
    bass_notes = generate_walking_bass(CHORDS, BARS, ticks_per_beat=TICKS_PER_BEAT,
                                        velocity=74, channel=2, bass_range=BASS_RANGE)
    drum_notes = generate_drum_groove("swing_ride", BARS, ticks_per_beat=TICKS_PER_BEAT)
    bass_track = RoleTrack(role="bass",  channel=2,  program=BASS_PROGRAM, notes=bass_notes)
    drum_track = RoleTrack(role="drums", channel=10, program=0,             notes=drum_notes)
    all_tracks = stamped + [bass_track, drum_track]

    polished   = _apply_coherence(all_tracks, bp, tension=tension)
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
    print("  tt-midi-maker  ▸  Quick Changes  (Bb major bebop, walking bass)")
    bar("═")

    devices = detect_tt_devices()
    backend = f"tt-forge ({len(devices)} P300C)" if devices else "CPU"
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {LOOP_SECS:.1f}s per {BARS}-bar phrase")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    f1 = generate("p1_head.mid",
                  max_events=128, hw_context_interval=2,
                  label="Pattern 1 — head (cold start)",
                  tension=0.0)
    f2 = generate("p2_solo1.mid",
                  source_midi=f1,
                  max_events=128, hw_context_interval=2,
                  label="Pattern 2 — first chorus",
                  tension=0.4)
    f3 = generate("p3_solo2.mid",
                  source_midi=f2,
                  max_events=160, hw_context_interval=2,
                  label="Pattern 3 — second chorus (peak improv)",
                  tension=0.7)

    bar("═")
    valid = [f for f in [f1, f2, f3] if f]
    print(f"  Done.  {len(valid)}/3 patterns saved to {OUTPUT_DIR}")
    bar("═")


if __name__ == "__main__":
    main()
