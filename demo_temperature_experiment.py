#!/usr/bin/env python3
"""
Alternate-model experiment: temperature variation on skytnt/midi-model.

The skytnt model is fundamentally a token distribution. Temperature controls
how peaked (conservative) or flat (experimental) that distribution is:

  temp=0.7  — conservative: model picks most-probable tokens → predictable,
               can sound repetitive or "safe"
  temp=1.0  — default calibration: balanced creativity vs. coherence
  temp=1.5  — experimental: flat distribution → more jumps, more surprises,
               noisier raw output that the coherence layer must discipline

This script runs three monosynth generations at different temperatures,
applies the full coherence layer to each, scores quality, and saves for A/B
comparison. It demonstrates both:
  (a) how temperature shapes raw model output
  (b) how the coherence layer's scale+harmony filter stabilizes high-temp output

All runs use TT hardware (4× P300C). If hardware is unavailable, CPU fallback
is used automatically.

Output:
  examples/temperature-experiment/
    temp_0.7_conservative.mid   + .mp3
    temp_1.0_standard.mid       + .mp3
    temp_1.5_experimental.mid   + .mp3
    comparison.txt              — quality scores side-by-side
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
logger = logging.getLogger("temp-exp")

import yaml

_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

SUITE_NAME  = "temperature-experiment"
OUTPUT_DIR  = Path(__file__).parent / "examples" / SUITE_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from tt_midi_maker.assembler import TICKS_PER_BEAT, build_midi_file
from tt_midi_maker.coherence.harmony import chord_aware_filter
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing, scale_velocity_by_role
from tt_midi_maker.coherence.improv import add_approach_notes
from tt_midi_maker.coherence.judge import judge_tracks
from tt_midi_maker.coherence.scale import build_scale_set, parse_key, scale_quantize
from tt_midi_maker.generation.hardware import detect_tt_devices
from tt_midi_maker.generation.midi_backend import generate_from_blueprint
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

# ── Musical parameters (same as monosynth demo) ───────────────────────────────

KEY    = "C major"
BPM    = 120
BARS   = 8
CHORDS = ["Cmaj7", "Am7", "Fmaj7", "G7"]   # I-vi-IV-V: universally recognisable

# Single melody voice — pure model output, nothing else in the mix
LEAD_PROGRAM = 80   # Lead 1 (Square wave)  — the classic monosynth voice
LEAD_RANGE   = [60, 96]   # C4–C7: bright register

SUITE_ROLES: dict = {}
for name, cfg in ROLES_CONFIG.items():
    overrides: dict = {}
    if name == "melody":
        overrides = {"program": LEAD_PROGRAM, "note_range": LEAD_RANGE}
    SUITE_ROLES[name] = {**cfg, **overrides}


# ── Temperature variants ──────────────────────────────────────────────────────

VARIANTS = [
    (0.7,  "conservative", "safe choices, lower rhythmic variance"),
    (1.0,  "standard",     "default calibration — balanced"),
    (1.5,  "experimental", "high entropy — requires coherence discipline"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blueprint() -> MusicalBlueprint:
    roles = {
        name: RoleConfig(density=cfg["density_default"] if name == "melody" else 0.0)
        for name, cfg in ROLES_CONFIG.items()
    }
    return MusicalBlueprint(
        key=KEY, bpm=BPM, bars=BARS,
        style="monosynth", mode="loop",
        chord_progression=CHORDS,
        roles=roles,
    )


def _apply_coherence(tracks, tension: float = 0.2) -> list:
    root, mode = parse_key(KEY)
    scale_set  = build_scale_set(root, mode)
    tpb        = TICKS_PER_BEAT
    tpbar      = tpb * 4
    out        = []
    for track in tracks:
        notes = scale_quantize(track.notes, KEY)
        notes = chord_aware_filter(notes, CHORDS, tpbar, tpb, scale_set)
        if track.role == "melody":
            notes = add_approach_notes(notes, CHORDS, tpbar, tension=tension, seed=42)
        notes = scale_velocity_by_role(notes, track.role)
        notes = humanize_velocities(notes)
        notes = nudge_timing(notes)
        out.append(replace(track, notes=notes))
    return out


def run_variant(temp: float, label: str, desc: str) -> dict:
    """Generate one pattern at `temp`, apply coherence, save, return quality info."""
    bp = _blueprint()
    filename = f"temp_{temp:.1f}_{label}.mid"

    print(f"\n  temperature={temp:.1f}  [{label}]  — {desc}")
    t0 = time.time()

    tracks = generate_from_blueprint(
        bp, SUITE_ROLES,
        max_events=96,
        hw_context_interval=4,
        temperature=temp,
        max_attempts=1,           # single attempt — we WANT to see raw temp effect
        judge_threshold=0.0,      # no re-rolling — keep whatever comes out
    )

    dt_gen = time.time() - t0

    if not tracks:
        print(f"  [!] empty — skipping")
        return {"temp": temp, "label": label, "notes": 0, "score": 0.0, "gen_s": dt_gen}

    # Quality before coherence
    raw_report = judge_tracks(tracks, bars=BARS, bpm=BPM, pass_threshold=0.0)

    # Apply coherence
    polished = _apply_coherence(tracks, tension=0.2)
    post_report = judge_tracks(polished, bars=BARS, bpm=BPM, pass_threshold=0.0)

    out = OUTPUT_DIR / filename
    build_midi_file(polished, BPM, out)

    melody_notes = sum(len(t.notes) for t in polished if t.role == "melody")
    loop_secs    = BARS * 4 * (60.0 / BPM)
    print(f"  generated in {dt_gen:.1f}s ({dt_gen/loop_secs:.2f}× loop)")
    print(f"  melody notes: {melody_notes}")
    print(f"  quality (raw):      {raw_report.rule_score:.2f}  issues={raw_report.issues}")
    print(f"  quality (coherence):{post_report.rule_score:.2f}  issues={post_report.issues}")
    print(f"  saved: {out.name}")

    return {
        "temp": temp,
        "label": label,
        "notes": melody_notes,
        "score_raw": raw_report.rule_score,
        "score_post": post_report.rule_score,
        "issues_raw": raw_report.issues,
        "issues_post": post_report.issues,
        "gen_s": dt_gen,
    }


def bar(char="─", width=60):
    print(char * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bar("═")
    print("  tt-midi-maker  ▸  Temperature Experiment")
    print("  Alternate-model run: 3 temperatures, same coherence layer")
    bar("═")

    devices  = detect_tt_devices()
    backend  = f"tt-forge ({len(devices)} P300C)" if devices else "CPU (no hardware)"
    loop_secs = BARS * 4 * (60.0 / BPM)
    print(f"  Backend : {backend}")
    print(f"  Key     : {KEY}   BPM: {BPM}   Bars: {BARS}")
    print(f"  Chords  : {' → '.join(CHORDS)}")
    print(f"  Loop    : {loop_secs:.1f}s")
    print(f"  Voice   : Lead Square (prog {LEAD_PROGRAM}), C4–C7")
    print(f"  Output  : {OUTPUT_DIR}")
    bar()

    results = []
    for temp, label, desc in VARIANTS:
        bar()
        r = run_variant(temp, label, desc)
        results.append(r)

    # Summary comparison table
    bar("═")
    print("\n  COMPARISON SUMMARY\n")
    print(f"  {'temp':>6}  {'label':<14}  {'notes':>5}  {'raw':>6}  {'post':>6}  {'issues (raw → post)'}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*30}")
    for r in results:
        issues_before = len(r.get("issues_raw", []))
        issues_after  = len(r.get("issues_post", []))
        score_raw  = r.get("score_raw", 0.0)
        score_post = r.get("score_post", 0.0)
        delta      = score_post - score_raw
        sign       = "+" if delta >= 0 else ""
        print(f"  {r['temp']:>6.1f}  {r['label']:<14}  {r['notes']:>5}  {score_raw:>6.2f}  "
              f"{score_post:>6.2f}  {issues_before}→{issues_after}  (Δ{sign}{delta:.2f})")
    print()

    # Write comparison text file
    comp_path = OUTPUT_DIR / "comparison.txt"
    lines = [
        "Temperature Experiment — tt-midi-maker",
        f"Key: {KEY}   BPM: {BPM}   Bars: {BARS}   Backend: {backend}",
        "",
        f"{'Temp':>6}  {'Label':<14}  {'Notes':>5}  {'Raw':>6}  {'Post':>6}  Delta",
        "-" * 60,
    ]
    for r in results:
        score_raw  = r.get("score_raw", 0.0)
        score_post = r.get("score_post", 0.0)
        delta      = score_post - score_raw
        lines.append(
            f"{r['temp']:>6.1f}  {r['label']:<14}  {r['notes']:>5}  "
            f"{score_raw:>6.2f}  {score_post:>6.2f}  {delta:+.2f}"
        )
    lines += [
        "",
        "Raw  = quality score before coherence layer",
        "Post = quality score after scale/harmony/approach-note coherence",
        "",
        "Higher temperature → more random tokens → lower raw score",
        "Coherence layer partially recovers quality by forcing scale/harmony alignment",
        "But high-temp generation retains rhythmic unpredictability even after coherence",
    ]
    comp_path.write_text("\n".join(lines) + "\n")

    bar("═")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"  Comparison table: {comp_path.name}")
    bar("═")


if __name__ == "__main__":
    main()
