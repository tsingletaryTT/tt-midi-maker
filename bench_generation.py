#!/usr/bin/env python3
"""
Generation timing benchmark.

Measures wall time vs max_events, active role count, and source_midi context,
then cross-references against loop durations to find the "safe budget" —
how many events you can generate before the current loop ends.

Run:
    python bench_generation.py
"""
from __future__ import annotations

import json
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
_CONFIG_DIR = Path(__file__).parent / "config"
ROLES_CONFIG: dict = yaml.safe_load((_CONFIG_DIR / "roles.yaml").read_text())["roles"]

from tt_midi_maker.generation.midi_backend import generate_from_blueprint, reset_model
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bp(active_roles: list[str], bpm: int = 120, bars: int = 8) -> MusicalBlueprint:
    return MusicalBlueprint(
        key="C major", bpm=bpm, bars=bars,
        style="jazz", mode="loop",
        chord_progression=["Dm7", "G7", "Cmaj7", "Am7"],
        roles={
            name: RoleConfig(density=cfg["density_default"] if name in active_roles else 0.0)
            for name, cfg in ROLES_CONFIG.items()
        },
    )


def _loop_duration(bpm: int, bars: int) -> float:
    return bars * 4 * (60.0 / bpm)


def measure(active_roles: list[str], max_events: int, source_midi: str | None,
            bpm: int = 120, bars: int = 8, label: str = "") -> dict:
    bp = _bp(active_roles, bpm=bpm, bars=bars)
    t0 = time.perf_counter()
    tracks = generate_from_blueprint(
        bp, ROLES_CONFIG, max_events=max_events,
        source_midi=source_midi, source_context_bars=8,
    )
    elapsed = time.perf_counter() - t0
    total_notes = sum(len(t.notes) for t in tracks)
    return {
        "label":        label,
        "max_events":   max_events,
        "roles":        active_roles,
        "n_roles":      len(active_roles),
        "source_midi":  source_midi is not None,
        "bpm":          bpm,
        "bars":         bars,
        "loop_secs":    round(_loop_duration(bpm, bars), 2),
        "gen_secs":     round(elapsed, 2),
        "notes_out":    total_notes,
        "ev_per_sec":   round(max_events / elapsed, 1),
        "loops_needed": round(elapsed / _loop_duration(bpm, bars), 2),
    }


def section(title: str):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print('─'*64)


def show(r: dict):
    src = "  +src" if r["source_midi"] else "      "
    ok  = "OK" if r["loops_needed"] <= 1.0 else ("~ok" if r["loops_needed"] <= 1.5 else "SLOW")
    print(
        f"  {r['label']:<30s} {src}  "
        f"{r['gen_secs']:5.1f}s / {r['loop_secs']:4.1f}s loop  "
        f"= {r['loops_needed']:.2f}x  {ok}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  tt-midi-maker  ▸  generation timing benchmark")
    print("=" * 64)
    print("  Warming up model (first call loads weights) …")
    # Warm-up: tiny generation to load the model and let torch settle
    warmup = measure(["bass", "drums"], max_events=32, source_midi=None, label="warmup")
    print(f"  Model warm — warmup took {warmup['gen_secs']:.1f}s")

    results = []

    # ── 1. max_events sweep (fixed: 2 roles, 120 BPM, 8 bars) ───────────────
    section("1. max_events sweep  (bass+drums, 120 BPM, 8 bars = 16.0s loop)")
    for n in [32, 64, 128, 192, 256, 384, 512]:
        r = measure(["bass", "drums"], max_events=n, source_midi=None,
                    bpm=120, bars=8, label=f"max_events={n}")
        show(r)
        results.append(r)

    # ── 2. Role count sweep (fixed: max_events=256, 120 BPM, 8 bars) ────────
    section("2. Role count sweep  (max_events=256, 120 BPM, 8 bars = 16.0s loop)")
    role_sets = [
        (["bass"],                         "1 role  (bass)"),
        (["bass", "drums"],                "2 roles (bass+drums)"),
        (["bass", "drums", "harmony"],     "3 roles (bass+drums+harmony)"),
        (["bass", "drums", "harmony", "melody"], "4 roles (+melody)"),
    ]
    for roles, label in role_sets:
        r = measure(roles, max_events=256, source_midi=None,
                    bpm=120, bars=8, label=label)
        show(r)
        results.append(r)

    # ── 3. Source context overhead ───────────────────────────────────────────
    section("3. Source MIDI context overhead  (bass+drums+harmony, max_events=256)")
    # generate a real MIDI file first
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        src_path = f.name

    from tt_midi_maker.assembler import build_midi_file
    bp_src = _bp(["bass", "drums", "harmony"], bpm=120, bars=8)
    tracks_src = generate_from_blueprint(bp_src, ROLES_CONFIG, max_events=128)
    build_midi_file(tracks_src, 120, Path(src_path))

    r_no  = measure(["bass", "drums", "harmony"], max_events=256, source_midi=None,
                    bpm=120, bars=8, label="no source")
    r_src = measure(["bass", "drums", "harmony"], max_events=256, source_midi=src_path,
                    bpm=120, bars=8, label="with source (8 bars)")
    show(r_no)
    show(r_src)
    results += [r_no, r_src]

    overhead = r_src["gen_secs"] - r_no["gen_secs"]
    print(f"\n  Source context overhead: {overhead:+.1f}s")

    # ── 4. BPM / bars: loop budget vs generation time ─────────────────────────
    section("4. Loop duration vs generation time  (bass+drums+harmony, max_events=256)")
    configs = [
        (80,  4,  "80 BPM 4 bars"),
        (80,  8,  "80 BPM 8 bars"),
        (100, 4,  "100 BPM 4 bars"),
        (100, 8,  "100 BPM 8 bars"),
        (120, 4,  "120 BPM 4 bars"),
        (120, 8,  "120 BPM 8 bars"),
        (120, 16, "120 BPM 16 bars"),
        (140, 8,  "140 BPM 8 bars"),
        (160, 8,  "160 BPM 8 bars"),
    ]
    for bpm, bars, label in configs:
        r = measure(["bass", "drums", "harmony"], max_events=256, source_midi=None,
                    bpm=bpm, bars=bars, label=label)
        show(r)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    ev_per_sec_vals = [r["ev_per_sec"] for r in results if not r["label"].startswith("warmup")]
    avg_eps = sum(ev_per_sec_vals) / len(ev_per_sec_vals)

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Avg events/sec (CPU):  {avg_eps:.1f}")
    print()
    print("  Safe max_events budget per loop duration:")
    for bpm, bars in [(80, 8), (100, 8), (120, 8), (120, 16), (140, 8), (160, 8)]:
        loop = _loop_duration(bpm, bars)
        safe = int(loop * avg_eps * 0.85)   # 15% headroom
        print(f"    {bpm:3d} BPM / {bars:2d} bars  ({loop:5.1f}s loop)  →  max_events ≤ {safe:3d}  for 1-loop generation")
    print()
    print("  To always finish in 1 loop, scale max_events with loop duration.")
    print("  For 1.5-loop tolerance (pleasant repeat before handoff), multiply by 1.5.")
    print("=" * 64)

    # Save results JSON for further analysis
    out = Path(__file__).parent / "bench_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    main()
