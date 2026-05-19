#!/usr/bin/env python3
"""
One-time quality audit of all generated MIDI suites.

Runs rule-based + perplexity analysis on every MIDI under examples/,
prints a formatted report, and writes results to docs/quality_report.json.

Usage:
  python scripts/analyze_quality.py           # full analysis (loads model for perplexity)
  python scripts/analyze_quality.py --no-ppl  # rule-based only, no model loading
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tt_midi_maker.coherence.judge import judge_midi_file, score_perplexity

# ── suite metadata (bars/bpm per suite directory name) ────────────────────────
SUITE_META: dict[str, dict] = {
    "silicon-road":        {"bars": 8,  "bpm": 84},
    "aria-d-minor":        {"bars": 8,  "bpm": 76},
    "midnight-blues":      {"bars": 8,  "bpm": 80},
    "slow-light":          {"bars": 8,  "bpm": 72},
    "bebop-quick-changes": {"bars": 8,  "bpm": 138},
    "monosynth":           {"bars": 4,  "bpm": 120},
}
DEFAULT_META = {"bars": 8, "bpm": 100}


def col(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(s):  return col(s, 32)
def red(s):    return col(s, 31)
def yellow(s): return col(s, 33)
def cyan(s):   return col(s, 36)
def bold(s):   return col(s, 1)
def dim(s):    return col(s, 2)


def bar(char="═", width=72):
    print(char * width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ppl", action="store_true", help="skip perplexity scoring")
    args = ap.parse_args()

    midi_files = sorted((ROOT / "examples").rglob("*.mid"))
    if not midi_files:
        print("No MIDI files found under examples/")
        sys.exit(1)

    bar("═")
    print(bold("  tt-midi-maker  ▸  Musical Quality Audit"))
    print(dim(f"  {len(midi_files)} patterns across {len(set(f.parent for f in midi_files))} suites"))
    bar("═")

    # Load model once if computing perplexity
    model, tokenizer = None, None
    if not args.no_ppl:
        print(dim("  Loading model for perplexity scoring (CPU)…"))
        t0 = time.time()
        from tt_midi_maker.generation.midi_backend import _get_model
        model, tokenizer = _get_model()
        print(dim(f"  Model loaded in {time.time()-t0:.1f}s"))
        print()

    results: list[dict] = []
    total_pass = 0
    total_fail = 0

    current_suite = None
    for midi_path in midi_files:
        suite = midi_path.parent.name
        meta  = SUITE_META.get(suite, DEFAULT_META)

        if suite != current_suite:
            if current_suite is not None:
                print()
            bar("─")
            print(f"  {bold(suite.upper())}  —  {meta['bpm']} BPM · {meta['bars']} bars")
            bar("─")
            current_suite = suite

        report = judge_midi_file(midi_path, bars=meta["bars"], bpm=meta["bpm"])

        # Perplexity
        nll = None
        if not args.no_ppl:
            t0 = time.time()
            nll = score_perplexity(midi_path, model=model, tokenizer=tokenizer)
            elapsed = time.time() - t0
            report.event_nll = nll

        # Render
        stem    = midi_path.stem
        passed  = report.passed
        score_s = green(f"{report.rule_score:.2f}") if passed else red(f"{report.rule_score:.2f}")
        nll_s   = (f"  NLL={yellow(f'{nll:.3f}')}" if nll is not None else "")
        status  = green("PASS") if passed else red("FAIL")

        # Note counts from metric keys (skip non-dict entries like register_overlap_semitones)
        note_counts = []
        for role, m in report.metrics.items():
            if isinstance(m, dict) and "notes_per_bar" in m:
                n = int(m["notes_per_bar"] * meta["bars"])
                note_counts.append(f"{role}:{n}n")

        print(f"  {stem:<28}  score={score_s}  {status}{nll_s}")
        if note_counts:
            print(dim(f"    notes: {' · '.join(note_counts)}"))
        for issue in report.issues:
            print(f"    {red('✗')} {issue}")
        if not report.issues:
            print(dim("    ✓ no issues detected"))

        if passed:
            total_pass += 1
        else:
            total_fail += 1

        results.append({
            "suite":      suite,
            "file":       midi_path.name,
            "bars":       meta["bars"],
            "bpm":        meta["bpm"],
            "rule_score": report.rule_score,
            "passed":     report.passed,
            "issues":     report.issues,
            "metrics":    report.metrics,
            "event_nll":  nll,
        })

    # Summary
    print()
    bar("═")
    total = total_pass + total_fail
    pct   = total_pass / total * 100 if total else 0
    print(bold(f"  Summary: {total_pass}/{total} patterns passed ({pct:.0f}%)"))

    # Worst patterns (by rule score)
    worst = sorted(results, key=lambda r: r["rule_score"])[:5]
    print(f"\n  {bold('Worst 5 by rule score:')}")
    for r in worst:
        issues_str = " | ".join(r["issues"][:2]) if r["issues"] else "—"
        print(f"    {r['suite']}/{r['file']}  score={r['rule_score']:.2f}  {red(issues_str)}")

    if not args.no_ppl:
        by_nll = [r for r in results if r["event_nll"] is not None]
        if by_nll:
            worst_nll = sorted(by_nll, key=lambda r: -r["event_nll"])[:5]
            print(f"\n  {bold('Highest NLL (model most surprised):')}")
            for r in worst_nll:
                print(f"    {r['suite']}/{r['file']}  NLL={r['event_nll']:.3f}")

    bar("═")

    # Write JSON report
    out = ROOT / "docs" / "quality_report.json"
    out.write_text(json.dumps({"patterns": results}, indent=2))
    print(dim(f"\n  Report saved to {out.relative_to(ROOT)}"))


if __name__ == "__main__":
    main()
