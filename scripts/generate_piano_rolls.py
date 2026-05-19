#!/usr/bin/env python3
"""Generate SVG piano-roll thumbnails for all MIDI files under docs/midi/.

Output: docs/assets/piano-rolls/<suite>/<stem>.svg  (top-level MIDIs → no subdir)
SVG size: 800×64px.  One colored rect per note, colored by MIDI channel.

Channel colors (matches site palette):
  ch0  → #4FD1C5  teal      (melody / lead)
  ch1  → #F4C471  gold      (bass)
  ch2  → #81E6D9  teal-light (harmony)
  ch9  → #8AAABB  text-dim  (drums)
  else → #EC96B8  pink      (other melodic)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mido

MIDI_DIR = ROOT / "docs" / "midi"
OUT_DIR  = ROOT / "docs" / "assets" / "piano-rolls"
SVG_W    = 800
SVG_H    = 64
PADDING  = 4   # semitones above/below pitch range

CH_COLORS = {
    0: "#4FD1C5",   # teal — melody/lead
    1: "#F4C471",   # gold — bass
    2: "#81E6D9",   # teal-light — harmony
    9: "#8AAABB",   # text-dim — drums
}
DEFAULT_COLOR = "#EC96B8"   # pink — other melodic channels


def _extract_notes(path: Path) -> tuple[list[dict], int]:
    """Return (notes, total_ticks). Each note: {ch, pitch, start, dur, vel}."""
    mid   = mido.MidiFile(str(path))
    notes: list[dict] = []
    total = 0

    for track in mid.tracks:
        abs_tick = 0
        active: dict[tuple[int, int], tuple[int, int]] = {}  # (ch, pitch) → (start, vel)
        for msg in track:
            abs_tick += msg.time
            total = max(total, abs_tick)
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key)
                    notes.append({
                        "ch": msg.channel,
                        "pitch": msg.note,
                        "start": start,
                        "dur": max(1, abs_tick - start),
                        "vel": vel,
                    })
    # flush any unclosed notes (give them one beat of duration)
    tpb = mid.ticks_per_beat or 480
    for (ch, pitch), (start, vel) in active.items():
        notes.append({"ch": ch, "pitch": pitch, "start": start, "dur": tpb, "vel": vel})

    return notes, max(total, 1)


def _render_svg(notes: list[dict], total_ticks: int) -> str:
    if not notes:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{SVG_W}" height="{SVG_H}" '
            f'style="background:#0B1E26;border-radius:6px"></svg>'
        )

    lo = max(0,   min(n["pitch"] for n in notes) - PADDING)
    hi = min(127, max(n["pitch"] for n in notes) + PADDING)
    pitch_span = max(1, hi - lo)
    px_per_semi = SVG_H / pitch_span

    rects: list[str] = []
    for n in notes:
        x     = n["start"] / total_ticks * SVG_W
        w     = max(1.5, n["dur"] / total_ticks * SVG_W)
        y     = (hi - n["pitch"]) / pitch_span * SVG_H
        h     = max(2.0, px_per_semi)
        color = CH_COLORS.get(n["ch"], DEFAULT_COLOR)
        alpha = 0.55 + (n["vel"] / 127) * 0.45
        rects.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" opacity="{alpha:.2f}" rx="1"/>'
        )

    body = "\n".join(rects)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
        f'style="background:#0B1E26;border-radius:6px">\n{body}\n</svg>'
    )


def main() -> None:
    mid_files = sorted(MIDI_DIR.rglob("*.mid"))
    if not mid_files:
        print("No MIDI files found under", MIDI_DIR)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mid_path in mid_files:
        rel      = mid_path.relative_to(MIDI_DIR)
        out_path = OUT_DIR / rel.with_suffix(".svg")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        notes, total_ticks = _extract_notes(mid_path)
        svg = _render_svg(notes, total_ticks)
        out_path.write_text(svg, encoding="utf-8")
        print(f"  {rel}  ({len(notes)} notes)  →  {out_path.relative_to(ROOT)}")

    print(f"\nDone. {len(mid_files)} piano rolls written to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
