#!/usr/bin/env python3
"""Play a tt-midi-maker MIDI file through FluidSynth.

Usage:
    python scripts/play_midi.py examples/midnight-blues/p1_intro.mid
    python scripts/play_midi.py file.mid --loop

Requires:
    fluidsynth + MuseScore_General.sf3 or FluidR3_GM.sf2
    (sudo apt install fluidsynth fluid-soundfont-gm)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SOUNDFONT = (
    Path("/usr/share/sounds/sf3/MuseScore_General.sf3")
    if Path("/usr/share/sounds/sf3/MuseScore_General.sf3").exists()
    else Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
)


def play(midi_path: Path, loop: bool) -> None:
    cmd = [
        "fluidsynth", "-a", "alsa", "-m", "alsa_seq",
        "-g", "2.5", "--quiet", "-i",
        str(SOUNDFONT), str(midi_path.resolve()),
    ]
    print(f"  Playing {midi_path.name} ...")
    try:
        while True:
            subprocess.run(cmd, check=False)
            if not loop:
                break
            print("  Looping...")
    except KeyboardInterrupt:
        pass
    print("  Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play MIDI through FluidSynth")
    ap.add_argument("midi", help="Path to MIDI file")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    args = ap.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        sys.exit(1)
    if not SOUNDFONT.exists():
        print(f"Soundfont not found: {SOUNDFONT}")
        sys.exit(1)

    print(f"  MIDI : {midi_path}")
    print(f"  Font : {SOUNDFONT.name}")
    play(midi_path, loop=args.loop)


if __name__ == "__main__":
    main()
