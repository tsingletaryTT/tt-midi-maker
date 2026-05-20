#!/usr/bin/env python3
"""Play a tt-midi-maker MIDI file with synced audio + falling-note visualization.

MIDIVisualizer loads the file (for correct note layout), then xdotool fires
Space to start playback at the same instant FluidSynth begins. Both run from
the same start signal so audio and visuals stay in sync.

Usage:
    python scripts/play_midi.py examples/midnight-blues/p1_intro.mid
    python scripts/play_midi.py file.mid --loop
    python scripts/play_midi.py file.mid --no-visual

Requires:
    fluidsynth   (sudo apt install fluidsynth)
    MuseScore_General.sf3 or FluidR3_GM.sf2
    xdotool      (sudo apt install xdotool)
    MIDIVisualizer v7.3+  (~/.local/bin/MIDIVisualizer)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SOUNDFONT = (
    Path("/usr/share/sounds/sf3/MuseScore_General.sf3")
    if Path("/usr/share/sounds/sf3/MuseScore_General.sf3").exists()
    else Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
)
MIDI_VIZ = Path.home() / ".local/bin/MIDIVisualizer"
DISPLAY  = ":0"

import os as _os
VIZ_ENV = {**_os.environ, "DISPLAY": DISPLAY}

VIZ_COLORS = [
    "--color-bg",        "0.06", "0.16", "0.21",
    "--color-major",     "0.31", "0.82", "0.77",
    "--color-minor",     "0.93", "0.59", "0.72",
    "--color-particles", "0.31", "0.82", "0.77",
]


def play(midi_path: Path, loop: bool, visual: bool) -> None:
    midi_str = str(midi_path.resolve())
    viz_proc: subprocess.Popen | None = None

    # --- MIDIVisualizer: load file, start paused ---
    if visual and MIDI_VIZ.exists():
        viz_proc = subprocess.Popen(
            [str(MIDI_VIZ),
             "--midi",    midi_str,
             "--loop",    "1" if loop else "0",
             "--preroll", "0",
             "--show-notes", "1", "--show-particles", "1",
             "--show-flashes", "1", "--show-blur", "1", "--show-keyboard", "1",
             "--min-key", "36", "--max-key", "84",
             "--sets-mode", "0",
             "--filter-show-channels", "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
             "--quality", "HIGH",
             *VIZ_COLORS],
            env=VIZ_ENV,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for the window to open fully
        time.sleep(2.5)
        # Hide settings panel
        subprocess.run(
            ["xdotool", "search", "--name", "MIDI", "windowfocus", "--sync", "key", "h"],
            env=VIZ_ENV, capture_output=True,
        )
        time.sleep(0.2)
        print(f"  Visual : MIDIVisualizer ready")

    # --- Fire both at once: Space to start viz + FluidSynth for audio ---
    fluid_cmd = [
        "fluidsynth", "-a", "alsa", "-m", "alsa_seq",
        "-g", "2.5", "--quiet", "-i", str(SOUNDFONT), midi_str,
    ]

    if visual and viz_proc:
        # Start FluidSynth and send Space to MIDIVisualizer simultaneously
        fluid_proc = subprocess.Popen(fluid_cmd)
        subprocess.run(
            ["xdotool", "search", "--name", "MIDI", "windowfocus", "--sync", "key", "space"],
            env=VIZ_ENV, capture_output=True,
        )
        print("  Playing... (Ctrl-C to stop)")
        try:
            if loop:
                while True:
                    fluid_proc.wait()
                    fluid_proc = subprocess.Popen(fluid_cmd)
            else:
                fluid_proc.wait()
        except KeyboardInterrupt:
            fluid_proc.terminate()
    else:
        if loop:
            print("  Audio  : looping (Ctrl-C to stop)")
            try:
                while True:
                    subprocess.run(fluid_cmd, check=False)
            except KeyboardInterrupt:
                pass
        else:
            print("  Playing...")
            try:
                subprocess.run(fluid_cmd, check=False)
            except KeyboardInterrupt:
                pass

    if viz_proc:
        viz_proc.terminate()
    print("  Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play MIDI with synced audio + visualization")
    ap.add_argument("midi", help="Path to MIDI file")
    ap.add_argument("--loop",      action="store_true", help="Loop playback")
    ap.add_argument("--no-visual", action="store_true", help="Skip MIDIVisualizer")
    args = ap.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        sys.exit(1)
    if not SOUNDFONT.exists():
        print(f"Soundfont not found: {SOUNDFONT}")
        sys.exit(1)

    print(f"  MIDI   : {midi_path}")
    print(f"  Font   : {SOUNDFONT.name}")
    play(midi_path, loop=args.loop, visual=not args.no_visual)


if __name__ == "__main__":
    main()
