#!/usr/bin/env python3
"""Play a tt-midi-maker MIDI file with audio + falling-note visualization.

Usage:
    python scripts/play_midi.py examples/midnight-blues/p1_intro.mid
    python scripts/play_midi.py examples/midnight-blues/p1_intro.mid --no-visual
    python scripts/play_midi.py examples/midnight-blues/p1_intro.mid --loop

Audio:   FluidSynth + FluidR3_GM soundfont
Visual:  MIDIVisualizer falling-note display (requires DISPLAY)

Requires:
    fluidsynth       (sudo apt install fluidsynth fluid-soundfont-gm)
    MIDIVisualizer   (~/.local/bin/MIDIVisualizer)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SOUNDFONT    = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
MIDI_VIZ     = Path.home() / ".local/bin/MIDIVisualizer"
DISPLAY      = ":0"

# Tenstorrent color palette for MIDIVisualizer
VIZ_COLORS = [
    "--color-bg",       "0.06", "0.16", "0.21",   # #0F2A35 deep blue-gray
    "--color-major",    "0.31", "0.82", "0.77",   # #4FD1C5 teal
    "--color-minor",    "0.93", "0.59", "0.72",   # #EC96B8 pink
    "--color-particles","0.31", "0.82", "0.77",
]


def play(midi_path: Path, loop: bool, visual: bool) -> None:
    midi_str = str(midi_path.resolve())
    procs: list[subprocess.Popen] = []

    # --- MIDIVisualizer (non-blocking, X11 window) ---
    if visual and MIDI_VIZ.exists():
        viz_cmd = [
            str(MIDI_VIZ),
            "--midi", midi_str,
            "--loop", "1" if loop else "0",
            "--show-particles", "1",
            "--show-flashes", "1",
            "--quality", "HIGH",
            *VIZ_COLORS,
        ]
        env = {"DISPLAY": DISPLAY, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        procs.append(subprocess.Popen(viz_cmd, env=env,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        print(f"  Visual : MIDIVisualizer launched (DISPLAY={DISPLAY})")
        time.sleep(0.5)  # let window open before audio starts

    # --- FluidSynth audio ---
    fluid_cmd = [
        "fluidsynth",
        "-a", "alsa",
        "-m", "alsa_seq",
        "-g", "1.2",          # gain — slightly louder than default
        "--quiet",
        SOUNDFONT,
        midi_str,
    ]
    if not loop:
        fluid_cmd.insert(1, "-i")  # non-interactive, exit when done
    else:
        # loop: use interactive mode and send a loop command
        # simplest approach: repeat file with shell loop
        pass

    if loop:
        print(f"  Audio  : FluidSynth looping (Ctrl-C to stop)...")
        try:
            while True:
                p = subprocess.run(
                    ["fluidsynth", "-a", "alsa", "-m", "alsa_seq",
                     "-g", "1.2", "--quiet", "-i", str(SOUNDFONT), midi_str],
                    check=False,
                )
                if p.returncode != 0:
                    break
        except KeyboardInterrupt:
            pass
    else:
        import mido
        duration = mido.MidiFile(midi_str).length
        print(f"  Audio  : FluidSynth playing {duration:.1f}s ...")
        proc = subprocess.Popen(
            ["fluidsynth", "-a", "alsa", "-m", "alsa_seq",
             "-g", "1.2", "--quiet", "-i", str(SOUNDFONT), midi_str],
        )
        procs.append(proc)
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass

    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass

    print("  Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play MIDI with audio + visualization")
    ap.add_argument("midi", help="Path to MIDI file")
    ap.add_argument("--loop",      action="store_true", help="Loop playback")
    ap.add_argument("--no-visual", action="store_true", help="Skip MIDIVisualizer")
    args = ap.parse_args()

    midi_path = Path(args.midi)
    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        sys.exit(1)

    if not SOUNDFONT.exists():
        print("FluidR3_GM soundfont not found. Install: sudo apt install fluid-soundfont-gm")
        sys.exit(1)

    print(f"  MIDI   : {midi_path}")
    print(f"  Font   : {SOUNDFONT.name}")
    play(midi_path, loop=args.loop, visual=not args.no_visual)


if __name__ == "__main__":
    main()
