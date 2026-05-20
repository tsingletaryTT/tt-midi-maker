#!/usr/bin/env python3
"""Render all MIDI files in examples/ to MP3 using fluidsynth + ffmpeg.

Usage:
    python scripts/render_mp3s.py [--suite NAME] [--gain GAIN]

Renders every *.mid under examples/*/  (or just the named suite).
Output MP3s land alongside the MID files.
Also copies to docs/audio/ for the GitHub Pages site.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
EXAMPLES    = REPO_ROOT / "examples"
DOCS_AUDIO  = REPO_ROOT / "docs" / "audio"
SOUNDFONT   = Path("/usr/share/sounds/sf3/MuseScore_General.sf3")

# Fallback to FluidR3 if MuseScore isn't installed
if not SOUNDFONT.exists():
    SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


def render_midi_to_mp3(midi_path: Path, gain: float = 2.5) -> Path | None:
    """Render a single MIDI file to MP3 using fluidsynth (temp WAV) → ffmpeg."""
    import tempfile, os
    mp3_path = midi_path.with_suffix(".mp3")

    try:
        # fluidsynth -F - (stdout) is unreliable in v2.3; write to a temp raw file
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
            raw_path = tf.name

        fluid_result = subprocess.run(
            ["fluidsynth", "-ni", "-g", str(gain), "-F", raw_path,
             str(SOUNDFONT), str(midi_path)],
            capture_output=True,
        )
        if fluid_result.returncode != 0 or not Path(raw_path).stat().st_size:
            print(f"  [!] fluidsynth failed for {midi_path.name}: "
                  f"{fluid_result.stderr[-200:].decode()}")
            return None

        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-f", "s32le", "-ar", "44100", "-ac", "2",
             "-i", raw_path, "-ab", "192k", str(mp3_path)],
            capture_output=True,
        )
        if ffmpeg_result.returncode != 0 or not mp3_path.exists():
            print(f"  [!] ffmpeg failed for {midi_path.name}: "
                  f"{ffmpeg_result.stderr[-200:].decode()}")
            return None
        return mp3_path
    except Exception as e:
        print(f"  [!] render error for {midi_path.name}: {e}")
        return None
    finally:
        try:
            os.unlink(raw_path)
        except Exception:
            pass


DOCS_MIDI = REPO_ROOT / "docs" / "midi"


def sync_midi_to_docs(suite_dir: Path) -> int:
    """Copy MIDI files from examples/{suite}/ to docs/midi/{suite}/. Returns count synced."""
    dest_dir = DOCS_MIDI / suite_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    synced = 0
    for midi in suite_dir.glob("*.mid"):
        shutil.copy2(midi, dest_dir / midi.name)
        synced += 1
    return synced


def cleanup_stray_files(docs_dir: Path, suffix: str) -> int:
    """Remove loose files (not in subdirs) from docs_dir. Returns count removed."""
    removed = 0
    for f in docs_dir.glob(f"*{suffix}"):
        if f.is_file():
            print(f"    [cleanup] removing stray {f.name} from {docs_dir.name}/")
            f.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MIDI → MP3 and sync docs/")
    parser.add_argument("--suite", default=None,
                        help="Render only this suite (directory name under examples/)")
    parser.add_argument("--gain", type=float, default=2.5,
                        help="FluidSynth gain (default 2.5)")
    parser.add_argument("--midi-only", action="store_true",
                        help="Only sync MIDI files (skip MP3 rendering)")
    args = parser.parse_args()

    if not args.midi_only and not SOUNDFONT.exists():
        print(f"ERROR: soundfont not found at {SOUNDFONT}")
        sys.exit(1)

    # Clean up stray files in docs/ roots (old patterns not in subdirectories)
    if not args.suite:
        cleanup_stray_files(DOCS_AUDIO, ".mp3")
        cleanup_stray_files(DOCS_MIDI, ".mid")

    # Collect MIDI files
    if args.suite:
        suites = [EXAMPLES / args.suite]
    else:
        suites = sorted(d for d in EXAMPLES.iterdir() if d.is_dir())

    total = ok = 0
    midi_synced = 0

    for suite_dir in suites:
        midi_files = sorted(suite_dir.glob("*.mid"))
        if not midi_files:
            continue
        print(f"\n  {suite_dir.name}/")

        # Sync MIDIs to docs/midi/
        n = sync_midi_to_docs(suite_dir)
        midi_synced += n
        print(f"    synced {n} MIDI files to docs/midi/{suite_dir.name}/")

        if args.midi_only:
            continue

        for midi in midi_files:
            total += 1
            print(f"    {midi.name} → ", end="", flush=True)
            mp3 = render_midi_to_mp3(midi, gain=args.gain)
            if mp3:
                ok += 1
                print(f"{mp3.name}  ({mp3.stat().st_size // 1024} KB)")
                # Mirror to docs/audio/ for GH Pages
                dest_dir = DOCS_AUDIO / suite_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp3, dest_dir / mp3.name)
            else:
                print("FAILED")

    if args.midi_only:
        print(f"\n  Synced {midi_synced} MIDI files to docs/midi/.")
    else:
        print(f"\n  Rendered {ok}/{total} MP3s.  Synced {midi_synced} MIDIs to docs/.")


if __name__ == "__main__":
    main()
