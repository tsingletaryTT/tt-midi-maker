"""
MIDI playback with multiple output backends and per-channel routing.

Backends:
  fluidsynth  — software GM synthesis → system audio (default)
  alsa        — raw MIDI to any ALSA sequencer port (USB, Bluetooth, loopback)

Per-channel routing (channel_map) lets each MIDI channel go to a different ALSA
port, e.g. melody → USB synth A, drums → Bluetooth drum machine.

Background jobs are tracked by job_id so stop_playback() can kill them.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

import mido

from .errors import MidiMakerError

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
_SOUNDFONT_DIRS   = [Path("/usr/share/sounds/sf2"), Path.home() / ".local/share/sounds/sf2"]
_RTMIDI_BACKEND   = "mido.backends.rtmidi"

# ── job registry ─────────────────────────────────────────────────────────────

_JOBS: dict[str, dict] = {}  # job_id → {"proc": …, "thread": …, "status": …, "file": …}


def _register_job(file_path: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {"status": "starting", "file": file_path, "proc": None, "thread": None}
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _JOBS:
        _JOBS[job_id].update(kwargs)


# ── device discovery ─────────────────────────────────────────────────────────

def list_output_ports() -> list[dict]:
    """Return all ALSA MIDI output ports via python-rtmidi."""
    try:
        mido.set_backend(_RTMIDI_BACKEND)
        names = mido.get_output_names()
        return [{"index": i, "name": n} for i, n in enumerate(names)]
    except Exception as exc:
        return [{"error": str(exc)}]


def list_soundfonts() -> list[str]:
    """Discover .sf2 files in standard locations."""
    found = []
    for d in _SOUNDFONT_DIRS:
        if d.is_dir():
            found.extend(str(p) for p in d.glob("*.sf2"))
    return sorted(found)


# ── internal playback workers ─────────────────────────────────────────────────

def _fluidsynth_worker(job_id: str, file_path: Path, soundfont: Path, gain: float) -> None:
    """Subprocess worker: fluidsynth -ni → PulseAudio."""
    cmd = [
        "fluidsynth", "-ni",
        "-a", "pulseaudio",
        "-g", str(gain),
        str(soundfont),
        str(file_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _update_job(job_id, proc=proc, status="playing")
        rc = proc.wait()
        _update_job(job_id, status="done" if rc == 0 else "error", proc=None)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))


def _alsa_worker(
    job_id: str,
    file_path: Path,
    channel_map: dict[str, str] | None,
    port_name: str | None,
) -> None:
    """
    Thread worker: play MIDI file to ALSA sequencer port(s) via mido.

    channel_map maps 1-indexed MIDI channel strings to port names:
        {"1": "USB Synth:0", "10": "BT Drum:1"}
    Channels not in channel_map fall back to port_name.
    Non-channel messages (meta, sysex) are silently dropped.
    """
    mido.set_backend(_RTMIDI_BACKEND)
    _update_job(job_id, status="playing")

    # Determine all ports we need to open
    ports_needed: set[str] = set()
    if channel_map:
        ports_needed.update(channel_map.values())
    if port_name:
        ports_needed.add(port_name)

    open_ports: dict[str, mido.ports.BaseOutput] = {}
    try:
        for name in ports_needed:
            try:
                open_ports[name] = mido.open_output(name)
            except Exception as exc:
                _update_job(job_id, status="error", error=f"Cannot open port {name!r}: {exc}")
                return

        mid = mido.MidiFile(str(file_path))

        # mid.play() yields messages with correct sleep intervals already applied
        for msg in mid.play():
            if _JOBS.get(job_id, {}).get("status") == "stopped":
                break
            if msg.is_meta:
                continue
            ch_str = str(getattr(msg, "channel", -1) + 1)  # mido ch is 0-indexed
            dest = (channel_map or {}).get(ch_str, port_name)
            if dest and dest in open_ports:
                open_ports[dest].send(msg)

        _update_job(job_id, status="done")
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        for p in open_ports.values():
            try:
                p.close()
            except Exception:
                pass


# ── public API ────────────────────────────────────────────────────────────────

def play(
    file_path: str | Path,
    backend: Literal["fluidsynth", "alsa"] = "fluidsynth",
    port: str | None = None,
    channel_map: dict[str, str] | None = None,
    soundfont: str | None = None,
    gain: float = 2.0,
    blocking: bool = False,
) -> dict:
    """
    Play a MIDI file.

    backend="fluidsynth"  — software GM synthesis, no hardware required.
    backend="alsa"        — send raw MIDI to an ALSA sequencer port.
    channel_map           — per-channel port routing, overrides port for mapped channels.
                            Keys are 1-indexed channel numbers (strings), values are port names.

    Returns a job dict with job_id, status, backend, and routing info.
    """
    path = Path(file_path)
    if not path.exists():
        raise MidiMakerError("FILE_NOT_FOUND", f"MIDI file not found: {path}",
                             "Generate a file first with generate_midi.")

    job_id = _register_job(str(path))

    if backend == "fluidsynth":
        sf = Path(soundfont) if soundfont else DEFAULT_SOUNDFONT
        if not sf.exists():
            raise MidiMakerError(
                "SOUNDFONT_NOT_FOUND",
                f"SoundFont not found: {sf}",
                "Install fluid-soundfont-gm: sudo apt install fluid-soundfont-gm",
            )
        if not shutil.which("fluidsynth"):
            raise MidiMakerError(
                "FLUIDSYNTH_NOT_FOUND",
                "fluidsynth binary not found.",
                "Install it: sudo apt install fluidsynth",
            )
        t = threading.Thread(
            target=_fluidsynth_worker,
            args=(job_id, path, sf, gain),
            daemon=True,
        )
        _update_job(job_id, thread=t, backend="fluidsynth", soundfont=str(sf), gain=gain)
        t.start()

    elif backend == "alsa":
        # Auto-select first available port if none specified and no channel_map
        if not port and not channel_map:
            ports = list_output_ports()
            if not ports or "error" in ports[0]:
                raise MidiMakerError(
                    "NO_MIDI_PORTS",
                    "No ALSA MIDI output ports found.",
                    "Connect a USB MIDI device or pair a Bluetooth MIDI device.",
                )
            port = ports[0]["name"]

        t = threading.Thread(
            target=_alsa_worker,
            args=(job_id, path, channel_map, port),
            daemon=True,
        )
        _update_job(job_id, thread=t, backend="alsa", port=port, channel_map=channel_map)
        t.start()

    else:
        raise MidiMakerError("UNKNOWN_BACKEND", f"Unknown backend: {backend!r}",
                             "Use 'fluidsynth' or 'alsa'.")

    if blocking:
        _JOBS[job_id]["thread"].join()

    return job_status(job_id)


def stop(job_id: str) -> dict:
    """Stop a background playback job."""
    job = _JOBS.get(job_id)
    if not job:
        raise MidiMakerError("JOB_NOT_FOUND", f"No job with id {job_id!r}.",
                             "List active jobs with list_midi_devices.")
    if job["status"] not in ("playing", "starting"):
        return {"job_id": job_id, "status": job["status"], "message": "Already finished."}

    # Signal thread workers to exit their loop
    _update_job(job_id, status="stopped")

    # Kill fluidsynth subprocess if present
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return {"job_id": job_id, "status": "stopped"}


def job_status(job_id: str) -> dict:
    """Return current status of a playback job."""
    job = _JOBS.get(job_id)
    if not job:
        raise MidiMakerError("JOB_NOT_FOUND", f"No job with id {job_id!r}.", "")
    return {
        "job_id": job_id,
        "file": job["file"],
        "backend": job.get("backend"),
        "status": job["status"],
        "port": job.get("port"),
        "channel_map": job.get("channel_map"),
        "soundfont": job.get("soundfont"),
        "error": job.get("error"),
    }


def active_jobs() -> list[dict]:
    """Return all jobs currently in playing or starting state."""
    return [
        {"job_id": jid, "file": j["file"], "backend": j.get("backend"), "status": j["status"]}
        for jid, j in _JOBS.items()
        if j["status"] in ("playing", "starting")
    ]
