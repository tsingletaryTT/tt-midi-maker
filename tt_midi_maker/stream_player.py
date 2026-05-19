"""
Real-time MIDI streaming player with seamless loop transitions.

Architecture:
  FluidSynth runs as a persistent ALSA sequencer server (stdin=PIPE keeps it alive).
  LoopPlayer drives it via mido/rtmidi using a background thread + monotonic clock.
  Patterns queue atomically at loop boundaries — no gap, no click.

Typical workflow:
  server = start_synth()         # once per session
  loop_play(file_path)           # start looping
  loop_queue(next_file_path)     # queue while playing — transitions at loop end
  loop_stop()                    # stop after current loop (or immediately=True)
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import mido

from .assembler import TICKS_PER_BEAT
from .errors import MidiMakerError

DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf3/MuseScore_General.sf3")
_RTMIDI_BACKEND = "mido.backends.rtmidi"


# ── MIDI file loading ─────────────────────────────────────────────────────────

def _events_from_file(path: Path) -> tuple[list, dict, int, float, int]:
    """
    Load a MIDI file and return everything needed to drive LoopPlayer.

    Returns:
      events       sorted list of (tick, 'on'/'off'/'prog', channel, pitch/prog, vel)
      programs     {channel: program} from program_change messages
      loop_ticks   length of one loop (rounded up to next bar)
      bpm          tempo from set_tempo meta message (default 120)
      ticks_per_beat  from the file header
    """
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    bpm = 120.0
    events: list[tuple] = []
    programs: dict[int, int] = {}

    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                bpm = 60_000_000 / msg.tempo
            elif msg.type == "program_change":
                programs[msg.channel] = msg.program
                events.append((tick, "prog", msg.channel, msg.program, 0))
            elif msg.type == "note_on" and msg.velocity > 0:
                events.append((tick, "on", msg.channel, msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                events.append((tick, "off", msg.channel, msg.note, 0))

    if not any(e[1] == "on" for e in events):
        raise MidiMakerError("NO_NOTES", f"No notes in {path.name}",
                             "Generate a file with generate_midi first.")

    # note_off before note_on at same tick (avoid same-tick re-attack glitch)
    events.sort(key=lambda e: (e[0], {"prog": 0, "off": 1, "on": 2}[e[1]]))

    # Round loop length up to next full bar
    last_tick = max(e[0] for e in events)
    bar_ticks = tpb * 4
    loop_ticks = ((last_tick + bar_ticks) // bar_ticks) * bar_ticks

    return events, programs, loop_ticks, bpm, tpb


# ── FluidSynth server ─────────────────────────────────────────────────────────

class FluidSynthServer:
    """
    Manages a FluidSynth process in ALSA sequencer server mode.
    stdin=PIPE keeps the process alive waiting for shell input (never sent).
    """

    def __init__(self, soundfont: Path = DEFAULT_SOUNDFONT, gain: float = 2.0):
        self._soundfont = soundfont
        self._gain = gain
        self._proc: subprocess.Popen | None = None
        self._port_name: str | None = None

    def start(self, driver: str = "pulseaudio") -> str:
        """Start server if not running; return the ALSA output port name."""
        if self.is_running and self._port_name:
            return self._port_name

        if not shutil.which("fluidsynth"):
            raise MidiMakerError("FLUIDSYNTH_NOT_FOUND", "fluidsynth binary not found",
                                 "sudo apt install fluidsynth")
        if not self._soundfont.exists():
            raise MidiMakerError("SOUNDFONT_NOT_FOUND", f"SoundFont not found: {self._soundfont}",
                                 "sudo apt install fluid-soundfont-gm")

        self._proc = subprocess.Popen(
            ["fluidsynth", "-a", driver, "-m", "alsa_seq",
             f"-g{self._gain}", str(self._soundfont)],
            stdin=subprocess.PIPE,    # keeps process alive waiting on shell stdin
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        mido.set_backend(_RTMIDI_BACKEND)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if self._proc.poll() is not None:
                raise MidiMakerError(
                    "FLUIDSYNTH_CRASHED",
                    "FluidSynth exited immediately on start",
                    "Try audio driver 'alsa' if 'pulseaudio' is unavailable",
                )
            for name in mido.get_output_names():
                if "FLUID" in name.upper():
                    self._port_name = name
                    return name

        raise MidiMakerError(
            "FLUIDSYNTH_PORT_TIMEOUT",
            "FluidSynth started but ALSA MIDI port did not appear within 5s",
            "Check that alsa_seq kernel module is loaded (modprobe snd_seq)",
        )

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._port_name = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def port_name(self) -> str | None:
        return self._port_name

    def to_dict(self) -> dict:
        return {
            "running": self.is_running,
            "port": self._port_name,
            "soundfont": str(self._soundfont),
            "gain": self._gain,
        }


# ── Loop player ───────────────────────────────────────────────────────────────

class LoopPlayer:
    """
    Background-thread MIDI loop player with atomic pattern transitions.

    Sends note_on/note_off/program_change events to a mido output port,
    sleeping between events using time.monotonic() for sub-millisecond accuracy.

    Queued patterns take over at the next loop boundary without any audible gap.
    """

    def __init__(self, port_name: str):
        mido.set_backend(_RTMIDI_BACKEND)
        self._out = mido.open_output(port_name)
        self._port_name = port_name

        # Active pattern
        self._events:     list   = []
        self._programs:   dict   = {}
        self._loop_ticks: int    = 0
        self._bpm:        float  = 120.0
        self._tpb:        int    = TICKS_PER_BEAT
        self._current_file: str | None = None

        # Queued next pattern (swapped in atomically at loop boundary)
        self._next: dict | None = None
        self._lock = threading.Lock()

        self._state = "stopped"   # stopped | playing | stopping
        self._thread: threading.Thread | None = None
        self._loop_count = 0

    @property
    def state(self) -> str:
        return self._state

    # ── helpers ──────────────────────────────────────────────────────────────

    def _spt(self) -> float:
        return 60.0 / (self._bpm * self._tpb)

    def _send_programs(self):
        for ch, prog in self._programs.items():
            self._out.send(mido.Message("program_change", channel=ch, program=prog))

    def _all_notes_off(self):
        for ch in range(16):
            try:
                self._out.send(mido.Message("control_change",
                                            channel=ch, control=123, value=0))
            except Exception:
                pass

    # ── main loop thread ─────────────────────────────────────────────────────

    def _run(self):
        self._send_programs()
        spt = self._spt()
        loop_origin = time.monotonic()

        while self._state == "playing":
            for tick, etype, chan, val, vel in self._events:
                if self._state != "playing":
                    self._all_notes_off()
                    self._state = "stopped"
                    return

                target = loop_origin + tick * spt
                delta = target - time.monotonic()
                if delta > 0.001:
                    time.sleep(delta)

                if etype == "on":
                    self._out.send(mido.Message("note_on",
                                                channel=chan, note=val, velocity=vel))
                elif etype == "off":
                    self._out.send(mido.Message("note_off",
                                                channel=chan, note=val, velocity=0))
                elif etype == "prog":
                    self._out.send(mido.Message("program_change",
                                                channel=chan, program=val))

            # ── loop boundary ─────────────────────────────────────────────
            loop_origin += self._loop_ticks * spt
            self._loop_count += 1

            with self._lock:
                nxt = self._next
                if nxt is not None:
                    self._events     = nxt["events"]
                    self._programs   = nxt["programs"]
                    self._loop_ticks = nxt["loop_ticks"]
                    self._bpm        = nxt["bpm"]
                    self._tpb        = nxt["tpb"]
                    self._current_file = nxt["file"]
                    self._next       = None
                    spt = self._spt()
                    self._send_programs()

            if self._state == "stopping":
                break

        self._all_notes_off()
        self._state = "stopped"

    # ── public API ───────────────────────────────────────────────────────────

    def play(self, events, programs, loop_ticks, bpm, tpb, file_path):
        """Start looping immediately, interrupting any current playback."""
        # Stop existing thread cleanly
        if self._state in ("playing", "stopping"):
            self._state = "stopped"
            self._all_notes_off()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)

        with self._lock:
            self._events     = events
            self._programs   = programs
            self._loop_ticks = loop_ticks
            self._bpm        = bpm
            self._tpb        = tpb
            self._current_file = file_path
            self._next       = None
        self._loop_count = 0
        self._state = "playing"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def queue_next(self, events, programs, loop_ticks, bpm, tpb, file_path):
        """Queue next pattern; it takes over at the next loop boundary."""
        with self._lock:
            self._next = {
                "events": events, "programs": programs,
                "loop_ticks": loop_ticks, "bpm": bpm, "tpb": tpb, "file": file_path,
            }

    def stop(self, immediately: bool = False):
        if immediately:
            self._state = "stopped"
            self._all_notes_off()
        else:
            self._state = "stopping"

    def close(self):
        self.stop(immediately=True)
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._out.close()
        except Exception:
            pass

    def to_dict(self) -> dict:
        bars = self._loop_ticks // (self._tpb * 4) if self._tpb else 0
        return {
            "state":        self._state,
            "current_file": self._current_file,
            "queued_file":  self._next["file"] if self._next else None,
            "bpm":          self._bpm,
            "loop_bars":    bars,
            "loops_played": self._loop_count,
        }


# ── module singletons ─────────────────────────────────────────────────────────

_server: FluidSynthServer | None = None
_player: LoopPlayer | None = None


def start_synth(soundfont: str | None = None,
                gain: float = 2.0,
                driver: str = "pulseaudio") -> dict:
    """Start FluidSynth server and open the loop player. Call once per session."""
    global _server, _player

    # Restart if requested or not running
    if _player:
        _player.close()
    if _server:
        _server.stop()

    sf = Path(soundfont) if soundfont else DEFAULT_SOUNDFONT
    _server = FluidSynthServer(soundfont=sf, gain=gain)
    port = _server.start(driver=driver)
    _player = LoopPlayer(port)
    return {"status": "started", "port": port, **_server.to_dict()}


def _require_player() -> LoopPlayer:
    if _player is None or _server is None or not _server.is_running:
        raise MidiMakerError(
            "SYNTH_NOT_STARTED",
            "FluidSynth server is not running.",
            "Call synth_start first.",
        )
    return _player


def loop_play(file_path: str) -> dict:
    """Start looping a MIDI file immediately."""
    player = _require_player()
    events, programs, loop_ticks, bpm, tpb = _events_from_file(Path(file_path))
    player.play(events, programs, loop_ticks, bpm, tpb, file_path)
    return {"status": "looping", **player.to_dict()}


def loop_queue(file_path: str) -> dict:
    """Queue a MIDI file to start at the next loop boundary."""
    player = _require_player()
    events, programs, loop_ticks, bpm, tpb = _events_from_file(Path(file_path))
    player.queue_next(events, programs, loop_ticks, bpm, tpb, file_path)
    return {"status": "queued", **player.to_dict()}


def loop_stop(immediately: bool = False) -> dict:
    """Stop after the current loop completes, or right now if immediately=True."""
    if _player is None:
        return {"status": "not_running"}
    _player.stop(immediately=immediately)
    return {"status": "stopping" if not immediately else "stopped", **_player.to_dict()}


def synth_status() -> dict:
    """Return server + player state."""
    return {
        "server": _server.to_dict() if _server else {"running": False},
        "player": _player.to_dict() if _player else {"state": "not_initialized"},
    }
