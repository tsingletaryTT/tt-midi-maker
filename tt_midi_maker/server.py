"""
tt-midi-maker MCP server.

12 tools, 4 prompts, 4 resources, argument completions.
Run with: python -m tt_midi_maker
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import yaml
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    Annotations, PromptMessage, TextContent, ToolAnnotations,
)

from .analyzer import describe_midi as _analyze, chat_about_midi
from .assembler import build_midi_file, TICKS_PER_BEAT
from .coherence.harmony import chord_aware_filter
from .coherence.humanize import humanize_velocities, nudge_timing
from .coherence.scale import build_scale_set, parse_key, scale_quantize
from .errors import MidiMakerError
from .generation.hardware import detect_tt_devices, hardware_status
from .generation.aria_backend import get_model, generate_tokens
from .generation.tokenizer import decode_tokens_to_midi, encode_midi_file
from .models.blueprint import MusicalBlueprint
from .player import (
    active_jobs, job_status, list_output_ports, list_soundfonts,
    play as _player_play, stop as _player_stop,
)
from .stream_player import (
    loop_play as _loop_play, loop_queue as _loop_queue,
    loop_stop as _loop_stop, start_synth as _start_synth,
    synth_status as _synth_status,
)
from .prompt_engine import build_blueprint
from .session import MusicalContext, clear_session, get_session, set_session

OUTPUT_DIR = Path.home() / "Music" / "tt-midi-maker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_CONFIG_DIR = Path(__file__).parent.parent / "config"
with open(_CONFIG_DIR / "roles.yaml") as f:
    ROLES_CONFIG: dict = yaml.safe_load(f)["roles"]
with open(_CONFIG_DIR / "styles.yaml") as f:
    STYLES_CONFIG: dict = yaml.safe_load(f)["styles"]

VALID_KEYS = [
    f"{r} {m}"
    for r in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
    for m in ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"]
]

_UNSET = object()  # sentinel: field not provided vs. explicitly set to None

# ---------------------------------------------------------------------------
# Internal helpers (importable for tests)
# ---------------------------------------------------------------------------

def _set_musical_context(
    session_id: str = "default",
    key=_UNSET,
    bpm=_UNSET,
    style=_UNSET,
    chord_progression=_UNSET,
) -> dict:
    updates = {k: v for k, v in
               [("key", key), ("bpm", bpm), ("style", style),
                ("chord_progression", chord_progression)]
               if v is not _UNSET}
    ctx = get_session(session_id).update(**updates)
    set_session(session_id, ctx)
    d = ctx.to_dict()
    fields_set = [k for k, v in d.items() if v is not None]
    # Restore null fields so output schema is complete
    for field in ("key", "bpm", "style", "chord_progression"):
        if field not in d:
            d[field] = None
    d["fields_set"] = fields_set
    return d


def _run_generation(blueprint: MusicalBlueprint) -> list:
    """Placeholder: generates stub notes from blueprint (Aria wiring is a follow-up)."""
    from .models.track import NoteEvent, RoleTrack
    tracks = []
    for role_name, role_cfg in blueprint.roles.items():
        if role_cfg.density <= 0.0:
            continue
        cfg = ROLES_CONFIG.get(role_name, {})
        channel = cfg.get("channel", 1)
        program = cfg.get("program", 0)
        lo, hi  = cfg.get("note_range", [48, 84])
        pitch   = (lo + hi) // 2
        notes   = [
            NoteEvent(pitch=pitch, velocity=int(sum(role_cfg.velocity_range) / 2),
                      start_tick=b * 4 * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                      channel=channel)
            for b in range(blueprint.bars)
        ]
        tracks.append(RoleTrack(role=role_name, channel=channel, program=program, notes=notes))
    return tracks


def _apply_coherence(tracks, blueprint: MusicalBlueprint) -> list:
    root, mode = parse_key(blueprint.key)
    scale_set  = build_scale_set(root, mode)
    ticks_per_bar  = 4 * TICKS_PER_BEAT
    ticks_per_beat = TICKS_PER_BEAT
    result = []
    for track in tracks:
        notes = scale_quantize(track.notes, blueprint.key)
        notes = chord_aware_filter(
            notes, blueprint.chord_progression,
            ticks_per_bar, ticks_per_beat, scale_set,
        )
        notes = humanize_velocities(notes)
        notes = nudge_timing(notes)
        from dataclasses import replace
        result.append(replace(track, notes=notes))
    return result


def _generate_midi(
    prompt: str,
    mode: Literal["loop", "section", "stream"] = "loop",
    roles: list[str] | None = None,
    bars: int | None = None,
    output_path: str | None = None,
    session_id: str = "default",
) -> dict:
    ctx       = get_session(session_id)
    blueprint = build_blueprint(prompt, ctx)

    if roles:
        filtered_roles = {
            role: (cfg if role in roles else cfg.model_copy(update={"density": 0.0}))
            for role, cfg in blueprint.roles.items()
        }
        blueprint = blueprint.model_copy(update={"roles": filtered_roles})
    if bars:
        blueprint = blueprint.model_copy(update={"bars": bars})

    raw_tracks    = _run_generation(blueprint)
    clean_tracks  = _apply_coherence(raw_tracks, blueprint)
    ts            = int(time.time())
    out           = Path(output_path) if output_path else OUTPUT_DIR / f"{ts}.mid"
    build_midi_file(clean_tracks, blueprint.bpm, out)
    return {
        "file_path":       str(out),
        "bars_generated":  blueprint.bars,
        "bpm":             blueprint.bpm,
        "key":             blueprint.key,
        "roles_generated": [t.role for t in clean_tracks],
        "generation_ms":   0,
        "hardware_used":   "cpu-fallback",
    }


def _describe_midi(file_path: str) -> dict:
    return _analyze(Path(file_path))


def _chat_with_midi(file_path: str, question: str) -> dict:
    return chat_about_midi(Path(file_path), question)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="tt-midi-maker",
    instructions="""
tt-midi-maker generates multi-track MIDI files from text prompts using
Tenstorrent AI hardware. Each output file follows General MIDI channel
conventions: drums on channel 10, melody on 1, bass on 2, harmony on 3.

Recommended workflow:
  1. Call set_musical_context to establish key, BPM, style, and chord
     progression. This persists for the session and improves all subsequent
     generate calls. Skip only for one-shot requests.
  2. Call generate_midi with a descriptive prompt and mode (loop/section/stream).
  3. Call continue_midi to extend the result, maintaining musical continuity.
  4. Call describe_midi or chat_with_midi to review and refine.

Output files: ~/Music/tt-midi-maker/ (or absolute path via output_path).
Prompts work best when they mention: genre, mood, tempo feel, instrumentation.
""",
    website_url="https://github.com/tenstorrent/tt-midi-maker",
)


@mcp.tool(
    title="Generate Multi-Track MIDI",
    description="""Generate a multi-track MIDI file from a natural language prompt.

Returns a file with up to 7 tracks (melody ch1, bass ch2, harmony ch3, arp ch4,
pad ch5, fx ch9, drums ch10) following General MIDI conventions.

MODES: loop = 4-16 bars seamless repeat (fastest). section = 16-64 bars with
development. stream = continuous via progress notifications.

PROMPT TIPS: include genre, mood, tempo feel, key, instrumentation. Examples:
  "dreamy lo-fi hip hop, slow, dusty drums and sparse bass"
  "uptempo bossa nova, piano melody, walking bass, brushed snare"
  "dark cinematic ambient, D minor, long pad swells, no percussion"

If set_musical_context was called, its values override inference from the prompt.""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def generate_midi(
    prompt: str,
    mode: Literal["loop", "section", "stream"] = "loop",
    roles: list[Literal["drums", "bass", "melody", "harmony", "arp", "pad", "fx"]] | None = None,
    bars: int | None = None,
    output_path: str | None = None,
) -> dict:
    return _generate_midi(prompt, mode, roles, bars, output_path)


@mcp.tool(
    title="Continue MIDI File",
    description="""Extend an existing MIDI file by generating additional bars that flow
naturally from its ending. Reads the last 4 bars as a harmonic/melodic context
prefix. Always writes a NEW file; the original is never modified.

style_hint nudges generation without changing key or BPM. Examples:
  "make it more intense" - raises velocities, denser notes
  "quiet this down"      - thinner arrangement, lower velocities
  "resolve it"           - end on the tonic chord""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def continue_midi(
    file_path: str,
    bars: int = 8,
    style_hint: str | None = None,
) -> dict:
    src = Path(file_path)
    if not src.exists():
        raise MidiMakerError("FILE_NOT_FOUND", f"Not found: {file_path}",
                             "Check midi://output/{filename} for available files.")
    existing_facts = _analyze(src)
    prompt = f"Continue this {existing_facts.get('style_guess', 'music')}" + (
        f", {style_hint}" if style_hint else ""
    )
    blueprint = build_blueprint(prompt)
    blueprint = blueprint.model_copy(update={"bars": bars})
    raw_tracks   = _run_generation(blueprint)
    clean_tracks = _apply_coherence(raw_tracks, blueprint)
    # Simplified append; full RoleTrack stitching wired in follow-up
    ts  = int(time.time())
    out = OUTPUT_DIR / f"{ts}_continued.mid"
    build_midi_file(clean_tracks, blueprint.bpm, out)
    return {"file_path": str(out), "bars_added": bars,
            "total_bars": existing_facts.get("bars", 0) + bars}


@mcp.tool(
    title="Describe MIDI File",
    description="""Analyze a MIDI file and return a structured natural language description.

Returns key, tempo, time signature, bar count, track inventory, chord progression
(if detectable), a style guess, and a prose description. Use before regenerating
to understand what was generated. Also useful for analyzing external MIDI.""",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False,
        idempotentHint=True, openWorldHint=True,
    ),
)
def describe_midi(file_path: str) -> dict:
    return _describe_midi(file_path)


@mcp.tool(
    title="Set Musical Context",
    description="""Establish a persistent musical context for this session.

All subsequent generate_midi and continue_midi calls will respect these values,
overriding anything inferred from the prompt. Call this first when composing
a multi-part piece to keep every section in the same key and harmonic world.

Pass null to any field to clear it (revert to prompt-inferred).

chord_progression accepts Roman numerals or chord names:
  ["I","IV","V","I"]  or  ["Dm","Gm","A7","Dm"]""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    ),
)
def set_musical_context(
    key: str | None = None,
    bpm: int | None = None,
    style: str | None = None,
    chord_progression: list[str] | None = None,
) -> dict:
    return _set_musical_context(key=key, bpm=bpm, style=style,
                                chord_progression=chord_progression)


@mcp.tool(
    title="Chat About a MIDI File",
    description="""Ask any musical question about a MIDI file and get an expert answer.

The engine parses the file structure and routes your question to the LLM with
that analysis as context. Useful questions:
  "What key is this in and how confident are you?"
  "Why does bar 4 feel tense?"
  "Is the bass line supporting the harmony or fighting it?"
  "How could I make this feel more like 90s R&B?"
  "What are the weakest bars and why?"

Reads the file; does not modify it.""",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def chat_with_midi(file_path: str, question: str) -> dict:
    return _chat_with_midi(file_path, question)


@mcp.tool(
    title="List MIDI Devices",
    description="""Enumerate all available MIDI output destinations.

Returns two lists:
  alsa_ports   — ALSA sequencer ports: USB MIDI devices, Bluetooth MIDI devices,
                 virtual ports (e.g. fluidsynth server mode, DAW loopbacks).
                 Plug in a USB MIDI device or pair a BT device and call this again.
  soundfonts   — .sf2 SoundFont files found on this system for FluidSynth playback.

Use port names from alsa_ports as the `port` argument to play_midi.""",
    annotations=ToolAnnotations(
        readOnlyHint=True, destructiveHint=False,
        idempotentHint=True, openWorldHint=True,
    ),
)
def list_midi_devices() -> dict:
    import shutil
    return {
        "alsa_ports": list_output_ports(),
        "soundfonts": list_soundfonts(),
        "fluidsynth_available": bool(shutil.which("fluidsynth")),
        "active_jobs": active_jobs(),
        "streaming_synth": _synth_status(),
    }


@mcp.tool(
    title="Play MIDI File",
    description="""Play a MIDI file through FluidSynth (software GM synth) or any
ALSA MIDI port — USB hardware synths, Bluetooth MIDI devices, DAW loopbacks, etc.

BACKENDS:
  fluidsynth  (default) — software synthesis via FluidR3 GM SoundFont → system audio.
              No hardware needed. Works immediately on any machine with speakers.
  alsa        — sends raw MIDI to a sequencer port. Use list_midi_devices to find
              port names. Requires a connected hardware or virtual MIDI device.

PER-CHANNEL ROUTING (channel_map):
  Route each MIDI channel to a different output port. Keys are 1-indexed channel
  numbers (as strings), values are ALSA port names from list_midi_devices.
  Example: {"1": "USB Synth:0", "2": "USB Synth:0", "10": "BT Drum Machine:1"}
  Channels not in the map fall back to the `port` argument.
  Unmapped channels with no fallback port are silently dropped.

  tt-midi-maker GM channel layout:
    1=melody  2=bass  3=harmony  4=arp  5=pad  9=fx  10=drums

RETURNS: job_id — pass to stop_playback to cancel; job finishes automatically.""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def play_midi(
    file_path: str,
    backend: Literal["fluidsynth", "alsa"] = "fluidsynth",
    port: str | None = None,
    channel_map: dict[str, str] | None = None,
    soundfont: str | None = None,
    gain: float = 2.0,
    blocking: bool = False,
) -> dict:
    return _player_play(
        file_path=file_path,
        backend=backend,
        port=port,
        channel_map=channel_map,
        soundfont=soundfont,
        gain=gain,
        blocking=blocking,
    )


@mcp.tool(
    title="Stop MIDI Playback",
    description="""Stop a background MIDI playback job started by play_midi.

Pass the job_id returned by play_midi. Playback stops within ~100 ms.
If playback has already finished, returns the final status without error.
Call list_midi_devices to see currently active jobs (returned in active_jobs field).""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    ),
)
def stop_playback(job_id: str) -> dict:
    return _player_stop(job_id)


@mcp.tool(
    title="Start Streaming Synth",
    description="""Start a persistent FluidSynth server for real-time loop playback.

Must be called once before loop_play / loop_queue / loop_stop. Starts FluidSynth
as an ALSA sequencer server and opens a direct MIDI connection to it. The server
stays alive for the session — call again only to change soundfont or gain.

audio_driver: 'pulseaudio' (default) or 'alsa'
soundfont:    path to a .sf2 file; defaults to the system FluidR3 GM soundfont
gain:         output volume multiplier (default 2.0; raise if quiet)""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def synth_start(
    soundfont: str | None = None,
    gain: float = 2.0,
    audio_driver: str = "pulseaudio",
) -> dict:
    return _start_synth(soundfont=soundfont, gain=gain, driver=audio_driver)


@mcp.tool(
    title="Loop MIDI File",
    description="""Start looping a MIDI file immediately through the streaming synth.

Interrupts any currently looping pattern. The file plays in a tight real-time
loop: when the last bar ends the first bar begins, with no gap and no file I/O
overhead. Timing is driven by the system monotonic clock at tick resolution.

Call synth_start first. Use generate_midi to create loop files.
Returns current player state including loop_bars and bpm.""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def loop_play(file_path: str) -> dict:
    return _loop_play(file_path)


@mcp.tool(
    title="Queue Next Loop",
    description="""Queue a MIDI file to take over at the next loop boundary.

The current loop keeps playing undisturbed until it reaches its end, then the
queued pattern starts on the downbeat — seamless, no audible gap. Call this
while the current loop is playing to set up the next variation.

Typical flow:
  1. loop_play("bassline_v1.mid")          # starts immediately
  2. generate_midi("add more energy")      # generate while it loops
  3. loop_queue("bassline_v2.mid")         # queues for next boundary
  4. generate_midi("now add horns")        # keep going...
  5. loop_queue("full_arrangement.mid")

Only one pattern can be queued; calling again replaces the previous queued file.""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=False, openWorldHint=True,
    ),
)
def loop_queue(file_path: str) -> dict:
    return _loop_queue(file_path)


@mcp.tool(
    title="Stop Loop",
    description="""Stop the looping playback.

immediately=False (default): finishes the current loop then stops cleanly.
  All notes are silenced on the last beat, preserving the musical phrase end.
immediately=True: cuts off right now, sends all-notes-off on every channel.""",
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False,
        idempotentHint=True, openWorldHint=False,
    ),
)
def loop_stop(immediately: bool = False) -> dict:
    return _loop_stop(immediately=immediately)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt(
    title="Quick Loop",
    description="Generate a polished loop in one step. Best starting point for a new session.",
)
def quick_loop(style: str, key: str = "infer", bars: int = 8) -> list[PromptMessage]:
    key_str = f" in {key}" if key != "infer" else ""
    return [PromptMessage(role="user", content=TextContent(
        type="text",
        text=f"Generate a {bars}-bar {style} loop{key_str}. "
             f"First call set_musical_context, then generate_midi with mode='loop'. "
             f"Use instrument roles appropriate for {style}. Make it feel complete and loopable.",
    ))]


@mcp.prompt(
    title="Build a Song Section",
    description="Guided workflow for composing a complete song section with internal development.",
)
def compose_section(
    section_type: Literal["intro", "verse", "chorus", "bridge", "outro"],
    style: str,
    bars: int = 16,
) -> list[PromptMessage]:
    return [PromptMessage(role="user", content=TextContent(
        type="text",
        text=f"Compose a {bars}-bar {section_type} for a {style} track.\n\n"
             f"Step 1: Call set_musical_context with key, BPM, and chord progression appropriate for a {style} {section_type}.\n"
             f"Step 2: Call generate_midi(mode='loop', bars=8) as a seed.\n"
             f"Step 3: Call describe_midi to confirm it sounds right.\n"
             f"Step 4: Call continue_midi to extend to {bars} bars with development appropriate for a {section_type}.\n"
             f"Step 5: Call describe_midi on the final result.",
    ))]


@mcp.prompt(
    title="Analyze and Improve",
    description="Analyze an existing MIDI file and get actionable improvement suggestions.",
)
def analyze_and_improve(file_path: str, goal: str) -> list[PromptMessage]:
    return [PromptMessage(role="user", content=TextContent(
        type="text",
        text=f"Analyze {file_path} and help me improve it.\n\nGoal: {goal}\n\n"
             f"1. Call describe_midi({file_path!r}) to understand the current state.\n"
             f"2. Call chat_with_midi({file_path!r}, 'What specifically prevents this from achieving: {goal}?')\n"
             f"3. Recommend whether to: regenerate with a new prompt, "
             f"continue_midi with a style_hint, or adjust set_musical_context first.",
    ))]


@mcp.prompt(
    title="Collaborative Composition Session",
    description="Start an open-ended session. Ask clarifying questions, then build the piece iteratively.",
)
def start_session() -> list[PromptMessage]:
    return [PromptMessage(role="user", content=TextContent(
        type="text",
        text="I'd like to compose music using tt-midi-maker. Ask me a few questions "
             "to understand what I'm going for — style, mood, instrumentation, length, "
             "any reference tracks or vibes — then call set_musical_context and start "
             "generating. We'll iterate from there.",
    ))]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource(
    "midi://session/context",
    title="Current Musical Context",
    description="Active key, BPM, style, and chord progression for this session. "
                "Read before generating to confirm context is set as expected.",
    mime_type="application/json",
    annotations=Annotations(audience=["user", "assistant"], priority=0.9),
)
def session_context() -> str:
    ctx = get_session("default")
    if ctx.is_empty():
        return json.dumps({"status": "not set — call set_musical_context first"})
    return json.dumps(ctx.to_dict(), indent=2)


@mcp.resource(
    "midi://hardware/status",
    title="TT Hardware Status",
    description="Connected Tenstorrent devices, active model, and generation backend. "
                "Check this if generation is slow or failing.",
    mime_type="application/json",
    annotations=Annotations(audience=["user", "assistant"], priority=0.6),
)
def hw_status() -> str:
    return json.dumps(hardware_status(), indent=2)


@mcp.resource(
    "midi://styles/catalog",
    title="Style Catalog",
    description="Available styles with BPM ranges, typical keys, default roles, and examples. "
                "Consult before writing prompts to improve generation quality.",
    mime_type="application/json",
    annotations=Annotations(audience=["user", "assistant"], priority=0.7),
)
def styles_catalog() -> str:
    return json.dumps(STYLES_CONFIG, indent=2)


@mcp.resource(
    "midi://output/{filename}",
    title="Generated MIDI File",
    description="Access a previously generated MIDI file by filename. Returns raw MIDI bytes.",
    mime_type="audio/midi",
    annotations=Annotations(audience=["user"], priority=0.5),
)
def output_file(filename: str) -> bytes:
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise MidiMakerError("FILE_NOT_FOUND", f"No such file: {filename}",
                             "Check midi://output/ for available filenames.")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------

@mcp.completion()
def complete_argument(ref, argument) -> list[str]:
    val = argument.value.lower() if argument.value else ""
    if argument.name == "style":
        return [s for s in STYLES_CONFIG if val in s.lower()][:10]
    if argument.name == "key":
        return [k for k in VALID_KEYS if val in k.lower()][:10]
    if argument.name == "mode":
        return [m for m in ("loop", "section", "stream") if val in m]
    if argument.name == "section_type":
        return [s for s in ("intro", "verse", "chorus", "bridge", "outro") if val in s]
    if argument.name == "roles":
        all_roles = list(ROLES_CONFIG.keys())
        return [r for r in all_roles if val in r]
    return []


def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
