# tt-midi-maker

Multi-track MIDI generation from text prompts, accelerated by Tenstorrent hardware and
exposed as a fully-featured [MCP](https://modelcontextprotocol.io) server.

```
Prompt → LLM blueprint → Aria MIDI transformer (tt-forge) → coherence layer → GM MIDI
```

---

## What it does

Give it a text description; get back a polished, multi-track General MIDI file ready to
drop into any DAW, sampler, or hardware synth — or play directly through the built-in
FluidSynth software synthesizer.

```
"cool jazz vamp, D minor, sparse for soloing, 68 BPM"
  → Dm | Gm | Bb | C progression, 4-bar loop
  → melody (ch1) + bass (ch2) + harmony (ch3) + drums (ch10)
  → scale-quantized, chord-filtered, humanised
  → ~/Music/tt-midi-maker/1716000000.mid
  → synth_start() → loop_play("…/1716000000.mid")  ← plays right now
```

---

## Features

| Category | Details |
|---|---|
| **Generation** | Aria MIDI transformer via tt-forge; CPU fallback when no TT device is detected |
| **Coherence** | Scale quantisation, chord-aware note filtering, velocity humanisation, timing nudge, phrase stitching |
| **Roles** | 12 GM roles across channels 1–12: melody, bass, harmony, arp, pad, lead, strings, brass, fx, drums, guitar, organ |
| **Styles** | 20 built-in across electronic, jazz, rock, world, cinematic, and experimental genres |
| **File playback** | FluidSynth (GM software synth → system audio) or raw MIDI to any ALSA port with per-channel routing |
| **Streaming loops** | Real-time loop player with seamless pattern transitions at bar boundaries (no gap, no file I/O) |
| **MCP interface** | 12 tools · 4 prompts · 4 resources · argument completions |
| **Session state** | Persistent key / BPM / style / chord progression across calls |

---

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Install FluidSynth + GM SoundFont for local audio (Ubuntu/Debian)
sudo apt install fluidsynth fluid-soundfont-gm

# Point at your LLM (needs OpenAI-compatible /chat/completions)
export MIDI_LLM_URL=http://localhost:8000/v1
export MIDI_LLM_MODEL=qwen3          # or llama3, mistral, etc.

# Start the MCP server (streamable-HTTP on :8000 by default)
python -m tt_midi_maker
```

The server announces itself at `http://127.0.0.1:8000`.  
Output files land in `~/Music/tt-midi-maker/`.

---

## MCP Tools

### Generation

#### `generate_midi`

Generate a new multi-track MIDI file from a prompt.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | — | Natural language description of the music |
| `mode` | `loop\|section\|stream` | `loop` | `loop` = 4–16 bar seamless repeat; `section` = 16–64 bar developed piece; `stream` = continuous |
| `roles` | `list[str]` | all roles | Restrict output to these roles (e.g. `["melody","drums"]`) |
| `bars` | `int` | from LLM | Override bar count |
| `output_path` | `str` | auto-timestamped | Absolute path for the output file |

Returns: `file_path`, `bars_generated`, `bpm`, `key`, `roles_generated`, `hardware_used`

**Prompt tips — include:**
- Genre or style: "uptempo bossa nova", "dark cinematic ambient"
- Mood: "melancholic", "euphoric", "tense"
- Tempo feel: "slow", "driving", "half-time"
- Key (optional): "D minor", "F# major"
- Instruments: "dusty drums", "walking bass", "electric piano"

---

#### `continue_midi`

Extend an existing MIDI file, maintaining musical continuity.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | — | Path to an existing `.mid` file |
| `bars` | `int` | `8` | How many bars to add |
| `style_hint` | `str` | — | Nudge generation: `"make it more intense"`, `"resolve it"` |

Always writes a **new** file; the original is unchanged.  
Returns: `file_path`, `bars_added`, `total_bars`

---

#### `set_musical_context`

Establish a persistent session context that overrides anything inferred from prompts.

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | e.g. `"D minor"`, `"F# major"` — pass `null` to clear |
| `bpm` | `int` | 40–300 — pass `null` to clear |
| `style` | `str` | e.g. `"lo-fi hip hop"` |
| `chord_progression` | `list[str]` | Roman numerals or chord names: `["Dm","Gm","A7","Dm"]` |

Set this first when composing a multi-part piece. Returns: all fields plus `fields_set`.

---

### Analysis

#### `describe_midi`

Analyse a MIDI file and return a structured musical description.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to a `.mid` file |

Returns: `key`, `tempo_bpm`, `time_signature`, `bars`, `tracks`, `chord_progression`,
`style_guess`, `description` (prose)

---

#### `chat_with_midi`

Ask any musical question about a MIDI file.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to a `.mid` file |
| `question` | `str` | Free-text question |

Example questions:
- `"What key is this in and how confident are you?"`
- `"Why does bar 4 feel tense?"`
- `"Is the bass line supporting the harmony or fighting it?"`
- `"How could I make this feel more like 90s R&B?"`

Returns: `answer`, `analysis_context`

---

### File Playback

#### `list_midi_devices`

Enumerate all available MIDI output destinations.

Returns: `alsa_ports`, `soundfonts`, `fluidsynth_available`, `active_jobs`,
`streaming_synth` (status of the real-time loop player).

---

#### `play_midi`

Play a MIDI file once through FluidSynth or any ALSA MIDI port.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | — | Path to a `.mid` file |
| `backend` | `"fluidsynth"\|"alsa"` | `"fluidsynth"` | Synthesis engine |
| `port` | `str` | auto | ALSA output port name (alsa backend only) |
| `channel_map` | `dict[str,str]` | — | Per-channel port routing; keys are 1-indexed channel strings |
| `soundfont` | `str` | FluidR3 GM | Path to a `.sf2` SoundFont |
| `gain` | `float` | `2.0` | Output volume multiplier |
| `blocking` | `bool` | `false` | Wait for playback to finish before returning |

Returns: `job_id` — pass to `stop_playback` to cancel early.

**Per-channel routing example:**
```json
{
  "channel_map": {
    "1": "USB Synth A:0",
    "10": "Bluetooth Drum Machine:1"
  }
}
```
Channels not in the map fall back to `port`. GM layout: 1=melody, 2=bass,
3=harmony, 4=arp, 5=pad, 9=fx, 10=drums.

---

#### `stop_playback`

Stop a background playback job started by `play_midi`.

| Parameter | Type | Description |
|---|---|---|
| `job_id` | `str` | Job ID returned by `play_midi` |

---

### Streaming Loop Playback

The streaming player runs FluidSynth as a persistent ALSA sequencer server and drives
it from a background thread using the monotonic clock. Patterns transition at bar
boundaries with no audible gap — ideal for live performance, composition workflows,
and waiting-for-instruction loops.

#### `synth_start`

Start the FluidSynth server. Call once per session before any `loop_*` tool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `soundfont` | `str` | FluidR3 GM | Path to a `.sf2` SoundFont |
| `gain` | `float` | `2.0` | Output volume multiplier |
| `audio_driver` | `str` | `"pulseaudio"` | Audio backend: `"pulseaudio"` or `"alsa"` |

Returns: `status`, `port`, `running`, `soundfont`, `gain`

---

#### `loop_play`

Start looping a MIDI file immediately. Interrupts any current playback.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to a `.mid` file |

Returns: player state — `state`, `current_file`, `queued_file`, `bpm`, `loop_bars`,
`loops_played`

---

#### `loop_queue`

Queue a MIDI file to take over at the next loop boundary.

The current loop plays undisturbed until it reaches its natural end, then the queued
pattern starts on the downbeat — no gap, no click.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to a `.mid` file |

Only one pattern can be queued; calling again replaces the previous queued file.

---

#### `loop_stop`

Stop the loop player.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `immediately` | `bool` | `false` | `false` = finish current loop then stop; `true` = cut off now + all-notes-off |

---

**Typical streaming workflow:**

```
synth_start()
loop_play("dm_jazz_vamp.mid")          # starts immediately

# While it loops — generate a variation
generate_midi("add trombone solo")     # creates next_pattern.mid
loop_queue("next_pattern.mid")         # transitions on next bar boundary

# Generate another layer
generate_midi("make it more intense")
loop_queue("intense.mid")              # replaces queued slot

loop_stop()                            # finishes current loop, then stops
```

---

## MCP Prompts

Pre-built workflows for common composition tasks:

| Prompt | Description |
|---|---|
| `quick_loop(style, key?, bars?)` | One-shot loop: sets context then generates. Best starting point. |
| `compose_section(section_type, style, bars?)` | Guided multi-step workflow: intro / verse / chorus / bridge / outro |
| `analyze_and_improve(file_path, goal)` | Diagnose a MIDI file and recommend regeneration strategy |
| `start_session()` | Open-ended session: asks clarifying questions then iterates |

---

## MCP Resources

| URI | Description |
|---|---|
| `midi://session/context` | Active key, BPM, style, chord progression for the default session |
| `midi://hardware/status` | Connected TT devices and active generation backend |
| `midi://styles/catalog` | All styles with BPM ranges, typical keys, default roles |
| `midi://output/{filename}` | Raw bytes of a previously generated MIDI file |

---

## Style catalog

**20 styles** across electronic, jazz, rock, world, cinematic, and experimental genres.

### Electronic / Hip-Hop

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| lo-fi hip hop | 70–90 | drums, bass, melody, pad | 0.55 | pensive, oozy |
| hip hop | 80–100 | drums, bass, melody, arp | 0.55 | tense, exuberant |
| drum and bass | 160–180 | drums, bass, arp | 0 | tense, synthetic |
| synthwave | 90–118 | melody, bass, arp, pad, drums | 0 | atmospheric, synthetic |
| idm | 100–160 | drums, bass, arp, fx, pad | 0 | glitchy, synthetic |
| detroit techno | 128–138 | drums, bass, arp, pad | 0 | tense, atmospheric |

### Jazz / Swing

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| jazz | 120–200 | drums, bass, melody, harmony | 0.65 | pensive, organic |
| bossa nova | 120–160 | drums, bass, melody, harmony | 0 | pensive, ethereal |
| blues | 60–130 | melody, bass, harmony, drums | 0.65 | pensive, tense |
| ska | 160–200 | melody, bass, harmony, brass, drums | 0.5 | exuberant, organic |

### Rock

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| surf rock | 140–180 | melody, bass, harmony, drums | 0 | exuberant, tense |
| post-rock | 60–140 | melody, harmony, bass, strings, drums | 0 | atmospheric, pensive |

### World / Latin

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| afrobeat | 100–130 | drums, bass, melody, harmony, brass | 0 | exuberant, organic |
| cumbia | 100–120 | drums, bass, melody, harmony, arp | 0.2 | exuberant, organic |

### Cinematic / Classical

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| classical | 60–180 | melody, strings, harmony, bass | 0 | pensive, tense |
| nino rota | 80–140 | melody, harmony, bass, arp, drums | 0.4 | pensive, ethereal |
| dark cinematic | 60–110 | strings, pad, bass, fx, melody | 0 | tense, atmospheric |

### Ambient / Experimental

| Style | BPM range | Default roles | Swing | Mood |
|---|---|---|---|---|
| ambient | 60–90 | pad, melody, fx | 0 | spacey, ethereal |
| dark ambient | 40–70 | pad, fx, bass, melody | 0 | spacey, oozy |
| glitch | 90–150 | drums, bass, fx, pad, arp | 0 | glitchy, chaos |

---

## GM channel / role map

**12 roles** across 12 GM channels. Default programs are starting points — generation
overrides them based on style and prompt.

| Role | GM ch | Default program | Note range | Purpose |
|---|---|---|---|---|
| melody | 1 | 0 · Acoustic Grand Piano | C4–C7 | Primary melodic voice |
| bass | 2 | 32 · Acoustic Bass | E1–E3 | Root-motion bass line |
| harmony | 3 | 48 · String Ensemble 1 | C3–C5 | Chordal / comping support |
| arp | 4 | 4 · Electric Piano 1 | C4–C6 | Arpeggiated forward motion |
| pad | 5 | 89 · Pad 2 Warm | C2–C5 | Sustained background texture |
| lead | 6 | 56 · Trumpet | C4–C7 | Bright solo lead (trumpet, sax, guitar) |
| strings | 7 | 49 · String Ensemble 2 | E2–C6 | Orchestral strings, swells, pizzicato |
| brass | 8 | 61 · Brass Section | Bb2–F5 | Horn stabs, ska upstrokes, fanfares |
| fx | 9 | 88 · Pad 1 New Age | full | Drones, sweeps, cinematic noise |
| drums | **10** | — percussion | GM kit | Main drum kit |
| guitar | 11 | 25 · Acoustic Steel | E2–E5 | Guitar parts (program sets style) |
| organ | 12 | 16 · Drawbar Organ | C2–C7 | Hammond/church organ, blues comping |

---

## Architecture

```
tt_midi_maker/
├── server.py              MCP server — 12 tools, 4 prompts, 4 resources
├── stream_player.py       Real-time loop player: FluidSynthServer + LoopPlayer
├── player.py              File-based playback: FluidSynth subprocess or ALSA port
├── prompt_engine.py       LLM → MusicalBlueprint (Pydantic model)
├── assembler.py           RoleTrack list → Type-1 mido MidiFile
├── analyzer.py            mido parse → facts dict → LLM describe / chat
├── session.py             Per-session MusicalContext (key/bpm/style/chords)
├── errors.py              MidiMakerError dataclass (code / message / suggestion)
├── models/
│   ├── blueprint.py       MusicalBlueprint + RoleConfig Pydantic models
│   └── track.py           NoteEvent + RoleTrack dataclasses
├── coherence/
│   ├── scale.py           Scale quantisation (7 modes, probabilistic)
│   ├── harmony.py         Chord-aware note filtering (strong beats)
│   ├── humanize.py        Velocity humanisation + timing nudge
│   └── stitching.py       Crossfade phrase stitching
└── generation/
    ├── hardware.py        tt-smi device detection
    ├── tokenizer.py       MidiTok REMI+ wrapper (encode/decode)
    └── aria_backend.py    Aria model load + tt-forge compile + generate
config/
├── roles.yaml             GM channel assignments and note ranges
└── styles.yaml            Style definitions with BPM ranges and defaults
```

### Pipeline detail

```
Prompt + MusicalContext
        │
        ▼
   prompt_engine.py  ──LLM call──▶  MusicalBlueprint
        │                            (key, BPM, bars, chord_progression, roles)
        ▼
   aria_backend.py  ──Aria/tt-forge──▶  raw RoleTrack list
        │
        ▼  coherence passes (in order):
   scale.py          nearest in-key pitch
   harmony.py        filter off-beat non-chord tones
   humanize.py       ±velocity + timing jitter
   stitching.py      crossfade when appending to existing tracks
        │
        ▼
   assembler.py  ──────────────────▶  Type-1 .mid file
        │
        ├─▶ player.py (file-based)    fluidsynth -ni → audio, or mido → ALSA port
        └─▶ stream_player.py (loop)   FluidSynthServer + LoopPlayer → seamless loops
```

### Streaming player internals

```
FluidSynthServer
  subprocess.Popen(["fluidsynth", "-a", "pulseaudio", "-m", "alsa_seq", ...],
                   stdin=PIPE)     ← stdin=PIPE keeps the process alive
  polls mido.get_output_names() until "FLUID Synth ..." appears
  returns ALSA port name

LoopPlayer (background thread)
  time.monotonic() clock — sub-millisecond note scheduling
  spt = 60 / (bpm × ticks_per_beat)  seconds per tick
  loop_origin advances by loop_ticks × spt at each boundary
  _next pattern swapped in atomically under threading.Lock
  stop(immediately=False) → sets state="stopping", exits after current loop
  stop(immediately=True)  → sets state="stopped", sends CC#123 all-notes-off
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MIDI_LLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint |
| `MIDI_LLM_MODEL` | `qwen3` | Model name sent in `/chat/completions` requests |

---

## Testing

```bash
pytest tests/ -v
```

131 tests cover all coherence passes, the assembler, session state, MCP tool handlers,
Aria backend (mocked), tokenizer, analyzer, hardware detection, streaming loop player
(timing, loop count, queue transitions, all-notes-off), file-based player (both
backends), and end-to-end integration with mocked LLM and FluidSynth.

---

## Connecting to an MCP client

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tt-midi-maker": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Start the server first (`python -m tt_midi_maker`), then restart Claude Desktop.

### Claude Code

```bash
claude mcp add tt-midi-maker --transport http --url http://127.0.0.1:8000/mcp
```

### Any MCP client

The server uses the **streamable-HTTP** transport. Connect to:
`http://127.0.0.1:8000/mcp`

---

## Extending

**Add a style** — edit `config/styles.yaml` and restart the server.  
No code changes needed; styles are loaded at startup and served via the catalog resource.

**Add a role** — add an entry to `config/roles.yaml` (channel, program, note_range).  
Update `generate_midi`'s `roles` parameter type annotation in `server.py`.

**Swap the LLM** — set `MIDI_LLM_URL` / `MIDI_LLM_MODEL`.  
The prompt engine works with any OpenAI-compatible `/chat/completions` endpoint.

**Swap the generation backend** — replace `_run_generation` in `server.py`.  
It receives a `MusicalBlueprint` and must return `list[RoleTrack]`.

**Use a different SoundFont** — pass `soundfont=` to `synth_start` or `play_midi`.  
Drop any `.sf2` file into `~/.local/share/sounds/sf2/` and it appears in `list_midi_devices`.

---

## License

Apache 2.0 — see `LICENSE`.
