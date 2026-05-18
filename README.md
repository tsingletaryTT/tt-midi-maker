# tt-midi-maker

Multi-track MIDI generation from text prompts, accelerated by Tenstorrent hardware and
exposed as a fully-featured [MCP](https://modelcontextprotocol.io) server.

```
Prompt → LLM blueprint → Aria MIDI transformer (tt-forge) → coherence layer → GM MIDI
```

---

## What it does

Give it a text description; get back a polished, multi-track General MIDI file ready to
drop into any DAW or sampler.

```
"dreamy lo-fi hip hop, slow, dusty drums and sparse piano"
  → 120 BPM, C major, 8-bar loop
  → melody (ch1) + bass (ch2) + pad (ch5) + drums (ch10)
  → scale-quantized, chord-filtered, humanised
  → ~/Music/tt-midi-maker/1716000000.mid
```

---

## Features

| Category | Details |
|---|---|
| **Generation** | Aria MIDI transformer via tt-forge; CPU fallback when no TT device is detected |
| **Coherence** | Scale quantisation, chord-aware note filtering, velocity humanisation, timing nudge, phrase stitching |
| **Roles** | 7 GM roles: melody (ch1), bass (ch2), harmony (ch3), arp (ch4), pad (ch5), fx (ch9), drums (ch10) |
| **Styles** | 6 built-in: lo-fi hip hop, bossa nova, ambient, hip hop, jazz, drum and bass |
| **MCP interface** | 5 tools · 4 prompts · 4 resources · argument completions |
| **Session state** | Persistent key / BPM / style / chord progression across calls |

---

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Point at your LLM (needs OpenAI-compatible /chat/completions)
export MIDI_LLM_URL=http://localhost:8000/v1
export MIDI_LLM_MODEL=qwen3          # or llama3, mistral, etc.

# Start the MCP server (streamable-HTTP on :8000 by default)
python -m tt_midi_maker

# Or use the installed entry point
tt-midi-maker
```

The server announces itself at `http://127.0.0.1:8000`.  
Output files land in `~/Music/tt-midi-maker/`.

---

## MCP Tools

### `generate_midi`

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

### `continue_midi`

Extend an existing MIDI file, maintaining musical continuity.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str` | — | Path to an existing `.mid` file |
| `bars` | `int` | `8` | How many bars to add |
| `style_hint` | `str` | — | Nudge generation: `"make it more intense"`, `"resolve it"` |

Always writes a **new** file; the original is unchanged.  
Returns: `file_path`, `bars_added`, `total_bars`

---

### `describe_midi`

Analyse a MIDI file and return a structured musical description.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to a `.mid` file |

Returns: `key`, `tempo_bpm`, `time_signature`, `bars`, `tracks`, `chord_progression`,
`style_guess`, `description` (prose)

---

### `set_musical_context`

Establish a persistent session context that overrides anything inferred from prompts.

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | e.g. `"D minor"`, `"F# major"` — pass `null` to clear |
| `bpm` | `int` | 40–300 — pass `null` to clear |
| `style` | `str` | e.g. `"lo-fi hip hop"` |
| `chord_progression` | `list[str]` | Roman numerals or chord names: `["Dm","Gm","A7","Dm"]` |

Set this first when composing a multi-part piece to keep all sections in the same
harmonic world. Returns: all fields plus `fields_set`.

---

### `chat_with_midi`

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

| Style | BPM range | Default roles | Swing |
|---|---|---|---|
| lo-fi hip hop | 70–90 | drums, bass, melody, pad | 0.55 |
| bossa nova | 120–160 | drums, bass, melody, harmony | 0 |
| ambient | 60–90 | pad, melody, fx | 0 |
| hip hop | 80–100 | drums, bass, melody, arp | 0.55 |
| jazz | 120–200 | drums, bass, melody, harmony | 0.65 |
| drum and bass | 160–180 | drums, bass, arp | 0 |

---

## GM channel / role map

| Role | GM channel | Default program | Note range |
|---|---|---|---|
| melody | 1 | 0 (Acoustic Grand Piano) | C4–C8 |
| bass | 2 | 32 (Acoustic Bass) | E1–E3 |
| harmony | 3 | 48 (String Ensemble 1) | C3–C5 |
| arp | 4 | 4 (Electric Piano 1) | C4–C6 |
| pad | 5 | 89 (Pad 2 Warm) | C2–C5 |
| fx | 9 | 88 (Pad 1 New Age) | full range |
| drums | **10** | — (percussion) | GM kit |

---

## Architecture

```
tt_midi_maker/
├── server.py              MCP server — 5 tools, 4 prompts, 4 resources
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

98 tests cover all coherence passes, the assembler, session state, MCP tool handlers,
Aria backend (mocked), tokenizer, analyzer, hardware detection, and end-to-end
integration with mocked LLM.

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

---

## License

Apache 2.0 — see `LICENSE`.
