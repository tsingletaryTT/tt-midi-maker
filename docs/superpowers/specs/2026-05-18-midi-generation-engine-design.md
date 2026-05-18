# tt-midi-maker: MIDI Generation Engine Design

**Date:** 2026-05-18  
**Status:** Approved  
**Original prompt:** Design a sophisticated MIDI generation engine that converts written prompts into musical phrases on multiple MIDI tracks for routing to multiple instruments, using standard MIDI channels, leveraging TT hardware (tt-xla / tt-forge / tt-lang), and supplementing model output with musical coherence.

---

## 1. Overview

tt-midi-maker is a five-stage pipeline that converts text prompts into polished, multi-track MIDI files using Tenstorrent AI hardware. It exposes its capabilities as a fully-featured MCP server so any LLM client (Claude, Cursor, custom agents) can compose, extend, and analyze music programmatically.

**Core approach (Approach B — selected):** LLM blueprint → Aria multi-track MIDI transformer (tt-forge compiled) → music theory coherence layer → GM MIDI assembly → MCP server.

---

## 2. Architecture

```
[text prompt]
      ↓
┌──────────────────────────────────────────┐
│  Stage 1: Prompt Engine                  │
│  Qwen3/Llama via TT inference server     │
│  output: MusicalBlueprint (JSON)         │
└──────────────────────────────────────────┘
      ↓  key · BPM · chord progression · role densities · bars · mode
┌──────────────────────────────────────────┐
│  Stage 2: Generation Engine              │
│  Aria-medium (tt-forge compiled)         │
│  REMI+ token generation, per-role        │
└──────────────────────────────────────────┘
      ↓  raw NoteEvents per role (pitch · velocity · time · duration)
┌──────────────────────────────────────────┐
│  Stage 3: Coherence Layer                │
│  scale snap · chord tones · humanize     │
│  micro-timing · phrase stitching         │
└──────────────────────────────────────────┘
      ↓  musicalized NoteEvents
┌──────────────────────────────────────────┐
│  Stage 4: MIDI Assembler                 │
│  GM channel assignment · mido output     │
└──────────────────────────────────────────┘
      ↓  .mid file
┌──────────────────────────────────────────┐
│  Stage 5: MCP Server (FastMCP)           │
│  5 tools · 4 prompts · 4 resources       │
│  completions · session state             │
└──────────────────────────────────────────┘
```

---

## 3. Module Layout

```
tt_midi_maker/
├── server.py               # MCP server + all tools, prompts, resources
├── session.py              # MusicalContext session state (per-connection)
├── prompt_engine.py        # Stage 1: LLM call → MusicalBlueprint
├── models/
│   ├── blueprint.py        # Pydantic MusicalBlueprint + RoleConfig
│   └── track.py            # NoteEvent, RoleTrack, RoleConfig
├── generation/
│   ├── aria_backend.py     # Stage 2: Aria inference on TT hardware
│   ├── tokenizer.py        # MidiTok REMI+ encode/decode
│   └── hardware.py         # TT device detection + tt-forge compile
├── coherence/
│   ├── scale.py            # Pass 1: scale quantization
│   ├── harmony.py          # Pass 2: chord-aware note selection
│   ├── humanize.py         # Pass 3: velocity + micro-timing
│   └── stitching.py        # Pass 4: phrase continuation / crossfade
├── assembler.py            # Stage 4: GM channel map + mido file output
└── analyzer.py             # describe_midi + chat_with_midi LLM calls

config/
├── roles.yaml              # GM channel assignments, pitch ranges, programs
└── styles.yaml             # Style catalog with BPM ranges, typical keys, roles

docs/superpowers/specs/
└── 2026-05-18-midi-generation-engine-design.md

tests/
├── test_prompt_engine.py
├── test_generation.py
├── test_coherence.py
├── test_assembler.py
└── fixtures/               # sample .mid files for tests

CLAUDE.md
pyproject.toml
requirements.txt
```

---

## 4. Stage 1 — Prompt Engine

**Input:** text prompt + optional session MusicalContext  
**Output:** validated MusicalBlueprint

The prompt engine calls an LLM running on the TT inference server (HTTP, OpenAI-compatible API at `http://localhost:8000/v1`) with a system prompt that instructs structured JSON output.

### MusicalBlueprint (Pydantic)

```python
class RoleConfig(BaseModel):
    density: float             # 0.0 = silent, 1.0 = fully active
    velocity_range: tuple[int, int]   # e.g. (60, 100)
    pattern_hint: str          # "walking", "bossa", "sparse", "driving", etc.

class MusicalBlueprint(BaseModel):
    key: str                          # e.g. "D minor", "G major", "F# dorian"
    bpm: int                          # 40–300
    time_signature: str               # "4/4", "3/4", "6/8"
    style: str                        # "bossa nova", "hip hop", "ambient"
    chord_progression: list[str]      # ["Dm", "Gm", "A7", "Dm"]
    bars: int
    mode: Literal["loop", "section", "stream"]
    roles: dict[str, RoleConfig]      # keyed by role name
```

Session context (from `set_musical_context`) overrides any inferred field. Blueprint is validated before proceeding; `CONTEXT_NOT_SET` error is raised if a required field cannot be resolved.

---

## 5. Stage 2 — Generation Engine

### Model: Aria-medium

`nlp4music/aria-medium` — a decoder-only transformer (~300M params) pre-trained on multi-track symbolic MIDI using REMI+ tokenization. It natively handles instrument-role tokens, so each role is generated with awareness of what other roles are playing.

**Tokenization (MidiTok REMI+):** tokens encode Bar, Position, Tempo, Program, Pitch, Velocity, Duration. Multi-track generation interleaves role tokens with track-separator tokens. The musical blueprint is prepended as a style prefix in the same token vocabulary.

### TT Hardware Compilation

```python
# aria_backend.py
def load_aria(device_ids: list[int]) -> callable:
    model = AriaForCausalLM.from_pretrained("nlp4music/aria-medium")
    if device_ids:
        compiled = forge.compile(model, sample_inputs, module_name="aria_midi")
        return compiled
    return model  # CPU-only if no TT hardware detected
```

Follows the tt-forge-models loader pattern (same as musicgen_small, GPT-2, T5 wrappers).

### Fallback Chain

If `aria-medium` fails to compile on available hardware: `aria-mini` → `m-a-p/music-llm` (GPT-2 class). Same MidiTok tokenization and coherence layer apply regardless of which model is used. Fallback is logged and reported in the `hardware_used` field of tool output.

### Generation Modes

| Mode | Bars | Strategy |
|------|------|----------|
| `loop` | 4–16 | Full sequence in one pass; tail checked for head compatibility |
| `section` | 16–64 | 32-bar sliding context window; 4-bar overlap between segments |
| `stream` | ∞ | Last 4 bars always in prefix; each bar emitted as MCP progress notification |

---

## 6. Stage 3 — Coherence Layer

Four sequential passes that transform raw token-sequence output into music that sounds composed:

### Pass 1 — Scale Quantization (`coherence/scale.py`)

Build the chromatic pitch set for `blueprint.key`. For each non-drum note not in the scale, snap to the nearest scale tone. `strictness` parameter (0.0–1.0) controls aggressiveness; 1.0 snaps all off-scale notes, 0.5 snaps probabilistically (preserves chromatic color), 0.0 is off.

### Pass 2 — Chord-Aware Note Selection (`coherence/harmony.py`)

For melody, harmony, and arp tracks: notes on strong beats (beat 1 and beat 3 in 4/4) are compared against the current chord's tones (root, 3rd, 5th, 7th). Off-chord-tones on strong beats are moved to the nearest chord tone. Weak beats and off-beats are left untouched — passing tones, approach notes, and suspensions are preserved.

### Pass 3 — Velocity Humanization (`coherence/humanize.py`)

- Per-role velocity curves: drums emphasize backbeat (beats 2+4) or downbeat (beats 1+3) depending on style; melody has wider dynamic range; bass is steady with slight phrase accents
- Add ±5–10 velocity variation following a gentle contour (louder at phrase peaks, softer at phrase endings)
- Apply attack timing offsets ±5–15 ticks for human feel; swing quantization option for jazz/hip-hop styles

### Pass 4 — Phrase Stitching (`coherence/stitching.py`)

Used by `continue_midi` only. The last 4 bars of the existing file become the generation prefix. At the join point: the first strong-beat note of each new role is checked against the chord implied by the existing tail; if clashing, it's transposed by a scale step. Velocities crossfade at the boundary (last bar of existing: fade to 90%, first new bar: start from 80%, ramp to 100% over 2 bars).

---

## 7. Stage 4 — MIDI Assembler

### GM Channel Assignments (`config/roles.yaml`)

| Role | Ch | Default GM Program | Note Range | Notes |
|------|----|--------------------|------------|-------|
| melody | 1 | 0 Acoustic Piano / 80 Lead Synth | C4–C7 (60–96) | Primary melodic voice |
| bass | 2 | 32 Acoustic Bass | C1–C3 (28–52) | Root motion |
| harmony | 3 | 48 String Ensemble | C3–C5 (48–72) | Chord voicings |
| arp | 4 | 4 Electric Piano | C4–C6 (60–84) | Counter-melody / arpeggios |
| pad | 5 | 89 Warm Pad | C2–C5 (36–72) | Texture / atmosphere |
| fx | 9 | 88 Synth Pad | any | Special effects |
| **drums** | **10** | **— GM drum map** | **35–81** | **Always channel 10** |

Roles with `density: 0.0` are omitted — no track, no channel events. The assembler uses `mido` to build a `MidiFile(type=1)` (multi-track), sets tempo from BPM, and writes to the output path.

---

## 8. Stage 5 — MCP Server

### Server Identity

```python
mcp = FastMCP(
    name="tt-midi-maker",
    instructions="""
tt-midi-maker generates multi-track MIDI files from text prompts using
Tenstorrent AI hardware. Each output file follows General MIDI channel
conventions: drums on channel 10, melody on 1, bass on 2, harmony on 3.

Recommended workflow:
  1. Call set_musical_context to establish key, BPM, style, and chord
     progression. This persists for the session and improves all subsequent
     generate calls. Skip this step only for one-shot requests.
  2. Call generate_midi with a descriptive prompt and a mode:
       loop    – 4–16 bars, loops seamlessly. Fast.
       section – 16–64 bars with internal development.
       stream  – continuous; bars arrive via progress notifications.
  3. Call continue_midi to extend the result, maintaining musical continuity.
  4. Call describe_midi or chat_with_midi to review, understand, or refine.

Output files are written to ~/Music/tt-midi-maker/ by default.
File path arguments accept absolute paths or names relative to that directory.

For the best musical results, prompts should mention: genre or style, mood
or energy level, tempo feel (not just BPM), and whether percussion is wanted.
The engine infers key and BPM when not stated; set_musical_context overrides
any inference.
""",
    website_url="https://github.com/tenstorrent/tt-midi-maker",
)
```

### Capabilities

```python
ServerCapabilities(
    tools=ToolsCapability(listChanged=False),
    resources=ResourcesCapability(subscribe=False, listChanged=False),
    prompts=PromptsCapability(listChanged=False),
    completions=CompletionsCapability(),  # style/key/roles autocomplete
    logging=LoggingCapability(),
)
```

---

### 8.1 — Tools

All tools share a consistent output envelope:
```json
{
  "ok": true,
  "data": { ... },       // tool-specific structured result
  "error": null          // or { "code": "...", "message": "...", "suggestion": "..." }
}
```

#### `generate_midi` — Generate Multi-Track MIDI

- **Annotations:** `readOnlyHint=False`, `destructiveHint=False`, `idempotentHint=False`, `openWorldHint=True`
- **Required inputs:** `prompt` (string, 3–500 chars)
- **Optional inputs:**
  - `mode`: `"loop"` | `"section"` | `"stream"` (default: `"loop"`)
  - `roles`: array of `"drums"` | `"bass"` | `"melody"` | `"harmony"` | `"arp"` | `"pad"` | `"fx"` — defaults to prompt-inferred
  - `bars`: integer 1–256 — defaults: 8 (loop), 32 (section), ignored (stream)
  - `output_path`: string — defaults to `~/Music/tt-midi-maker/<timestamp>.mid`
- **Output schema:** `file_path`, `bars_generated`, `bpm`, `key`, `roles_generated`, `generation_ms`, `hardware_used`
- For `mode="stream"`: the tool returns `{ "status": "streaming", "file_path": "<path being written>" }` immediately; each completed bar arrives as a `notifications/progress` with `{ "bar": N, "total": null }`. The file at `file_path` is valid and playable at any point — it grows as bars are appended.
- **inputSchema examples on every field** — see section 8.4 for full JSON Schema

**Description (shown to LLM clients):**
> Generate a multi-track MIDI file from a natural language prompt. Returns a file with up to 7 tracks following General MIDI channel conventions. Modes: `loop` (4–16 bars, seamless repeat, fastest), `section` (16–64 bars with development), `stream` (continuous via progress notifications). Prompt tips: include genre, mood, tempo feel, key/scale, instrumentation. If `set_musical_context` has been called, key/BPM/style from the context takes precedence over anything inferred.

---

#### `continue_midi` — Continue MIDI File

- **Annotations:** `readOnlyHint=False`, `destructiveHint=False`, `idempotentHint=False`, `openWorldHint=True`
- **Required inputs:** `file_path` (string)
- **Optional inputs:**
  - `bars`: integer 1–128 (default: 8)
  - `style_hint`: string — e.g. `"make it more intense"`, `"quiet this down"`, `"resolve it"`
- **Output schema:** `file_path` (new file), `bars_added`, `total_bars`
- **Never modifies the source file** — always writes a new output

---

#### `describe_midi` — Describe MIDI File

- **Annotations:** `readOnlyHint=True`, `destructiveHint=False`, `idempotentHint=True`, `openWorldHint=True`
- **Required inputs:** `file_path`
- **Output schema:** `key`, `tempo_bpm`, `time_signature`, `bars`, `tracks` (array), `chord_progression` (array), `style_guess`, `description` (prose)

---

#### `set_musical_context` — Set Musical Context

- **Annotations:** `readOnlyHint=False`, `destructiveHint=False`, `idempotentHint=True`, `openWorldHint=False`
- **All inputs optional** (any combination can be set or cleared):
  - `key`: string — accepts `"C major"`, `"D minor"`, `"F# dorian"`, `"Bb mixolydian"`, etc.
  - `bpm`: integer 40–300
  - `style`: string
  - `chord_progression`: array of strings — Roman numerals or chord names
- **Output schema:** `{ "key": string|null, "bpm": integer|null, "style": string|null, "chord_progression": array|null, "fields_set": array[string] }`
- **Passing `null` to any field clears it** (reverts to prompt-inferred)

**chord_progression format:**
```
["I", "IV", "V", "I"]          Roman numerals, key-relative
["Dm", "Gm", "A7", "Dm"]      chord names
["i", "VI", "III", "VII"]     minor key Roman numerals
```

---

#### `chat_with_midi` — Chat About a MIDI File

- **Annotations:** `readOnlyHint=True`, `destructiveHint=False`, `idempotentHint=False`, `openWorldHint=True`
- **Required inputs:** `file_path`, `question` (string)
- **Output schema:** `answer` (string), `analysis_context` (parsed MIDI facts used)

**Example questions the tool handles well:**
- "What key is this in and how confident are you?"
- "Why does bar 4 feel tense?"
- "Is the bass line supporting the harmony or fighting it?"
- "How could I make this feel more like 90s R&B?"
- "What chord progression is the harmony track playing?"
- "What are the weakest bars and why?"

---

### 8.2 — Prompts

#### `quick-loop` — Quick Loop

**Description:** Generate a polished loop in one step. Sets musical context and generates in a single guided sequence. Best starting point for a new session.

**Arguments:** `style` (required), `key` (optional, default `"infer"`), `bars` (optional, default `8`)

**Message template:**
> I want to generate a {bars}-bar {style} loop{in {key} if given}. First, set the musical context, then generate the loop. Use roles appropriate for the style. Make it feel complete and loopable.

---

#### `compose-section` — Build a Song Section

**Description:** Guided workflow for composing a complete song section. Starts with context setup, generates a seed loop, extends and refines into a full section.

**Arguments:** `section_type` (required: intro/verse/chorus/bridge/outro), `style` (required), `bars` (optional, default `16`)

**Message template:** 5-step guided sequence — set context → generate seed → describe → extend → final describe.

---

#### `analyze-and-improve` — Analyze and Improve

**Description:** Analyze an existing MIDI file and get specific, actionable improvement suggestions. Use when a generated result is close but not quite right.

**Arguments:** `file_path` (required), `goal` (required, e.g. `"make it feel more alive"`)

**Message template:** Uses describe_midi → chat_with_midi to diagnose → suggests whether to regenerate, continue with style_hint, or adjust context.

---

#### `start-session` — Collaborative Composition Session

**Description:** Start an open-ended composition session. The assistant asks clarifying questions about style, mood, instrumentation, and structure, then builds the piece iteratively.

**Arguments:** none

**Message template:** Open-ended: "Please ask me a few questions to understand what I'm going for — style, mood, instrumentation, how long, any reference tracks — then set the musical context and start generating."

---

### 8.3 — Resources

| URI | Title | MIME | Audience | Priority | Description |
|-----|-------|------|----------|----------|-------------|
| `midi://session/context` | Current Musical Context | `application/json` | user+assistant | 0.9 | Active key, BPM, style, chord progression. Read before generating to confirm context is set. |
| `midi://hardware/status` | TT Hardware Status | `application/json` | user+assistant | 0.6 | Connected TT devices, active model, backend. Check if generation is slow or failing. |
| `midi://styles/catalog` | Style Catalog | `application/json` | user+assistant | 0.7 | Available styles with BPM ranges, typical keys, default roles, and example prompts. |
| `midi://output/{filename}` | Generated MIDI File | `audio/midi` | user | 0.5 | Access any previously generated file by filename. Returns raw MIDI bytes. |

---

### 8.4 — Argument Completions

Declared via `CompletionsCapability`. When clients support argument autocomplete, the server responds to `completion/complete` for:

- `style` — filtered list from `config/styles.yaml`
- `key` — filtered list of valid key/mode combinations (72 entries: 12 roots × 6 modes)
- `roles` — filtered list of `["drums","bass","melody","harmony","arp","pad","fx"]`
- `section_type` — filtered list of `["intro","verse","chorus","bridge","outro"]`
- `mode` — filtered list of `["loop","section","stream"]`

---

### 8.5 — Error Convention

All tool errors follow the same shape so clients can handle them programmatically:

```python
class MidiMakerError(Exception):
    code: str        # machine-readable
    message: str     # human-readable
    suggestion: str  # what to call next / how to recover
```

| Code | When raised | Suggestion |
|------|-------------|------------|
| `HARDWARE_UNAVAILABLE` | No TT chips found at startup | "Generation will proceed on CPU. Check midi://hardware/status." |
| `CONTEXT_NOT_SET` | Blueprint inference failed (no key/style inferable) | "Call set_musical_context with explicit key and style." |
| `FILE_NOT_FOUND` | file_path argument doesn't resolve | "Check midi://output/{filename} for available files." |
| `GENERATION_FAILED` | Inference crashed or produced empty output | "Try fewer roles or a simpler prompt. Aria-mini fallback is available." |
| `INVALID_KEY` | key string couldn't be parsed | "Use format 'C major', 'D minor', 'F# dorian'. Request completions on the key argument." |
| `MODEL_COMPILE_FAILED` | Aria failed to compile on TT hardware | "Falling back to aria-mini. Check midi://hardware/status." |

---

## 9. GM Channel Reference

| Ch | Role | Default Program | Note Range | Notes |
|----|------|-----------------|------------|-------|
| 1 | melody | 0 Acoustic Piano | C4–C7 | Primary voice |
| 2 | bass | 32 Acoustic Bass | C1–C3 | Root motion |
| 3 | harmony | 48 String Ensemble | C3–C5 | Chord voicings |
| 4 | arp | 4 Electric Piano | C4–C6 | Counter-melody |
| 5 | pad | 89 Warm Pad | C2–C5 | Texture |
| 9 | fx | 88 Synth Pad | any | Effects |
| **10** | **drums** | **— GM drum map** | **35–81** | **Always ch 10** |

Channels 6–8 and 11–16 are available for future extended ensemble use.

---

## 10. Dependencies

| Package | Purpose |
|---------|---------|
| `mcp>=1.27.0` | MCP server (FastMCP) |
| `mido` | MIDI file read/write |
| `miditok` | REMI+ tokenization for Aria |
| `transformers` | Aria model loading (HuggingFace) |
| `torch` | Model inference + tt-forge compilation |
| `forge` | TT hardware compilation (from tt-forge-fe) |
| `pydantic>=2` | Blueprint + config validation |
| `music21` | Optional: chord parsing, Roman numeral analysis |
| `httpx` | LLM inference server calls (async) |
| `pyyaml` | Config loading |

---

## 11. Open Questions / Decisions for Implementation

1. **Prompt engine LLM endpoint:** which model and endpoint does the TT inference server expose at time of implementation? The prompt engine should make this configurable via env var (`MIDI_LLM_URL`).

2. **Aria compilation verification:** aria-medium has not yet been run through tt-forge. Implementation should attempt compilation on first startup, log results, and fall back gracefully. A `--compile-check` CLI flag would help.

3. **MidiTok version compatibility:** verify the exact REMI+ tokenizer config Aria was trained with (vocab size, special tokens) before implementing tokenizer.py.

4. **Output directory:** `~/Music/tt-midi-maker/` — should be created on first use if absent. The MCP resource template for `midi://output/{filename}` lists only files in this directory.

5. **Session state isolation:** FastMCP's session model should be verified for multi-client isolation — each connection needs its own MusicalContext, not a shared global.
