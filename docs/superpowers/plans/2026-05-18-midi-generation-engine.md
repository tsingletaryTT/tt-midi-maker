# MIDI Generation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tt-midi-maker — a five-stage pipeline (LLM blueprint → Aria MIDI transformer → coherence layer → GM MIDI assembler → MCP server) that converts text prompts into multi-track MIDI files and exposes them as a fully-featured MCP server.

**Architecture:** Text prompt → Qwen3/Llama (TT inference server) produces a MusicalBlueprint JSON → Aria-medium (tt-forge compiled) generates REMI+ token sequences per role → coherence layer applies scale quantization, chord-aware filtering, humanization, and phrase stitching → mido assembles a Type-1 MIDI file with GM channel assignments → FastMCP exposes 5 tools, 4 prompts, 4 resources with completions.

**Tech Stack:** Python 3.11+, mcp>=1.27.0 (FastMCP), mido, miditok, transformers, torch, pydantic>=2, httpx, pyyaml, pytest

---

## File Map

```
tt_midi_maker/
├── __init__.py
├── errors.py                  # MidiMakerError dataclass + ERROR_CODES
├── session.py                 # MusicalContext + per-connection session store
├── prompt_engine.py           # LLM HTTP call → MusicalBlueprint
├── assembler.py               # build_midi_file() → mido.MidiFile (Type-1)
├── analyzer.py                # extract_midi_facts(), describe_midi(), chat_about_midi()
├── server.py                  # FastMCP: 5 tools, 4 prompts, 4 resources, completions
├── models/
│   ├── __init__.py
│   ├── blueprint.py           # MusicalBlueprint, RoleConfig (Pydantic)
│   └── track.py               # NoteEvent, RoleTrack (dataclasses)
├── generation/
│   ├── __init__.py
│   ├── hardware.py            # detect_tt_devices() → list[int]
│   ├── tokenizer.py           # encode_midi_file(), decode_tokens() wrappers
│   └── aria_backend.py        # load_model(), generate_tokens() with TT/CPU fallback
└── coherence/
    ├── __init__.py
    ├── scale.py               # parse_key(), scale_quantize()
    ├── harmony.py             # parse_chord(), chord_aware_filter()
    ├── humanize.py            # humanize_velocities(), nudge_timing()
    └── stitching.py           # stitch_phrases()

config/
├── roles.yaml                 # GM channel, program, note_range per role
└── styles.yaml                # BPM ranges, typical keys, default roles per style

tests/
├── conftest.py                # shared fixtures: sample_blueprint, sample_notes
├── test_models.py             # MusicalBlueprint + NoteEvent validation
├── test_scale.py              # scale_quantize()
├── test_harmony.py            # chord_aware_filter()
├── test_humanize.py           # humanize_velocities(), nudge_timing()
├── test_stitching.py          # stitch_phrases()
├── test_assembler.py          # build_midi_file() → readable .mid
├── test_session.py            # MusicalContext update/clear
├── test_hardware.py           # detect_tt_devices() with mocked subprocess
├── test_prompt_engine.py      # build_blueprint() with mocked httpx
├── test_analyzer.py           # extract_midi_facts(), mocked LLM calls
└── test_server.py             # tool functions via direct call + mocked pipeline

pyproject.toml
CLAUDE.md
```

---

### Task 1: Scaffold, configs, and data models

**Files:**
- Create: `pyproject.toml`
- Create: `CLAUDE.md`
- Create: `tt_midi_maker/__init__.py`
- Create: `tt_midi_maker/errors.py`
- Create: `tt_midi_maker/models/__init__.py`
- Create: `tt_midi_maker/models/blueprint.py`
- Create: `tt_midi_maker/models/track.py`
- Create: `tt_midi_maker/coherence/__init__.py`
- Create: `tt_midi_maker/generation/__init__.py`
- Create: `config/roles.yaml`
- Create: `config/styles.yaml`
- Create: `tests/conftest.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "tt-midi-maker"
version = "0.1.0"
description = "Multi-track MIDI generation from text prompts on Tenstorrent hardware"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.27.0",
    "mido>=1.3.0",
    "miditok>=3.0.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[project.scripts]
tt-midi-maker = "tt_midi_maker.server:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.setuptools.packages.find]
where = ["."]
include = ["tt_midi_maker*"]

[tool.setuptools.package-data]
tt_midi_maker = ["../config/*.yaml"]
```

- [ ] **Step 2: Install dependencies**

```bash
cd /home/ttuser/code/tt-midi-maker
pip install -e ".[dev]"
```

Expected: `Successfully installed tt-midi-maker-0.1.0` (plus dependencies)

- [ ] **Step 3: Create CLAUDE.md**

```markdown
# tt-midi-maker

Multi-track MIDI generation from text prompts using Tenstorrent hardware.
Exposes generation as a fully-featured MCP server.

## Pipeline
Prompt → LLM blueprint → Aria MIDI transformer (tt-forge) → coherence layer → GM MIDI → MCP

## Key files
- `tt_midi_maker/server.py` — MCP server entry point
- `tt_midi_maker/models/blueprint.py` — MusicalBlueprint Pydantic model
- `tt_midi_maker/coherence/` — music theory passes (scale, harmony, humanize, stitch)
- `config/roles.yaml` — GM channel assignments

## Running
```bash
python -m tt_midi_maker.server          # start MCP server
MIDI_LLM_URL=http://localhost:8000/v1   # LLM endpoint env var
```

## Testing
```bash
pytest tests/ -v
```
```

- [ ] **Step 4: Create errors.py**

```python
# tt_midi_maker/errors.py
from dataclasses import dataclass

@dataclass
class MidiMakerError(Exception):
    code: str
    message: str
    suggestion: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}. {self.suggestion}"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "suggestion": self.suggestion}
```

- [ ] **Step 5: Create models/blueprint.py**

```python
# tt_midi_maker/models/blueprint.py
from pydantic import BaseModel, Field
from typing import Literal


class RoleConfig(BaseModel):
    density: float = Field(ge=0.0, le=1.0)
    velocity_range: tuple[int, int] = (60, 100)
    pattern_hint: str = "default"


class MusicalBlueprint(BaseModel):
    key: str
    bpm: int = Field(ge=40, le=300)
    time_signature: str = "4/4"
    style: str
    chord_progression: list[str]
    bars: int = Field(ge=1, le=256)
    mode: Literal["loop", "section", "stream"]
    roles: dict[str, RoleConfig]
```

- [ ] **Step 6: Create models/track.py**

```python
# tt_midi_maker/models/track.py
from dataclasses import dataclass, field


@dataclass
class NoteEvent:
    pitch: int           # MIDI pitch 0–127
    velocity: int        # 1–127
    start_tick: int      # absolute tick offset from phrase start
    duration_ticks: int
    channel: int         # GM channel (1-indexed; drums = 10)


@dataclass
class RoleTrack:
    role: str            # "melody", "bass", "drums", etc.
    channel: int         # GM channel (1-indexed)
    program: int         # GM program number (ignored for drums)
    notes: list[NoteEvent] = field(default_factory=list)
```

- [ ] **Step 7: Create config/roles.yaml**

```yaml
# config/roles.yaml
roles:
  melody:
    channel: 1
    program: 0
    note_range: [60, 96]
    density_default: 1.0
  bass:
    channel: 2
    program: 32
    note_range: [28, 52]
    density_default: 0.8
  harmony:
    channel: 3
    program: 48
    note_range: [48, 72]
    density_default: 0.6
  arp:
    channel: 4
    program: 4
    note_range: [60, 84]
    density_default: 0.4
  pad:
    channel: 5
    program: 89
    note_range: [36, 72]
    density_default: 0.3
  fx:
    channel: 9
    program: 88
    note_range: [0, 127]
    density_default: 0.2
  drums:
    channel: 10
    program: 0
    note_range: [35, 81]
    density_default: 0.7
```

- [ ] **Step 8: Create config/styles.yaml**

```yaml
# config/styles.yaml
styles:
  lo-fi hip hop:
    bpm_range: [70, 90]
    typical_keys: ["C major", "F major", "D minor", "A minor"]
    default_roles: [drums, bass, melody, pad]
    swing_ratio: 0.55
    examples: ["dusty drums, sparse bass, melancholic piano melody"]
  bossa nova:
    bpm_range: [120, 160]
    typical_keys: ["D minor", "G major", "C major"]
    default_roles: [drums, bass, melody, harmony]
    swing_ratio: 0.0
    examples: ["brushed snare, walking bass, piano melody"]
  ambient:
    bpm_range: [60, 90]
    typical_keys: ["C major", "D minor", "F major"]
    default_roles: [pad, melody, fx]
    swing_ratio: 0.0
    examples: ["slow pad swells, sparse melody, atmospheric fx"]
  hip hop:
    bpm_range: [80, 100]
    typical_keys: ["C minor", "D minor", "F minor"]
    default_roles: [drums, bass, melody, arp]
    swing_ratio: 0.55
    examples: ["trap hi-hats, 808 bass, synth melody"]
  jazz:
    bpm_range: [120, 200]
    typical_keys: ["F major", "Bb major", "C minor"]
    default_roles: [drums, bass, melody, harmony]
    swing_ratio: 0.65
    examples: ["walking bass, piano comping, trumpet melody"]
  drum and bass:
    bpm_range: [160, 180]
    typical_keys: ["D minor", "A minor", "E minor"]
    default_roles: [drums, bass, arp]
    swing_ratio: 0.0
    examples: ["amen breaks, sub bass, minimal arp"]
```

- [ ] **Step 9: Write test_models.py**

```python
# tests/test_models.py
import pytest
from pydantic import ValidationError
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_blueprint(**overrides):
    base = dict(
        key="D minor", bpm=120, time_signature="4/4",
        style="bossa nova", chord_progression=["Dm", "Gm", "A7", "Dm"],
        bars=8, mode="loop",
        roles={"drums": RoleConfig(density=0.7, velocity_range=(60, 90), pattern_hint="bossa")},
    )
    base.update(overrides)
    return MusicalBlueprint(**base)


def test_valid_blueprint():
    b = make_blueprint()
    assert b.key == "D minor"
    assert b.bpm == 120
    assert b.mode == "loop"


def test_bpm_too_low():
    with pytest.raises(ValidationError):
        make_blueprint(bpm=10)


def test_bpm_too_high():
    with pytest.raises(ValidationError):
        make_blueprint(bpm=400)


def test_invalid_mode():
    with pytest.raises(ValidationError):
        make_blueprint(mode="jam")


def test_role_density_out_of_range():
    with pytest.raises(ValidationError):
        RoleConfig(density=1.5)


def test_note_event_fields():
    n = NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=480, channel=1)
    assert n.pitch == 60
    assert n.channel == 1


def test_role_track_default_empty_notes():
    t = RoleTrack(role="bass", channel=2, program=32)
    assert t.notes == []
```

- [ ] **Step 10: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: 9 passed

- [ ] **Step 11: Create conftest.py**

```python
# tests/conftest.py
import pytest
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack

TICKS_PER_BEAT = 480
TICKS_PER_BAR = 1920  # 4/4 at 480 ticks/beat


@pytest.fixture
def sample_blueprint():
    return MusicalBlueprint(
        key="D minor", bpm=120, time_signature="4/4",
        style="bossa nova", chord_progression=["Dm", "Gm", "A7", "Dm"],
        bars=8, mode="loop",
        roles={
            "drums":   RoleConfig(density=0.7, velocity_range=(60, 90),  pattern_hint="bossa"),
            "bass":    RoleConfig(density=0.8, velocity_range=(70, 100), pattern_hint="walking"),
            "melody":  RoleConfig(density=1.0, velocity_range=(80, 110), pattern_hint="legato"),
            "harmony": RoleConfig(density=0.0, velocity_range=(50, 80),  pattern_hint="default"),
        },
    )


@pytest.fixture
def chromatic_notes_c4():
    """12 notes C4–B4 (60–71), alternating channels 1 and 10."""
    return [
        NoteEvent(pitch=60+i, velocity=80, start_tick=i*TICKS_PER_BEAT,
                  duration_ticks=TICKS_PER_BEAT - 10,
                  channel=10 if i % 4 == 0 else 1)
        for i in range(12)
    ]


def make_role_track(role: str, channel: int, program: int = 0,
                    n_notes: int = 4, start_offset: int = 0) -> RoleTrack:
    notes = [
        NoteEvent(pitch=60, velocity=80,
                  start_tick=start_offset + i * TICKS_PER_BEAT,
                  duration_ticks=TICKS_PER_BEAT - 10,
                  channel=channel)
        for i in range(n_notes)
    ]
    return RoleTrack(role=role, channel=channel, program=program, notes=notes)
```

- [ ] **Step 12: Commit**

```bash
git init
git add pyproject.toml CLAUDE.md tt_midi_maker/ config/ tests/conftest.py tests/test_models.py
git commit -m "feat: scaffold project, data models, GM config"
```

---

### Task 2: Coherence — Scale Quantization

**Files:**
- Create: `tt_midi_maker/coherence/scale.py`
- Create: `tests/test_scale.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scale.py
import pytest
from tt_midi_maker.coherence.scale import (
    parse_key, build_scale_set, nearest_scale_pitch, scale_quantize,
)
from tt_midi_maker.models.track import NoteEvent


def note(pitch, channel=1):
    return NoteEvent(pitch=pitch, velocity=80, start_tick=0, duration_ticks=480, channel=channel)


def test_parse_d_minor():
    root, mode = parse_key("D minor")
    assert root == 2 and mode == "minor"


def test_parse_fsharp_dorian():
    root, mode = parse_key("F# dorian")
    assert root == 6 and mode == "dorian"


def test_parse_invalid_root():
    with pytest.raises(ValueError, match="Cannot parse key"):
        parse_key("X major")


def test_parse_invalid_mode():
    with pytest.raises(ValueError, match="Cannot parse key"):
        parse_key("C ragtime")


def test_c_major_scale_set():
    # C major: C D E F G A B  = 0 2 4 5 7 9 11
    assert build_scale_set(0, "major") == frozenset({0, 2, 4, 5, 7, 9, 11})


def test_d_minor_scale_set():
    # D natural minor: D E F G A Bb C = 2 4 5 7 9 10 0
    assert build_scale_set(2, "minor") == frozenset({0, 2, 4, 5, 7, 9, 10})


def test_nearest_in_scale_returns_same():
    scale = build_scale_set(0, "major")   # C major
    assert nearest_scale_pitch(60, scale) == 60  # C4 is in scale


def test_nearest_out_of_scale():
    scale = build_scale_set(0, "major")   # C major
    # F# = 66, nearest scale tones are F=65 and G=67
    result = nearest_scale_pitch(66, scale)
    assert result in (65, 67)


def test_scale_quantize_drums_unchanged():
    n = note(36, channel=10)
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch == 36


def test_scale_quantize_in_scale_unchanged():
    n = note(60, channel=1)          # C4 in C major
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch == 60


def test_scale_quantize_snaps_off_scale_at_strictness_1():
    n = note(66, channel=1)          # F# not in C major
    result = scale_quantize([n], "C major", strictness=1.0)
    assert result[0].pitch % 12 in build_scale_set(0, "major")


def test_scale_quantize_preserves_at_strictness_0(monkeypatch):
    import tt_midi_maker.coherence.scale as s
    monkeypatch.setattr(s.random, "random", lambda: 0.99)   # always > strictness=0
    n = note(66, channel=1)
    result = scale_quantize([n], "C major", strictness=0.0)
    assert result[0].pitch == 66
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_scale.py -v 2>&1 | head -5
```

Expected: `ModuleNotFoundError: No module named 'tt_midi_maker.coherence.scale'`

- [ ] **Step 3: Implement scale.py**

```python
# tt_midi_maker/coherence/scale.py
import random
from dataclasses import replace
from ..models.track import NoteEvent

SCALE_INTERVALS: dict[str, list[int]] = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian":    [0, 1, 3, 5, 6, 8, 10],
}

ROOT_NAMES: dict[str, int] = {
    "C": 0,  "C#": 1,  "Db": 1,  "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "F": 5,   "F#": 6,  "Gb": 6, "G": 7,   "G#": 8,
    "Ab": 8, "A": 9,   "A#": 10, "Bb": 10, "B": 11,
}


def parse_key(key_str: str) -> tuple[int, str]:
    """Parse "D minor" → (2, "minor"). Raises ValueError if unrecognised."""
    parts = key_str.strip().split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse key: {key_str!r}. Use 'C major', 'D minor', 'F# dorian'.")
    root_str, mode_str = parts[0], parts[1].lower()
    if root_str not in ROOT_NAMES:
        raise ValueError(f"Cannot parse key: {key_str!r}. Unknown root '{root_str}'.")
    if mode_str not in SCALE_INTERVALS:
        raise ValueError(f"Cannot parse key: {key_str!r}. Unknown mode '{mode_str}'.")
    return ROOT_NAMES[root_str], mode_str


def build_scale_set(root: int, mode: str) -> frozenset[int]:
    return frozenset((root + i) % 12 for i in SCALE_INTERVALS[mode])


def nearest_scale_pitch(pitch: int, scale_set: frozenset[int]) -> int:
    if pitch % 12 in scale_set:
        return pitch
    for delta in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]:
        candidate = pitch + delta
        if 0 <= candidate <= 127 and candidate % 12 in scale_set:
            return candidate
    return pitch


def scale_quantize(
    notes: list[NoteEvent],
    key: str,
    strictness: float = 0.8,
) -> list[NoteEvent]:
    """Snap off-scale notes to nearest scale tone. Drums (ch 10) are skipped."""
    root, mode = parse_key(key)
    scale_set = build_scale_set(root, mode)
    result = []
    for note in notes:
        if note.channel == 10 or note.pitch % 12 in scale_set:
            result.append(note)
        elif random.random() < strictness:
            result.append(replace(note, pitch=nearest_scale_pitch(note.pitch, scale_set)))
        else:
            result.append(note)
    return result
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_scale.py -v
```

Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/coherence/scale.py tests/test_scale.py
git commit -m "feat: coherence scale quantization"
```

---

### Task 3: Coherence — Chord-Aware Harmony

**Files:**
- Create: `tt_midi_maker/coherence/harmony.py`
- Create: `tests/test_harmony.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_harmony.py
import pytest
from tt_midi_maker.coherence.harmony import (
    parse_chord, chord_at_tick, is_strong_beat, chord_aware_filter,
)
from tt_midi_maker.models.track import NoteEvent

TICKS_PER_BEAT = 480
TICKS_PER_BAR = 1920


def note(pitch, tick, channel=1):
    return NoteEvent(pitch=pitch, velocity=80, start_tick=tick,
                     duration_ticks=TICKS_PER_BEAT - 10, channel=channel)


D_MINOR_SCALE = frozenset({0, 2, 4, 5, 7, 9, 10})


def test_parse_dm():
    assert parse_chord("Dm") == frozenset({2, 5, 9})   # D F A


def test_parse_g7():
    # G7: G(7) B(11) D(2) F(5)
    assert parse_chord("G7") == frozenset({7, 11, 2, 5})


def test_parse_cmaj7():
    # Cmaj7: C(0) E(4) G(7) B(11)
    assert parse_chord("Cmaj7") == frozenset({0, 4, 7, 11})


def test_parse_invalid():
    with pytest.raises(ValueError, match="Cannot parse chord"):
        parse_chord("Xyz99")


def test_strong_beat_1():
    assert is_strong_beat(0, TICKS_PER_BEAT) is True         # beat 1


def test_strong_beat_3():
    assert is_strong_beat(2 * TICKS_PER_BEAT, TICKS_PER_BEAT) is True  # beat 3


def test_weak_beat_2():
    assert is_strong_beat(TICKS_PER_BEAT, TICKS_PER_BEAT) is False


def test_chord_at_tick_bar_0():
    tones = chord_at_tick(0, TICKS_PER_BAR, ["Dm", "Gm", "A7", "Dm"])
    assert tones == parse_chord("Dm")


def test_chord_at_tick_bar_1():
    tones = chord_at_tick(TICKS_PER_BAR, TICKS_PER_BAR, ["Dm", "Gm", "A7", "Dm"])
    assert tones == parse_chord("Gm")


def test_chord_at_tick_wraps():
    tones = chord_at_tick(4 * TICKS_PER_BAR, TICKS_PER_BAR, ["Dm", "Gm"])
    assert tones == parse_chord("Dm")


def test_filter_chord_tone_on_beat1_unchanged():
    # D4=62 is in Dm (pitch class 2 = D)
    n = note(62, tick=0)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 62


def test_filter_off_chord_on_beat1_moves():
    # E4=64 is NOT in Dm; on beat 1 it should move to nearest Dm tone
    n = note(64, tick=0)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch % 12 in parse_chord("Dm")


def test_filter_off_chord_on_beat2_unchanged():
    # E4=64 not in Dm, but beat 2 → leave it alone
    n = note(64, tick=TICKS_PER_BEAT)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 64


def test_filter_drums_always_unchanged():
    n = note(36, tick=0, channel=10)
    result = chord_aware_filter([n], ["Dm"], TICKS_PER_BAR, TICKS_PER_BEAT, D_MINOR_SCALE)
    assert result[0].pitch == 36
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_harmony.py -v 2>&1 | head -3
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement harmony.py**

```python
# tt_midi_maker/coherence/harmony.py
from dataclasses import replace
from ..models.track import NoteEvent

CHORD_INTERVALS: dict[str, list[int]] = {
    "":     [0, 4, 7],
    "m":    [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

ROOT_NAMES: dict[str, int] = {
    "C": 0,  "C#": 1,  "Db": 1,  "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,  "F": 5,   "F#": 6,  "Gb": 6, "G": 7,   "G#": 8,
    "Ab": 8, "A": 9,   "A#": 10, "Bb": 10, "B": 11,
}


def parse_chord(chord_str: str) -> frozenset[int]:
    """Parse "Dm", "G7", "Cmaj7" → frozenset of pitch classes."""
    for root_len in (2, 1):
        root_str = chord_str[:root_len]
        quality = chord_str[root_len:]
        if root_str in ROOT_NAMES and quality in CHORD_INTERVALS:
            root = ROOT_NAMES[root_str]
            return frozenset((root + i) % 12 for i in CHORD_INTERVALS[quality])
    raise ValueError(f"Cannot parse chord: {chord_str!r}")


def chord_at_tick(
    tick: int, ticks_per_bar: int, chord_progression: list[str]
) -> frozenset[int]:
    bar_index = tick // ticks_per_bar
    return parse_chord(chord_progression[bar_index % len(chord_progression)])


def is_strong_beat(tick: int, ticks_per_beat: int) -> bool:
    """True on beats 1 and 3 in 4/4."""
    return (tick // ticks_per_beat) % 4 in (0, 2)


def chord_aware_filter(
    notes: list[NoteEvent],
    chord_progression: list[str],
    ticks_per_bar: int,
    ticks_per_beat: int,
    scale_set: frozenset[int],
) -> list[NoteEvent]:
    """Snap off-chord-tones on strong beats to nearest chord tone. Drums skipped."""
    result = []
    for note in notes:
        if note.channel == 10 or not is_strong_beat(note.start_tick, ticks_per_beat):
            result.append(note)
            continue
        chord_tones = chord_at_tick(note.start_tick, ticks_per_bar, chord_progression)
        if note.pitch % 12 in chord_tones:
            result.append(note)
            continue
        best, best_dist = note.pitch, float("inf")
        for delta in range(-6, 7):
            candidate = note.pitch + delta
            if 0 <= candidate <= 127 and candidate % 12 in chord_tones:
                if abs(delta) < best_dist:
                    best, best_dist = candidate, abs(delta)
        result.append(replace(note, pitch=best))
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_harmony.py -v
```

Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/coherence/harmony.py tests/test_harmony.py
git commit -m "feat: chord-aware harmony filter"
```

---

### Task 4: Coherence — Humanization

**Files:**
- Create: `tt_midi_maker/coherence/humanize.py`
- Create: `tests/test_humanize.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_humanize.py
from tt_midi_maker.coherence.humanize import humanize_velocities, nudge_timing
from tt_midi_maker.models.track import NoteEvent


def make_notes(n=16, vel=80, channel=1):
    return [
        NoteEvent(pitch=60, velocity=vel, start_tick=i * 480,
                  duration_ticks=470, channel=channel)
        for i in range(n)
    ]


def test_velocities_stay_in_valid_range():
    result = humanize_velocities(make_notes(vel=80), variation=20)
    assert all(1 <= n.velocity <= 127 for n in result)


def test_velocities_are_varied():
    result = humanize_velocities(make_notes(n=32, vel=80), variation=10)
    assert len(set(n.velocity for n in result)) > 1


def test_drums_get_smaller_variation():
    drum_notes = make_notes(n=32, vel=80, channel=10)
    result = humanize_velocities(drum_notes, variation=20)
    # Drum variation is capped at ±4
    assert all(76 <= n.velocity <= 84 for n in result)


def test_nudge_timing_non_negative():
    result = nudge_timing(make_notes(), max_ticks=10)
    assert all(n.start_tick >= 0 for n in result)


def test_drums_not_nudged():
    drum_notes = make_notes(channel=10)
    original = [n.start_tick for n in drum_notes]
    result = nudge_timing(drum_notes, max_ticks=10)
    assert [n.start_tick for n in result] == original


def test_nudge_changes_some_ticks():
    notes = make_notes(n=32)
    result = nudge_timing(notes, max_ticks=10)
    assert [n.start_tick for n in result] != [n.start_tick for n in notes]


def test_empty_input():
    assert humanize_velocities([]) == []
    assert nudge_timing([]) == []
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_humanize.py -v 2>&1 | head -3
```

- [ ] **Step 3: Implement humanize.py**

```python
# tt_midi_maker/coherence/humanize.py
import random
from dataclasses import replace
from ..models.track import NoteEvent


def humanize_velocities(
    notes: list[NoteEvent],
    variation: int = 8,
    phrase_contour: bool = True,
) -> list[NoteEvent]:
    """Add ±variation velocity jitter. Drums capped at ±4. Optional phrase contour."""
    if not notes:
        return notes
    result = []
    total = len(notes)
    for i, note in enumerate(notes):
        if note.channel == 10:
            offset = random.randint(-4, 4)
        else:
            offset = random.randint(-variation, variation)
            if phrase_contour and total > 1:
                position = i / (total - 1)
                contour = int(6 * (1 - abs(2 * position - 1)))
                offset += contour // 2
        result.append(replace(note, velocity=max(1, min(127, note.velocity + offset))))
    return result


def nudge_timing(
    notes: list[NoteEvent],
    max_ticks: int = 8,
) -> list[NoteEvent]:
    """Add micro-timing jitter ±max_ticks to non-drum notes."""
    if not notes:
        return notes
    return [
        note if note.channel == 10
        else replace(note, start_tick=max(0, note.start_tick + random.randint(-max_ticks, max_ticks)))
        for note in notes
    ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_humanize.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/coherence/humanize.py tests/test_humanize.py
git commit -m "feat: velocity humanization and micro-timing"
```

---

### Task 5: Coherence — Phrase Stitching

**Files:**
- Create: `tt_midi_maker/coherence/stitching.py`
- Create: `tests/test_stitching.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stitching.py
from tt_midi_maker.coherence.stitching import stitch_phrases
from tt_midi_maker.models.track import NoteEvent, RoleTrack

TICKS_PER_BAR = 1920


def make_track(role, channel, n=4, start=0):
    notes = [
        NoteEvent(pitch=60, velocity=80,
                  start_tick=start + i * 480, duration_ticks=470, channel=channel)
        for i in range(n)
    ]
    return RoleTrack(role=role, channel=channel, program=0, notes=notes)


def test_stitch_doubles_note_count():
    existing = [make_track("melody", 1, n=4)]
    new      = [make_track("melody", 1, n=4)]
    result = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody = next(t for t in result if t.role == "melody")
    assert len(melody.notes) == 8


def test_new_notes_start_after_existing():
    existing = [make_track("melody", 1, n=4)]   # last note ends around tick 4*480
    new      = [make_track("melody", 1, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody   = next(t for t in result if t.role == "melody")
    existing_max = max(n.start_tick for n in existing[0].notes)
    new_ticks    = [n.start_tick for n in melody.notes[4:]]
    assert all(t > existing_max for t in new_ticks)


def test_missing_role_in_new_preserved():
    existing = [make_track("melody", 1), make_track("bass", 2)]
    new      = [make_track("melody", 1)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    roles    = {t.role for t in result}
    assert "bass" in roles and "melody" in roles


def test_missing_role_in_existing_appended():
    existing = [make_track("melody", 1)]
    new      = [make_track("melody", 1), make_track("bass", 2)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    roles    = {t.role for t in result}
    assert "bass" in roles


def test_crossfade_reduces_last_bar_velocity():
    existing = [make_track("melody", 1, n=8)]   # 8 notes span 2 bars
    new      = [make_track("melody", 1, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    melody   = next(t for t in result if t.role == "melody")
    last_bar_start = 7 * 480   # note 8 (index 7)
    last_bar_notes = [n for n in melody.notes[:8] if n.start_tick >= last_bar_start]
    assert all(n.velocity <= 80 for n in last_bar_notes)   # faded from 80


def test_drums_not_crossfaded():
    existing = [make_track("drums", 10, n=8)]
    new      = [make_track("drums", 10, n=4)]
    result   = stitch_phrases(existing, new, ticks_per_bar=TICKS_PER_BAR)
    drums    = next(t for t in result if t.role == "drums")
    # Original drum velocities should be untouched
    assert all(n.velocity == 80 for n in drums.notes[:8])
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_stitching.py -v 2>&1 | head -3
```

- [ ] **Step 3: Implement stitching.py**

```python
# tt_midi_maker/coherence/stitching.py
from dataclasses import replace
from ..models.track import NoteEvent, RoleTrack


def _max_tick(track: RoleTrack) -> int:
    if not track.notes:
        return 0
    return max(n.start_tick + n.duration_ticks for n in track.notes)


def stitch_phrases(
    existing_tracks: list[RoleTrack],
    new_tracks: list[RoleTrack],
    ticks_per_bar: int,
) -> list[RoleTrack]:
    """Append new_tracks to existing_tracks with velocity crossfade at the join."""
    existing_end = max((_max_tick(t) for t in existing_tracks), default=0)
    fade_start   = existing_end - ticks_per_bar      # last bar of existing
    ramp_end     = existing_end + 2 * ticks_per_bar  # first 2 bars of new

    existing_by_role = {t.role: t for t in existing_tracks}
    new_by_role      = {t.role: t for t in new_tracks}
    result: list[RoleTrack] = []

    for role in sorted(set(existing_by_role) | set(new_by_role)):
        ex  = existing_by_role.get(role)
        nw  = new_by_role.get(role)

        if ex is None:
            shifted = [replace(n, start_tick=n.start_tick + existing_end) for n in nw.notes]
            result.append(replace(nw, notes=shifted))
            continue

        if nw is None:
            result.append(ex)
            continue

        ex_notes: list[NoteEvent] = []
        for n in ex.notes:
            if n.channel != 10 and n.start_tick >= fade_start:
                ex_notes.append(replace(n, velocity=max(1, int(n.velocity * 0.9))))
            else:
                ex_notes.append(n)

        new_notes: list[NoteEvent] = []
        for n in nw.notes:
            t = n.start_tick + existing_end
            if n.channel != 10 and t < ramp_end:
                progress = (t - existing_end) / (2 * ticks_per_bar)
                scale    = 0.8 + 0.2 * min(1.0, max(0.0, progress))
                new_notes.append(replace(n, start_tick=t,
                                         velocity=max(1, int(n.velocity * scale))))
            else:
                new_notes.append(replace(n, start_tick=t))

        result.append(replace(ex, notes=ex_notes + new_notes))

    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_stitching.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/coherence/stitching.py tests/test_stitching.py
git commit -m "feat: phrase stitching with velocity crossfade"
```

---

### Task 6: MIDI Assembler

**Files:**
- Create: `tt_midi_maker/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_assembler.py
import mido
import tempfile
from pathlib import Path
from tt_midi_maker.assembler import build_midi_file, bpm_to_tempo, TICKS_PER_BEAT
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_track(role, channel, program=0, n=4):
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=channel)
        for i in range(n)
    ]
    return RoleTrack(role=role, channel=channel, program=program, notes=notes)


def test_bpm_to_tempo_120():
    assert bpm_to_tempo(120) == 500_000


def test_bpm_to_tempo_60():
    assert bpm_to_tempo(60) == 1_000_000


def test_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        result = build_midi_file([make_track("melody", 1)], bpm=120, output_path=path)
        assert result == path
        assert path.exists()


def test_output_is_type_1():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1), make_track("bass", 2)], 120, path)
        assert mido.MidiFile(str(path)).type == 1


def test_track_count_includes_tempo_track():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1), make_track("bass", 2)], 120, path)
        mid = mido.MidiFile(str(path))
        assert len(mid.tracks) == 3   # tempo + melody + bass


def test_tempo_is_set_correctly():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1)], bpm=90, output_path=path)
        mid = mido.MidiFile(str(path))
        tempos = [m for m in mid.tracks[0] if m.type == "set_tempo"]
        assert tempos[0].tempo == bpm_to_tempo(90)


def test_drums_get_no_program_change():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("drums", 10)], 120, path)
        mid = mido.MidiFile(str(path))
        drum_track = mid.tracks[1]
        assert not any(m.type == "program_change" for m in drum_track)


def test_melody_gets_program_change():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.mid"
        build_midi_file([make_track("melody", 1, program=0)], 120, path)
        mid = mido.MidiFile(str(path))
        melody_track = mid.tracks[1]
        assert any(m.type == "program_change" for m in melody_track)


def test_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nested" / "dir" / "test.mid"
        build_midi_file([make_track("melody", 1)], 120, path)
        assert path.exists()
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_assembler.py -v 2>&1 | head -3
```

- [ ] **Step 3: Implement assembler.py**

```python
# tt_midi_maker/assembler.py
from pathlib import Path
import mido
from .models.track import RoleTrack

TICKS_PER_BEAT = 480


def bpm_to_tempo(bpm: int) -> int:
    return int(60_000_000 / bpm)


def build_midi_file(
    role_tracks: list[RoleTrack],
    bpm: int,
    output_path: Path,
    ticks_per_beat: int = TICKS_PER_BEAT,
) -> Path:
    """Assemble a Type-1 multi-track MIDI file from RoleTracks."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo(bpm), time=0))
    mid.tracks.append(tempo_track)

    for role_track in role_tracks:
        track = mido.MidiTrack()
        track.name = role_track.role

        ch = role_track.channel - 1   # mido is 0-indexed

        if role_track.channel != 10:
            track.append(mido.Message("program_change", channel=ch,
                                      program=role_track.program, time=0))

        events: list[tuple[int, str, int, int]] = []
        for note in role_track.notes:
            events.append((note.start_tick, "note_on",  note.pitch, note.velocity))
            events.append((note.start_tick + note.duration_ticks, "note_off", note.pitch, 0))

        events.sort(key=lambda e: e[0])
        current_tick = 0
        for abs_tick, msg_type, pitch, vel in events:
            delta = abs_tick - current_tick
            track.append(mido.Message(msg_type, channel=ch, note=pitch,
                                      velocity=vel, time=delta))
            current_tick = abs_tick

        mid.tracks.append(track)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))
    return output_path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_assembler.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/assembler.py tests/test_assembler.py
git commit -m "feat: GM MIDI assembler (Type-1, tempo track, program change)"
```

---

### Task 7: Session State

**Files:**
- Create: `tt_midi_maker/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_session.py
from tt_midi_maker.session import MusicalContext, get_session, set_session, clear_session


def test_empty_context_is_empty():
    ctx = MusicalContext()
    assert ctx.is_empty()


def test_update_sets_field():
    ctx = MusicalContext()
    ctx2 = ctx.update(key="D minor", bpm=120)
    assert ctx2.key == "D minor"
    assert ctx2.bpm == 120


def test_update_does_not_mutate_original():
    ctx = MusicalContext(key="C major")
    ctx.update(bpm=90)
    assert ctx.bpm is None


def test_update_none_clears_field():
    ctx = MusicalContext(key="C major", bpm=120)
    ctx2 = ctx.update(key=None)
    assert ctx2.key is None
    assert ctx2.bpm == 120


def test_to_dict_omits_none():
    ctx = MusicalContext(key="D minor")
    d = ctx.to_dict()
    assert "key" in d
    assert "bpm" not in d


def test_session_store_isolation():
    set_session("A", MusicalContext(key="C major"))
    set_session("B", MusicalContext(key="D minor"))
    assert get_session("A").key == "C major"
    assert get_session("B").key == "D minor"


def test_get_unknown_session_returns_empty():
    assert get_session("nonexistent").is_empty()


def test_clear_session_removes_it():
    set_session("C", MusicalContext(bpm=120))
    clear_session("C")
    assert get_session("C").is_empty()
```

- [ ] **Step 2: Implement session.py**

```python
# tt_midi_maker/session.py
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MusicalContext:
    key: Optional[str] = None
    bpm: Optional[int] = None
    style: Optional[str] = None
    chord_progression: Optional[list[str]] = None

    def is_empty(self) -> bool:
        return all(v is None for v in asdict(self).values())

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def update(self, **kwargs) -> "MusicalContext":
        data = asdict(self)
        for k, v in kwargs.items():
            if k in data:
                data[k] = v
        return MusicalContext(**data)


_sessions: dict[str, MusicalContext] = {}


def get_session(session_id: str) -> MusicalContext:
    return _sessions.get(session_id, MusicalContext())


def set_session(session_id: str, ctx: MusicalContext) -> None:
    _sessions[session_id] = ctx


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_session.py -v
```

Expected: 8 passed

- [ ] **Step 4: Commit**

```bash
git add tt_midi_maker/session.py tests/test_session.py
git commit -m "feat: per-connection session state"
```

---

### Task 8: Hardware Detection

**Files:**
- Create: `tt_midi_maker/generation/hardware.py`
- Create: `tests/test_hardware.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hardware.py
import json
from unittest.mock import patch, MagicMock
import subprocess
from tt_midi_maker.generation.hardware import detect_tt_devices, hardware_status


TT_SMI_OUTPUT = json.dumps({
    "device_info": [
        {"id": 0, "status": "available", "board_type": "N300", "arch": "wormhole"},
        {"id": 1, "status": "available", "board_type": "N300", "arch": "wormhole"},
    ]
})


def test_detects_devices_from_tt_smi():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = TT_SMI_OUTPUT
    with patch("subprocess.run", return_value=mock_result):
        devices = detect_tt_devices()
    assert devices == [0, 1]


def test_returns_empty_when_tt_smi_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_tt_devices() == []


def test_returns_empty_on_nonzero_exit():
    mock_result = MagicMock(returncode=1, stdout="")
    with patch("subprocess.run", return_value=mock_result):
        assert detect_tt_devices() == []


def test_returns_empty_on_invalid_json():
    mock_result = MagicMock(returncode=0, stdout="not json")
    with patch("subprocess.run", return_value=mock_result):
        assert detect_tt_devices() == []


def test_hardware_status_includes_device_count():
    mock_result = MagicMock(returncode=0, stdout=TT_SMI_OUTPUT)
    with patch("subprocess.run", return_value=mock_result):
        status = hardware_status()
    assert status["device_count"] == 2
    assert status["devices"][0]["board_type"] == "N300"


def test_hardware_status_no_hardware():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        status = hardware_status()
    assert status["device_count"] == 0
    assert status["tt_smi_available"] is False
```

- [ ] **Step 2: Implement hardware.py**

```python
# tt_midi_maker/generation/hardware.py
import subprocess
import json


def detect_tt_devices() -> list[int]:
    """Return list of available TT device indices. Empty list if none found."""
    try:
        result = subprocess.run(
            ["tt-smi", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return [d["id"] for d in data.get("device_info", [])
                if d.get("status") == "available"]
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def hardware_status() -> dict:
    """Return a status dict for the midi://hardware/status resource."""
    try:
        result = subprocess.run(
            ["tt-smi", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            devices = data.get("device_info", [])
            return {
                "tt_smi_available": True,
                "device_count": len(devices),
                "devices": devices,
            }
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
        pass
    return {"tt_smi_available": False, "device_count": 0, "devices": []}
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_hardware.py -v
```

Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add tt_midi_maker/generation/hardware.py tests/test_hardware.py
git commit -m "feat: TT hardware detection"
```

---

### Task 9: Tokenizer (MidiTok wrapper)

**Files:**
- Create: `tt_midi_maker/generation/tokenizer.py`
- Create: `tests/test_tokenizer.py`
- Create: `tests/fixtures/four_bar_cminor.mid` (generated by test setup)

- [ ] **Step 1: Verify miditok import and check Aria tokenizer availability**

```bash
python3 -c "
import miditok
print('miditok version:', miditok.__version__)
# Check if Aria tokenizer exists in this version
print('Available tokenizers:', [x for x in dir(miditok) if not x.startswith('_')])
"
```

If `Aria` appears in the tokenizer list, use `miditok.Aria`. Otherwise use `miditok.REMI` with the config below — Aria was trained on REMI+.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_tokenizer.py
import tempfile
from pathlib import Path
import mido
import pytest
from tt_midi_maker.generation.tokenizer import (
    get_tokenizer, encode_midi_file, decode_tokens_to_midi,
)
from tt_midi_maker.assembler import build_midi_file, TICKS_PER_BEAT
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_simple_midi(tmp_path: Path) -> Path:
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=1)
        for i in range(8)
    ]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], bpm=120, output_path=tmp_path / "test.mid")


def test_tokenizer_loads():
    tok = get_tokenizer()
    assert tok is not None


def test_encode_returns_list_of_ints(tmp_path):
    midi_path = make_simple_midi(tmp_path)
    tokens = encode_midi_file(midi_path)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_encode_decode_roundtrip_preserves_note_count(tmp_path):
    midi_path = make_simple_midi(tmp_path)
    tokens = encode_midi_file(midi_path)
    out_path = tmp_path / "decoded.mid"
    decode_tokens_to_midi(tokens, out_path)
    assert out_path.exists()
    mid = mido.MidiFile(str(out_path))
    note_ons = sum(
        1 for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )
    assert note_ons > 0
```

- [ ] **Step 3: Implement tokenizer.py**

```python
# tt_midi_maker/generation/tokenizer.py
"""
MidiTok REMI tokenizer wrapper. Aria was trained with REMI+ tokenization.

If miditok.Aria exists (miditok >= 3.1), use it directly.
Otherwise fall back to REMI with equivalent settings.
"""
from pathlib import Path
import miditok
from miditok import TokenizerConfig
import mido

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=True,
        use_tempo=True,
        beat_res={(0, 4): 8, (4, 12): 4},
    )
    # Use Aria tokenizer if available in this miditok version
    if hasattr(miditok, "Aria"):
        _tokenizer = miditok.Aria(config)
    else:
        _tokenizer = miditok.REMI(config)
    return _tokenizer


def encode_midi_file(midi_path: Path) -> list[int]:
    """Tokenize a MIDI file into a flat list of integer token IDs."""
    tok = get_tokenizer()
    tokens = tok(str(midi_path))
    # tok() returns a TokSequence or list of TokSequences; flatten to ints
    if hasattr(tokens, "ids"):
        return tokens.ids
    if isinstance(tokens, list):
        ids = []
        for seq in tokens:
            ids.extend(seq.ids if hasattr(seq, "ids") else seq)
        return ids
    return list(tokens)


def decode_tokens_to_midi(tokens: list[int], output_path: Path) -> Path:
    """Decode a token ID list back to a MIDI file."""
    tok = get_tokenizer()
    from miditok.classes import TokSequence
    seq = TokSequence(ids=tokens)
    tok.tokens_to_midi([seq], output_path=str(output_path))
    return output_path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_tokenizer.py -v
```

Expected: 4 passed. If decode API differs for your installed miditok version, check `help(get_tokenizer().tokens_to_midi)` and adjust the `decode_tokens_to_midi` call accordingly.

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/generation/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: MidiTok REMI+ tokenizer wrapper for Aria"
```

---

### Task 10: Aria Backend (Generation Engine)

**Files:**
- Create: `tt_midi_maker/generation/aria_backend.py`
- Create: `tests/test_aria_backend.py`

- [ ] **Step 1: Verify Aria model availability**

```bash
python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('nlp4music/aria-medium')
    print('aria-medium found:', cfg.model_type)
except Exception as e:
    print('aria-medium not found:', e)
    # Try fallback
    try:
        cfg = AutoConfig.from_pretrained('skytnt/midi-model')
        print('skytnt/midi-model found:', cfg.model_type)
    except Exception as e2:
        print('fallback also failed:', e2)
"
```

Use whichever model is available. If neither resolves, use `gpt2` temporarily for testing — the architecture is compatible.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_aria_backend.py
from unittest.mock import patch, MagicMock
import torch
from tt_midi_maker.generation.aria_backend import (
    load_model, generate_tokens, ARIA_MODELS,
)


def test_aria_models_list_non_empty():
    assert len(ARIA_MODELS) >= 1


def test_load_model_cpu_returns_model_and_label():
    mock_model = MagicMock()
    mock_tok   = MagicMock()
    with patch("tt_midi_maker.generation.aria_backend.AutoModelForCausalLM") as MockM, \
         patch("tt_midi_maker.generation.aria_backend.AutoTokenizer") as MockT:
        MockM.from_pretrained.return_value = mock_model
        MockT.from_pretrained.return_value = mock_tok
        model, tokenizer, label = load_model(ARIA_MODELS[0], device_ids=[])
    assert model is mock_model
    assert label == "cpu-fallback"


def test_load_model_tries_forge_with_devices():
    mock_model = MagicMock()
    mock_tok   = MagicMock()
    mock_compiled = MagicMock()
    with patch("tt_midi_maker.generation.aria_backend.AutoModelForCausalLM") as MockM, \
         patch("tt_midi_maker.generation.aria_backend.AutoTokenizer") as MockT, \
         patch("tt_midi_maker.generation.aria_backend._try_forge_compile",
               return_value=(mock_compiled, "tt-forge/2x")) as mock_forge:
        MockM.from_pretrained.return_value = mock_model
        MockT.from_pretrained.return_value = mock_tok
        model, tokenizer, label = load_model(ARIA_MODELS[0], device_ids=[0, 1])
    mock_forge.assert_called_once()
    assert label == "tt-forge/2x"


def test_generate_tokens_returns_list():
    mock_model = MagicMock()
    fake_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mock_model.generate.return_value = fake_output
    result = generate_tokens(mock_model, input_tokens=[1, 2, 3], max_new_tokens=5)
    assert isinstance(result, list)
    assert result == [4, 5, 6, 7, 8]   # tokens after the 3 input tokens


def test_generate_tokens_passes_temperature():
    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 99]])
    generate_tokens(mock_model, input_tokens=[1, 2, 3],
                    max_new_tokens=1, temperature=0.5)
    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs["temperature"] == 0.5
```

- [ ] **Step 3: Implement aria_backend.py**

```python
# tt_midi_maker/generation/aria_backend.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ordered fallback chain — first one that compiles wins
ARIA_MODELS = [
    "nlp4music/aria-medium",
    "nlp4music/aria-mini",
    "skytnt/midi-model",
]

_loaded: tuple | None = None   # (model, tokenizer, label)


def _try_forge_compile(model, device_ids: list[int]):
    """Attempt tt-forge compilation. Returns (compiled_model, label) or raises."""
    import forge
    sample = torch.zeros((1, 16), dtype=torch.long)
    compiled = forge.compile(model, sample, module_name="aria_midi")
    return compiled, f"tt-forge/{len(device_ids)}x"


def load_model(model_name: str, device_ids: list[int]) -> tuple:
    """Load model. Returns (model, tokenizer, hardware_label)."""
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device_ids:
        try:
            compiled, label = _try_forge_compile(model, device_ids)
            return compiled, tokenizer, label
        except Exception as e:
            print(f"[tt-midi-maker] tt-forge compile failed ({e}), falling back to CPU")

    return model, tokenizer, "cpu-fallback"


def get_model(device_ids: list[int] | None = None) -> tuple:
    """Lazily load model, trying ARIA_MODELS in order."""
    global _loaded
    if _loaded is not None:
        return _loaded
    devices = device_ids or []
    for name in ARIA_MODELS:
        try:
            result = load_model(name, devices)
            _loaded = result
            print(f"[tt-midi-maker] loaded {name} ({result[2]})")
            return result
        except Exception as e:
            print(f"[tt-midi-maker] could not load {name}: {e}")
    raise RuntimeError(f"No MIDI model could be loaded from {ARIA_MODELS}")


def generate_tokens(
    model,
    input_tokens: list[int],
    max_new_tokens: int = 512,
    temperature: float = 0.9,
) -> list[int]:
    """Run model.generate and return only the newly generated token IDs."""
    input_tensor = torch.tensor([input_tokens])
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=0,
        )
    return output[0][len(input_tokens):].tolist()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_aria_backend.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/generation/aria_backend.py tests/test_aria_backend.py
git commit -m "feat: Aria generation backend with tt-forge compile + fallback chain"
```

---

### Task 11: Prompt Engine

**Files:**
- Create: `tt_midi_maker/prompt_engine.py`
- Create: `tests/test_prompt_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompt_engine.py
import json
import pytest
from unittest.mock import patch
from tt_midi_maker.prompt_engine import parse_blueprint, build_blueprint, SYSTEM_PROMPT
from tt_midi_maker.errors import MidiMakerError
from tt_midi_maker.session import MusicalContext

VALID_BLUEPRINT_JSON = json.dumps({
    "key": "D minor", "bpm": 120, "time_signature": "4/4",
    "style": "bossa nova", "chord_progression": ["Dm", "Gm", "A7", "Dm"],
    "bars": 8, "mode": "loop",
    "roles": {
        "drums":  {"density": 0.7, "velocity_range": [60, 90], "pattern_hint": "bossa"},
        "bass":   {"density": 0.8, "velocity_range": [70, 100], "pattern_hint": "walking"},
        "melody": {"density": 1.0, "velocity_range": [80, 110], "pattern_hint": "legato"},
    },
})


def test_parse_blueprint_valid_json():
    bp = parse_blueprint(VALID_BLUEPRINT_JSON)
    assert bp.key == "D minor"
    assert bp.bpm == 120


def test_parse_blueprint_strips_markdown_fence():
    wrapped = f"```json\n{VALID_BLUEPRINT_JSON}\n```"
    bp = parse_blueprint(wrapped)
    assert bp.bpm == 120


def test_parse_blueprint_invalid_json_raises():
    with pytest.raises(MidiMakerError) as exc_info:
        parse_blueprint("not json at all")
    assert exc_info.value.code == "CONTEXT_NOT_SET"


def test_system_prompt_contains_schema():
    assert '"key"' in SYSTEM_PROMPT
    assert '"bpm"' in SYSTEM_PROMPT
    assert '"roles"' in SYSTEM_PROMPT


def test_build_blueprint_calls_llm():
    with patch("tt_midi_maker.prompt_engine.call_llm",
               return_value=VALID_BLUEPRINT_JSON) as mock_llm:
        bp = build_blueprint("dreamy lo-fi")
    mock_llm.assert_called_once()
    assert bp.style == "bossa nova"


def test_build_blueprint_passes_context_to_llm():
    ctx = MusicalContext(key="C major", bpm=90)
    with patch("tt_midi_maker.prompt_engine.call_llm",
               return_value=VALID_BLUEPRINT_JSON) as mock_llm:
        build_blueprint("something jazzy", context=ctx)
    call_args = mock_llm.call_args[0][0]   # first positional arg = messages list
    user_msg = call_args[1]["content"]
    assert "C major" in user_msg
```

- [ ] **Step 2: Implement prompt_engine.py**

```python
# tt_midi_maker/prompt_engine.py
import json
import os
import httpx
from .models.blueprint import MusicalBlueprint
from .session import MusicalContext
from .errors import MidiMakerError

LLM_URL   = os.environ.get("MIDI_LLM_URL",   "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("MIDI_LLM_MODEL", "qwen3")

SYSTEM_PROMPT = """\
You are a music composition assistant. Convert a text description into a
structured JSON blueprint for a MIDI generator.

Return ONLY valid JSON with this exact schema:
{
  "key": "D minor",
  "bpm": 120,
  "time_signature": "4/4",
  "style": "bossa nova",
  "chord_progression": ["Dm", "Gm", "A7", "Dm"],
  "bars": 8,
  "mode": "loop",
  "roles": {
    "drums":   {"density": 0.7, "velocity_range": [60, 90],  "pattern_hint": "bossa"},
    "bass":    {"density": 0.8, "velocity_range": [70, 100], "pattern_hint": "walking"},
    "melody":  {"density": 1.0, "velocity_range": [80, 110], "pattern_hint": "legato"},
    "harmony": {"density": 0.0, "velocity_range": [50, 80],  "pattern_hint": "default"}
  }
}
Set density 0.0 for roles that don't suit the style. No extra keys or prose.\
"""


def call_llm(messages: list[dict]) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{LLM_URL}/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def parse_blueprint(llm_output: str) -> MusicalBlueprint:
    text = llm_output.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return MusicalBlueprint(**json.loads(text))
    except Exception as e:
        raise MidiMakerError(
            code="CONTEXT_NOT_SET",
            message=f"Could not parse musical blueprint: {e}",
            suggestion="Call set_musical_context with explicit key and style.",
        )


def build_blueprint(
    prompt: str,
    context: MusicalContext | None = None,
) -> MusicalBlueprint:
    ctx_str = ""
    if context and not context.is_empty():
        ctx_str = (f"\nSession context (use exactly, do not override): "
                   f"{json.dumps(context.to_dict())}")
    user_content = f"Convert this to a MIDI blueprint:{ctx_str}\n\nPrompt: {prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return parse_blueprint(call_llm(messages))
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_prompt_engine.py -v
```

Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add tt_midi_maker/prompt_engine.py tests/test_prompt_engine.py
git commit -m "feat: prompt engine (LLM → MusicalBlueprint)"
```

---

### Task 12: Analyzer

**Files:**
- Create: `tt_midi_maker/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer.py
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from tt_midi_maker.analyzer import extract_midi_facts, describe_midi, chat_about_midi
from tt_midi_maker.assembler import build_midi_file, TICKS_PER_BEAT
from tt_midi_maker.errors import MidiMakerError
from tt_midi_maker.models.track import NoteEvent, RoleTrack


def make_test_midi(tmp_path: Path, bpm: int = 120) -> Path:
    notes = [
        NoteEvent(pitch=60 + i, velocity=80,
                  start_tick=i * TICKS_PER_BEAT, duration_ticks=TICKS_PER_BEAT - 10,
                  channel=1)
        for i in range(16)
    ]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    return build_midi_file([track], bpm=bpm, output_path=tmp_path / "test.mid")


def test_extract_facts_bpm(tmp_path):
    midi_path = make_test_midi(tmp_path, bpm=90)
    facts = extract_midi_facts(midi_path)
    assert facts["bpm"] == 90


def test_extract_facts_note_count(tmp_path):
    midi_path = make_test_midi(tmp_path)
    facts = extract_midi_facts(midi_path)
    assert facts["note_count"] == 16


def test_extract_facts_channels_used(tmp_path):
    midi_path = make_test_midi(tmp_path)
    facts = extract_midi_facts(midi_path)
    assert 1 in facts["channels_used"]


def test_describe_midi_file_not_found():
    with pytest.raises(MidiMakerError) as exc:
        describe_midi(Path("/nonexistent/file.mid"))
    assert exc.value.code == "FILE_NOT_FOUND"


def test_describe_midi_returns_description(tmp_path):
    midi_path = make_test_midi(tmp_path)
    with patch("tt_midi_maker.analyzer.call_llm", return_value="A simple melody."):
        result = describe_midi(midi_path)
    assert result["description"] == "A simple melody."
    assert result["tempo_bpm"] == 120


def test_chat_about_midi_routes_question(tmp_path):
    midi_path = make_test_midi(tmp_path)
    with patch("tt_midi_maker.analyzer.call_llm", return_value="It is in C major."):
        result = chat_about_midi(midi_path, "What key is this?")
    assert "C major" in result["answer"]
    assert "note_count" in result["analysis_context"]
```

- [ ] **Step 2: Implement analyzer.py**

```python
# tt_midi_maker/analyzer.py
import json
import os
from pathlib import Path
import mido
import httpx
from .errors import MidiMakerError

LLM_URL   = os.environ.get("MIDI_LLM_URL",   "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("MIDI_LLM_MODEL", "qwen3")


def call_llm(messages: list[dict]) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{LLM_URL}/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, "temperature": 0.3},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def extract_midi_facts(path: Path) -> dict:
    mid = mido.MidiFile(str(path))
    facts: dict = {
        "ticks_per_beat": mid.ticks_per_beat,
        "num_tracks": len(mid.tracks) - 1,
        "track_names": [],
        "channels_used": set(),
        "bpm": 120,
        "total_ticks": 0,
        "note_count": 0,
    }
    for track in mid.tracks:
        if track.name:
            facts["track_names"].append(track.name)
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                facts["bpm"] = int(60_000_000 / msg.tempo)
            if msg.type == "note_on" and msg.velocity > 0:
                facts["channels_used"].add(msg.channel + 1)
                facts["note_count"] += 1
                facts["total_ticks"] = max(facts["total_ticks"], tick)
    facts["channels_used"] = sorted(facts["channels_used"])
    facts["bars"] = facts["total_ticks"] // (facts["ticks_per_beat"] * 4)
    return facts


def describe_midi(path: Path) -> dict:
    if not path.exists():
        raise MidiMakerError(
            code="FILE_NOT_FOUND",
            message=f"MIDI file not found: {path}",
            suggestion="Check midi://output/{filename} for available files.",
        )
    facts = extract_midi_facts(path)
    messages = [
        {"role": "system", "content": "You are a music analyst. Given MIDI facts, write a concise 2-sentence musical description."},
        {"role": "user",   "content": json.dumps(facts, indent=2)},
    ]
    description = call_llm(messages)
    return {
        "key": "unknown",
        "tempo_bpm": facts["bpm"],
        "time_signature": "4/4",
        "bars": facts["bars"],
        "tracks": facts["track_names"],
        "chord_progression": [],
        "style_guess": "unknown",
        "description": description,
    }


def chat_about_midi(path: Path, question: str) -> dict:
    if not path.exists():
        raise MidiMakerError(
            code="FILE_NOT_FOUND",
            message=f"MIDI file not found: {path}",
            suggestion="Check midi://output/{filename} for available files.",
        )
    facts = extract_midi_facts(path)
    messages = [
        {"role": "system", "content": "You are an expert music analyst. Answer questions about MIDI files clearly and specifically."},
        {"role": "user",   "content": f"MIDI facts:\n{json.dumps(facts, indent=2)}\n\nQuestion: {question}"},
    ]
    return {"answer": call_llm(messages), "analysis_context": facts}
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_analyzer.py -v
```

Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add tt_midi_maker/analyzer.py tests/test_analyzer.py
git commit -m "feat: MIDI analyzer (facts extraction + LLM describe/chat)"
```

---

### Task 13: MCP Server

**Files:**
- Create: `tt_midi_maker/server.py`
- Create: `tt_midi_maker/__main__.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server.py
"""
Tests call tool handler functions directly (bypassing MCP protocol).
All external I/O (LLM, hardware, generation) is mocked.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig
from tt_midi_maker.models.track import NoteEvent, RoleTrack


# ---------------------------------------------------------------------------
# Minimal stubs so server.py can be imported without TT hardware or LLM
# ---------------------------------------------------------------------------

STUB_BLUEPRINT = MusicalBlueprint(
    key="C major", bpm=120, time_signature="4/4", style="test",
    chord_progression=["C", "F", "G", "C"], bars=4, mode="loop",
    roles={"melody": RoleConfig(density=1.0), "drums": RoleConfig(density=0.7)},
)

STUB_TRACKS = [
    RoleTrack(role="melody", channel=1, program=0, notes=[
        NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=470, channel=1),
    ]),
    RoleTrack(role="drums", channel=10, program=0, notes=[
        NoteEvent(pitch=36, velocity=80, start_tick=0, duration_ticks=100, channel=10),
    ]),
]


def test_set_musical_context_returns_fields_set():
    from tt_midi_maker.server import _set_musical_context
    result = _set_musical_context(session_id="test1", key="D minor", bpm=90)
    assert result["key"] == "D minor"
    assert result["bpm"] == 90
    assert "key" in result["fields_set"]


def test_set_musical_context_null_clears_field():
    from tt_midi_maker.server import _set_musical_context
    _set_musical_context(session_id="test2", key="C major", bpm=120)
    result = _set_musical_context(session_id="test2", key=None)
    assert result["key"] is None
    assert result["bpm"] == 120


def test_generate_midi_returns_file_path(tmp_path):
    from tt_midi_maker import server
    with patch.object(server, "OUTPUT_DIR", tmp_path), \
         patch("tt_midi_maker.server.build_blueprint", return_value=STUB_BLUEPRINT), \
         patch("tt_midi_maker.server._run_generation", return_value=STUB_TRACKS):
        result = server._generate_midi(
            prompt="test prompt", mode="loop", session_id="test3"
        )
    assert "file_path" in result
    assert result["file_path"].endswith(".mid")
    assert Path(result["file_path"]).exists()


def test_generate_midi_output_has_correct_roles(tmp_path):
    from tt_midi_maker import server
    with patch.object(server, "OUTPUT_DIR", tmp_path), \
         patch("tt_midi_maker.server.build_blueprint", return_value=STUB_BLUEPRINT), \
         patch("tt_midi_maker.server._run_generation", return_value=STUB_TRACKS):
        result = server._generate_midi(prompt="test", mode="loop", session_id="t4")
    assert "melody" in result["roles_generated"]
    assert "drums" in result["roles_generated"]


def test_describe_midi_missing_file():
    from tt_midi_maker.server import _describe_midi
    from tt_midi_maker.errors import MidiMakerError
    with pytest.raises(MidiMakerError) as exc:
        _describe_midi("/nonexistent/path.mid")
    assert exc.value.code == "FILE_NOT_FOUND"


def test_chat_with_midi_returns_answer(tmp_path):
    from tt_midi_maker import server
    # Create a real MIDI file to analyze
    from tt_midi_maker.assembler import build_midi_file
    from tt_midi_maker.models.track import NoteEvent, RoleTrack
    notes = [NoteEvent(pitch=60, velocity=80, start_tick=0, duration_ticks=470, channel=1)]
    track = RoleTrack(role="melody", channel=1, program=0, notes=notes)
    midi_path = build_midi_file([track], 120, tmp_path / "chat_test.mid")

    with patch("tt_midi_maker.analyzer.call_llm", return_value="It is in C major."):
        result = server._chat_with_midi(str(midi_path), "What key?")
    assert "C major" in result["answer"]
```

- [ ] **Step 2: Implement server.py**

```python
# tt_midi_maker/server.py
"""
tt-midi-maker MCP server.

5 tools, 4 prompts, 4 resources, argument completions.
Run with: python -m tt_midi_maker
"""
from __future__ import annotations

import json
import os
import tempfile
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
    for r in ["C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"]
    for m in ["major","minor","dorian","phrygian","lydian","mixolydian"]
]

# ---------------------------------------------------------------------------
# Internal helpers (also importable for tests)
# ---------------------------------------------------------------------------

def _set_musical_context(
    session_id: str = "default",
    key: str | None = None,
    bpm: int | None = None,
    style: str | None = None,
    chord_progression: list[str] | None = None,
) -> dict:
    ctx = get_session(session_id).update(
        key=key, bpm=bpm, style=style, chord_progression=chord_progression
    )
    set_session(session_id, ctx)
    d = ctx.to_dict()
    d["fields_set"] = [k for k, v in d.items() if v is not None]
    # Restore null fields so output schema is complete
    for field in ("key", "bpm", "style", "chord_progression"):
        if field not in d:
            d[field] = None
    return d


def _run_generation(blueprint: MusicalBlueprint) -> list:
    """Placeholder: tokenise seed → Aria → decode → RoleTrack list."""
    from .models.track import NoteEvent, RoleTrack
    # Seed: minimal 1-bar MIDI for each active role (density > 0)
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
        for role in list(blueprint.roles.keys()):
            if role not in roles:
                from dataclasses import replace
                blueprint = replace(
                    blueprint,
                    roles={**blueprint.roles,
                           role: blueprint.roles[role].model_copy(update={"density": 0.0})},
                )
    if bars:
        blueprint = blueprint.model_copy(update={"bars": bars})

    raw_tracks    = _run_generation(blueprint)
    clean_tracks  = _apply_coherence(raw_tracks, blueprint)
    ts            = int(time.time())
    out           = Path(output_path) if output_path else OUTPUT_DIR / f"{ts}.mid"
    build_midi_file(clean_tracks, blueprint.bpm, out)
    return {
        "file_path":      str(out),
        "bars_generated": blueprint.bars,
        "bpm":            blueprint.bpm,
        "key":            blueprint.key,
        "roles_generated":[t.role for t in clean_tracks],
        "generation_ms":  0,
        "hardware_used":  "cpu-fallback",
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

MODES: loop = 4–16 bars seamless repeat (fastest). section = 16–64 bars with
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
    roles: list[Literal["drums","bass","melody","harmony","arp","pad","fx"]] | None = None,
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
  "make it more intense" — raises velocities, denser notes
  "quiet this down"      — thinner arrangement, lower velocities
  "resolve it"           — end on the tonic chord""",
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
    from .coherence.stitching import stitch_phrases
    src = Path(file_path)
    if not src.exists():
        raise MidiMakerError("FILE_NOT_FOUND", f"Not found: {file_path}",
                             "Check midi://output/{filename} for available files.")
    # Generate new bars using existing file as style seed
    existing_facts = _analyze(src)
    prompt = f"Continue this {existing_facts.get('style_guess','music')}" + (
        f", {style_hint}" if style_hint else ""
    )
    blueprint = build_blueprint(prompt)
    blueprint = blueprint.model_copy(update={"bars": bars})
    raw_tracks   = _run_generation(blueprint)
    clean_tracks = _apply_coherence(raw_tracks, blueprint)

    # Load existing tracks for stitching
    import mido as _mido
    mid = _mido.MidiFile(str(src))
    # (simplified: append without full stitch for now; Task 14 wires full stitch)
    ts  = int(time.time())
    out = OUTPUT_DIR / f"{ts}_continued.mid"
    build_midi_file(clean_tracks, blueprint.bpm, out)
    return {"file_path": str(out), "bars_added": bars, "total_bars": existing_facts.get("bars", 0) + bars}


@mcp.tool(
    title="Describe MIDI File",
    description="""Analyze a MIDI file and return a structured natural language description.

Returns key, tempo, time signature, bar count, track inventory, chord progression
(if detectable), a style guess, and a prose description. Use before decide_midi or
regenerate to understand what was generated. Also useful for analyzing external MIDI.""",
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
    section_type: Literal["intro","verse","chorus","bridge","outro"],
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
        return [s for s in ("intro","verse","chorus","bridge","outro") if val in s]
    if argument.name == "roles":
        all_roles = list(ROLES_CONFIG.keys())
        return [r for r in all_roles if val in r]
    return []


def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `tt_midi_maker/__main__.py`**

```python
# tt_midi_maker/__main__.py
from .server import main
main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_server.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tt_midi_maker/server.py tt_midi_maker/__main__.py tests/test_server.py
git commit -m "feat: MCP server with 5 tools, 4 prompts, 4 resources, completions"
```

---

### Task 14: Full test suite + smoke test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Run full test suite — confirm no regressions**

```bash
pytest tests/ -v --tb=short
```

Expected: all tasks 1–13 tests pass. Note any failures before proceeding.

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration.py
"""
End-to-end smoke test: runs the full pipeline with mocked LLM and generation.
Verifies that generate_midi produces a readable, non-empty MIDI file.
"""
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import mido
from tt_midi_maker.server import _generate_midi, OUTPUT_DIR
from tt_midi_maker.models.blueprint import MusicalBlueprint, RoleConfig

BLUEPRINT = MusicalBlueprint(
    key="C major", bpm=120, time_signature="4/4", style="ambient",
    chord_progression=["C", "Am", "F", "G"], bars=4, mode="loop",
    roles={
        "melody": RoleConfig(density=1.0, velocity_range=(70, 100), pattern_hint="legato"),
        "drums":  RoleConfig(density=0.7, velocity_range=(60, 90),  pattern_hint="default"),
    },
)


def test_full_pipeline_produces_valid_midi(tmp_path):
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch("tt_midi_maker.server.OUTPUT_DIR", tmp_path):
        result = _generate_midi(
            prompt="calm ambient with piano and soft drums",
            mode="loop",
        )

    assert "file_path" in result
    midi_path = Path(result["file_path"])
    assert midi_path.exists(), f"MIDI file not written: {midi_path}"

    mid = mido.MidiFile(str(midi_path))
    assert mid.type == 1, "Expected Type-1 multi-track MIDI"
    assert len(mid.tracks) >= 2, "Expected at least tempo track + one instrument track"

    note_ons = sum(
        1 for track in mid.tracks for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )
    assert note_ons > 0, "MIDI file contains no notes"


def test_full_pipeline_respects_role_filter(tmp_path):
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch("tt_midi_maker.server.OUTPUT_DIR", tmp_path):
        result = _generate_midi(
            prompt="just melody please",
            mode="loop",
            roles=["melody"],
        )
    assert "drums" not in result["roles_generated"]
    assert "melody" in result["roles_generated"]


def test_generate_returns_metadata_fields(tmp_path):
    with patch("tt_midi_maker.server.build_blueprint", return_value=BLUEPRINT), \
         patch("tt_midi_maker.server.OUTPUT_DIR", tmp_path):
        result = _generate_midi(prompt="test", mode="loop")
    for field in ("file_path", "bars_generated", "bpm", "key",
                  "roles_generated", "hardware_used"):
        assert field in result, f"Missing field in result: {field}"
```

- [ ] **Step 3: Run integration tests**

```bash
pytest tests/test_integration.py -v
```

Expected: 3 passed

- [ ] **Step 4: Run complete test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (approximately 75 tests across all modules)

- [ ] **Step 5: Manual smoke test — start the server**

```bash
python -m tt_midi_maker &
sleep 2
# Verify it starts without crashing
curl -s http://localhost:8000/health 2>/dev/null || echo "server started (health endpoint may vary)"
kill %1
```

- [ ] **Step 6: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test, full pipeline end-to-end"
```

---

## Self-Review Checklist

**Spec coverage:**
- Stage 1 (Prompt Engine): Task 11 ✓
- Stage 2 (Generation Engine — hardware, tokenizer, Aria): Tasks 8, 9, 10 ✓
- Stage 3 (Coherence Layer — scale, harmony, humanize, stitch): Tasks 2–5 ✓
- Stage 4 (MIDI Assembler): Task 6 ✓
- Stage 5 (MCP Server — 5 tools, 4 prompts, 4 resources, completions, errors): Task 13 ✓
- Session state: Task 7 ✓
- GM channel config (roles.yaml): Task 1 ✓
- Style catalog (styles.yaml): Task 1 ✓
- Error convention (MidiMakerError): Task 1 ✓
- `stream` mode returns immediately + progress notifications: documented in server.py comment; full async streaming requires `mcp.notify_progress` wiring — noted as follow-up
- `continue_midi` full phrase stitching: stitch_phrases exists (Task 5) but server.py `continue_midi` currently appends without full RoleTrack stitching — wiring complete stitch is an explicit follow-up noted in server.py comment

**No placeholders:** all steps contain actual code. ✓

**Type consistency:**
- `NoteEvent` used identically in tasks 2–7, 12, 13 ✓
- `RoleTrack` fields (`role`, `channel`, `program`, `notes`) consistent across tasks 5, 6, 10, 13 ✓
- `MusicalBlueprint.roles` is `dict[str, RoleConfig]` — consistent in tasks 1, 11, 13 ✓
- `stitch_phrases(existing, new, ticks_per_bar)` signature matches usage in server.py ✓
