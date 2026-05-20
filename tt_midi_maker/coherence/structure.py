"""Genre structure layer — deterministic skeleton for each musical style.

Provides:
  GenreStructure          — dataclass holding resolved structure for one generation
  build_genre_structure() — reads genres.yaml, resolves roman numeral chords
  generate_walking_bass() — 4 quarter notes per bar (root, 3rd, 5th, approach)
  generate_drum_groove()  — deterministic drum patterns (shuffle, swing_ride, straight)
  enforce_phrase_gaps()   — thin melody in "response" bars for call-response phrasing
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..models.track import NoteEvent
from .harmony import ROOT_NAMES, parse_chord

# ── Roman numeral → semitone offset ──────────────────────────────────────────

_ROMAN_INTERVALS: dict[str, int] = {
    "I": 0, "II": 2, "III": 4, "IV": 5,
    "V": 7, "VI": 9, "VII": 11,
}
_PC_TO_NAME: dict[int, str] = {
    0: "C", 1: "C#", 2: "D",  3: "Eb", 4: "E",  5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A",  10: "Bb", 11: "B",
}


def _chord_root_pc(chord_name: str) -> int:
    """Extract pitch class of root from chord name: "D7" -> 2, "Eb7" -> 3."""
    for root_len in (2, 1):
        candidate = chord_name[:root_len]
        if candidate in ROOT_NAMES:
            return ROOT_NAMES[candidate]
    raise ValueError(f"Cannot find root in chord: {chord_name!r}")


def _roman_to_chord(roman: str, key_root_pc: int) -> str:
    """Resolve a roman numeral chord symbol to a concrete chord name.

    Example: _roman_to_chord("IV7", 9) -> "D7"   (A is root, IV = D)

    Tries longest numeral match first so "IV" doesn't match "I".
    The quality suffix (everything after the numeral) is appended unchanged.
    """
    upper = roman.upper()
    for numeral in ("VII", "VI", "IV", "III", "II", "I", "V"):
        if upper.startswith(numeral):
            quality = roman[len(numeral):]
            semitones = _ROMAN_INTERVALS[numeral]
            chord_root = (key_root_pc + semitones) % 12
            return _PC_TO_NAME[chord_root] + quality
    raise ValueError(f"Cannot parse roman numeral: {roman!r}")


# ── GenreStructure ────────────────────────────────────────────────────────────

@dataclass
class GenreStructure:
    """Fully resolved structural description for one generation pass."""
    genre: str
    bars: int
    chord_progression: list[str]      # resolved chord names, length == bars
    call_bars: list[int]              # 1-indexed bar numbers for "call" phrases
    response_bars: list[int]          # 1-indexed bar numbers for "response" gaps
    walking_bass: bool
    swing_ratio: float                # 0.0 = straight, 0.67 = shuffle
    drum_groove: str                  # "shuffle" | "swing_ride" | "straight" | "none"
    tension: float = 0.0              # 0.0–1.0, improv layer density


_GENRES_YAML = Path(__file__).parent.parent.parent / "config" / "genres.yaml"


def build_genre_structure(
    genre: str,
    key: str,                          # e.g. "A minor"
    chord_progression: list[str],      # from blueprint (used if template is null)
    bars: int,
    tension: float = 0.0,
) -> GenreStructure:
    """Load genre config and resolve roman numeral chord template.

    If the genre's chord_template is null, the passed chord_progression is
    cycled to fill `bars` bars.  If chord_template is a list of roman numerals,
    they are resolved against the key root.
    """
    genres_cfg = yaml.safe_load(_GENRES_YAML.read_text())["genres"]
    cfg = genres_cfg.get(genre, {})

    # Parse key root (e.g. "A minor" -> root_pc=9)
    root_name = key.split()[0]
    key_root_pc = ROOT_NAMES.get(root_name, 0)

    template = cfg.get("chord_template")
    if template:
        resolved = [_roman_to_chord(ch, key_root_pc) for ch in template]
    else:
        resolved = [chord_progression[i % len(chord_progression)] for i in range(bars)]

    return GenreStructure(
        genre=genre,
        bars=bars,
        chord_progression=resolved,
        call_bars=cfg.get("call_bars", []),
        response_bars=cfg.get("response_bars", []),
        walking_bass=cfg.get("walking_bass", False),
        swing_ratio=cfg.get("swing_ratio", 0.0),
        drum_groove=cfg.get("drum_groove", "none"),
        tension=tension,
    )


# ── Walking bass generator ────────────────────────────────────────────────────

_BASS_RANGE = (33, 57)  # A1 – A3 in MIDI


def _nearest_in_range(pitch: int, lo: int, hi: int) -> int:
    """Transpose pitch into [lo, hi] by octave shifts."""
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def generate_walking_bass(
    chord_progression: list[str],
    bars: int,
    ticks_per_beat: int = 480,
    velocity: int = 78,
    channel: int = 2,
    bass_range: tuple[int, int] = _BASS_RANGE,
) -> list[NoteEvent]:
    """Generate a deterministic walking bass line — 4 quarter notes per bar.

    Beat layout per bar:
      1: chord root
      2: chord 3rd (first non-root chord tone by pitch class)
      3: chord 5th (second non-root chord tone)
      4: chromatic approach (half-step below next bar's root)
    """
    notes: list[NoteEvent] = []
    ticks_per_bar = ticks_per_beat * 4
    note_dur = ticks_per_beat - 30  # slight gap for articulation feel

    for bar_idx in range(bars):
        chord_name = chord_progression[bar_idx % len(chord_progression)]
        next_name  = chord_progression[(bar_idx + 1) % len(chord_progression)]

        root_pc   = _chord_root_pc(chord_name)
        chord_pcs = sorted(parse_chord(chord_name))  # sorted pitch classes

        # Build bass pitches, anchored around octave 3 (MIDI 48) then clamped into range
        root     = _nearest_in_range(root_pc + 48, *bass_range)
        third    = _nearest_in_range((chord_pcs[1] if len(chord_pcs) > 1 else chord_pcs[0]) + 48, *bass_range)
        fifth    = _nearest_in_range((chord_pcs[2] if len(chord_pcs) > 2 else chord_pcs[0]) + 48, *bass_range)
        next_root_pc = _chord_root_pc(next_name)
        approach = _nearest_in_range(next_root_pc + 48 - 1, *bass_range)  # half-step below next root

        bar_start = bar_idx * ticks_per_bar
        for beat, pitch in enumerate([root, third, fifth, approach]):
            # Accent beat 1 slightly more than walking beats
            vel = velocity if beat == 0 else velocity - (beat * 2)
            notes.append(NoteEvent(
                pitch=pitch,
                velocity=max(40, vel),
                start_tick=bar_start + beat * ticks_per_beat,
                duration_ticks=note_dur,
                channel=channel,
            ))

    return notes


# ── Drum groove generator ─────────────────────────────────────────────────────

# GM Standard Kit pitches
_KICK   = 36   # Bass Drum 1
_SNARE  = 38   # Acoustic Snare
_HIHAT  = 42   # Closed Hi-Hat
_RIDE   = 51   # Ride Cymbal 1
_HIHAT2 = 44   # Pedal Hi-Hat (backbeat for swing)
_DRUM_CHANNEL = 10  # 1-indexed; drums always ch10 in MIDI


def generate_drum_groove(
    groove_type: str,
    bars: int,
    ticks_per_beat: int = 480,
    velocity_kick: int = 95,
    velocity_snare: int = 82,
    velocity_hat: int = 58,
) -> list[NoteEvent]:
    """Generate a deterministic drum pattern for `bars` bars.

    groove_type:
      "shuffle"    — blues triplet shuffle (kick 1+3, snare 2+4, triplet hi-hat)
      "swing_ride" — jazz swing (kick beat-1, snare 2+4, ride all beats+upbeats)
      "straight"   — basic 4/4 (kick 1+3, snare 2+4, 8th-note hi-hat)
      "none"       — returns empty list
    """
    if groove_type == "none":
        return []

    notes: list[NoteEvent] = []
    tpb   = ticks_per_beat
    tpbar = tpb * 4
    triplet = tpb * 2 // 3   # 320 ticks at 480 tpb — swung off-beat position

    for bar in range(bars):
        b0 = bar * tpbar

        if groove_type == "shuffle":
            # Kick: beats 1 and 3
            for beat in (0, 2):
                notes.append(NoteEvent(_KICK, velocity_kick, b0 + beat * tpb, tpb // 4, _DRUM_CHANNEL))
            # Snare: beats 2 and 4
            for beat in (1, 3):
                notes.append(NoteEvent(_SNARE, velocity_snare, b0 + beat * tpb, tpb // 4, _DRUM_CHANNEL))
            # Hi-hat: shuffle triplet pattern (straight 8th + swung 8th per beat)
            for beat in range(4):
                for offset in (0, triplet):
                    notes.append(NoteEvent(_HIHAT, velocity_hat, b0 + beat * tpb + offset, tpb // 8, _DRUM_CHANNEL))

        elif groove_type == "swing_ride":
            # Kick: beat 1 only (jazz feel — lighter than blues)
            notes.append(NoteEvent(_KICK, velocity_kick, b0, tpb // 4, _DRUM_CHANNEL))
            # Snare: beats 2 and 4 (brushed — lower velocity)
            for beat in (1, 3):
                notes.append(NoteEvent(_SNARE, velocity_snare - 15, b0 + beat * tpb, tpb // 4, _DRUM_CHANNEL))
            # Ride: all 4 beats + swung off-beats (0 and triplet per beat)
            for beat in range(4):
                for offset in (0, triplet):
                    notes.append(NoteEvent(_RIDE, velocity_hat, b0 + beat * tpb + offset, tpb // 8, _DRUM_CHANNEL))
            # Pedal hi-hat: beats 2 and 4 (classic jazz hi-hat foot)
            for beat in (1, 3):
                notes.append(NoteEvent(_HIHAT2, velocity_hat + 5, b0 + beat * tpb, tpb // 6, _DRUM_CHANNEL))

        elif groove_type == "straight":
            for beat in (0, 2):
                notes.append(NoteEvent(_KICK, velocity_kick, b0 + beat * tpb, tpb // 4, _DRUM_CHANNEL))
            for beat in (1, 3):
                notes.append(NoteEvent(_SNARE, velocity_snare, b0 + beat * tpb, tpb // 4, _DRUM_CHANNEL))
            for beat in range(8):
                notes.append(NoteEvent(_HIHAT, velocity_hat, b0 + beat * (tpb // 2), tpb // 6, _DRUM_CHANNEL))

    return notes


# ── Phrase gap enforcement ────────────────────────────────────────────────────

def enforce_phrase_gaps(
    notes: list[NoteEvent],
    response_bars: list[int],          # 1-indexed
    ticks_per_beat: int = 480,
    gap_start_beat: int = 2,           # keep only notes before this beat in response bars
) -> list[NoteEvent]:
    """Remove notes at or after beat `gap_start_beat` in response bars.

    Creates call-response phrasing: call bars have full melody, response bars
    have only a brief downbeat "answer" note then silence before the next call.
    gap_start_beat=2 (default) keeps only beat 1 of each response bar.
    """
    if not response_bars:
        return notes
    ticks_per_bar = ticks_per_beat * 4
    gap_offset = (gap_start_beat - 1) * ticks_per_beat  # ticks from bar start
    result = []
    for note in notes:
        bar_1indexed = note.start_tick // ticks_per_bar + 1
        if bar_1indexed in response_bars:
            offset_in_bar = note.start_tick % ticks_per_bar
            if offset_in_bar < gap_offset + ticks_per_beat:
                result.append(note)
            # else: drop — creates the silence gap
        else:
            result.append(note)
    return result
