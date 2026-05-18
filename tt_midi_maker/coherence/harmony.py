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
    """Parse "Dm", "G7", "Cmaj7" -> frozenset of pitch classes."""
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
