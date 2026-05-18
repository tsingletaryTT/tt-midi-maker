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
    """Parse "D minor" -> (2, "minor"). Raises ValueError if unrecognised."""
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
