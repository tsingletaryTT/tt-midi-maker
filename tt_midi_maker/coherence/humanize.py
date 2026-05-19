import random
from dataclasses import replace
from ..models.track import NoteEvent

TICKS_PER_BEAT = 480

# Per-role velocity windows.  Mirrors a typical mix hierarchy:
#   melody > drums > harmony > bass (ceiling order)
# scale_velocity_by_role() maps [1,127] → [lo, hi] before humanize_velocities()
# applies jitter, so the jitter pass refines within the already-shaped window.
_ROLE_VELOCITY_RANGES: dict[str, tuple[int, int]] = {
    "melody":  (80, 110),   # lead voice — prominent in the mix
    "harmony": (55,  80),   # supporting pads/comps — below melody
    "bass":    (50,  75),   # anchoring, steady, below harmony
    "drums":   (60, 100),   # wide dynamic range kept as-is
}


def scale_velocity_by_role(
    notes: list[NoteEvent],
    role: str,
    ranges: dict | None = None,
) -> list[NoteEvent]:
    """Remap note velocities into a role-appropriate window.

    Normalizes the input velocity (1–127) to a [0,1] fraction then maps it
    into [lo, hi] for the given role. Call this before humanize_velocities()
    so the jitter pass refines within the already-shaped window.

    Args:
        notes:  Input NoteEvent list (may be empty).
        role:   Role name — one of 'melody', 'harmony', 'bass', 'drums'.
                Unknown roles fall back to the default range (60, 100).
        ranges: Optional override dict mapping role → (lo, hi).  When
                provided, replaces _ROLE_VELOCITY_RANGES entirely.

    Returns:
        New list of NoteEvents with remapped velocities; original list is
        not mutated.
    """
    if not notes:
        return []
    velocity_ranges = ranges or _ROLE_VELOCITY_RANGES
    lo, hi = velocity_ranges.get(role, (60, 100))
    result = []
    for note in notes:
        frac = (note.velocity - 1) / 126.0          # 0.0 at vel=1, 1.0 at vel=127
        new_vel = int(lo + frac * (hi - lo))
        result.append(replace(note, velocity=max(1, min(127, new_vel))))
    return result


def humanize_velocities(
    notes: list[NoteEvent],
    variation: int = 8,
    phrase_contour: bool = True,
) -> list[NoteEvent]:
    """Add +-variation velocity jitter. Drums capped at +-4. Optional phrase contour."""
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
    """Add micro-timing jitter +-max_ticks to non-drum notes."""
    if not notes:
        return notes
    return [
        note if note.channel == 10
        else replace(note, start_tick=max(0, note.start_tick + random.randint(-max_ticks, max_ticks)))
        for note in notes
    ]


def swing_timing(
    notes: list[NoteEvent],
    swing_ratio: float = 0.67,
    ticks_per_beat: int = TICKS_PER_BEAT,
) -> list[NoteEvent]:
    """Shift off-beat eighth notes to create swing feel. Drums are skipped.

    In straight time, off-beat eighths land at beat + ticks_per_beat/2 (beat/2).
    In swing, they land at beat + ticks_per_beat * swing_ratio.

    swing_ratio=0.67 → triplet feel (2:1 long-short), classic jazz/blues shuffle.
    swing_ratio=0.63 → medium swing, bebop/Latin.
    swing_ratio=0.50 → straight (no effect).

    Notes within ±tol ticks of the straight off-beat position are snapped to
    the swung position. All other notes are left untouched.
    """
    if not notes or swing_ratio <= 0.5:
        return notes

    eighth      = ticks_per_beat // 2                   # 240: straight off-beat
    swung       = int(ticks_per_beat * swing_ratio)     # 320 at 0.67
    tol         = eighth // 3                           # ±80 ticks = "near off-beat"

    result = []
    for note in notes:
        if note.channel == 10:
            result.append(note)
            continue
        beat_phase = note.start_tick % ticks_per_beat
        if abs(beat_phase - eighth) <= tol:
            beat_base = note.start_tick - beat_phase
            result.append(replace(note, start_tick=max(0, beat_base + swung)))
        else:
            result.append(note)
    return result
