import random
from dataclasses import replace
from ..models.track import NoteEvent


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
