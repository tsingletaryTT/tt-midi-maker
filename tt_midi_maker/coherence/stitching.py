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
