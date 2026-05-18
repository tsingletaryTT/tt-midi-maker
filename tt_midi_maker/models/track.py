from dataclasses import dataclass, field


@dataclass
class NoteEvent:
    pitch: int           # MIDI pitch 0-127
    velocity: int        # 1-127
    start_tick: int      # absolute tick offset from phrase start
    duration_ticks: int
    channel: int         # GM channel (1-indexed; drums = 10)


@dataclass
class RoleTrack:
    role: str            # "melody", "bass", "drums", etc.
    channel: int         # GM channel (1-indexed)
    program: int         # GM program number (ignored for drums)
    notes: list[NoteEvent] = field(default_factory=list)
