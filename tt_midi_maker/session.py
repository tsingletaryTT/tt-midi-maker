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
