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
