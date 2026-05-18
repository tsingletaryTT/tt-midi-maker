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
