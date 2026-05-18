import json
import os
import httpx
from .models.blueprint import MusicalBlueprint
from .session import MusicalContext
from .errors import MidiMakerError

LLM_URL   = os.environ.get("MIDI_LLM_URL",   "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("MIDI_LLM_MODEL", "qwen3")

SYSTEM_PROMPT = """\
You are a music composition assistant. Convert a text description into a
structured JSON blueprint for a MIDI generator.

Return ONLY valid JSON with this exact schema:
{
  "key": "D minor",
  "bpm": 120,
  "time_signature": "4/4",
  "style": "bossa nova",
  "chord_progression": ["Dm", "Gm", "A7", "Dm"],
  "bars": 8,
  "mode": "loop",
  "roles": {
    "drums":   {"density": 0.7, "velocity_range": [60, 90],  "pattern_hint": "bossa"},
    "bass":    {"density": 0.8, "velocity_range": [70, 100], "pattern_hint": "walking"},
    "melody":  {"density": 1.0, "velocity_range": [80, 110], "pattern_hint": "legato"},
    "harmony": {"density": 0.0, "velocity_range": [50, 80],  "pattern_hint": "default"}
  }
}
Set density 0.0 for roles that don't suit the style. No extra keys or prose.\
"""


def call_llm(messages: list[dict]) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{LLM_URL}/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def parse_blueprint(llm_output: str) -> MusicalBlueprint:
    text = llm_output.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return MusicalBlueprint(**json.loads(text))
    except Exception as e:
        raise MidiMakerError(
            code="CONTEXT_NOT_SET",
            message=f"Could not parse musical blueprint: {e}",
            suggestion="Call set_musical_context with explicit key and style.",
        )


def build_blueprint(
    prompt: str,
    context: MusicalContext | None = None,
) -> MusicalBlueprint:
    ctx_str = ""
    if context and not context.is_empty():
        ctx_str = (f"\nSession context (use exactly, do not override): "
                   f"{json.dumps(context.to_dict())}")
    user_content = f"Convert this to a MIDI blueprint:{ctx_str}\n\nPrompt: {prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return parse_blueprint(call_llm(messages))
