import json
import pytest
from unittest.mock import patch
from tt_midi_maker.prompt_engine import parse_blueprint, build_blueprint, SYSTEM_PROMPT
from tt_midi_maker.errors import MidiMakerError
from tt_midi_maker.session import MusicalContext

VALID_BLUEPRINT_JSON = json.dumps({
    "key": "D minor", "bpm": 120, "time_signature": "4/4",
    "style": "bossa nova", "chord_progression": ["Dm", "Gm", "A7", "Dm"],
    "bars": 8, "mode": "loop",
    "roles": {
        "drums":  {"density": 0.7, "velocity_range": [60, 90], "pattern_hint": "bossa"},
        "bass":   {"density": 0.8, "velocity_range": [70, 100], "pattern_hint": "walking"},
        "melody": {"density": 1.0, "velocity_range": [80, 110], "pattern_hint": "legato"},
    },
})


def test_parse_blueprint_valid_json():
    bp = parse_blueprint(VALID_BLUEPRINT_JSON)
    assert bp.key == "D minor"
    assert bp.bpm == 120


def test_parse_blueprint_strips_markdown_fence():
    wrapped = f"```json\n{VALID_BLUEPRINT_JSON}\n```"
    bp = parse_blueprint(wrapped)
    assert bp.bpm == 120


def test_parse_blueprint_invalid_json_raises():
    with pytest.raises(MidiMakerError) as exc_info:
        parse_blueprint("not json at all")
    assert exc_info.value.code == "CONTEXT_NOT_SET"


def test_system_prompt_contains_schema():
    assert '"key"' in SYSTEM_PROMPT
    assert '"bpm"' in SYSTEM_PROMPT
    assert '"roles"' in SYSTEM_PROMPT


def test_build_blueprint_calls_llm():
    with patch("tt_midi_maker.prompt_engine.call_llm",
               return_value=VALID_BLUEPRINT_JSON) as mock_llm:
        bp = build_blueprint("dreamy lo-fi")
    mock_llm.assert_called_once()
    assert bp.style == "bossa nova"


def test_build_blueprint_passes_context_to_llm():
    ctx = MusicalContext(key="C major", bpm=90)
    with patch("tt_midi_maker.prompt_engine.call_llm",
               return_value=VALID_BLUEPRINT_JSON) as mock_llm:
        build_blueprint("something jazzy", context=ctx)
    call_args = mock_llm.call_args[0][0]   # first positional arg = messages list
    user_msg = call_args[1]["content"]
    assert "C major" in user_msg
