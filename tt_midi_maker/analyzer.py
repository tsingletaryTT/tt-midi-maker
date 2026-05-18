import json
import os
from pathlib import Path
import mido
import httpx
from .errors import MidiMakerError

LLM_URL   = os.environ.get("MIDI_LLM_URL",   "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("MIDI_LLM_MODEL", "qwen3")


def call_llm(messages: list[dict]) -> str:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{LLM_URL}/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, "temperature": 0.3},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def extract_midi_facts(path: Path) -> dict:
    mid = mido.MidiFile(str(path))
    facts: dict = {
        "ticks_per_beat": mid.ticks_per_beat,
        "num_tracks": len(mid.tracks) - 1,
        "track_names": [],
        "channels_used": set(),
        "bpm": 120,
        "total_ticks": 0,
        "note_count": 0,
    }
    for track in mid.tracks:
        if track.name:
            facts["track_names"].append(track.name)
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                facts["bpm"] = int(60_000_000 / msg.tempo)
            if msg.type == "note_on" and msg.velocity > 0:
                facts["channels_used"].add(msg.channel + 1)
                facts["note_count"] += 1
                facts["total_ticks"] = max(facts["total_ticks"], tick)
    facts["channels_used"] = sorted(facts["channels_used"])
    facts["bars"] = facts["total_ticks"] // (facts["ticks_per_beat"] * 4)
    return facts


def describe_midi(path: Path) -> dict:
    if not path.exists():
        raise MidiMakerError(
            code="FILE_NOT_FOUND",
            message=f"MIDI file not found: {path}",
            suggestion="Check midi://output/{filename} for available files.",
        )
    facts = extract_midi_facts(path)
    messages = [
        {"role": "system", "content": "You are a music analyst. Given MIDI facts, write a concise 2-sentence musical description."},
        {"role": "user",   "content": json.dumps(facts, indent=2)},
    ]
    description = call_llm(messages)
    return {
        "key": "unknown",
        "tempo_bpm": facts["bpm"],
        "time_signature": "4/4",
        "bars": facts["bars"],
        "tracks": facts["track_names"],
        "chord_progression": [],
        "style_guess": "unknown",
        "description": description,
    }


def chat_about_midi(path: Path, question: str) -> dict:
    if not path.exists():
        raise MidiMakerError(
            code="FILE_NOT_FOUND",
            message=f"MIDI file not found: {path}",
            suggestion="Check midi://output/{filename} for available files.",
        )
    facts = extract_midi_facts(path)
    messages = [
        {"role": "system", "content": "You are an expert music analyst. Answer questions about MIDI files clearly and specifically."},
        {"role": "user",   "content": f"MIDI facts:\n{json.dumps(facts, indent=2)}\n\nQuestion: {question}"},
    ]
    return {"answer": call_llm(messages), "analysis_context": facts}
