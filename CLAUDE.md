# tt-midi-maker

Multi-track MIDI generation from text prompts using Tenstorrent hardware.
Exposes generation as a fully-featured MCP server.

## Pipeline
Prompt → LLM blueprint → Aria MIDI transformer (tt-forge) → coherence layer → GM MIDI → MCP

## Key files
- `tt_midi_maker/server.py` — MCP server entry point
- `tt_midi_maker/models/blueprint.py` — MusicalBlueprint Pydantic model
- `tt_midi_maker/coherence/` — music theory passes (scale, harmony, humanize, stitch)
- `config/roles.yaml` — GM channel assignments

## Running
```bash
python -m tt_midi_maker          # start MCP server
MIDI_LLM_URL=http://localhost:8000/v1   # LLM endpoint env var
```

## Testing
```bash
pytest tests/ -v
```
