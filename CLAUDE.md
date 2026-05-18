# tt-midi-maker

Multi-track MIDI generation from text prompts using Tenstorrent hardware.
Exposes generation as a fully-featured MCP server with local audio playback.

## Pipeline
Prompt → LLM blueprint → Aria MIDI transformer (tt-forge) → coherence layer → GM MIDI → MCP

## Key files
- `tt_midi_maker/server.py` — MCP server entry point (12 tools, 4 prompts, 4 resources)
- `tt_midi_maker/stream_player.py` — real-time loop player (FluidSynthServer + LoopPlayer)
- `tt_midi_maker/player.py` — file-based playback (fluidsynth subprocess or ALSA port)
- `tt_midi_maker/models/blueprint.py` — MusicalBlueprint Pydantic model
- `tt_midi_maker/coherence/` — music theory passes (scale, harmony, humanize, stitch)
- `config/roles.yaml` — GM channel assignments

## Running
```bash
sudo apt install fluidsynth fluid-soundfont-gm   # required for local audio
python -m tt_midi_maker                          # start MCP server
MIDI_LLM_URL=http://localhost:8000/v1            # LLM endpoint env var
```

## Testing
```bash
pytest tests/ -v   # 131 tests
```

## Streaming playback tools (MCP)
- `synth_start` — launch FluidSynth as ALSA sequencer server (call once per session)
- `loop_play(file)` — start looping immediately
- `loop_queue(file)` — queue next pattern; transitions at loop boundary with no gap
- `loop_stop(immediately?)` — stop after current loop or immediately with all-notes-off

## File playback tools (MCP)
- `play_midi(file, backend, port, channel_map, gain)` — one-shot playback
- `stop_playback(job_id)` — cancel a background playback job
- `list_midi_devices` — enumerate ALSA ports, soundfonts, active jobs, synth status
