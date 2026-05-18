# tt-midi-maker

Multi-track MIDI generation from text prompts using Tenstorrent hardware.
Exposes generation as a fully-featured MCP server with local audio playback.

## Pipeline
Prompt → LLM blueprint → skytnt/midi-model (LlamaModel, forge-compiled on TT hardware) → coherence layer → GM MIDI → MCP

## Generation backend
- `tt_midi_maker/generation/midi_backend.py` — orchestration; auto-selects hardware or CPU path
- `tt_midi_maker/generation/forge_backend.py` — forge.compile wrapper; compiles 12-layer net to TT P300C
- `tt_midi_maker/generation/skytnt_model.py` — vendored MIDIModel (Apache 2.0)
- `tt_midi_maker/generation/skytnt_tokenizer.py` — vendored MIDITokenizerV1/V2
- `tt_midi_maker/generation/hardware.py` — detect_tt_devices() via tt-smi

Hardware activation:
```bash
source /home/ttuser/tt-forge-fe/forge-venv/bin/activate
export PYTHONPATH="/home/ttuser/tt-forge-fe/third_party/tvm/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/home/ttuser/tt-forge-fe/third_party/tvm/build:$LD_LIBRARY_PATH"
```

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
pytest tests/ -v   # 178 tests
```

## Hardware generation performance (P300C, 4× chips)

`hw_context_interval` in `generate_from_blueprint` / `generate_hardware` controls how often
the hardware 12-layer net is called to refresh the context vector.  Between refreshes the CPU
3-layer net_token runs standalone.  Benchmarked on 4× P300C at 138 BPM / 8 bars (13.9s loop):

| hw_context_interval | max_events | time   | ev/s | loop ratio |
|---------------------|------------|--------|------|------------|
| 1 (every step)      | 64         | 35.3s  | 2    | 2.5×       |
| 4 (default)         | 96         | 12.3s  | 7.8  | 0.88×  ✓  |
| 8                   | 128        | 10.8s  | 11.9 | 0.77×  ✓  |

Recommended for real-time loop generation: `max_events=96, hw_context_interval=4` (fits in 1 loop).
The stride-mismatch bug that caused CPU fallback at max_events≥256 is also fixed.

## Source MIDI context
`generate_from_blueprint(bp, roles_config, source_midi=path, source_context_bars=8)` accepts a path to a previous MIDI file. The last N bars are tokenized and prepended to the prompt so the model can continue from or respond to existing material. Bad paths are silently ignored. `demo_postrock.py` uses this to chain patterns so each generation builds on the previous one.

## Streaming playback tools (MCP)
- `synth_start` — launch FluidSynth as ALSA sequencer server (call once per session)
- `loop_play(file)` — start looping immediately
- `loop_queue(file)` — queue next pattern; transitions at loop boundary with no gap
- `loop_stop(immediately?)` — stop after current loop or immediately with all-notes-off

## File playback tools (MCP)
- `play_midi(file, backend, port, channel_map, gain)` — one-shot playback
- `stop_playback(job_id)` — cancel a background playback job
- `list_midi_devices` — enumerate ALSA ports, soundfonts, active jobs, synth status
