# tt-midi-maker

Multi-track MIDI generation from text prompts using Tenstorrent hardware.
Exposes generation as a fully-featured MCP server with local audio playback.

## Pipeline
Prompt → LLM blueprint → skytnt/midi-model (LlamaModel, forge-compiled on TT hardware) → coherence layer → quality judge → GM MIDI → MCP

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

## Quality judge

`tt_midi_maker/coherence/judge.py` — rule-based quality filter applied after generation.

**Rule-based metrics per track:**
- `notes_per_bar`: density — flags sparse (<0.5 npb) or machine-gun (>30 npb)
- `pitch_span`: semitone range — melody/bass flagged if <4 (monotonous) or >36 (scattered)
- `unique_pitches`: distinct notes — melody flagged if <3
- `mean_interval` / `max_interval`: average and max semitone jumps in melody — >8 mean or >24 max flagged
- `direction_reversal_ratio`: fraction of intervals that reverse direction — >75% = melodic zigzag
- `silence_ratio`: fraction of loop with no notes — <8% = no breathing room, >96% = nearly silent
- `cluster_ratio`: fraction of notes landing within 24 ticks (half-beat) of another — >65% = rhythmic pile-up
- `register_overlap_semitones`: how many semitones bass top invades melody bottom — >6 flagged

**Scoring:** `rule_score = max(0, 1 - 0.12 * n_issues)`. A pattern with 4 issues scores 0.52 (FAIL).

**Re-rolling** is wired into `generate_from_blueprint` via `max_attempts` and `judge_threshold` params.
All demo scripts set `max_attempts=3, judge_threshold=0.55` — the model re-rolls up to 3 times and keeps
the best-scoring attempt. Adds 1–2 extra generation passes when needed (~20–40s on hardware).

**Perplexity scoring** (`score_perplexity()` in judge.py): two batched forward passes through the model —
`model.forward()` then `model.forward_token()` — returns mean NLL of event-type tokens.
Lower = model more expected the sequence. Used by `scripts/analyze_quality.py --no-ppl` off.

**One-time audit:**
```bash
python scripts/analyze_quality.py --no-ppl   # fast rule-based only (no model loading)
python scripts/analyze_quality.py            # full analysis including perplexity
```
Writes JSON to `docs/quality_report.json`. Audit of 22 existing patterns: 19/22 passed (86%).
Most common issues: rhythmic clustering, melodic zigzag direction reversals, silence_ratio=0%.
Single-voice monosynth patterns scored best (2× perfect 1.00, 1× 0.88 for one jarring leap).

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
